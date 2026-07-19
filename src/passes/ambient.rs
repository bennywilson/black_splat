//! Screen-space ambient occlusion + a cheap single-bounce screen-space
//! diffuse GI (`engine_assets/shaders/ambient_probe.wgsl`), read by
//! `LightingPass`'s skylight shader. Runs once per frame, right after
//! `GBufferPass` and before `LightingPass`, on the G-buffer that
//! `GBufferPass` just wrote. See `ambient_probe.wgsl`'s header comment for
//! the technique (same-frame approximation, no history buffer).

use wgpu::{
    util::DeviceExt, BindGroupLayoutEntry, BindingType, SamplerBindingType, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use crate::{assets::*, game_object::*, log, resource::*};

pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub const GI_FORMAT: wgpu::TextureFormat = SCENE_COLOR_FORMAT;

/// Toggles for `AmbientPass`'s two screen-space effects. Both default on;
/// projects that only want the static baked skylight cubemap (no local
/// reactive ambient) can turn either or both off. The pass still runs every
/// frame either way -- disabling just short-circuits the per-pixel work in
/// `ambient_probe.wgsl` and clears the corresponding output to a neutral
/// value (AO 1.0 = unoccluded, GI 0.0 = no bounce), rather than skipping the
/// render pass, since `LightingPass` always samples both textures.
#[derive(Clone, Copy, Debug)]
pub struct AmbientSettings {
    pub ssao_enabled: bool,
    pub ssgi_enabled: bool,
    // Straight multiplier on the GI hit contribution before it's written to
    // gi_texture (default 1.0). GI's hit color is unlit albedo * facing --
    // it doesn't know whether the hit surface is actually in shadow, so it
    // can wash out dynamic shadows/AO with false bounce light. Turning this
    // down (rather than the GI toggle) keeps some indirect bounce while
    // taming that.
    pub gi_intensity: f32,
    // How many of AO_KERNEL's 12 taps ambient_probe.wgsl spends per pixel.
    // Clamped to [1, 12] before upload -- more taps = less noise, more cost.
    pub ao_samples: u32,
    // How many hemisphere rays ambient_probe.wgsl traces per pixel for GI.
    // Clamped to >= 1 before upload -- more rays = less noise, more cost.
    pub gi_samples: u32,
    // Cross-bilateral denoise controls (ambient_denoise.wgsl). Radius is taps
    // per side, clamped to [0, 4] (0 disables spatial blur for that pass);
    // strength scales the depth edge-stopping tolerance, so higher values
    // let the blur bleed across bigger depth discontinuities.
    pub denoise_radius: u32,
    pub denoise_strength: f32,
    // How many times the horizontal+vertical blur pair runs, each pass
    // feeding off the previous pass's output. Clamped to [1, 3].
    pub denoise_iterations: u32,
}

impl Default for AmbientSettings {
    fn default() -> Self {
        Self {
            ssao_enabled: true,
            ssgi_enabled: true,
            gi_intensity: 1.0,
            ao_samples: 12,
            gi_samples: 4,
            denoise_radius: 4,
            denoise_strength: 1.0,
            denoise_iterations: 1,
        }
    }
}

crate::editor_properties!(AmbientSettings {
    ssao_enabled: bool("Screen-Space AO"),
    ssgi_enabled: bool("Screen-Space GI"),
    gi_intensity: float("GI Intensity"),
    ao_samples: int("AO Samples"),
    gi_samples: int("GI Samples"),
    denoise_radius: int("Denoise Radius"),
    denoise_strength: float("Denoise Strength"),
    denoise_iterations: int("Denoise Iterations"),
});

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DenoiseUniform {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    // xy render target size in pixels, zw texel step direction (1,0 or 0,1).
    params: [f32; 4],
    // x: blur radius in taps, y: depth edge-tolerance multiplier, zw unused.
    blur_params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AmbientUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    // xyz camera world position, w: 1.0 if a baked skylight is feeding the
    // GI miss-fallback this frame.
    camera_pos: [f32; 4],
    // xy render target size in pixels, z: ssao_enabled, w: ssgi_enabled
    // (see AmbientSettings).
    target_dims: [f32; 4],
    // x: gi_intensity, y: ao_samples, z: gi_samples, w unused.
    gi_params: [f32; 4],
}

pub struct AmbientPass {
    pipeline: wgpu::RenderPipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: wgpu::BindGroup,
    env_bind_group_layout: wgpu::BindGroupLayout,
    // Bound in place of a real skylight bake before one exists -- same role
    // as LightingPass::fallback_cube.
    fallback_cube: CubeTexture,
    // Cross-bilateral denoise (ambient_denoise.wgsl), run as two separable
    // passes over the raw probe output below. See DenoisePass docs there.
    denoise_pipeline: wgpu::RenderPipeline,
    denoise_input_bind_group_layout: wgpu::BindGroupLayout,
    denoise_uniform_bind_group_layout: wgpu::BindGroupLayout,
    // fs_main's raw, noisy AO/GI output -- input to the horizontal blur pass.
    raw_ao_texture: Texture,
    raw_gi_texture: Texture,
    // Horizontal blur output / vertical blur input.
    tmp_ao_texture: Texture,
    tmp_gi_texture: Texture,
    // Final, denoised AO/GI -- what LightingPass actually samples.
    pub ao_texture: Texture,
    pub gi_texture: Texture,
    pub settings: AmbientSettings,
}

impl AmbientPass {
    fn make_gbuffer_bind_group(
        device_resources: &DeviceResources,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        device_resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &device_resources.gbuffer_textures[0].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &device_resources.gbuffer_textures[1].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &device_resources.gbuffer_textures[2].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &device_resources.render_textures[1].view,
                        ),
                    },
                ],
                label: Some("AmbientPass_gbuffer_bind_group"),
            })
    }

    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating AmbientPass");
        let device = &device_resources.device;

        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Depth,
                        },
                        count: None,
                    },
                ],
                label: Some("AmbientPass_gbuffer_bind_group_layout"),
            });
        let gbuffer_bind_group =
            Self::make_gbuffer_bind_group(device_resources, &gbuffer_bind_group_layout);

        let env_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::Cube,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("AmbientPass_env_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("AmbientPass_pipeline_layout"),
            bind_group_layouts: &[Some(&gbuffer_bind_group_layout), Some(&env_bind_group_layout)],
            immediate_size: 0,
        });

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/ambient_probe.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("AmbientPass_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: AO_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GI_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let fallback_cube = CubeTexture::new(device, SCENE_COLOR_FORMAT, 1);
        let size = device_resources.render_textures[1].texture.size();
        let make_ao = || {
            Texture::new_render_texture_with_format(device, AO_FORMAT, size.width, size.height)
                .unwrap()
        };
        let make_gi = || {
            Texture::new_render_texture_with_format(device, GI_FORMAT, size.width, size.height)
                .unwrap()
        };
        let raw_ao_texture = make_ao();
        let raw_gi_texture = make_gi();
        let tmp_ao_texture = make_ao();
        let tmp_gi_texture = make_gi();
        let ao_texture = make_ao();
        let gi_texture = make_gi();

        // Denoise: takes t_ao/t_gi (whichever generation is the input for
        // this pass) plus the G-buffer normal/depth as edge-stopping guides.
        let denoise_input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Depth,
                        },
                        count: None,
                    },
                ],
                label: Some("AmbientPass_denoise_input_bind_group_layout"),
            });
        let denoise_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("AmbientPass_denoise_uniform_bind_group_layout"),
            });
        let denoise_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("AmbientPass_denoise_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&denoise_input_bind_group_layout),
                    Some(&denoise_uniform_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let denoise_shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/ambient_denoise.wgsl", device_resources)
            .await;
        let denoise_shader = asset_manager.get_shader(&denoise_shader_handle);
        let denoise_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("AmbientPass_denoise_pipeline"),
            layout: Some(&denoise_pipeline_layout),
            vertex: wgpu::VertexState {
                module: denoise_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: denoise_shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: AO_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GI_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        AmbientPass {
            pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group,
            env_bind_group_layout,
            fallback_cube,
            denoise_pipeline,
            denoise_input_bind_group_layout,
            denoise_uniform_bind_group_layout,
            raw_ao_texture,
            raw_gi_texture,
            tmp_ao_texture,
            tmp_gi_texture,
            ao_texture,
            gi_texture,
            settings: AmbientSettings::default(),
        }
    }

    /// Rebuilds the AO/GI targets and the (recreated) G-buffer bind group
    /// after a window resize.
    pub fn resize(&mut self, device_resources: &DeviceResources) {
        let device = &device_resources.device;
        let size = device_resources.render_textures[1].texture.size();
        let make_ao = || {
            Texture::new_render_texture_with_format(device, AO_FORMAT, size.width, size.height)
                .unwrap()
        };
        let make_gi = || {
            Texture::new_render_texture_with_format(device, GI_FORMAT, size.width, size.height)
                .unwrap()
        };
        self.raw_ao_texture = make_ao();
        self.raw_gi_texture = make_gi();
        self.tmp_ao_texture = make_ao();
        self.tmp_gi_texture = make_gi();
        self.ao_texture = make_ao();
        self.gi_texture = make_gi();
        self.gbuffer_bind_group = Self::make_gbuffer_bind_group(
            device_resources,
            &self.gbuffer_bind_group_layout,
        );
    }

    /// Writes this frame's AO + GI targets from the G-buffer `GBufferPass`
    /// just filled. `lights`/`env_cubemaps` are only consulted to find the
    /// lowest-id baked skylight, whose SH irradiance feeds GI rays that miss
    /// all on-screen geometry; with no baked skylight, misses contribute
    /// zero.
    pub fn render(
        &mut self,
        ctx: &mut RenderContext,
        lights: &std::collections::HashMap<u32, Light>,
        env_cubemaps: &std::collections::HashMap<u32, CubeTexture>,
    ) {
        let device_resources = &mut *ctx.device;
        let game_camera = ctx.camera;
        let game_config = ctx.config;

        let (view_matrix, _, _) = game_camera.calculate_view_matrix();
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );
        let view_proj = proj_matrix * view_matrix;
        let inv_view_proj: [[f32; 4]; 4] = {
            use cgmath::SquareMatrix;
            view_proj.invert().unwrap_or_else(cgmath::Matrix4::identity).into()
        };
        let camera_pos = game_camera.get_position();
        let (rw, rh) = game_config.render_resolution();

        let mut skylight_ids: Vec<u32> = lights
            .values()
            .filter(|l| l.get_light_type() == LightType::Skylight)
            .map(|l| l.id)
            .collect();
        skylight_ids.sort_unstable();
        let env = skylight_ids.iter().find_map(|id| env_cubemaps.get(id));
        let has_env = env.is_some();
        let env = env.unwrap_or(&self.fallback_cube);

        let uniform = AmbientUniform {
            view_proj: view_proj.into(),
            inv_view_proj,
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, if has_env { 1.0 } else { 0.0 }],
            target_dims: [
                rw as f32,
                rh as f32,
                if self.settings.ssao_enabled { 1.0 } else { 0.0 },
                if self.settings.ssgi_enabled { 1.0 } else { 0.0 },
            ],
            gi_params: [
                self.settings.gi_intensity.max(0.0),
                self.settings.ao_samples.clamp(1, 12) as f32,
                self.settings.gi_samples.max(1) as f32,
                0.0,
            ],
        };
        let uniform_buffer =
            device_resources.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AmbientPass_uniform"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let env_bind_group = device_resources.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.env_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&env.cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&env.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: env.sh_buffer.as_entire_binding(),
                },
            ],
            label: Some("AmbientPass_env_bind_group"),
        });

        let mut encoder =
            device_resources.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("AmbientPass render"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Ambient AO+GI"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.raw_ao_texture.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.raw_gi_texture.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
            render_pass.set_bind_group(1, &env_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // Cross-bilateral denoise, two separable passes: raw -> tmp
        // (horizontal) -> ao_texture/gi_texture (vertical). Guided by the
        // same G-buffer normal/depth AmbientPass itself reads, so it stays
        // in lockstep with this frame's geometry even under camera motion.
        let normal_view = &device_resources.gbuffer_textures[1].view;
        let depth_view = &device_resources.render_textures[1].view;
        let make_denoise_input_bind_group =
            |device: &wgpu::Device, ao_view: &wgpu::TextureView, gi_view: &wgpu::TextureView| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.denoise_input_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(ao_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gi_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                    ],
                    label: Some("AmbientPass_denoise_input_bind_group"),
                })
            };
        let blur_params =
            [self.settings.denoise_radius.min(4) as f32, self.settings.denoise_strength.max(0.0)];
        let make_denoise_uniform_bind_group = |device: &wgpu::Device, direction: [f32; 2]| {
            let uniform = DenoiseUniform {
                inv_view_proj,
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
                params: [rw as f32, rh as f32, direction[0], direction[1]],
                blur_params: [blur_params[0], blur_params[1], 0.0, 0.0],
            };
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AmbientPass_denoise_uniform"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.denoise_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
                label: Some("AmbientPass_denoise_uniform_bind_group"),
            })
        };

        let device = &device_resources.device;
        // Each iteration's horizontal pass reads the previous iteration's
        // final output (raw probe output for the first iteration), so
        // iterations > 1 compound rather than just re-running the same blur.
        let mut input_ao = &self.raw_ao_texture.view;
        let mut input_gi = &self.raw_gi_texture.view;
        let iterations = self.settings.denoise_iterations.clamp(1, 3);
        for _ in 0..iterations {
            let horizontal_input = make_denoise_input_bind_group(device, input_ao, input_gi);
            let horizontal_uniform = make_denoise_uniform_bind_group(device, [1.0, 0.0]);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Ambient Denoise Horizontal"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.tmp_ao_texture.view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.tmp_gi_texture.view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.denoise_pipeline);
                pass.set_bind_group(0, &horizontal_input, &[]);
                pass.set_bind_group(1, &horizontal_uniform, &[]);
                pass.draw(0..3, 0..1);
            }

            let vertical_input = make_denoise_input_bind_group(
                device,
                &self.tmp_ao_texture.view,
                &self.tmp_gi_texture.view,
            );
            let vertical_uniform = make_denoise_uniform_bind_group(device, [0.0, 1.0]);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Ambient Denoise Vertical"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.ao_texture.view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.gi_texture.view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.denoise_pipeline);
                pass.set_bind_group(0, &vertical_input, &[]);
                pass.set_bind_group(1, &vertical_uniform, &[]);
                pass.draw(0..3, 0..1);
            }

            input_ao = &self.ao_texture.view;
            input_gi = &self.gi_texture.view;
        }

        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }
}

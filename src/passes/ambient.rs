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
}

impl Default for AmbientSettings {
    fn default() -> Self {
        Self { ssao_enabled: true, ssgi_enabled: true, gi_intensity: 1.0 }
    }
}

crate::editor_properties!(AmbientSettings {
    ssao_enabled: bool("Screen-Space AO"),
    ssgi_enabled: bool("Screen-Space GI"),
    gi_intensity: float("GI Intensity"),
});

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
    // x: gi_intensity, yzw unused.
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
        let ao_texture =
            Texture::new_render_texture_with_format(device, AO_FORMAT, size.width, size.height)
                .unwrap();
        let gi_texture =
            Texture::new_render_texture_with_format(device, GI_FORMAT, size.width, size.height)
                .unwrap();

        AmbientPass {
            pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group,
            env_bind_group_layout,
            fallback_cube,
            ao_texture,
            gi_texture,
            settings: AmbientSettings::default(),
        }
    }

    /// Rebuilds the AO/GI targets and the (recreated) G-buffer bind group
    /// after a window resize.
    pub fn resize(&mut self, device_resources: &DeviceResources) {
        let size = device_resources.render_textures[1].texture.size();
        self.ao_texture = Texture::new_render_texture_with_format(
            &device_resources.device,
            AO_FORMAT,
            size.width,
            size.height,
        )
        .unwrap();
        self.gi_texture = Texture::new_render_texture_with_format(
            &device_resources.device,
            GI_FORMAT,
            size.width,
            size.height,
        )
        .unwrap();
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
            gi_params: [self.settings.gi_intensity.max(0.0), 0.0, 0.0, 0.0],
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
                        view: &self.ao_texture.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gi_texture.view,
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
        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }
}

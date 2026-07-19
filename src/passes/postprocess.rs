use wgpu::util::DeviceExt;

use crate::{assets::*, resource::*};

/// Post-process / tonemap settings for the scene.  The tonemap curve is
/// `y = (x(Ax+B)) / (x(Cx+D)+E)` applied to linear radiance after an `exposure`
/// multiply; the postprocess pass runs it, and the gaussian-splat pass pre-applies
/// its exact inverse so display-referred splats survive it unchanged.
///
/// The five curve params are kept *invertible* by [`enforce_invertible`], which
/// the editor calls after every edit:
///   * `A*D >= B*C` (highlight_scale*midtone_curve >= midtone_scale*highlight_curve)
///     keeps the curve monotonic.  Otherwise it rolls over past a peak and the
///     inverse grabs the wrong branch -- the "bad values when midtone is high".
///   * `A/C >= 1` (highlight_scale/highlight_curve) keeps display white reachable,
///     so every value in [0,1] has an inverse.
///
/// [`enforce_invertible`]: PostProcessSettings::enforce_invertible
#[derive(Clone, Copy, Debug)]
pub struct PostProcessSettings {
    pub tonemap_enabled: bool,
    pub exposure: f32,
    pub highlight_scale: f32, // A
    pub midtone_scale: f32,   // B
    pub highlight_curve: f32, // C
    pub midtone_curve: f32,   // D
    pub shadow_offset: f32,   // E
    /// Strength of the post-upscale sharpen (0 disables).  The scene renders at
    /// `render_scale` and the postprocess pass upscales it with a Catmull-Rom
    /// kernel; this wins back the acutance that costs.  Above ~0.5 the limiter
    /// stops hiding the sharpening halos.
    pub sharpen_strength: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        // Narkowicz 2015 ACES filmic constants -- already invertible
        // (A*D = 1.48 >= B*C = 0.073, and A/C = 1.033 >= 1).
        Self {
            tonemap_enabled: true,
            exposure: 1.0,
            highlight_scale: 2.51,
            midtone_scale: 0.03,
            highlight_curve: 2.43,
            midtone_curve: 0.59,
            shadow_offset: 0.14,
            sharpen_strength: 0.25,
        }
    }
}

impl PostProcessSettings {
    /// Nudges the curve params back into the invertible region (see the struct
    /// docs) so the midtone knob can't push the curve non-monotonic.  Raises
    /// `midtone_curve` (D) to satisfy `A*D >= B*C` and `highlight_scale` (A) to
    /// satisfy `A/C >= 1` -- the least-surprising nudges, since both only ever
    /// *increase* a value the user pushed too far relative to the others.
    pub fn enforce_invertible(&mut self) {
        // Keep everything positive / non-degenerate first.
        self.highlight_scale = self.highlight_scale.max(1e-3);
        self.highlight_curve = self.highlight_curve.max(1e-3);
        self.shadow_offset = self.shadow_offset.max(1e-4);
        self.midtone_scale = self.midtone_scale.max(0.0);
        self.midtone_curve = self.midtone_curve.max(0.0);
        self.exposure = self.exposure.max(1e-3);
        self.sharpen_strength = self.sharpen_strength.clamp(0.0, 1.0);

        // A/C >= 1: display white is reachable.
        self.highlight_scale = self.highlight_scale.max(self.highlight_curve);
        // A*D >= B*C: curve stays monotonic (single-valued inverse).
        let min_midtone_curve = self.midtone_scale * self.highlight_curve / self.highlight_scale;
        self.midtone_curve = self.midtone_curve.max(min_midtone_curve);
    }
}

crate::editor_properties!(PostProcessSettings {
    tonemap_enabled: bool("Tonemap Enabled"),
    exposure: float("Exposure"),
    highlight_scale: float("Highlight Scale"),
    midtone_scale: float("Midtone Scale"),
    highlight_curve: float("Highlight Curve"),
    midtone_curve: float("Midtone Curve"),
    shadow_offset: float("Shadow Offset"),
    sharpen_strength: float("Upscale Sharpen"),
});

pub struct PostprocessPass {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub pipeline: wgpu::RenderPipeline,
    pub postprocess_uniform: PostProcessUniform,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub bind_group: wgpu::BindGroup,
    pub postprocess_tex_handle: TextureHandle,
    pub scene_sampler: wgpu::Sampler,
    pub settings: PostProcessSettings,
}

impl PostprocessPass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;
        let render_texture = &device_resources.render_textures[0];

        // Post Process Pipeline
        let postprocess_shader_handle = asset_manager
            .load_shader(
                "/engine_assets/shaders/postprocess_uber.wgsl",
                device_resources,
            )
            .await;
        let postprocess_shader = asset_manager.get_shader(&postprocess_shader_handle);

        let postprocess_uniform = PostProcessUniform {
            ..Default::default()
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buffer"),
            contents: bytemuck::cast_slice(&[postprocess_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("bind_group"),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&uniform_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: postprocess_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), SpriteDrawInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: postprocess_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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

        let postprocess_tex_handle = asset_manager
            .load_texture(
                "/engine_assets/textures/postprocess_filter.png",
                device_resources,
                TextureFilter::Linear,
            )
            .await;
        // Linear filtering so a sub-native render_scale upscales smoothly onto the
        // surface.  Repeat address mode matches the filter texture's sampler so the
        // scanline/warp effects keep wrapping as before.
        let scene_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let postprocess_tex = asset_manager.get_texture(&postprocess_tex_handle);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&postprocess_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&scene_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&render_texture.view),
                },
            ],
            label: Some("bind_group"),
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        PostprocessPass {
            pipeline,
            postprocess_uniform,
            uniform_buffer,
            uniform_bind_group,
            bind_group,
            vertex_buffer,
            index_buffer,
            postprocess_tex_handle,
            scene_sampler,
            settings: PostProcessSettings::default(),
        }
    }

    pub fn render(
        &mut self,
        ctx: &mut RenderContext,
        target_view: &wgpu::TextureView,
        postprocess_override: Option<PostProcessMode>,
    ) {
        let device_resources = &mut *ctx.device;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PostprocessPass::render()"),
                });

        let color_attachment = Some(wgpu::RenderPassColorAttachment {
            view: target_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        });

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Postprocess"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, device_resources.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        self.postprocess_uniform.time_mode_srgb_tonemap[0] =
            game_config.start_time.elapsed().as_secs_f32();
        self.postprocess_uniform.time_mode_srgb_tonemap[1] = {
            let postprocess_mode = match postprocess_override {
                Some(p) => p.clone(),
                None => game_config.postprocess_mode.clone(),
            };
            match postprocess_mode {
                PostProcessMode::Desaturation => 1.0,
                PostProcessMode::ScanLines => 2.0,
                PostProcessMode::Warp => 3.0,
                _ => 0.0,
            }
        };
        // When the surface isn't sRGB (e.g. Chrome WebGPU), the hardware won't
        // gamma-encode on present, so the shader must do it or everything looks dark.
        self.postprocess_uniform.time_mode_srgb_tonemap[2] =
            if device_resources.surface_config.format.is_srgb() {
                0.0
            } else {
                1.0
            };
        // Tonemap toggle + curve params (applied in linear space before the
        // sRGB encode; the splat pass pre-applies the exact inverse of this same
        // curve so display-referred splats pass through unchanged).
        let s = &self.settings;
        self.postprocess_uniform.time_mode_srgb_tonemap[3] =
            if s.tonemap_enabled { 1.0 } else { 0.0 };
        self.postprocess_uniform.tonemap_abcd = [
            s.highlight_scale,
            s.midtone_scale,
            s.highlight_curve,
            s.midtone_curve,
        ];
        self.postprocess_uniform.tonemap_e_exposure =
            [s.shadow_offset, s.exposure, s.sharpen_strength, 0.0];

        device_resources.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.postprocess_uniform]),
        );
        render_pass.draw_indexed(0..6, 0, 0..1);
        drop(render_pass);

        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }

    pub fn resize(
        &mut self,
        device_resources: &mut DeviceResources,
        asset_manager: &AssetManager,
    ) {
        let postprocess_tex = asset_manager.get_texture(&self.postprocess_tex_handle);
        let bind_group_layout =
            device_resources
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                    ],
                    label: Some("bind_group_layout"),
                });
        let render_texture = &device_resources.render_textures[0];
        self.bind_group = device_resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&postprocess_tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.scene_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&render_texture.view),
                    },
                ],
                label: Some("bind_group"),
            });
    }
}

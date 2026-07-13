use wgpu::util::DeviceExt;

use crate::passes::postprocess::PostProcessSettings;
use crate::{assets::*, resource::*};

/// Uniform mirror of the tonemap curve, matching `TonemapUniform` in
/// splat_composite.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct TonemapUniform {
    abcd: [f32; 4],          // A, B, C, D
    e_exp_enabled: [f32; 4], // E, exposure, enabled 1/0, _
}

/// Composites the display-referred gaussian-splat buffer (`render_textures[2]`)
/// into the linear HDR scene color (`render_textures[0]`), converting the
/// finished composite once (sRGB-decode + inverse tonemap).  See
/// splat_composite.wgsl for the why.
pub struct SplatCompositePass {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    tex_bind_group_layout: wgpu::BindGroupLayout,
    tex_bind_group: wgpu::BindGroup,
    uniform: TonemapUniform,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    pub settings: PostProcessSettings,
}

impl SplatCompositePass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        let device = &device_resources.device;

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/splat_composite.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        // Splat source is 1:1 with the scene color, so nearest sampling is exact.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let tex_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SplatComposite_tex_layout"),
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
                ],
            });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SplatComposite_uniform_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SplatComposite_pipeline_layout"),
            bind_group_layouts: &[Some(&tex_bind_group_layout), Some(&uniform_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SplatComposite_pipeline"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_COLOR_FORMAT,
                    // Non-premultiplied over: linear*a + scene*(1-a).
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
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

        let uniform = TonemapUniform::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SplatComposite_uniform_buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SplatComposite_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let tex_bind_group = Self::make_tex_bind_group(
            device,
            &tex_bind_group_layout,
            &device_resources.render_textures[2].view,
            &sampler,
        );

        SplatCompositePass {
            pipeline,
            sampler,
            tex_bind_group_layout,
            tex_bind_group,
            uniform,
            uniform_buffer,
            uniform_bind_group,
            settings: PostProcessSettings::default(),
        }
    }

    fn make_tex_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        splat_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SplatComposite_tex_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(splat_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Rebuilds the splat-texture binding after the offscreen targets are
    /// recreated (window resize / render-scale change).
    pub fn resize(&mut self, device_resources: &DeviceResources) {
        self.tex_bind_group = Self::make_tex_bind_group(
            &device_resources.device,
            &self.tex_bind_group_layout,
            &device_resources.render_textures[2].view,
            &self.sampler,
        );
    }

    pub fn render(&mut self, ctx: &mut RenderContext) {
        let device_resources = &mut *ctx.device;

        let s = &self.settings;
        self.uniform.abcd = [
            s.highlight_scale,
            s.midtone_scale,
            s.highlight_curve,
            s.midtone_curve,
        ];
        self.uniform.e_exp_enabled = [
            s.shadow_offset,
            s.exposure,
            if s.tonemap_enabled { 1.0 } else { 0.0 },
            0.0,
        ];
        device_resources.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );

        let mut encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SplatCompositePass::render()"),
                });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Splat Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &device_resources.render_textures[0].view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.tex_bind_group, &[]);
            pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        device_resources
            .queue
            .submit(std::iter::once(encoder.finish()));
    }
}

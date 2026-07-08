use cgmath::InnerSpace;
use cgmath::SquareMatrix;
use wgpu::util::DeviceExt;

use crate::{assets::*, config::*, game_object::*, resource::*, utils::*, log};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BulletHoleUniform {
    pub trace_hit: [f32; 4],
    pub trace_dir: [f32; 4],
}

#[allow(dead_code)]
pub struct BulletHoleRenderGroup {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform: BulletHoleUniform,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub first_run: bool,
}

#[allow(dead_code)]
impl BulletHoleRenderGroup {
    pub async fn new(
        shader_path: &str,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating BulletHoleRenderGroup with shader {shader_path}");
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;

        // Uniform buffer
        let uniform = BulletHoleUniform {
            ..Default::default()
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BulletHoleRenderGroup::uniform_buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("BulletHoleRenderGroup::bind_group_layout"),
        });

        let scorch_texture = asset_manager
            .load_texture("/engine_assets/textures/scorch_t.png", device_resources)
            .await;
        let scorch_tex = asset_manager.get_texture(&scorch_texture);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scorch_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&scorch_tex.sampler),
                },
            ],
            label: Some("BulletHoleRenderGroup::bind_group"),
        });

        log!("  Creating pipeline");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ModelRenderGroup_render_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader_handle = asset_manager
            .load_shader(shader_path, device_resources)
            .await;
        let model_shader = asset_manager.get_shader(&shader_handle);

        let mut cull_mode = Some(wgpu::Face::Back);
        if shader_path.contains("decal") {
            cull_mode = None;
        }

        let blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("BulletHoleRenderGroup::pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: model_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: model_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format.add_srgb_suffix(),
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
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
        let mut surface_config = device_resources.surface_config.clone();
        surface_config.width = 1024;
        surface_config.height = 1024;

        //  let render_texture = Texture::new_render_texture(&device, &surface_config).unwrap();
        BulletHoleRenderGroup {
            pipeline,
            uniform,
            uniform_buffer,
            bind_group,
            first_run: true,
            //render_texture,
        }
    }

    pub fn render(
        &mut self,
        device_resources: &mut DeviceResources,
        asset_manager: &mut AssetManager,
        _game_config: &Config,
        actor: &Actor,
        traces: &(CgVec3, CgVec3),
    ) {
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ModelRenderGroup::render()"),
                });

        let model_mappings = asset_manager.get_model_mappings();
        let model = &model_mappings[&actor.get_model()];
        let color_attachment = wgpu::RenderPassColorAttachment {
            view: &model.hole_texture.as_ref().unwrap().view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: if !self.first_run {
                    wgpu::LoadOp::Load
                } else {
                    wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    })
                },
                store: wgpu::StoreOp::Store,
            },
        };
        self.first_run = false;

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("BulletHoleRenderGroup::render_pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });

        let inv_world_matrix = cgmath::Matrix4::from_translation(actor.get_position())
            * cgmath::Matrix4::from(actor.get_rotation())
            * cgmath::Matrix4::from_nonuniform_scale(
                actor.get_scale().x,
                actor.get_scale().y,
                actor.get_scale().z,
            )
            .invert()
            .unwrap();
        let local_pos = inv_world_matrix * CgVec4::new(traces.0.x, traces.0.y, traces.0.z, 1.0);
        let local_dir = inv_world_matrix * CgVec4::new(traces.1.x, traces.1.y, traces.1.z, 0.0);
        let local_dir = local_dir.normalize();
        let uniform_data = BulletHoleUniform {
            trace_hit: [local_pos.x, local_pos.y, local_pos.z, 0.0],
            trace_dir: [local_dir.x, local_dir.y, local_dir.z, 0.0],
        };
        device_resources.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
        render_pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..model.num_indices, 0, 0..1);

        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}

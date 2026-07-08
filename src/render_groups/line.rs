use cgmath::InnerSpace;
use std::mem::size_of;
use wgpu::util::DeviceExt;

use crate::{assets::*, resource::*, utils::*, log};

#[derive(Clone, Copy)]
pub struct Line {
    pub start: CgVec3,
    pub end: CgVec3,
    pub color: CgVec4,
    pub thickness: f32,
    pub end_time: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineUniform {
    pub mvp_matrix: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub camera_dir: [f32; 4],
    pub screen_dimensions: [f32; 4],
    pub model_color: [f32; 4],
}

pub struct LineRenderGroup {
    pub vertex_buffer: wgpu::Buffer,
    pub pipeline: wgpu::RenderPipeline,
    pub uniform: LineUniform,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
}

const MAX_LINES: usize = 1000;
const MAX_VERTS: usize = 4 * MAX_LINES;

impl LineRenderGroup {
    pub async fn new(
        shader_path: &str,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating ModelRenderGroup with shader {shader_path}");
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LineRenderGroup_vertex_buffer"),
            mapped_at_creation: false,
            size: (size_of::<Vertex>() * MAX_VERTS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Uniform buffer
        let uniform = LineUniform {
            ..Default::default()
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LineRenderGroup_uniform_buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
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
                label: Some("LineRenderGroup_uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("LineRenderGroup_uniform_bind_group"),
        });

        log!("  Creating pipeline");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LineRenderGroup_render_pipeline_layout"),
                bind_group_layouts: &[Some(&uniform_bind_group_layout)],
                immediate_size: 0,
            });

        let shader_handle = asset_manager
            .load_shader(shader_path, device_resources)
            .await;
        let model_shader = asset_manager.get_shader(&shader_handle);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("LineRenderGroup_opaque_pipeline"),
            layout: Some(&render_pipeline_layout),
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        LineRenderGroup {
            vertex_buffer,
            pipeline,
            uniform,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    pub fn render(&mut self, ctx: &mut RenderContext, lines: &Vec<Line>) {
        let device_resources = &mut *ctx.device;
        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("LineGroup::render()"),
                });

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: &device_resources.render_textures[0].view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let depth_attachment = wgpu::RenderPassDepthStencilAttachment {
            view: &device_resources.render_textures[1].view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        };

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Line"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(depth_attachment),
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline);

        let (view_matrix, view_dir, _) = game_camera.calculate_view_matrix();
        let view_pos = game_camera.get_position();
        let view_pos = [view_pos.x, view_pos.y, view_pos.z, 1.0];
        let fov = game_config.fov;
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );

        let uniform_buffer = &self.uniform_buffer;
        let mut uniform_data = LineUniform {
            ..Default::default()
        };
        uniform_data.mvp_matrix = (proj_matrix * view_matrix).into();
        uniform_data.camera_dir = [view_dir.x, view_dir.y, view_dir.z, 0.0];
        uniform_data.camera_pos = view_pos;
        uniform_data.screen_dimensions = [
            game_config.window_width as f32,
            game_config.window_height as f32,
            (game_config.window_height as f32) / (game_config.window_width as f32),
            0.0,
        ];
        uniform_data.model_color = [1.0, 0.1, 1.0, 1.0];
        device_resources.queue.write_buffer(
            uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );

        let mut vertices = Vec::<Vertex>::new();
        for line in lines {
            let center_pos = (line.end + line.start) * 0.5;
            let right_vec = center_pos - game_camera.get_position();
            let forward_vec = (line.end - line.start).normalize();
            let up_vec = right_vec.cross(forward_vec).normalize() * line.thickness;

            let start_1 = line.start + up_vec;
            let start_2 = line.start - up_vec;
            let end_1 = line.end + up_vec;
            let end_2 = line.end - up_vec;

            let vertex_1 = Vertex {
                position: [start_1.x, start_1.y, start_1.z],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                color: line.color.into(),
            };
            let vertex_2 = Vertex {
                position: [start_2.x, start_2.y, start_2.z],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                color: line.color.into(),
            };
            let vertex_3 = Vertex {
                position: [end_2.x, end_2.y, end_2.z],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                color: line.color.into(),
            };
            let vertex_4 = Vertex {
                position: [end_1.x, end_1.y, end_1.z],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                color: line.color.into(),
            };

            vertices.push(vertex_1);
            vertices.push(vertex_2);
            vertices.push(vertex_3);

            vertices.push(vertex_1);
            vertices.push(vertex_3);
            vertices.push(vertex_4);
        }
        device_resources.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(vertices.as_slice()),
        );

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.draw(0..vertices.len() as u32, 0..1);

        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}

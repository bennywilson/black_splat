use std::mem::size_of;

use wgpu::{
    util::DeviceExt, BindGroupLayoutEntry, BindingType, SamplerBindingType, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use crate::{
    assets::*, config::*, game_object::*, resource::*, utils::*, log, PERF_SCOPE,
};

pub struct SpriteRenderGroup {
    pub opaque_pipeline: wgpu::RenderPipeline,
    pub alpha_blend_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub uniform: SpriteUniform,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub tex_bind_group: wgpu::BindGroup,
    pub texture_index: u32,
}

impl SpriteRenderGroup {
    pub async fn new(
        texture_file: String,
        texture_index: u32,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
        game_config: &Config,
    ) -> Self {
        log!("Creating SpriteRenderGroup...");

        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("kbSpritePipeline: texture_bind_group_layout"),
            });
        let sprite_tex_handle = asset_manager
            .load_texture(&texture_file, device_resources)
            .await;

        let postprocess_tex_handle = asset_manager
            .load_texture(
                "/engine_assets/textures/postprocess_filter.png",
                device_resources,
            )
            .await;

        let sprite_tex = asset_manager.get_texture(&sprite_tex_handle);
        let postprocess_tex = asset_manager.get_texture(&postprocess_tex_handle);
        let tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sprite_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sprite_tex.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&postprocess_tex.view),
                },
            ],
            label: Some("kbSpritePipeline: tex_bind_group"),
        });

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/basic_sprite.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        let uniform = SpriteUniform {
            ..Default::default()
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sprite_uniform_buffer"),
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
                label: Some("sprite_uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("model_bind_group"),
        });

        log!("  Creating pipeline");

        // Render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&texture_bind_group_layout), Some(&uniform_bind_group_layout)],
                immediate_size: 0,
            });

        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), SpriteDrawInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
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

        let transparent_shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/cloud_sprite.wgsl", device_resources)
            .await;
        let transparent_shader = asset_manager.get_shader(&transparent_shader_handle);

        let alpha_blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: transparent_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), SpriteDrawInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: transparent_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
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

        log!("  Creating vertex/index buffers");

        // Vertex/Index buffer
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

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            mapped_at_creation: false,
            size: (size_of::<SpriteDrawInstance>() * game_config.max_render_instances as usize)
                as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        SpriteRenderGroup {
            opaque_pipeline,
            alpha_blend_pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            uniform,
            uniform_buffer,
            uniform_bind_group,
            tex_bind_group,
            texture_index,
        }
    }

    pub fn render(
        &mut self,
        ctx: &mut RenderContext,
        render_pass_type: RenderPassType,
        game_objects: &Vec<GameObject>,
    ) {
        if game_objects.is_empty() {
            return;
        }

        let device_resources = &mut *ctx.device;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SpriteRenderGroup::render()"),
                });

        let mut frame_instances = Vec::<SpriteDrawInstance>::new();

        // Create instances
        let u_scale = 1.0 / 8.0;
        let v_scale = 1.0 / 8.0;
        let extra_scale = 1.0;
        let extra_offset: CgVec3 = CgVec3::new(0.0, 0.0, 0.0);

        for game_object in game_objects {
            PERF_SCOPE!("Creating instances");
            if game_object.texture_index != self.texture_index {
                continue;
            }

            if frame_instances.len() >= game_config.max_render_instances as usize {
                log::warn!(
                    "SpriteRenderGroup: hit max_render_instances ({}); dropping remaining sprites this frame",
                    game_config.max_render_instances
                );
                break;
            }

            let game_object_position = game_object.position + extra_offset;
            let sprite_index = game_object.sprite_index + game_object.anim_frame;
            let mut u_offset = ((sprite_index % 8) as f32) * u_scale;
            let v_offset = ((sprite_index / 8) as f32) * v_scale;
            let mul = if game_object.direction.x > 0.0 {
                1.0
            } else {
                -1.0
            };
            if mul < 0.0 {
                u_offset += u_scale;
            }

            let new_instance = SpriteDrawInstance {
                pos_scale: [
                    game_object_position.x,
                    game_object_position.y,
                    game_object.scale.x * extra_scale,
                    game_object.scale.y * extra_scale,
                ],
                uv_scale_bias: [
                    u_scale * mul * game_object.uv_tiles.0,
                    v_scale * game_object.uv_tiles.1,
                    u_offset,
                    v_offset,
                ],
                per_instance_data: [game_object.random_val, 0.0, 0.0, 0.0],
            };
            frame_instances.push(new_instance);
        }

        if frame_instances.is_empty() {
            return;
        }

        device_resources.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(frame_instances.as_slice()),
        );

        let color_attachment = Some(wgpu::RenderPassColorAttachment {
            view: &device_resources.render_textures[0].view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        });

        let label = format!("Sprite {:?}", render_pass_type);
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&label),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &device_resources.render_textures[1].view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        device_resources.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );

        if matches!(render_pass_type, RenderPassType::Opaque) {
            render_pass.set_pipeline(&self.opaque_pipeline);
        } else {
            render_pass.set_pipeline(&self.alpha_blend_pipeline);
        }

        self.uniform.screen_dimensions = [
            game_config.window_width as f32,
            game_config.window_height as f32,
            (game_config.window_height as f32) / (game_config.window_width as f32),
            0.0,
        ]; //[self.game_config.window_width as f32, self.game_config.window_height as f32, (self.game_config.window_height as f32) / (self.game_config.window_width as f32), 0.0]));
        self.uniform.time[0] = game_config.start_time.elapsed().as_secs_f32();

        render_pass.set_bind_group(0, &self.tex_bind_group, &[]);
        render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..frame_instances.len() as _);
        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}

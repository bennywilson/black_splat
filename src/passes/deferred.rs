//! Deferred world rendering: `GBufferPass` draws the `SceneLayer::World`
//! actors into the G-buffer (albedo / world normal / specular + depth), then
//! `LightingPass` accumulates every scene light into the scene color target
//! with one fullscreen pass per light (skylight, directional, point, spot --
//! no shadows yet).  Everything after (holes, custom passes, splats,
//! particles) still renders forward on top, sharing the same depth buffer.

use cgmath::{InnerSpace, SquareMatrix};
use std::collections::HashMap;
use wgpu::{
    util::DeviceExt, BindGroupLayoutEntry, BindingType, SamplerBindingType, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use crate::{assets::*, game_object::*, log, passes::model::*, resource::*, utils::*};

/// Most lights drawn per frame; extras are ignored.
pub const MAX_LIGHTS: usize = 32;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub inv_view_proj: [[f32; 4]; 4],
    // xyz world position, w range (point/spot).
    pub position_range: [f32; 4],
    // xyz direction the light points, w cos(outer cone angle) (spot).
    pub direction_cone: [f32; 4],
    // rgb color * intensity, w cos(inner cone angle) (spot).
    pub color_cone: [f32; 4],
    // Skylight bottom-hemisphere color * intensity.
    pub color2: [f32; 4],
    pub camera_pos: [f32; 4],
    // xy render target size in pixels.
    pub target_dims: [f32; 4],
}

/// Renders the world-layer actors into the G-buffer.  Mirrors
/// `ModelPass::render`'s actor walk, but writes albedo/normal/specular to
/// three color targets and lets an actor's material override its model's
/// textures and constants.
pub struct GBufferPass {
    pipeline: wgpu::RenderPipeline,
}

impl GBufferPass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating GBufferPass");
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;

        // Same layouts a Model bakes its bind groups against (see Model::from_bytes).
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
                label: Some("GBufferPass_uniform_bind_group_layout"),
            });

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
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("GBufferPass_texture_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GBufferPass_pipeline_layout"),
            bind_group_layouts: &[
                Some(&texture_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/gbuffer.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GBufferPass_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: surface_config.format.add_srgb_suffix(),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_NORMAL_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_SPEC_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
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

        GBufferPass { pipeline }
    }

    /// Draws the World-layer actors into the G-buffer, clearing it (and the
    /// shared depth buffer) first -- this is the frame's first scene pass.
    pub fn render(&mut self, ctx: &mut RenderContext, actors: &HashMap<u32, Actor>) {
        let device_resources = &mut *ctx.device;
        let asset_manager = &mut *ctx.assets;
        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GBufferPass::render()"),
                });

        let clear_ops = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };
        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &device_resources.gbuffer_textures[0].view,
                resolve_target: None,
                depth_slice: None,
                ops: clear_ops,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &device_resources.gbuffer_textures[1].view,
                resolve_target: None,
                depth_slice: None,
                ops: clear_ops,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &device_resources.gbuffer_textures[2].view,
                resolve_target: None,
                depth_slice: None,
                ops: clear_ops,
            }),
        ];
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GBuffer"),
            color_attachments: &color_attachments,
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

        render_pass.set_pipeline(&self.pipeline);

        let (view_matrix, view_dir, _) = game_camera.calculate_view_matrix();
        let view_pos = game_camera.get_position();
        let view_pos = [view_pos.x, view_pos.y, view_pos.z, 1.0];
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );

        // Fill each actor's uniform slot, remembering which material (if any)
        // goes with each slot so the draw loop below can bind it.
        let mut models_to_render = Vec::<ModelHandle>::new();
        let mut slot_materials = HashMap::<ModelHandle, Vec<MaterialHandle>>::new();
        for actor in actors.values() {
            let (actor_layer, _) = actor.get_layer();
            if actor_layer != SceneLayer::World {
                continue;
            }
            let model_handle = actor.get_model();

            // Material constants: multiplied into the actor color / written as
            // the specular constant.  No material = albedo only, no specular.
            // Fetched before get_model so the borrows don't overlap.
            let material_handle = actor.get_material();
            let (color_const, spec_const) = asset_manager
                .get_material(&material_handle)
                .map_or((CG_VEC4_ONE, CG_VEC4_ZERO), |m| {
                    (m.color_constant, m.spec_constant)
                });

            // Editor-placed actors can exist before a model is assigned.
            let Some(model) = asset_manager.get_model(&model_handle) else {
                continue;
            };

            if !models_to_render.contains(&model_handle) {
                models_to_render.push(model_handle);
            }

            let uniform_buffer = model.alloc_uniform_buffer();
            let mut uniform_data = ModelUniform {
                ..Default::default()
            };
            let world_matrix = cgmath::Matrix4::from_translation(actor.get_position())
                * cgmath::Matrix4::from(actor.get_rotation())
                * cgmath::Matrix4::from_nonuniform_scale(
                    actor.get_scale().x,
                    actor.get_scale().y,
                    actor.get_scale().z,
                );
            uniform_data.world = world_matrix.into();
            // Same zero-scale guard as ModelPass::render.
            uniform_data.inv_world = world_matrix
                .invert()
                .unwrap_or_else(cgmath::Matrix4::identity)
                .into();
            uniform_data.mvp_matrix = (proj_matrix * view_matrix * world_matrix).into();
            uniform_data.view_proj = (proj_matrix * view_matrix).into();
            uniform_data.camera_dir = [view_dir.x, view_dir.y, view_dir.z, 0.0];
            uniform_data.camera_pos = view_pos;
            uniform_data.screen_dimensions = [
                game_config.window_width as f32,
                game_config.window_height as f32,
                (game_config.window_height as f32) / (game_config.window_width as f32),
                0.0,
            ];
            uniform_data.time[0] = game_config.start_time.elapsed().as_secs_f32();
            uniform_data.time[1] = 1.0;

            let actor_color = actor.get_color();
            uniform_data.model_color = [
                actor_color.x * color_const.x,
                actor_color.y * color_const.y,
                actor_color.z * color_const.z,
                actor_color.w * color_const.w,
            ];
            uniform_data.spec_color = spec_const.into();
            uniform_data.custom_data_1 = [
                actor.get_custom_data_1().x,
                actor.get_custom_data_1().y,
                actor.get_custom_data_1().z,
                actor.get_custom_data_1().w,
            ];
            device_resources.queue.write_buffer(
                uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniform_data]),
            );
            slot_materials
                .entry(model_handle)
                .or_default()
                .push(material_handle);
        }

        // Draw every filled slot, binding the slot's material over the model's
        // own textures when one is assigned.
        let (model_mappings, material_mappings) = asset_manager.get_models_and_materials();
        for model_handle in &models_to_render {
            let model = &model_mappings[model_handle];
            render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
            render_pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            let materials = &slot_materials[model_handle];
            for i in 0..model.get_uniform_info_count() {
                let tex_bind_group = materials
                    .get(i)
                    .and_then(|handle| material_mappings.get(handle))
                    .map_or(&model.tex_bind_group, |material| &material.bind_group);
                render_pass.set_bind_group(0, tex_bind_group, &[]);
                render_pass.set_bind_group(1, model.get_uniform_bind_group(i), &[]);
                render_pass.draw_indexed(0..model.num_indices, 0, 0..1);
            }
        }

        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));

        for model_handle in &models_to_render {
            let model = &mut model_mappings.get_mut(model_handle).unwrap();
            model.free_uniform_buffers();
        }
    }
}

/// Accumulates the scene's lights onto the scene color target from the
/// G-buffer: the pass clears to the config's clear color, then draws one
/// additive fullscreen triangle per light with the pipeline matching its type.
pub struct LightingPass {
    skylight_pipeline: wgpu::RenderPipeline,
    directional_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    spot_pipeline: wgpu::RenderPipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: wgpu::BindGroup,
    // One pre-built uniform buffer + bind group per light slot, so every
    // light's constants can be written before the frame's single submit.
    light_uniforms: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl LightingPass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating LightingPass");
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;

        // G-buffer inputs: albedo / normal / specular / depth, all read with
        // textureLoad (no sampler).
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
                label: Some("LightingPass_gbuffer_bind_group_layout"),
            });
        let gbuffer_bind_group =
            Self::make_gbuffer_bind_group(device_resources, &gbuffer_bind_group_layout);

        let light_bind_group_layout =
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
                label: Some("LightingPass_light_bind_group_layout"),
            });

        let mut light_uniforms = Vec::with_capacity(MAX_LIGHTS);
        for i in 0..MAX_LIGHTS {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("LightingPass_light_uniform_{i}")),
                contents: bytemuck::cast_slice(&[LightUniform::default()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &light_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
                label: Some(&format!("LightingPass_light_bind_group_{i}")),
            });
            light_uniforms.push((buffer, bind_group));
        }

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LightingPass_pipeline_layout"),
            bind_group_layouts: &[
                Some(&gbuffer_bind_group_layout),
                Some(&light_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Lights accumulate: add rgb onto whatever earlier lights wrote, force
        // alpha to 1 wherever geometry is lit (background keeps the clear alpha).
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Shader loading is the only async step; fetch all four up front, then
        // build the pipelines synchronously.
        let shader_paths = [
            "/engine_assets/shaders/light_skylight.wgsl",
            "/engine_assets/shaders/light_directional.wgsl",
            "/engine_assets/shaders/light_point.wgsl",
            "/engine_assets/shaders/light_spot.wgsl",
        ];
        let mut shader_handles = Vec::with_capacity(shader_paths.len());
        for path in shader_paths {
            shader_handles.push(asset_manager.load_shader(path, device_resources).await);
        }

        let make_pipeline = |handle: &ShaderHandle, label: &str| -> wgpu::RenderPipeline {
            let shader = asset_manager.get_shader(handle);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                        format: surface_config.format.add_srgb_suffix(),
                        blend: Some(additive),
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let skylight_pipeline = make_pipeline(&shader_handles[0], "LightingPass_skylight");
        let directional_pipeline = make_pipeline(&shader_handles[1], "LightingPass_directional");
        let point_pipeline = make_pipeline(&shader_handles[2], "LightingPass_point");
        let spot_pipeline = make_pipeline(&shader_handles[3], "LightingPass_spot");

        LightingPass {
            skylight_pipeline,
            directional_pipeline,
            point_pipeline,
            spot_pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group,
            light_uniforms,
        }
    }

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
                label: Some("LightingPass_gbuffer_bind_group"),
            })
    }

    /// Rebind the (recreated) G-buffer/depth textures after a window resize.
    pub fn resize(&mut self, device_resources: &DeviceResources) {
        self.gbuffer_bind_group =
            Self::make_gbuffer_bind_group(device_resources, &self.gbuffer_bind_group_layout);
    }

    /// Clears the scene color target to the config's clear color and adds
    /// every light's contribution from the G-buffer.
    pub fn render(&mut self, ctx: &mut RenderContext, lights: &HashMap<u32, Light>) {
        let device_resources = &mut *ctx.device;
        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("LightingPass::render()"),
                });

        let (view_matrix, _, _) = game_camera.calculate_view_matrix();
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );
        let inv_view_proj = (proj_matrix * view_matrix)
            .invert()
            .unwrap_or_else(cgmath::Matrix4::identity);
        let camera_pos = game_camera.get_position();
        // The G-buffer renders at render_resolution (not the window size);
        // the shaders rebuild UVs from pixel coords with these dimensions.
        let (rw, rh) = game_config.render_resolution();

        // Fill one uniform slot per light (the writes land before the submit).
        let mut draws = Vec::<(usize, LightType)>::new();
        for (slot, light) in lights.values().enumerate().take(MAX_LIGHTS) {
            let position = light.get_position();
            let direction = {
                let dir = light.get_direction();
                if dir.magnitude2() > 0.0001 {
                    dir.normalize()
                } else {
                    CgVec3::new(0.0, 0.0, 1.0)
                }
            };
            let color = light.get_color() * light.get_intensity();
            let color2 = light.get_color2() * light.get_intensity();
            let outer_rad = light.get_spot_angle().clamp(0.5, 89.0).to_radians();
            // The cone fades in over the outer 20% of the angle.
            let inner_rad = outer_rad * 0.8;

            let uniform = LightUniform {
                inv_view_proj: inv_view_proj.into(),
                position_range: [position.x, position.y, position.z, light.get_range()],
                direction_cone: [direction.x, direction.y, direction.z, outer_rad.cos()],
                color_cone: [color.x, color.y, color.z, inner_rad.cos()],
                color2: [color2.x, color2.y, color2.z, 0.0],
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
                target_dims: [rw as f32, rh as f32, 0.0, 0.0],
            };
            device_resources.queue.write_buffer(
                &self.light_uniforms[slot].0,
                0,
                bytemuck::cast_slice(&[uniform]),
            );
            draws.push((slot, light.get_light_type()));
        }

        let clear_color = game_config.clear_color;
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Deferred Lighting"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &device_resources.render_textures[0].view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear_color.x as f64,
                        g: clear_color.y as f64,
                        b: clear_color.z as f64,
                        a: clear_color.w as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });

        render_pass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
        for (slot, light_type) in &draws {
            let pipeline = match light_type {
                LightType::Skylight => &self.skylight_pipeline,
                LightType::Directional => &self.directional_pipeline,
                LightType::Point => &self.point_pipeline,
                LightType::Spot => &self.spot_pipeline,
            };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(1, &self.light_uniforms[*slot].1, &[]);
            render_pass.draw(0..3, 0..1);
        }

        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}

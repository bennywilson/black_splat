//! Deferred world rendering: `GBufferPass` draws the `SceneLayer::World`
//! actors into the G-buffer (albedo / world normal / metallic+roughness +
//! depth), then `LightingPass` accumulates every scene light into the scene
//! color target with one fullscreen pass per light (skylight, directional,
//! point, spot).  Everything after (holes, custom passes, splats, particles)
//! still renders forward on top, sharing the same depth buffer.
//!
//! Shadows are screen-space masks (see `ShadowPass`): a shadowed light first
//! renders casters into a depth map -- the directional light into a 2x2
//! cascade atlas, each spot light serially into one shared depth tile -- then
//! a fullscreen mask pass projects that map against the G-buffer depth into a
//! per-light screen-space shadow factor.  The light's shading pass multiplies
//! by its mask, and every mask also multiplies into a persistent screen-space
//! accumulation texture that will later darken the Gaussian splats (which
//! never sample shadow maps themselves).  Because each spot's map is consumed
//! into a mask before the next spot renders, any number of spot lights can
//! cast shadows.

use cgmath::{EuclideanSpace, InnerSpace, SquareMatrix};
use std::collections::HashMap;
use wgpu::{
    util::DeviceExt, BindGroupLayoutEntry, BindingType, SamplerBindingType, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use crate::{assets::*, game_object::*, log, passes::model::*, resource::*, utils::*};

/// Most lights drawn per frame; extras are ignored.
pub const MAX_LIGHTS: usize = 32;

/// Cascade tiles in the shadow atlas (and the most a directional light uses).
pub const MAX_CASCADES: usize = 4;
// Bias subtracted from the receiver depth when comparing against the shadow
// map (on top of the shadow pipeline's slope-scaled rasterizer bias).
const SHADOW_DEPTH_BIAS: f32 = 0.0015;
// Uniform pool for shadow draws: one 256-byte slot (dynamic offset) per
// caster-per-tile draw.
const SHADOW_DRAW_STRIDE: usize = 256;
const MAX_SHADOW_DRAWS: usize = 1024;

/// Skylight-only extra data: a 9-term spherical-harmonic projection of the
/// baked cube (diffuse irradiance) plus the mip chain's roughness range.
/// Kept separate from `LightUniform` (rather than appended to it) since that
/// struct's tail is shared shadow-tile data every other light type writes --
/// inserting fields there would shift offsets in every light's WGSL struct.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkylightEnvUniform {
    // 9 SH band coefficients, RGB in xyz (w unused, kept for 16-byte align).
    pub sh: [[f32; 4]; 9],
    // x: highest mip index (roughness 1.0 samples this level).
    pub mip_params: [f32; 4],
}

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
    // xy render target size in pixels, zw this light's shadow map size in
    // pixels (the cascade atlas or the spot tile).
    pub target_dims: [f32; 4],
    // Light view-projection per shadow tile: the directional light's cascades
    // (near to far), or the spot's single projection in [0].  Consumed by the
    // shadow *mask* pass; the shading pass only reads shadow_params.x.
    pub shadow_matrices: [[[f32; 4]; 4]; MAX_CASCADES],
    // Each tile's rect in its shadow map: xy uv offset, zw uv scale.
    pub shadow_rects: [[f32; 4]; MAX_CASCADES],
    // x: shadow tile count (0 = unshadowed), y: depth bias.
    pub shadow_params: [f32; 4],
}

/// Shadow quality settings, tweakable at runtime (the editor exposes them in
/// its Settings tab).  Changing the resolution recreates the shadow maps.
/// Per-light on/off stays on the light (`Light::casts_shadow`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ShadowSettings {
    /// Side of one shadow tile in texels (the cascade atlas is 2x2 tiles; the
    /// shared spot tile is one).
    pub resolution: u32,
    /// Directional-light cascade count, 1..=MAX_CASCADES.
    pub num_cascades: u32,
    /// How far from the camera the directional cascades reach, in world units.
    pub distance: f32,
    /// Artist control over how black shadow-catcher shadows land: a multiplier
    /// on the projected darkening amount.  1 = as projected; > 1 deepens toward
    /// black; 0 disables the catcher darkening.
    pub density: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        ShadowSettings {
            resolution: 1024,
            num_cascades: 3,
            distance: 75.0,
            density: 1.0,
        }
    }
}

impl ShadowSettings {
    fn clamped(&self) -> Self {
        ShadowSettings {
            resolution: self.resolution.clamp(256, 2048),
            num_cascades: self.num_cascades.clamp(1, MAX_CASCADES as u32),
            distance: self.distance.max(5.0),
            density: self.density.clamp(0.0, 4.0),
        }
    }
}

/// One rendered shadow-map tile: the light's view-projection and where the
/// tile sits in its depth map (uv offset + scale).
#[derive(Clone, Copy)]
struct ShadowTile {
    view_proj: CgMat4,
    rect: [f32; 4],
}

// Right-handed orthographic projection with wgpu's 0..1 clip z, for a box
// centered on the view axis.  (cgmath::ortho targets OpenGL's -1..1 z, which
// would clip away half the casters on wgpu.)
fn ortho_wgpu(half_extent: f32, near: f32, far: f32) -> CgMat4 {
    let inv_depth = 1.0 / (far - near);
    #[rustfmt::skip]
    let m = CgMat4::new(
        1.0 / half_extent, 0.0, 0.0, 0.0,
        0.0, 1.0 / half_extent, 0.0, 0.0,
        0.0, 0.0, -inv_depth, 0.0,
        0.0, 0.0, -near * inv_depth, 1.0,
    );
    m
}

// Right-handed perspective projection with wgpu's 0..1 clip z (for the spot
// shadow projection; see ortho_wgpu on why cgmath::perspective isn't used).
fn perspective_wgpu(fovy_rad: f32, aspect: f32, near: f32, far: f32) -> CgMat4 {
    let f = 1.0 / (fovy_rad * 0.5).tan();
    #[rustfmt::skip]
    let m = CgMat4::new(
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, far / (near - far), -1.0,
        0.0, 0.0, near * far / (near - far), 0.0,
    );
    m
}

// An up vector that isn't parallel to `dir`, for the light's look_at.
fn shadow_up(dir: CgVec3) -> CgVec3 {
    if dir.y.abs() > 0.99 {
        CgVec3::new(0.0, 0.0, 1.0)
    } else {
        CgVec3::new(0.0, 1.0, 0.0)
    }
}

/// A world-layer shadow caster gathered once per frame: its model and world
/// matrix, re-projected per shadow tile.
struct ShadowCaster {
    model: ModelHandle,
    world: CgMat4,
}

/// Renders the world-layer actors into the G-buffer.  Mirrors
/// `ModelPass::render`'s actor walk, but writes albedo/normal/metallic-
/// roughness to three color targets and lets an actor's material override its
/// model's textures and constants.
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
                    BindGroupLayoutEntry {
                        binding: 3,
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
            // Shadow-catcher proxies are invisible: they never write the
            // G-buffer (the deferred catcher pass renders them into their own
            // depth instead -- see ShadowPass::render_catcher_depth).
            if actor.is_shadow_catcher() {
                continue;
            }
            let model_handle = actor.get_model();

            // Material constants: color multiplied into the actor color;
            // metallic/roughness written as the spec constant.  No material =
            // albedo only (dielectric, fairly rough).  Fetched before
            // get_model so the borrows don't overlap.
            let material_handle = actor.get_material();
            let (color_const, mr_const) = asset_manager
                .get_material(&material_handle)
                .map_or((CG_VEC4_ONE, CgVec4::new(0.0, 0.85, 0.0, 0.0)), |m| {
                    (m.color_constant, m.mr_constant)
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
            uniform_data.spec_color = mr_const.into();
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

/// Which shadow mask shader a mask pass uses.
#[derive(Clone, Copy)]
enum MaskKind {
    Cascades,
    Spot,
}

/// Everything shadows: the caster depth pipeline, the directional cascade
/// atlas, the shared serially-reused spot depth tile, and the screen-space
/// mask plumbing (per-light temp mask + the multiplied accumulation texture
/// for the future Gaussian-splat overlay).
pub struct ShadowPass {
    settings: ShadowSettings,
    pending_settings: Option<ShadowSettings>,

    // Caster depth rendering.
    depth_pipeline: wgpu::RenderPipeline,
    // 2x2 cascade tiles for the one shadow-casting directional light.
    atlas: Texture,
    // One tile, re-rendered per shadow-casting spot light (its contents are
    // consumed into a screen-space mask before the next spot needs it).
    spot_depth: Texture,
    // Dynamic-offset uniform pool: one 256-byte slot per caster-per-tile draw.
    draw_uniform_buffer: wgpu::Buffer,
    draw_bind_group: wgpu::BindGroup,
    next_draw_slot: usize,

    // Screen-space masks.
    mask_cascades_pipeline: wgpu::RenderPipeline,
    mask_spot_pipeline: wgpu::RenderPipeline,
    mask_bind_group_layout: wgpu::BindGroupLayout,
    // Scene depth + cascade atlas / spot tile + comparison sampler.
    cascades_mask_bind_group: wgpu::BindGroup,
    spot_mask_bind_group: wgpu::BindGroup,
    // This light's shadow factor (1 = lit), sampled by its shading pass.
    pub mask_temp: Texture,
    // Product of every light's mask this frame; the later splat-overlay pass
    // darkens the Gaussians with it.
    pub mask_accum: Texture,

    // --- Shadow catchers -------------------------------------------------
    // Invisible catcher proxies rendered from the camera into their own depth
    // (they must NOT write the shared scene depth, or they would cull the
    // ground splats behind them).  The catcher mask passes project the frame's
    // shadow maps onto this depth to build `catcher_shadow`, and the overlay
    // multiplies the splats by it.
    catcher_depth_pipeline: wgpu::RenderPipeline,
    catcher_depth: Texture,
    // Per-light scratch (mask target 0) + the multiplied catcher shadow factor
    // (target 1); mirrors mask_temp / mask_accum but keyed to catcher depth.
    catcher_temp: Texture,
    pub catcher_shadow: Texture,
    // Cascade / spot mask bind groups whose receiver-depth slot is the catcher
    // depth instead of the scene depth (they reuse the mask pipelines).
    catcher_cascades_mask_bind_group: wgpu::BindGroup,
    catcher_spot_mask_bind_group: wgpu::BindGroup,
    // Fullscreen overlay that darkens the composited splats by catcher_shadow.
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_bind_group_layout: wgpu::BindGroupLayout,
    overlay_bind_group: wgpu::BindGroup,
    // Artist params for the overlay (shadow density); written each frame.
    overlay_params_buffer: wgpu::Buffer,
}

impl ShadowPass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating ShadowPass");
        let device = &device_resources.device;
        let settings = ShadowSettings::default();

        // Caster depth pipeline: vertex-only, depth-writes into a tile of the
        // atlas / the spot map, with slope-scaled bias against acne.
        let draw_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
                label: Some("ShadowPass_draw_uniform_layout"),
            });
        let draw_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowPass_draw_uniform_buffer"),
            size: (SHADOW_DRAW_STRIDE * MAX_SHADOW_DRAWS) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &draw_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &draw_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
            label: Some("ShadowPass_draw_bind_group"),
        });

        let depth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ShadowPass_depth_pipeline_layout"),
                bind_group_layouts: &[Some(&draw_uniform_layout)],
                immediate_size: 0,
            });
        let depth_shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/shadow_depth.wgsl", device_resources)
            .await;
        let depth_shader = asset_manager.get_shader(&depth_shader_handle);
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ShadowPass_depth_pipeline"),
            layout: Some(&depth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
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
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let (atlas, spot_depth) = Self::make_shadow_maps(device_resources, &settings);

        // Mask pass inputs: scene depth + the light's shadow map + a
        // comparison sampler (hardware PCF taps).
        let mask_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
                label: Some("ShadowPass_mask_bind_group_layout"),
            });

        // Same layout the lighting pass binds its per-light uniform with, so
        // a light's uniform slot serves both its mask and shading passes.
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
                label: Some("ShadowPass_light_bind_group_layout"),
            });

        let mask_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ShadowPass_mask_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&mask_bind_group_layout),
                    Some(&light_bind_group_layout),
                ],
                immediate_size: 0,
            });

        // Two targets: the per-light temp mask (replace) and the accumulation
        // texture (multiply: accum *= mask).
        let multiply_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let mask_targets = [
            Some(wgpu::ColorTargetState {
                format: SHADOW_MASK_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: SHADOW_MASK_FORMAT,
                blend: Some(multiply_blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];

        let mut mask_shader_handles = Vec::new();
        for path in [
            "/engine_assets/shaders/projected_shadow_directional.wgsl",
            "/engine_assets/shaders/projected_shadow_spot.wgsl",
        ] {
            mask_shader_handles.push(asset_manager.load_shader(path, device_resources).await);
        }
        let make_mask_pipeline = |handle: &ShaderHandle, label: &str| -> wgpu::RenderPipeline {
            let shader = asset_manager.get_shader(handle);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&mask_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &mask_targets,
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
        let mask_cascades_pipeline =
            make_mask_pipeline(&mask_shader_handles[0], "ShadowPass_mask_cascades");
        let mask_spot_pipeline = make_mask_pipeline(&mask_shader_handles[1], "ShadowPass_mask_spot");

        let (rw, rh) = (
            device_resources.render_textures[0].texture.width(),
            device_resources.render_textures[0].texture.height(),
        );
        let mask_temp = Texture::new_render_texture_with_format(
            device,
            SHADOW_MASK_FORMAT,
            rw,
            rh,
        )
        .unwrap();
        let mask_accum = Texture::new_render_texture_with_format(
            device,
            SHADOW_MASK_FORMAT,
            rw,
            rh,
        )
        .unwrap();

        let cascades_mask_bind_group = Self::make_mask_bind_group(
            device_resources,
            &mask_bind_group_layout,
            &atlas,
            "cascades",
        );
        let spot_mask_bind_group = Self::make_mask_bind_group(
            device_resources,
            &mask_bind_group_layout,
            &spot_depth,
            "spot",
        );

        // --- Shadow-catcher resources ----------------------------------------
        // Camera-space depth pipeline for the invisible proxies: reuses the
        // depth-only shadow shader (a plain MVP -> depth) and the draw-uniform
        // pool, but without the shadow bias and with no back-face culling (the
        // proxy is viewed from the camera, either side may face us).  Re-fetch
        // the depth shader (the mask/overlay loads above took a mutable borrow
        // of the asset manager, ending the earlier immutable one).
        let depth_shader = asset_manager.get_shader(&depth_shader_handle);
        let catcher_depth_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ShadowPass_catcher_depth_pipeline"),
                layout: Some(&depth_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: depth_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: None,
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

        let catcher_depth =
            Texture::new_depth_texture(device, &device_resources.surface_config, rw, rh).unwrap();
        let catcher_temp =
            Texture::new_render_texture_with_format(device, SHADOW_MASK_FORMAT, rw, rh).unwrap();
        let catcher_shadow =
            Texture::new_render_texture_with_format(device, SHADOW_MASK_FORMAT, rw, rh).unwrap();

        let catcher_cascades_mask_bind_group = Self::make_catcher_mask_bind_group(
            device_resources,
            &mask_bind_group_layout,
            &catcher_depth,
            &atlas,
            "cascades",
        );
        let catcher_spot_mask_bind_group = Self::make_catcher_mask_bind_group(
            device_resources,
            &mask_bind_group_layout,
            &catcher_depth,
            &spot_depth,
            "spot",
        );

        // Fullscreen overlay that multiplies the composited splats by the
        // catcher shadow factor.  Samples the catcher shadow + both depths via
        // textureLoad, so it needs no sampler.
        let overlay_bind_group_layout =
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
                            sample_type: TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(16),
                        },
                        count: None,
                    },
                ],
                label: Some("ShadowPass_overlay_bind_group_layout"),
            });
        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ShadowPass_overlay_pipeline_layout"),
                bind_group_layouts: &[Some(&overlay_bind_group_layout)],
                immediate_size: 0,
            });
        // Multiply color (scene *= factor), keep the scene's alpha untouched.
        let overlay_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let overlay_shader_handle = asset_manager
            .load_shader(
                "/engine_assets/shaders/shadow_catcher_overlay.wgsl",
                device_resources,
            )
            .await;
        let overlay_shader = asset_manager.get_shader(&overlay_shader_handle);
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ShadowPass_overlay_pipeline"),
            layout: Some(&overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: overlay_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: overlay_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::resource::SCENE_COLOR_FORMAT,
                    blend: Some(overlay_blend),
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
        });
        let overlay_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowPass_overlay_params_buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_bind_group = Self::make_overlay_bind_group(
            device_resources,
            &overlay_bind_group_layout,
            &catcher_shadow,
            &catcher_depth,
            &overlay_params_buffer,
        );

        ShadowPass {
            settings,
            pending_settings: None,
            depth_pipeline,
            atlas,
            spot_depth,
            draw_uniform_buffer,
            draw_bind_group,
            next_draw_slot: 0,
            mask_cascades_pipeline,
            mask_spot_pipeline,
            mask_bind_group_layout,
            cascades_mask_bind_group,
            spot_mask_bind_group,
            mask_temp,
            mask_accum,
            catcher_depth_pipeline,
            catcher_depth,
            catcher_temp,
            catcher_shadow,
            catcher_cascades_mask_bind_group,
            catcher_spot_mask_bind_group,
            overlay_pipeline,
            overlay_bind_group_layout,
            overlay_bind_group,
            overlay_params_buffer,
        }
    }

    fn make_shadow_maps(
        device_resources: &DeviceResources,
        settings: &ShadowSettings,
    ) -> (Texture, Texture) {
        let res = settings.resolution;
        let atlas = Texture::new_depth_texture(
            &device_resources.device,
            &device_resources.surface_config,
            res * 2,
            res * 2,
        )
        .unwrap();
        let spot = Texture::new_depth_texture(
            &device_resources.device,
            &device_resources.surface_config,
            res,
            res,
        )
        .unwrap();
        (atlas, spot)
    }

    fn make_mask_bind_group(
        device_resources: &DeviceResources,
        layout: &wgpu::BindGroupLayout,
        shadow_map: &Texture,
        label: &str,
    ) -> wgpu::BindGroup {
        device_resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &device_resources.render_textures[1].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&shadow_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
                    },
                ],
                label: Some(&format!("ShadowPass_mask_bind_group_{label}")),
            })
    }

    // Like make_mask_bind_group, but the receiver-depth slot (binding 0) is the
    // catcher depth instead of the scene depth, so the mask projects the shadow
    // map onto the catcher proxy.
    fn make_catcher_mask_bind_group(
        device_resources: &DeviceResources,
        layout: &wgpu::BindGroupLayout,
        catcher_depth: &Texture,
        shadow_map: &Texture,
        label: &str,
    ) -> wgpu::BindGroup {
        device_resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&catcher_depth.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&shadow_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
                    },
                ],
                label: Some(&format!("ShadowPass_catcher_mask_bind_group_{label}")),
            })
    }

    // The overlay's inputs: the catcher shadow factor plus the catcher and
    // scene depths (all read via textureLoad, so no sampler).
    fn make_overlay_bind_group(
        device_resources: &DeviceResources,
        layout: &wgpu::BindGroupLayout,
        catcher_shadow: &Texture,
        catcher_depth: &Texture,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device_resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&catcher_shadow.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&catcher_depth.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &device_resources.render_textures[1].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
                label: Some("ShadowPass_overlay_bind_group"),
            })
    }

    /// Requests new quality settings; applied at the start of the next frame.
    pub fn request_settings(&mut self, settings: &ShadowSettings) {
        let clamped = settings.clamped();
        if clamped != self.settings {
            self.pending_settings = Some(clamped);
        }
    }

    pub fn settings(&self) -> ShadowSettings {
        self.pending_settings.unwrap_or(self.settings)
    }

    // Applies pending settings, recreating the shadow maps (and their mask
    // bind groups) if the resolution changed.
    fn apply_pending(&mut self, device_resources: &DeviceResources) {
        let Some(pending) = self.pending_settings.take() else {
            return;
        };
        let resolution_changed = pending.resolution != self.settings.resolution;
        self.settings = pending;
        if resolution_changed {
            let (atlas, spot) = Self::make_shadow_maps(device_resources, &self.settings);
            self.atlas = atlas;
            self.spot_depth = spot;
            self.rebuild_mask_bind_groups(device_resources);
        }
    }

    fn rebuild_mask_bind_groups(&mut self, device_resources: &DeviceResources) {
        self.cascades_mask_bind_group = Self::make_mask_bind_group(
            device_resources,
            &self.mask_bind_group_layout,
            &self.atlas,
            "cascades",
        );
        self.spot_mask_bind_group = Self::make_mask_bind_group(
            device_resources,
            &self.mask_bind_group_layout,
            &self.spot_depth,
            "spot",
        );
        self.catcher_cascades_mask_bind_group = Self::make_catcher_mask_bind_group(
            device_resources,
            &self.mask_bind_group_layout,
            &self.catcher_depth,
            &self.atlas,
            "cascades",
        );
        self.catcher_spot_mask_bind_group = Self::make_catcher_mask_bind_group(
            device_resources,
            &self.mask_bind_group_layout,
            &self.catcher_depth,
            &self.spot_depth,
            "spot",
        );
    }

    /// Window resize: the screen-space mask textures track the render
    /// resolution, and the mask bind groups reference the recreated scene
    /// depth.  The lighting pass must rebuild its bind group after this (it
    /// samples `mask_temp`).
    pub fn resize(&mut self, device_resources: &DeviceResources) {
        let (rw, rh) = (
            device_resources.render_textures[0].texture.width(),
            device_resources.render_textures[0].texture.height(),
        );
        self.mask_temp = Texture::new_render_texture_with_format(
            &device_resources.device,
            SHADOW_MASK_FORMAT,
            rw,
            rh,
        )
        .unwrap();
        self.mask_accum = Texture::new_render_texture_with_format(
            &device_resources.device,
            SHADOW_MASK_FORMAT,
            rw,
            rh,
        )
        .unwrap();
        // Catcher targets track the render resolution too; recreate them before
        // rebuild_mask_bind_groups so the catcher mask groups pick up the new
        // catcher depth, then refresh the overlay (it references scene depth).
        self.catcher_depth = Texture::new_depth_texture(
            &device_resources.device,
            &device_resources.surface_config,
            rw,
            rh,
        )
        .unwrap();
        self.catcher_temp = Texture::new_render_texture_with_format(
            &device_resources.device,
            SHADOW_MASK_FORMAT,
            rw,
            rh,
        )
        .unwrap();
        self.catcher_shadow = Texture::new_render_texture_with_format(
            &device_resources.device,
            SHADOW_MASK_FORMAT,
            rw,
            rh,
        )
        .unwrap();
        self.rebuild_mask_bind_groups(device_resources);
        self.overlay_bind_group = Self::make_overlay_bind_group(
            device_resources,
            &self.overlay_bind_group_layout,
            &self.catcher_shadow,
            &self.catcher_depth,
            &self.overlay_params_buffer,
        );
    }

    // Resets the per-frame draw pool and clears the accumulation texture to
    // fully lit (each light's mask multiplies into it).
    fn begin_frame(&mut self, ctx: &mut RenderContext) {
        self.apply_pending(ctx.device);
        self.next_draw_slot = 0;

        let mut encoder =
            ctx.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowPass::begin_frame()"),
                });
        // Both the scene and the catcher shadow accumulators start fully lit;
        // each shadow-casting light multiplies its mask into them.
        for view in [&self.mask_accum.view, &self.catcher_shadow.view] {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow accum clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
        }
        ctx.device.queue.submit(std::iter::once(encoder.finish()));
    }

    // Renders `casters` into tiles of a shadow map.  `tiles` gives each
    // tile's pixel origin and light view-projection; the whole map is cleared
    // first (each light's map is consumed into a mask before the next light
    // renders, so nothing outlives the frame).
    fn render_depth(
        &mut self,
        ctx: &mut RenderContext,
        casters: &[ShadowCaster],
        tiles: &[(u32, u32, CgMat4)],
        use_atlas: bool,
    ) {
        let device_resources = &mut *ctx.device;
        let asset_manager = &mut *ctx.assets;
        let resolution = self.settings.resolution;

        let mut encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowPass::render_depth()"),
                });
        let target_view = if use_atlas {
            &self.atlas.view
        } else {
            &self.spot_depth.view
        };
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shadow depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target_view,
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
        render_pass.set_pipeline(&self.depth_pipeline);

        let model_mappings = asset_manager.get_model_mappings();
        for (x, y, view_proj) in tiles {
            render_pass.set_viewport(
                *x as f32,
                *y as f32,
                resolution as f32,
                resolution as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(*x, *y, resolution, resolution);
            for caster in casters {
                if self.next_draw_slot >= MAX_SHADOW_DRAWS {
                    break; // Pool exhausted; remaining casters drop out this frame.
                }
                let Some(model) = model_mappings.get(&caster.model) else {
                    continue;
                };
                let mvp: [[f32; 4]; 4] = (view_proj * caster.world).into();
                let offset = (self.next_draw_slot * SHADOW_DRAW_STRIDE) as u64;
                device_resources.queue.write_buffer(
                    &self.draw_uniform_buffer,
                    offset,
                    bytemuck::cast_slice(&[mvp]),
                );
                render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.set_bind_group(0, &self.draw_bind_group, &[offset as u32]);
                render_pass.draw_indexed(0..model.num_indices, 0, 0..1);
                self.next_draw_slot += 1;
            }
        }

        drop(render_pass);
        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }

    // Projects the just-rendered shadow map into screen space: writes this
    // light's shadow factor into mask_temp and multiplies it into mask_accum.
    // `light_bind_group` is the light's uniform slot (it holds the tile
    // matrices/rects the mask shader needs).
    fn render_mask(
        &mut self,
        ctx: &mut RenderContext,
        kind: MaskKind,
        light_bind_group: &wgpu::BindGroup,
    ) {
        let device_resources = &mut *ctx.device;
        let mut encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowPass::render_mask()"),
                });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shadow mask"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.mask_temp.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.mask_accum.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        let (pipeline, bind_group) = match kind {
            MaskKind::Cascades => (&self.mask_cascades_pipeline, &self.cascades_mask_bind_group),
            MaskKind::Spot => (&self.mask_spot_pipeline, &self.spot_mask_bind_group),
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(1, light_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        drop(render_pass);
        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }

    // Renders the invisible catcher proxies into their own full-screen depth
    // from the camera's view (clearing it to far first).  Kept separate from
    // the shared scene depth so the proxies don't occlude the ground splats
    // behind them.  `view_proj` must match the scene's camera projection so the
    // overlay can compare catcher and scene depth.  Always call it (even with
    // no casters) so the depth is cleared to "no catcher" for the frame.
    fn render_catcher_depth(
        &mut self,
        ctx: &mut RenderContext,
        casters: &[ShadowCaster],
        view_proj: CgMat4,
    ) {
        let device_resources = &mut *ctx.device;
        let asset_manager = &mut *ctx.assets;

        let mut encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowPass::render_catcher_depth()"),
                });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Catcher depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.catcher_depth.view,
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
        render_pass.set_pipeline(&self.catcher_depth_pipeline);

        let model_mappings = asset_manager.get_model_mappings();
        for caster in casters {
            if self.next_draw_slot >= MAX_SHADOW_DRAWS {
                break; // Pool exhausted; remaining catchers drop out this frame.
            }
            let Some(model) = model_mappings.get(&caster.model) else {
                continue;
            };
            let mvp: [[f32; 4]; 4] = (view_proj * caster.world).into();
            let offset = (self.next_draw_slot * SHADOW_DRAW_STRIDE) as u64;
            device_resources.queue.write_buffer(
                &self.draw_uniform_buffer,
                offset,
                bytemuck::cast_slice(&[mvp]),
            );
            render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
            render_pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.draw_bind_group, &[offset as u32]);
            render_pass.draw_indexed(0..model.num_indices, 0, 0..1);
            self.next_draw_slot += 1;
        }

        drop(render_pass);
        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }

    // Same as render_mask, but projects the light's shadow map onto the catcher
    // depth (not the scene depth) and multiplies the result into catcher_shadow.
    // catcher_temp is a throwaway (the mask pipeline's replace target).
    fn render_catcher_mask(
        &mut self,
        ctx: &mut RenderContext,
        kind: MaskKind,
        light_bind_group: &wgpu::BindGroup,
    ) {
        let device_resources = &mut *ctx.device;
        let mut encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowPass::render_catcher_mask()"),
                });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Catcher shadow mask"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.catcher_temp.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.catcher_shadow.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        let (pipeline, bind_group) = match kind {
            MaskKind::Cascades => (
                &self.mask_cascades_pipeline,
                &self.catcher_cascades_mask_bind_group,
            ),
            MaskKind::Spot => (&self.mask_spot_pipeline, &self.catcher_spot_mask_bind_group),
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(1, light_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        drop(render_pass);
        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Darkens the composited scene color (the Gaussian splats) by this frame's
    /// catcher shadow factor, where a catcher proxy is the frontmost surface.
    /// Runs after the splat pass; a no-op wherever no catcher was rendered
    /// (catcher depth stays at far -> factor 1).
    pub fn render_catcher_overlay(&mut self, ctx: &mut RenderContext) {
        let device_resources = &mut *ctx.device;
        // Upload the current shadow density (padded to 16 bytes) for the overlay.
        let params = [self.settings.density, 0.0, 0.0, 0.0];
        device_resources.queue.write_buffer(
            &self.overlay_params_buffer,
            0,
            bytemuck::cast_slice(&params),
        );
        let mut encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowPass::render_catcher_overlay()"),
                });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Catcher overlay"),
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
            render_pass.set_pipeline(&self.overlay_pipeline);
            render_pass.set_bind_group(0, &self.overlay_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        device_resources.queue.submit(std::iter::once(encoder.finish()));
    }

    // Cascade tiles for the shadowed directional light: split the camera
    // frustum out to settings.distance, fit a texel-snapped light-space ortho
    // box around each slice's bounding sphere, and render the casters into
    // the 2x2 atlas.
    fn render_cascades(
        &mut self,
        ctx: &mut RenderContext,
        casters: &[ShadowCaster],
        light: &Light,
    ) -> Vec<ShadowTile> {
        let camera = ctx.camera;
        let config = ctx.config;
        let num_cascades = self.settings.num_cascades as usize;
        let resolution = self.settings.resolution;

        let light_dir = {
            let dir = light.get_direction();
            if dir.magnitude2() > 0.0001 {
                dir.normalize()
            } else {
                CgVec3::new(0.0, -1.0, 0.0)
            }
        };
        let up = shadow_up(light_dir);

        let (_, forward, right) = camera.calculate_view_matrix();
        let cam_pos = camera.get_position();
        let cam_up = forward.cross(right).normalize();
        let tan_v = (config.fov.to_radians() * 0.5).tan();
        let tan_h = tan_v * (config.window_width as f32 / config.window_height as f32);

        // Practical split scheme: halfway between uniform and logarithmic.
        let near = 0.1_f32;
        let far = self.settings.distance;
        let split = |t: f32| -> f32 {
            if t <= 0.0 {
                return near;
            }
            let uniform = near + (far - near) * t;
            let logarithmic = near * (far / near).powf(t);
            (uniform + logarithmic) * 0.5
        };

        let mut tiles = Vec::with_capacity(num_cascades);
        let mut tile_origins = Vec::with_capacity(num_cascades);
        for i in 0..num_cascades {
            let slice_near = split(i as f32 / num_cascades as f32);
            let slice_far = split((i + 1) as f32 / num_cascades as f32);

            // Bounding sphere of the frustum slice (corner spans are symmetric
            // around the view axis, so the centroid sits on it).
            let center = cam_pos + forward * (slice_near + slice_far) * 0.5;
            let corner_dist = |d: f32| -> f32 {
                let corner = cam_pos + forward * d + right * (d * tan_h) + cam_up * (d * tan_v);
                (corner - center).magnitude()
            };
            let mut radius = corner_dist(slice_near).max(corner_dist(slice_far));
            // Quantize the radius so tiny camera moves don't resize the box
            // (resizing re-scales texels and makes edges shimmer).
            radius = (radius * 16.0).ceil() / 16.0;

            // Pull the light eye back far enough to catch casters behind the
            // slice (e.g. a roof above the view) on their way to it.
            let backup = radius.max(20.0);
            let eye = center - light_dir * (radius + backup);
            let view = CgMat4::look_at_rh(
                CgPoint::from_vec(eye),
                CgPoint::from_vec(center),
                up,
            );
            let proj = ortho_wgpu(radius, 0.0, backup + radius * 2.0);
            let mut view_proj = proj * view;

            // Texel snapping: shift the projection so the world origin lands
            // on a texel boundary, killing edge shimmer as the camera pans.
            let origin = view_proj * CgVec4::new(0.0, 0.0, 0.0, 1.0);
            let half_res = resolution as f32 * 0.5;
            let snap_x = ((origin.x * half_res).round() - origin.x * half_res) / half_res;
            let snap_y = ((origin.y * half_res).round() - origin.y * half_res) / half_res;
            view_proj =
                CgMat4::from_translation(CgVec3::new(snap_x, snap_y, 0.0)) * view_proj;

            // 2x2 atlas layout.
            let (tx, ty) = ((i as u32 % 2) * resolution, (i as u32 / 2) * resolution);
            tiles.push(ShadowTile {
                view_proj,
                rect: [
                    (i % 2) as f32 * 0.5,
                    (i / 2) as f32 * 0.5,
                    0.5,
                    0.5,
                ],
            });
            tile_origins.push((tx, ty, view_proj));
        }

        self.render_depth(ctx, casters, &tile_origins, true);
        tiles
    }

    // The spot light's single projected shadow map, rendered into the shared
    // spot tile (consumed into a screen-space mask right after).
    fn render_spot(
        &mut self,
        ctx: &mut RenderContext,
        casters: &[ShadowCaster],
        light: &Light,
    ) -> ShadowTile {
        let position = light.get_position();
        let direction = {
            let dir = light.get_direction();
            if dir.magnitude2() > 0.0001 {
                dir.normalize()
            } else {
                CgVec3::new(0.0, 0.0, 1.0)
            }
        };
        let fovy = (light.get_spot_angle().clamp(1.0, 85.0) * 2.0).to_radians();
        let proj = perspective_wgpu(fovy, 1.0, 0.05, light.get_range().max(0.5));
        let view = CgMat4::look_at_rh(
            CgPoint::from_vec(position),
            CgPoint::from_vec(position + direction),
            shadow_up(direction),
        );
        let view_proj = proj * view;

        self.render_depth(ctx, casters, &[(0, 0, view_proj)], false);
        ShadowTile {
            view_proj,
            rect: [0.0, 0.0, 1.0, 1.0],
        }
    }

    /// The screen-space product of every shadow mask this frame (1 = fully
    /// lit).  The Gaussian-splat overlay will multiply the splats by it.
    pub fn shadow_accum(&self) -> &Texture {
        &self.mask_accum
    }
}

/// Accumulates the scene's lights onto the scene color target from the
/// G-buffer: the pass clears to the config's clear color, then draws one
/// additive fullscreen triangle per light with the pipeline matching its
/// type.  Shadowed lights (the first directional caster, every spot caster)
/// first get their shadow map + screen-space mask rendered via `ShadowPass`,
/// and their shading pass multiplies by the mask.
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
    // The skylight pipeline alone binds an extra texture_cube + sampler (its
    // baked environment map, see Renderer::bake_skylight_cubemap), so it gets
    // its own bind-group-1 layout instead of sharing `light_uniforms`'
    // uniform-only layout. Built fresh each frame a skylight renders (rare
    // enough not to warrant a pre-built pool like `light_uniforms`).
    skylight_env_bind_group_layout: wgpu::BindGroupLayout,
    // A 1x1 black cube texture bound in place of a light's real environment
    // map before it has one baked -- keeps the bind group valid without a
    // branch in the pipeline/layout.
    fallback_cube: CubeTexture,
    // GGX-prefilters a skylight's baked mip-0 cube into its roughness mip
    // chain (see Renderer::bake_skylight_cubemap). Built once here since it's
    // only ever invoked from the one-shot bake, not per frame.
    skylight_prefilter_pipeline: wgpu::RenderPipeline,
    skylight_prefilter_bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkylightPrefilterUniform {
    inv_view_proj: [[f32; 4]; 4],
    // x: roughness for this mip, y: this mip's face size in texels.
    params: [f32; 4],
}

impl LightingPass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
        shadow_pass: &ShadowPass,
    ) -> Self {
        log!("Creating LightingPass");
        let device = &device_resources.device;

        // G-buffer inputs: albedo / normal / metallic-roughness / depth read
        // with textureLoad, plus the current light's screen-space shadow mask.
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
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                ],
                label: Some("LightingPass_gbuffer_bind_group_layout"),
            });
        let gbuffer_bind_group = Self::make_gbuffer_bind_group(
            device_resources,
            &gbuffer_bind_group_layout,
            shadow_pass,
        );

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

        // The skylight's bind-group-1 layout: the same uniform buffer plus a
        // texture_cube + sampler for its baked environment map.
        let skylight_env_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("LightingPass_skylight_env_bind_group_layout"),
            });
        let skylight_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LightingPass_skylight_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&gbuffer_bind_group_layout),
                    Some(&skylight_env_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let fallback_cube = CubeTexture::new(device, SCENE_COLOR_FORMAT, 1);

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

        // Shader loading is the only async step; fetch everything up front,
        // then build the pipelines synchronously.
        let shader_paths = [
            "/engine_assets/shaders/light_skylight.wgsl",
            "/engine_assets/shaders/light_directional.wgsl",
            "/engine_assets/shaders/light_point.wgsl",
            "/engine_assets/shaders/light_spot.wgsl",
            "/engine_assets/shaders/skylight_prefilter.wgsl",
        ];
        let mut shader_handles = Vec::with_capacity(shader_paths.len());
        for path in shader_paths {
            shader_handles.push(asset_manager.load_shader(path, device_resources).await);
        }

        let make_pipeline = |handle: &ShaderHandle,
                              label: &str,
                              layout: &wgpu::PipelineLayout|
         -> wgpu::RenderPipeline {
            let shader = asset_manager.get_shader(handle);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
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
                        format: crate::resource::SCENE_COLOR_FORMAT,
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

        let skylight_pipeline = make_pipeline(
            &shader_handles[0],
            "LightingPass_skylight",
            &skylight_pipeline_layout,
        );
        let directional_pipeline =
            make_pipeline(&shader_handles[1], "LightingPass_directional", &pipeline_layout);
        let point_pipeline =
            make_pipeline(&shader_handles[2], "LightingPass_point", &pipeline_layout);
        let spot_pipeline =
            make_pipeline(&shader_handles[3], "LightingPass_spot", &pipeline_layout);

        // Prefilter pass: uniform (view/roughness) + the source cube (its
        // mip-0-only view, so it's never sampled and rendered-to at once) +
        // sampler. Writes replace (no blend) since each mip/face texel is
        // computed fresh from scratch every bake.
        let skylight_prefilter_bind_group_layout =
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
                ],
                label: Some("LightingPass_skylight_prefilter_bind_group_layout"),
            });
        let skylight_prefilter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LightingPass_skylight_prefilter_pipeline_layout"),
                bind_group_layouts: &[Some(&skylight_prefilter_bind_group_layout)],
                immediate_size: 0,
            });
        let prefilter_shader = asset_manager.get_shader(&shader_handles[4]);
        let skylight_prefilter_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("LightingPass_skylight_prefilter"),
                layout: Some(&skylight_prefilter_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: prefilter_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: prefilter_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: crate::resource::SCENE_COLOR_FORMAT,
                        blend: None,
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
            });

        LightingPass {
            skylight_pipeline,
            directional_pipeline,
            point_pipeline,
            spot_pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group,
            light_uniforms,
            skylight_env_bind_group_layout,
            fallback_cube,
            skylight_prefilter_pipeline,
            skylight_prefilter_bind_group_layout,
        }
    }

    /// GGX-prefilters `cube_texture`'s mip-0 capture into its mips 1.. (see
    /// `CubeTexture::mip_count_for`), one fullscreen draw per (mip, face). A
    /// no-op if the texture only has one mip (e.g. the 1x1 fallback cube).
    /// One-shot cost, called only from `Renderer::bake_skylight_cubemap`.
    pub fn prefilter_skylight_mips(
        &self,
        device_resources: &DeviceResources,
        cube_texture: &CubeTexture,
        face_size: u32,
    ) {
        if cube_texture.mip_count <= 1 {
            return;
        }
        let device = &device_resources.device;

        // Mip 0 only: the prefilter must never sample a mip it could also be
        // writing to in the same pass.
        let source_view = cube_texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Skylight Env Cubemap Mip0 Source View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: Some(1),
            array_layer_count: Some(6),
            ..Default::default()
        });

        let max_mip = cube_texture.mip_count - 1;
        for mip in 1..cube_texture.mip_count {
            let mip_size = (face_size >> mip).max(1);
            let roughness = mip as f32 / max_mip as f32;
            for (face, (dir, up)) in Texture::CUBE_FACE_DIRECTIONS.iter().enumerate() {
                let cam = Camera::from_look(
                    CgVec3::new(0.0, 0.0, 0.0),
                    CgVec3::new(dir[0], dir[1], dir[2]),
                    CgVec3::new(up[0], up[1], up[2]),
                );
                let (view_matrix, _, _) = cam.calculate_view_matrix();
                let proj_matrix = cgmath::perspective(cgmath::Deg(90.0), 1.0, 0.1, 10.0);
                let inv_view_proj: [[f32; 4]; 4] = (proj_matrix * view_matrix)
                    .invert()
                    .unwrap_or_else(cgmath::Matrix4::identity)
                    .into();

                let uniform = SkylightPrefilterUniform {
                    inv_view_proj,
                    params: [roughness, mip_size as f32, 0.0, 0.0],
                };
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LightingPass_skylight_prefilter_uniform"),
                    contents: bytemuck::cast_slice(&[uniform]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.skylight_prefilter_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&source_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&cube_texture.sampler),
                        },
                    ],
                    label: Some("LightingPass_skylight_prefilter_bind_group"),
                });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("skylight prefilter"),
                    });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("skylight prefilter mip"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &cube_texture.face_views[mip as usize][face],
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.skylight_prefilter_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
                device_resources.queue.submit(std::iter::once(encoder.finish()));
            }
        }
    }

    fn make_gbuffer_bind_group(
        device_resources: &DeviceResources,
        layout: &wgpu::BindGroupLayout,
        shadow_pass: &ShadowPass,
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &shadow_pass.mask_temp.view,
                        ),
                    },
                ],
                label: Some("LightingPass_gbuffer_bind_group"),
            })
    }

    /// Rebind the (recreated) G-buffer / depth / shadow-mask textures after a
    /// window resize.
    pub fn resize(&mut self, device_resources: &DeviceResources, shadow_pass: &ShadowPass) {
        self.gbuffer_bind_group = Self::make_gbuffer_bind_group(
            device_resources,
            &self.gbuffer_bind_group_layout,
            shadow_pass,
        );
    }

    /// Clears the scene color target to the config's clear color and adds
    /// every light's contribution from the G-buffer, rendering shadow maps +
    /// screen-space masks along the way for the lights that cast them.
    pub fn render(
        &mut self,
        ctx: &mut RenderContext,
        lights: &HashMap<u32, Light>,
        actors: &HashMap<u32, Actor>,
        shadow_pass: &mut ShadowPass,
        env_cubemaps: &HashMap<u32, CubeTexture>,
    ) {
        shadow_pass.begin_frame(ctx);

        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let (view_matrix, _, _) = game_camera.calculate_view_matrix();
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );
        let inv_view_proj: [[f32; 4]; 4] = (proj_matrix * view_matrix)
            .invert()
            .unwrap_or_else(cgmath::Matrix4::identity)
            .into();
        let camera_pos = game_camera.get_position();
        // The G-buffer renders at render_resolution (not the window size);
        // the shaders rebuild UVs from pixel coords with these dimensions.
        let (rw, rh) = game_config.render_resolution();
        let clear_color = game_config.clear_color;
        let settings = shadow_pass.settings;

        // World-space transform of an actor, shared by the caster and catcher
        // gathers below.
        let actor_world = |actor: &Actor| -> CgMat4 {
            cgmath::Matrix4::from_translation(actor.get_position())
                * cgmath::Matrix4::from(actor.get_rotation())
                * cgmath::Matrix4::from_nonuniform_scale(
                    actor.get_scale().x,
                    actor.get_scale().y,
                    actor.get_scale().z,
                )
        };

        // World-layer casters, gathered once and re-projected per shadow tile.
        // Catcher proxies are excluded -- they receive shadow, they don't cast.
        let casters: Vec<ShadowCaster> = actors
            .values()
            .filter(|actor| {
                actor.get_layer().0 == SceneLayer::World && !actor.is_shadow_catcher()
            })
            .map(|actor| ShadowCaster {
                model: actor.get_model(),
                world: actor_world(actor),
            })
            .collect();

        // The invisible catcher proxies, rendered into their own depth so the
        // masks can project the casters' shadows onto them.
        let catcher_casters: Vec<ShadowCaster> = actors
            .values()
            .filter(|actor| {
                actor.get_layer().0 == SceneLayer::World && actor.is_shadow_catcher()
            })
            .map(|actor| ShadowCaster {
                model: actor.get_model(),
                world: actor_world(actor),
            })
            .collect();

        // Camera-view catcher depth (matches the scene projection so the overlay
        // can compare depths).  Always run so the depth is cleared each frame.
        shadow_pass.render_catcher_depth(ctx, &catcher_casters, proj_matrix * view_matrix);

        // Clear the scene target up front; each light then adds onto it in
        // its own pass (shadowed lights interleave their depth/mask renders).
        {
            let mut encoder =
                ctx.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("LightingPass clear"),
                    });
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Deferred Lighting clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.device.render_textures[0].view,
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
            drop(_clear);
            ctx.device.queue.submit(std::iter::once(encoder.finish()));
        }

        // Deterministic order, and the first shadow-casting directional light
        // owns the cascade atlas (only one directional can cast per frame).
        let mut sorted_lights: Vec<&Light> = lights.values().collect();
        sorted_lights.sort_by_key(|light| light.id);
        let cascade_owner = sorted_lights
            .iter()
            .find(|l| l.get_light_type() == LightType::Directional && l.casts_shadow())
            .map(|l| l.id);

        for (slot, light) in sorted_lights.iter().enumerate().take(MAX_LIGHTS) {
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

            let has_env_cubemap = light.use_env_cubemap() && env_cubemaps.contains_key(&light.id);
            let mut uniform = LightUniform {
                inv_view_proj,
                position_range: [position.x, position.y, position.z, light.get_range()],
                // w: for a skylight, its plain intensity (the cubemap sample is
                // already full-color radiance -- unlike the analytic gradient,
                // it must not also be tinted by the light's Color swatch).
                // Unused by other light types, which keep cos(outer angle) here.
                direction_cone: [
                    direction.x,
                    direction.y,
                    direction.z,
                    if light.get_light_type() == LightType::Skylight {
                        light.get_intensity()
                    } else {
                        outer_rad.cos()
                    },
                ],
                color_cone: [color.x, color.y, color.z, inner_rad.cos()],
                // w: 1.0 if this skylight has a baked environment cubemap to
                // sample instead of the top/bottom gradient (unused by other
                // light types).
                color2: [color2.x, color2.y, color2.z, if has_env_cubemap { 1.0 } else { 0.0 }],
                // w: 1.0 if this skylight should also draw its cubemap as the
                // background where no geometry was rendered -- a debug view of
                // the bake (unused by other light types).
                camera_pos: [
                    camera_pos.x,
                    camera_pos.y,
                    camera_pos.z,
                    if has_env_cubemap && light.show_env_as_skybox() {
                        1.0
                    } else {
                        0.0
                    },
                ],
                target_dims: [rw as f32, rh as f32, 0.0, 0.0],
                ..Default::default()
            };

            // Shadowed lights: render the depth map, fill the tile matrices,
            // then bake the screen-space mask their shading pass will read.
            let light_type = light.get_light_type();
            let mask_kind = if light_type == LightType::Directional
                && cascade_owner == Some(light.id)
            {
                let tiles = shadow_pass.render_cascades(ctx, &casters, light);
                for (i, tile) in tiles.iter().enumerate() {
                    uniform.shadow_matrices[i] = tile.view_proj.into();
                    uniform.shadow_rects[i] = tile.rect;
                }
                uniform.shadow_params = [tiles.len() as f32, SHADOW_DEPTH_BIAS, 0.0, 0.0];
                let atlas_size = (settings.resolution * 2) as f32;
                uniform.target_dims[2] = atlas_size;
                uniform.target_dims[3] = atlas_size;
                Some(MaskKind::Cascades)
            } else if light_type == LightType::Spot && light.casts_shadow() {
                let tile = shadow_pass.render_spot(ctx, &casters, light);
                uniform.shadow_matrices[0] = tile.view_proj.into();
                uniform.shadow_rects[0] = tile.rect;
                uniform.shadow_params = [1.0, SHADOW_DEPTH_BIAS, 0.0, 0.0];
                uniform.target_dims[2] = settings.resolution as f32;
                uniform.target_dims[3] = settings.resolution as f32;
                Some(MaskKind::Spot)
            } else {
                None
            };

            ctx.device.queue.write_buffer(
                &self.light_uniforms[slot].0,
                0,
                bytemuck::cast_slice(&[uniform]),
            );
            if let Some(kind) = mask_kind {
                shadow_pass.render_mask(ctx, kind, &self.light_uniforms[slot].1);
                // Project the same shadow map onto the catcher proxies too.
                if !catcher_casters.is_empty() {
                    shadow_pass.render_catcher_mask(ctx, kind, &self.light_uniforms[slot].1);
                }
            }

            // The light's additive shading pass.
            let pipeline = match light_type {
                LightType::Skylight => &self.skylight_pipeline,
                LightType::Directional => &self.directional_pipeline,
                LightType::Point => &self.point_pipeline,
                LightType::Spot => &self.spot_pipeline,
            };
            // Skylights bind an extra texture_cube/sampler pair, so they get a
            // one-off bind group each frame instead of the pre-built
            // `light_uniforms` pool (see `skylight_env_bind_group_layout`).
            let skylight_bind_group = (light_type == LightType::Skylight).then(|| {
                let env = env_cubemaps.get(&light.id).unwrap_or(&self.fallback_cube);
                let buffer = ctx.device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LightingPass_skylight_uniform"),
                    contents: bytemuck::cast_slice(&[uniform]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let env_uniform = SkylightEnvUniform {
                    sh: env.sh_coeffs,
                    // z: 1.0 if `sh` holds a real SH projection. wasm never
                    // populates it (see Renderer::bake_skylight_cubemap --
                    // the CPU readback it needs isn't available there), so
                    // the shader falls back to sampling the roughest
                    // GPU-prefiltered mip by normal instead.
                    mip_params: [
                        (env.mip_count.max(1) - 1) as f32,
                        0.0,
                        if cfg!(target_arch = "wasm32") { 0.0 } else { 1.0 },
                        0.0,
                    ],
                };
                let env_buffer = ctx.device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LightingPass_skylight_env_uniform"),
                    contents: bytemuck::cast_slice(&[env_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                ctx.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.skylight_env_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
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
                            resource: env_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("LightingPass_skylight_bind_group"),
                })
            });
            let mut encoder =
                ctx.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("LightingPass light"),
                    });
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Deferred Light"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.device.render_textures[0].view,
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
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
            render_pass.set_bind_group(
                1,
                skylight_bind_group.as_ref().unwrap_or(&self.light_uniforms[slot].1),
                &[],
            );
            render_pass.draw(0..3, 0..1);
            drop(render_pass);
            ctx.device.queue.submit(std::iter::once(encoder.finish()));
        }
    }
}

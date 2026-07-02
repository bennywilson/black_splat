use std::mem::size_of;

use crate::{kb_assets::*, kb_config::*, kb_resource::*, log};

/// A screen-space button: a solid quad with a centered text label, drawn over
/// the final frame.  `rect` is (x, y, width, height) in physical pixels.  The
/// caller owns all styling (e.g. brightening `background` on hover/press); the
/// label is drawn by the renderer's text brush at a size derived from the rect
/// height, so buttons scale up cleanly on high-DPI touch screens.
#[derive(Clone)]
pub struct KbUiButton {
    pub label: String,
    pub rect: (f32, f32, f32, f32),
    pub background: [f32; 4],
    pub border: [f32; 4],
    pub text_color: [f32; 4],
}

const MAX_UI_QUADS: usize = 64;
const BORDER_PX: f32 = 2.0;

/// Draws `KbUiButton` background quads (labels are handled by the text pass).
/// Vertices are converted to NDC on the CPU each frame -- button counts are
/// tiny, so one buffer upload beats carrying a screen-size uniform around.
pub struct KbUiRenderGroup {
    vertex_buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
}

impl KbUiRenderGroup {
    pub async fn new(
        device_resources: &KbDeviceResources<'_>,
        asset_manager: &mut KbAssetManager,
    ) -> Self {
        log!("Creating KbUiRenderGroup");
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KbUiRenderGroup_vertex_buffer"),
            mapped_at_creation: false,
            size: (size_of::<KbVertex>() * 6 * MAX_UI_QUADS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/ui_overlay.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("KbUiRenderGroup_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("KbUiRenderGroup_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[KbVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // This pass targets the swapchain view directly, so it must
                    // match the surface format EXACTLY -- no add_srgb_suffix().
                    // Chrome WebGPU surfaces are non-sRGB, and a suffixed format
                    // there fails pipeline validation and the pass never draws.
                    // (Offscreen render_textures are sRGB; the surface may not be.)
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            // UI draws last, straight onto the final view; nothing occludes it.
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        KbUiRenderGroup {
            vertex_buffer,
            pipeline,
        }
    }

    /// Records a pass into `view` drawing each button as a border quad with the
    /// background quad inset on top.  Call before the text pass so labels land
    /// on top of the quads.
    pub fn render(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        device_resources: &KbDeviceResources,
        buttons: &[KbUiButton],
        game_config: &KbConfig,
    ) {
        if buttons.is_empty() {
            return;
        }

        let screen_w = (game_config.window_width.max(1)) as f32;
        let screen_h = (game_config.window_height.max(1)) as f32;
        let to_ndc_x = |x: f32| (x / screen_w) * 2.0 - 1.0;
        let to_ndc_y = |y: f32| 1.0 - (y / screen_h) * 2.0;

        // On an sRGB surface (native) the hardware gamma-encodes our linear
        // colors at write time.  A non-sRGB surface (e.g. Chrome WebGPU) stores
        // them raw, which displays much darker -- so encode on the CPU there to
        // keep the two platforms looking identical.  Alpha stays linear.
        let encode = if device_resources.surface_config.format.is_srgb() {
            |c: [f32; 4]| c
        } else {
            |c: [f32; 4]| {
                let srgb = |v: f32| {
                    if v <= 0.003_130_8 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    }
                };
                [srgb(c[0]), srgb(c[1]), srgb(c[2]), c[3]]
            }
        };

        let mut vertices = Vec::<KbVertex>::new();
        let mut push_quad = |rect: (f32, f32, f32, f32), color: [f32; 4]| {
            if vertices.len() + 6 > 6 * MAX_UI_QUADS {
                return;
            }
            let (x0, y0) = (to_ndc_x(rect.0), to_ndc_y(rect.1));
            let (x1, y1) = (to_ndc_x(rect.0 + rect.2), to_ndc_y(rect.1 + rect.3));
            let corner = |x: f32, y: f32| KbVertex {
                position: [x, y, 0.0],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                color,
            };
            let (tl, tr, bl, br) = (corner(x0, y0), corner(x1, y0), corner(x0, y1), corner(x1, y1));
            vertices.extend_from_slice(&[tl, bl, br, tl, br, tr]);
        };

        for button in buttons {
            push_quad(button.rect, encode(button.border));
            let (x, y, w, h) = button.rect;
            push_quad(
                (
                    x + BORDER_PX,
                    y + BORDER_PX,
                    (w - 2.0 * BORDER_PX).max(0.0),
                    (h - 2.0 * BORDER_PX).max(0.0),
                ),
                encode(button.background),
            );
        }

        device_resources.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(vertices.as_slice()),
        );

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("UI Overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..vertices.len() as u32, 0..1);
    }
}

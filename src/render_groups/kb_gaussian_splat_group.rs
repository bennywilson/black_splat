use std::collections::HashMap;
use std::mem::size_of;
use wgpu::util::DeviceExt;

use crate::{kb_assets::*, kb_config::*, kb_game_object::*, kb_resource::*, log};

/// 700k splats * 160 bytes = 112 MB, comfortably under wgpu's default 128 MiB
/// storage-buffer binding limit.  Larger point clouds are clamped to this.
pub const MAX_SPLATS: usize = 700_000;

/// GPU-side gaussian splat record.  Layout mirrors the WGSL `Splat` struct in
/// gaussian_splat.wgsl (160 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KbSplatInstance {
    pub position: [f32; 4],      // xyz position, w unused
    pub scale_opacity: [f32; 4], // linear scale xyz, normalized opacity w
    pub rotation: [f32; 4],      // quaternion x,y,z,w
    pub sh0: [f32; 4],           // degree-0 SH (base color), w unused
    pub sh_rest: [f32; 24],      // 8 higher-order coeffs * 3 channels
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct KbSplatUniform {
    pub view: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub splat_params: [f32; 4],   // falloff, scale, contrast, num_splats
    pub splat_params_2: [f32; 4], // max_sh_degree, overall_scale, _, _
}

/// Tunable rendering parameters (see the original blk GaussianSplatComponent).
#[derive(Clone, Copy)]
pub struct KbSplatParams {
    pub falloff: f32,
    pub scale: f32,
    pub contrast: f32,
    pub max_sh_degree: f32,
    pub overall_scale: f32,
}

impl Default for KbSplatParams {
    fn default() -> Self {
        KbSplatParams {
            falloff: 8.0,
            scale: 1.0,
            contrast: 1.0,
            max_sh_degree: 0.0,
            overall_scale: 1.0,
        }
    }
}

/// A loaded point cloud: the GPU splat buffer, the per-frame sort-index buffer,
/// and the CPU-side positions used to sort back-to-front.
#[allow(dead_code)] // splat_buffer is retained to keep the storage binding alive
pub struct KbSplatModel {
    pub splat_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub storage_bind_group: wgpu::BindGroup,
    pub positions: Vec<[f32; 3]>,
    pub num_splats: u32,

    // Reused each frame to avoid per-frame allocation.
    sort_indices: Vec<u32>,
    sort_depths: Vec<f32>,
}

fn type_size(ty: &str) -> usize {
    match ty {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "double" | "float64" => 8,
        // float / float32 / int / uint / int32 / uint32 and anything else
        _ => 4,
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Parses a binary-little-endian 3D gaussian splat .ply into GPU instances.
///
/// Mirrors kbModel::load_ply + Renderer_Dx12::initialize_gaussian_splatting from
/// the blk reference: y is flipped, the quaternion is reordered, scale is taken
/// out of log space, opacity through a sigmoid, and the SH "rest" coefficients
/// are de-interleaved from [all R | all G | all B] into per-coefficient RGB.
pub async fn load_splat_ply(path: &str) -> Vec<KbSplatInstance> {
    log!("Loading gaussian splat ply: {path}");
    let bytes = load_binary(path)
        .await
        .unwrap_or_else(|e| panic!("Failed to read splat ply {path}: {e}"));

    let header_pos =
        find_subsequence(&bytes, b"end_header").expect("ply is missing an end_header marker");
    let header = std::str::from_utf8(&bytes[0..header_pos]).expect("ply header is not valid utf8");

    let mut vertex_count = 0usize;
    let mut stride = 0usize;
    let mut offsets: HashMap<String, usize> = HashMap::new();

    for line in header.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("element") => {
                if it.next() == Some("vertex") {
                    vertex_count = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
            }
            Some("property") => {
                let ty = it.next().unwrap_or("float");
                if let Some(name) = it.next() {
                    offsets.insert(name.to_string(), stride);
                    stride += type_size(ty);
                }
            }
            _ => {}
        }
    }

    // Binary data begins right after the newline following "end_header".
    let mut data_start = header_pos + "end_header".len();
    while data_start < bytes.len() && bytes[data_start] != b'\n' {
        data_start += 1;
    }
    data_start += 1;

    if vertex_count == 0 || stride == 0 {
        panic!("ply {path} has no vertices or zero stride");
    }

    let available = (bytes.len() - data_start) / stride;
    if available < vertex_count {
        log!(
            "Warning: ply {path} header claims {vertex_count} verts but only {available} present"
        );
        vertex_count = available;
    }

    if vertex_count > MAX_SPLATS {
        log!(
            "Warning: ply {path} has {vertex_count} splats; clamping to MAX_SPLATS ({MAX_SPLATS})"
        );
        vertex_count = MAX_SPLATS;
    }

    let get = |base: usize, name: &str| -> f32 {
        match offsets.get(name) {
            Some(off) => {
                let o = base + off;
                f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
            }
            None => 0.0,
        }
    };

    let mut instances = Vec::<KbSplatInstance>::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let base = data_start + i * stride;

        // Position: flip Y to go from the 3DGS data frame into this engine's
        // right-handed (look_at_rh) frame.
        let position = [get(base, "x"), -get(base, "y"), get(base, "z"), 0.0];

        // Quaternion stored as rot_0=w, rot_1=x, rot_2=y, rot_3=z, so the
        // standard form is xyzw = (rot_1, rot_2, rot_3, rot_0).  Flipping Y on the
        // positions is a reflection M = diag(1,-1,1); to keep each splat's
        // orientation consistent the rotation is conjugated by M, which for a
        // quaternion (w,x,y,z) yields (w,-x,y,-z) -- i.e. negate x and z.
        let rotation = [
            -get(base, "rot_1"),
            get(base, "rot_2"),
            -get(base, "rot_3"),
            get(base, "rot_0"),
        ];

        // Scale out of log space, opacity through a sigmoid.
        let scale_opacity = [
            get(base, "scale_0").exp(),
            get(base, "scale_1").exp(),
            get(base, "scale_2").exp(),
            1.0 / (1.0 + (-get(base, "opacity")).exp()),
        ];

        let sh0 = [
            get(base, "f_dc_0"),
            get(base, "f_dc_1"),
            get(base, "f_dc_2"),
            0.0,
        ];

        // De-interleave f_rest: ply packs it as [R0..14, G15..29, B30..44].
        // We assemble per-coefficient RGB triples for the first 8 coefficients.
        let mut sh_rest = [0.0f32; 24];
        for n in 0..8 {
            sh_rest[n * 3] = get(base, &format!("f_rest_{n}"));
            sh_rest[n * 3 + 1] = get(base, &format!("f_rest_{}", n + 15));
            sh_rest[n * 3 + 2] = get(base, &format!("f_rest_{}", n + 30));
        }

        instances.push(KbSplatInstance {
            position,
            scale_opacity,
            rotation,
            sh0,
            sh_rest,
        });
    }

    log!("Loaded {} splats from {path}", instances.len());
    instances
}

pub struct KbGaussianSplatRenderGroup {
    pipeline: wgpu::RenderPipeline,
    uniform: KbSplatUniform,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    storage_bind_group_layout: wgpu::BindGroupLayout,
    model: Option<KbSplatModel>,
    params: KbSplatParams,
}

impl KbGaussianSplatRenderGroup {
    pub async fn new(
        device_resources: &KbDeviceResources<'_>,
        asset_manager: &mut KbAssetManager,
    ) -> Self {
        log!("Creating KbGaussianSplatRenderGroup");
        let device = &device_resources.device;
        let surface_config = &device_resources.surface_config;

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/gaussian_splat.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        let uniform = KbSplatUniform::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KbSplat_uniform_buffer"),
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
                label: Some("KbSplat_uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("KbSplat_uniform_bind_group"),
        });

        // Read-only storage for the splat records and the per-frame sort indices.
        let storage_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("KbSplat_storage_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("KbSplat_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &storage_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("KbSplat_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
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
            // Splats are CPU-sorted and alpha-composited; no depth test.
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        KbGaussianSplatRenderGroup {
            pipeline,
            uniform,
            uniform_buffer,
            uniform_bind_group,
            storage_bind_group_layout,
            model: None,
            params: KbSplatParams::default(),
        }
    }

    pub fn set_params(&mut self, params: &KbSplatParams) {
        self.params = *params;
    }

    pub async fn load(&mut self, path: &str, device_resources: &KbDeviceResources<'_>) {
        let instances = load_splat_ply(path).await;
        let num_splats = instances.len() as u32;
        let device = &device_resources.device;

        let positions: Vec<[f32; 3]> = instances
            .iter()
            .map(|s| [s.position[0], s.position[1], s.position[2]])
            .collect();

        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KbSplat_splat_buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let initial_indices: Vec<u32> = (0..num_splats).collect();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KbSplat_index_buffer"),
            contents: bytemuck::cast_slice(&initial_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
            label: Some("KbSplat_storage_bind_group"),
        });

        self.model = Some(KbSplatModel {
            splat_buffer,
            index_buffer,
            storage_bind_group,
            positions,
            num_splats,
            sort_indices: initial_indices,
            sort_depths: vec![0.0; num_splats as usize],
        });
    }

    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    pub fn render(
        &mut self,
        device_resources: &mut KbDeviceResources,
        game_camera: &KbCamera,
        game_config: &KbConfig,
    ) {
        let model = match self.model.as_mut() {
            Some(m) => m,
            None => return,
        };
        if model.num_splats == 0 {
            return;
        }

        let (view_matrix, _, _) = game_camera.calculate_view_matrix();
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );
        let view_proj = proj_matrix * view_matrix;

        // CPU depth sort, back-to-front (farthest first) for correct alpha
        // compositing.  In a right-handed view space the camera looks down -Z,
        // so farther fragments have a smaller (more negative) Z.
        let zc = [
            view_matrix.x.z,
            view_matrix.y.z,
            view_matrix.z.z,
            view_matrix.w.z,
        ];
        let positions = &model.positions;
        let depths = &mut model.sort_depths;
        for (i, p) in positions.iter().enumerate() {
            depths[i] = zc[0] * p[0] + zc[1] * p[1] + zc[2] * p[2] + zc[3];
        }
        model
            .sort_indices
            .sort_unstable_by(|&a, &b| depths[a as usize].total_cmp(&depths[b as usize]));

        device_resources.queue.write_buffer(
            &model.index_buffer,
            0,
            bytemuck::cast_slice(&model.sort_indices),
        );

        let cam_pos = game_camera.get_position();
        self.uniform.view = view_matrix.into();
        self.uniform.view_proj = view_proj.into();
        self.uniform.camera_pos = [cam_pos.x, cam_pos.y, cam_pos.z, 1.0];
        self.uniform.splat_params = [
            self.params.falloff,
            self.params.scale,
            self.params.contrast,
            model.num_splats as f32,
        ];
        self.uniform.splat_params_2 = [
            self.params.max_sh_degree,
            self.params.overall_scale,
            0.0,
            0.0,
        ];
        device_resources.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );

        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("KbGaussianSplatRenderGroup::render()"),
                });
        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Gaussian Splats"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &device_resources.render_textures[0].view,
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
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &model.storage_bind_group, &[]);
            render_pass.draw(0..model.num_splats * 6, 0..1);
        }

        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}

// Keep the GPU struct size in lockstep with the WGSL layout.
const _: () = assert!(size_of::<KbSplatInstance>() == 160);

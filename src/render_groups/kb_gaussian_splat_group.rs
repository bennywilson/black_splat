use std::collections::HashMap;
use std::mem::size_of;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use crate::{kb_assets::*, kb_config::*, kb_game_object::*, kb_resource::*, log};

/// 700k splats * 160 bytes = 112 MB, comfortably under wgpu's default 128 MiB
/// storage-buffer binding limit.  Larger point clouds are clamped to this.
pub const MAX_SPLATS: usize = 700_000;

const SORT_WORKGROUP_SIZE: u32 = 256;

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

/// Matches `SortGlobals` in gaussian_splat_radix.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct KbSortGlobals {
    zc: [f32; 4],         // view-matrix depth row
    num_elements: u32,    // real splat count (no padding)
    num_tiles: u32,       // ceil(num_elements / SORT_WORKGROUP_SIZE)
    _pad0: u32,
    _pad1: u32,
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

/// Number of 8-bit digit passes for a 32-bit radix sort.
const RADIX_PASSES: u32 = 4;

/// A loaded point cloud plus everything the GPU radix sort needs.  The index
/// buffer is sorted on the GPU each frame; the CPU never touches it.
#[allow(dead_code)] // some buffers are retained only to keep their bindings alive
pub struct KbSplatModel {
    pub splat_buffer: wgpu::Buffer,
    pub num_splats: u32,

    // Ping-pong key + payload buffers.  cs_compute_keys fills A; each of the 4
    // digit passes reads one side and writes the other.
    keys_a: wgpu::Buffer,
    keys_b: wgpu::Buffer,
    vals_a: wgpu::Buffer,
    vals_b: wgpu::Buffer,
    // Per-tile bucket histogram (bucket-major) + the scan spine.
    hist_buffer: wgpu::Buffer,
    block_sums_buffer: wgpu::Buffer,

    // Draw bindings (group 1 of the draw pipeline): reads the vals buffer the sort
    // finishes in.  With RADIX_PASSES even, that is vals_a.
    storage_bind_group: wgpu::BindGroup,

    // Compute-sort bindings + state.
    num_tiles: u32,
    sort_globals: KbSortGlobals,
    sort_globals_buffer: wgpu::Buffer,
    // cs_compute_keys + odd digit passes write A (in=B, out=A); even passes A->B.
    sort_bind_group_a_to_b: wgpu::BindGroup,
    sort_bind_group_b_to_a: wgpu::BindGroup,
    // One uniform slot per digit pass holding its bit shift (0/8/16/24).
    pass_stride: u32,
    pass_buffer: wgpu::Buffer,
    pass_bind_group: wgpu::BindGroup,
    // View depth row used for the last sort; lets us skip re-sorting a static
    // camera.  NaN forces a sort on the first frame.
    last_sort_zc: [f32; 4],
    // When the last sort ran.  Re-sorts are rate-limited (splat order tolerates
    // being a few frames stale) to avoid flooding the GPU -- which on the browser
    // also starves the rest of the UI.
    last_sort_time: instant::Instant,

    // World-space bounding sphere for view-frustum culling.  When the sphere is
    // entirely outside the frustum both the sort and the draw are skipped.
    bounding_center: [f32; 3],
    bounding_radius: f32,
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

// Returns false when the sphere (world-space center + radius) is entirely outside
// any of the six frustum planes extracted from the view-projection matrix
// (Gribb-Hartmann method, column-major cgmath convention).
fn sphere_in_frustum(vp: cgmath::Matrix4<f32>, center: [f32; 3], radius: f32) -> bool {
    let r0 = [vp.x.x, vp.y.x, vp.z.x, vp.w.x];
    let r1 = [vp.x.y, vp.y.y, vp.z.y, vp.w.y];
    let r2 = [vp.x.z, vp.y.z, vp.z.z, vp.w.z];
    let r3 = [vp.x.w, vp.y.w, vp.z.w, vp.w.w];
    let planes = [
        [r3[0]+r0[0], r3[1]+r0[1], r3[2]+r0[2], r3[3]+r0[3]], // left
        [r3[0]-r0[0], r3[1]-r0[1], r3[2]-r0[2], r3[3]-r0[3]], // right
        [r3[0]+r1[0], r3[1]+r1[1], r3[2]+r1[2], r3[3]+r1[3]], // bottom
        [r3[0]-r1[0], r3[1]-r1[1], r3[2]-r1[2], r3[3]-r1[3]], // top
        [r3[0]+r2[0], r3[1]+r2[1], r3[2]+r2[2], r3[3]+r2[3]], // near
        [r3[0]-r2[0], r3[1]-r2[1], r3[2]-r2[2], r3[3]-r2[3]], // far
    ];
    let (cx, cy, cz) = (center[0], center[1], center[2]);
    for p in &planes {
        let dot = p[0]*cx + p[1]*cy + p[2]*cz + p[3];
        let len = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
        if dot + radius * len < 0.0 {
            return false;
        }
    }
    true
}

pub struct KbGaussianSplatRenderGroup {
    pipeline: wgpu::RenderPipeline,
    uniform: KbSplatUniform,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    storage_bind_group_layout: wgpu::BindGroupLayout,

    // GPU radix sort: one pipeline per RSSS phase, all sharing the sort layouts.
    sort_pipeline_compute_keys: wgpu::ComputePipeline,
    sort_pipeline_histogram: wgpu::ComputePipeline,
    sort_pipeline_scan_reduce: wgpu::ComputePipeline,
    sort_pipeline_scan_spine: wgpu::ComputePipeline,
    sort_pipeline_scan_add: wgpu::ComputePipeline,
    sort_pipeline_scatter: wgpu::ComputePipeline,
    sort_globals_layout: wgpu::BindGroupLayout,
    sort_pass_layout: wgpu::BindGroupLayout,

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

        // Read-only storage for the splat records and the sorted indices (draw).
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
            // Splats are sorted and alpha-composited; no depth test.
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // --- GPU radix sort pipelines ----------------------------------------
        let sort_shader_handle = asset_manager
            .load_shader(
                "/engine_assets/shaders/gaussian_splat_radix.wgsl",
                device_resources,
            )
            .await;
        let sort_shader = asset_manager.get_shader(&sort_shader_handle);

        // Helper for the storage-buffer bindings in the sort globals layout.
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let sort_globals_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("KbSplat_sort_globals_layout"),
                entries: &[
                    // 0: SortGlobals uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage_entry(1, true),  // splats (read-only)
                    storage_entry(2, true),  // keys_in
                    storage_entry(3, false), // keys_out
                    storage_entry(4, true),  // vals_in
                    storage_entry(5, false), // vals_out
                    storage_entry(6, false), // hist
                    storage_entry(7, false), // block_sums
                ],
            });

        let sort_pass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("KbSplat_sort_pass_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });

        let sort_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("KbSplat_sort_pipeline_layout"),
                bind_group_layouts: &[&sort_globals_layout, &sort_pass_layout],
                push_constant_ranges: &[],
            });

        let make_pipeline = |label: &str, entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&sort_pipeline_layout),
                module: sort_shader,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };
        let sort_pipeline_compute_keys = make_pipeline("KbSplat_sort_compute_keys", "cs_compute_keys");
        let sort_pipeline_histogram = make_pipeline("KbSplat_sort_histogram", "cs_histogram");
        let sort_pipeline_scan_reduce = make_pipeline("KbSplat_sort_scan_reduce", "cs_scan_reduce");
        let sort_pipeline_scan_spine = make_pipeline("KbSplat_sort_scan_spine", "cs_scan_spine");
        let sort_pipeline_scan_add = make_pipeline("KbSplat_sort_scan_add", "cs_scan_add");
        let sort_pipeline_scatter = make_pipeline("KbSplat_sort_scatter", "cs_scatter");

        KbGaussianSplatRenderGroup {
            pipeline,
            uniform,
            uniform_buffer,
            uniform_bind_group,
            storage_bind_group_layout,
            sort_pipeline_compute_keys,
            sort_pipeline_histogram,
            sort_pipeline_scan_reduce,
            sort_pipeline_scan_spine,
            sort_pipeline_scan_add,
            sort_pipeline_scatter,
            sort_globals_layout,
            sort_pass_layout,
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

        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KbSplat_splat_buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Radix sort needs no power-of-two padding, but the per-tile kernels run
        // over whole workgroup-sized tiles, so allocate up to a tile multiple.  The
        // tail slots past num_splats are never read or written (kernels guard on
        // num_elements), they just keep indexing in-bounds.
        let num_tiles = num_splats.max(1).div_ceil(SORT_WORKGROUP_SIZE);
        let alloc = (num_tiles * SORT_WORKGROUP_SIZE) as u64;
        let alloc_bytes = alloc * 4;

        // Ping-pong key + payload buffers.  Contents are written by cs_compute_keys
        // each sort, so no initial data is needed.
        let make_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: alloc_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let keys_a = make_storage("KbSplat_keys_a");
        let keys_b = make_storage("KbSplat_keys_b");
        let vals_a = make_storage("KbSplat_vals_a");
        let vals_b = make_storage("KbSplat_vals_b");

        // Histogram is bucket-major: RADIX (256) buckets * num_tiles.  The scan
        // spine has one entry per tile (== one per 256-wide histogram block).
        let hist_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KbSplat_hist_buffer"),
            size: (256 * num_tiles) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KbSplat_block_sums_buffer"),
            size: num_tiles as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // RADIX_PASSES is even, so the sort finishes back in the A buffers.
        let final_vals = &vals_a;

        let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: final_vals.as_entire_binding(),
                },
            ],
            label: Some("KbSplat_storage_bind_group"),
        });

        // Sort globals (depth row updated per frame; counts are constant).
        let sort_globals = KbSortGlobals {
            zc: [0.0; 4],
            num_elements: num_splats,
            num_tiles,
            _pad0: 0,
            _pad1: 0,
        };
        let sort_globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KbSplat_sort_globals_buffer"),
            contents: bytemuck::cast_slice(&[sort_globals]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Two bind groups swapping the in/out key+val buffers between passes.
        let make_sort_bg = |label: &str,
                            keys_in: &wgpu::Buffer,
                            keys_out: &wgpu::Buffer,
                            vals_in: &wgpu::Buffer,
                            vals_out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.sort_globals_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: sort_globals_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: splat_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: keys_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: keys_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: vals_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: vals_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: hist_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: block_sums_buffer.as_entire_binding() },
                ],
            })
        };
        let sort_bind_group_a_to_b =
            make_sort_bg("KbSplat_sort_bg_a_to_b", &keys_a, &keys_b, &vals_a, &vals_b);
        let sort_bind_group_b_to_a =
            make_sort_bg("KbSplat_sort_bg_b_to_a", &keys_b, &keys_a, &vals_b, &vals_a);

        // One uniform slot per digit pass holding its bit shift (0/8/16/24), each
        // aligned for use as a dynamic-offset binding.
        let pass_stride = device.limits().min_uniform_buffer_offset_alignment.max(16);
        let mut pass_bytes = vec![0u8; RADIX_PASSES as usize * pass_stride as usize];
        for p in 0..RADIX_PASSES {
            let shift = p * 8;
            let o = p as usize * pass_stride as usize;
            pass_bytes[o..o + 4].copy_from_slice(&shift.to_le_bytes());
        }
        let pass_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KbSplat_pass_buffer"),
            contents: &pass_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pass_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KbSplat_pass_bind_group"),
            layout: &self.sort_pass_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pass_buffer,
                    offset: 0,
                    size: NonZeroU64::new(16),
                }),
            }],
        });

        // Bounding sphere (world-space, for frustum culling).  Center = mean
        // position; radius = max distance from center.
        let (bounding_center, bounding_radius) = {
            let n = instances.len() as f32;
            let cx = instances.iter().map(|s| s.position[0]).sum::<f32>() / n;
            let cy = instances.iter().map(|s| s.position[1]).sum::<f32>() / n;
            let cz = instances.iter().map(|s| s.position[2]).sum::<f32>() / n;
            let r = instances
                .iter()
                .map(|s| {
                    let dx = s.position[0] - cx;
                    let dy = s.position[1] - cy;
                    let dz = s.position[2] - cz;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(0.0_f32, f32::max);
            ([cx, cy, cz], r)
        };

        self.model = Some(KbSplatModel {
            splat_buffer,
            num_splats,
            keys_a,
            keys_b,
            vals_a,
            vals_b,
            hist_buffer,
            block_sums_buffer,
            storage_bind_group,
            num_tiles,
            sort_globals,
            sort_globals_buffer,
            sort_bind_group_a_to_b,
            sort_bind_group_b_to_a,
            pass_stride,
            pass_buffer,
            pass_bind_group,
            last_sort_zc: [f32::NAN; 4],
            last_sort_time: instant::Instant::now(),
            bounding_center,
            bounding_radius,
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

        // Skip the sort and draw entirely if the splat cloud is off-screen.
        if !sphere_in_frustum(view_proj, model.bounding_center, model.bounding_radius) {
            return;
        }

        // Depth ordering depends only on the view matrix's third (Z) row, so we
        // only re-run the GPU sort when that row changes -- a static camera reuses
        // the previously sorted index buffer.
        let zc = [
            view_matrix.x.z,
            view_matrix.y.z,
            view_matrix.z.z,
            view_matrix.w.z,
        ];
        // Sort order depends only on the view DIRECTION (zc[0..3]); zc[3] is the
        // translation, which shifts every splat's depth equally and so never
        // changes their order.  So only re-sort when the camera rotates -- pure
        // translation (WASD, any direction) reuses the existing order for free.
        // The radix sort is cheap enough to run every rotated frame (measured well
        // over 200 fps sorting every frame), so there is no rate limit: re-sorting
        // each frame keeps rotation smooth instead of updating in visible steps.
        let first_sort = model.last_sort_zc[0].is_nan();
        let needs_sort = first_sort
            || model.last_sort_zc[0..3]
                .iter()
                .zip(zc[0..3].iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);

        if needs_sort {
            model.sort_globals.zc = zc;
            device_resources.queue.write_buffer(
                &model.sort_globals_buffer,
                0,
                bytemuck::cast_slice(&[model.sort_globals]),
            );
            model.last_sort_zc = zc;
            model.last_sort_time = instant::Instant::now();
        }

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

        // GPU radix sort (Reduce-Scan-Scan-Scatter).  Every phase is its own compute
        // pass: WebGPU gives no barrier between dispatches inside one pass, and the
        // separate passes also remove the in-place write hazard that flickered on the
        // web backend.  cs_compute_keys fills the A buffers; each of the 4 digit
        // passes then reads one side and writes the other (A->B, B->A, ...), landing
        // back in A after the (even) RADIX_PASSES.
        if needs_sort {
            let groups = model.num_tiles;
            let pass_bg = &model.pass_bind_group;
            let dispatch = |enc: &mut wgpu::CommandEncoder,
                            pipe: &wgpu::ComputePipeline,
                            bg: &wgpu::BindGroup,
                            offset: u32,
                            n_groups: u32| {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Gaussian Splat Radix Phase"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, bg, &[]);
                cp.set_bind_group(1, pass_bg, &[offset]);
                cp.dispatch_workgroups(n_groups, 1, 1);
            };

            // Phase 0: compute keys + payloads into the A buffers (b_to_a writes A).
            dispatch(
                &mut command_encoder,
                &self.sort_pipeline_compute_keys,
                &model.sort_bind_group_b_to_a,
                0,
                groups,
            );

            for p in 0..RADIX_PASSES {
                let bg = if p % 2 == 0 {
                    &model.sort_bind_group_a_to_b
                } else {
                    &model.sort_bind_group_b_to_a
                };
                let offset = p * model.pass_stride;
                dispatch(&mut command_encoder, &self.sort_pipeline_histogram, bg, offset, groups);
                dispatch(&mut command_encoder, &self.sort_pipeline_scan_reduce, bg, offset, groups);
                dispatch(&mut command_encoder, &self.sort_pipeline_scan_spine, bg, offset, 1);
                dispatch(&mut command_encoder, &self.sort_pipeline_scan_add, bg, offset, groups);
                dispatch(&mut command_encoder, &self.sort_pipeline_scatter, bg, offset, groups);
            }
        }

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
            // The sorted vals buffer holds exactly num_splats indices, back-to-front.
            render_pass.draw(0..model.num_splats * 6, 0..1);
        }

        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}

// Keep the GPU struct size in lockstep with the WGSL layout.
const _: () = assert!(size_of::<KbSplatInstance>() == 160);

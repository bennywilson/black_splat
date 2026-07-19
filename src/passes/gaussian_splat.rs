use std::collections::HashMap;
use std::mem::size_of;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use crate::{assets::*, resource::*, log};

/// Per-platform cap on splats per cloud.  The GPU-side ceiling is computed at
/// load time from the device's actual limits (see `load`); this constant bounds
/// how much CPU memory a parse may commit before that ceiling is known.
///
/// Native: effectively "whatever the GPU can bind" -- also kept safely under the
/// radix sort's hard dispatch ceiling of 65535 workgroups * 256 = ~16.7M splats.
/// Wasm: browsers commonly cap a storage binding near the spec's 128 MiB
/// default, and the 32-bit address space must hold the raw .ply AND the parsed
/// instances, so stay conservative: 700k * 160 bytes = 112 MB.
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_SPLATS: usize = 16_000_000;
#[cfg(target_arch = "wasm32")]
pub const MAX_SPLATS: usize = 700_000;

const SORT_WORKGROUP_SIZE: u32 = 256;

/// GPU-side gaussian splat record.  Layout mirrors the WGSL `Splat` struct in
/// gaussian_splat.wgsl (160 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatInstance {
    pub position: [f32; 4],      // xyz position, w unused
    pub scale_opacity: [f32; 4], // linear scale xyz, normalized opacity w
    pub rotation: [f32; 4],      // quaternion x,y,z,w
    pub sh0: [f32; 4],           // degree-0 SH (base color), w unused
    pub sh_rest: [f32; 24],      // 8 higher-order coeffs * 3 channels
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct SplatUniform {
    pub view: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub splat_params: [f32; 4],   // falloff, scale, contrast, num_splats
    pub splat_params_2: [f32; 4], // max_sh_degree, overall_scale, _, _
    pub model: [[f32; 4]; 4],     // cloud world transform (editor gizmo)
    // Normalized world-space frustum planes, so the vertex shader can reject an
    // off-screen splat with six dot products before doing any real work.
    pub frustum_planes: [[f32; 4]; 6],
}

/// Matches `SortGlobals` in gaussian_splat_radix.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct SortGlobals {
    zc: [f32; 4],         // view-matrix depth row
    num_elements: u32,    // real splat count (no padding)
    num_tiles: u32,       // ceil(num_elements / SORT_WORKGROUP_SIZE)
    _pad0: u32,
    _pad1: u32,
    // Everything cs_compute_keys needs to frustum-cull a splat: the cloud
    // transform to reach world space, the planes to test against, and the scales
    // that size its billboard.
    model: [[f32; 4]; 4],
    frustum_planes: [[f32; 4]; 6],
    // x: overall scale, y: splat scale, z: cloud scale, w: unused
    cull_params: [f32; 4],
}

/// Outcome of a successful runtime splat load: how many splats made it onto the
/// GPU, and the file's original count when the cloud had to be clamped to the
/// platform/device budget (`None` = loaded in full).
#[derive(Clone, Copy)]
pub struct SplatLoadInfo {
    pub num_splats: u32,
    pub clamped_from: Option<usize>,
}

/// Tunable rendering parameters (see the original blk GaussianSplatComponent).
#[derive(Clone, Copy)]
pub struct SplatParams {
    pub falloff: f32,
    pub scale: f32,
    pub contrast: f32,
    pub max_sh_degree: f32,
    pub overall_scale: f32,
}

impl Default for SplatParams {
    fn default() -> Self {
        SplatParams {
            falloff: 8.0,
            scale: 1.0,
            contrast: 1.0,
            max_sh_degree: 0.0,
            overall_scale: 1.0,
        }
    }
}

// Editor markup: the splat rendering knobs shown in the editor's Details panel
// when a splat is selected (see crate::editor).  SH degree is an integer 0..2
// stored as a float; edited as a drag value like the rest.
crate::editor_properties!(SplatParams {
    falloff: float("Falloff"),
    scale: float("Splat Scale"),
    contrast: float("Contrast"),
    overall_scale: float("Overall Scale"),
    max_sh_degree: float("SH Degree"),
});

/// Number of 8-bit digit passes for a 32-bit radix sort.
const RADIX_PASSES: u32 = 4;

/// A loaded point cloud plus everything the GPU radix sort needs.  The index
/// buffer is sorted on the GPU each frame; the CPU never touches it.
#[allow(dead_code)] // some buffers are retained only to keep their bindings alive
pub struct SplatModel {
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
    // Indirect draw args: [vertex_count, instance_count, first_vertex,
    // first_instance].  vertex_count is a fixed 4 (one strip per splat);
    // cs_compute_keys tallies the splats that survive the frustum cull into
    // instance_count, so the draw covers exactly the visible prefix of the
    // sorted index buffer without the CPU ever learning the count.
    draw_args: wgpu::Buffer,

    // Draw bindings (group 1 of the draw pipeline): reads the vals buffer the sort
    // finishes in.  With RADIX_PASSES even, that is vals_a.
    storage_bind_group: wgpu::BindGroup,

    // Compute-sort bindings + state.
    num_tiles: u32,
    sort_globals: SortGlobals,
    sort_globals_buffer: wgpu::Buffer,
    // cs_compute_keys + odd digit passes write A (in=B, out=A); even passes A->B.
    sort_bind_group_a_to_b: wgpu::BindGroup,
    sort_bind_group_b_to_a: wgpu::BindGroup,
    // One uniform slot per digit pass holding its bit shift (0/8/16/24).
    pass_stride: u32,
    pass_buffer: wgpu::Buffer,
    pass_bind_group: wgpu::BindGroup,
    // Model-view-projection the last sort ran against; a static camera and cloud
    // reuse that sort's result.  NaN forces a sort on the first frame.  The whole
    // MVP is the key (not just the depth row) because cs_compute_keys also culls,
    // and visibility depends on translation as well as orientation.
    last_sort_mvp: [[f32; 4]; 4],
    // When the last sort ran.
    last_sort_time: instant::Instant,

    // World-space bounding sphere for view-frustum culling.  When the sphere is
    // entirely outside the frustum both the sort and the draw are skipped.
    // Fitted to the cloud's core rather than its full extent -- see the
    // CULL_PERCENTILE comment where these are computed.
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

/// What `parse_splat_ply` produced: the GPU instances plus, when the file held
/// more splats than the platform/device budget allowed, the original count (so
/// callers can warn the user that the cloud was truncated).
#[derive(Debug)]
pub struct SplatParse {
    pub instances: Vec<SplatInstance>,
    pub clamped_from: Option<usize>,
}

/// Loads a splat .ply from a path and parses it (see `parse_splat_ply`).
/// Returns `None` when the file can't be read or parsed, so a demo cycling
/// several optional .ply files keeps running with whatever is actually present.
pub async fn load_splat_ply(path: &str, max_splats: usize) -> Option<Vec<SplatInstance>> {
    log!("Loading gaussian splat ply: {path}");
    let bytes = match load_binary(path).await {
        Ok(bytes) => bytes,
        Err(e) => {
            log!("Skipping splat ply {path}: {e}");
            return None;
        }
    };
    match parse_splat_ply(&bytes, path, max_splats) {
        Ok(parse) => Some(parse.instances),
        Err(e) => {
            log!("Skipping splat ply {path}: {e}");
            None
        }
    }
}

/// Parses a binary-little-endian 3D gaussian splat .ply into GPU instances,
/// clamped to `max_splats` (callers derive it from the device's storage-binding
/// limits).  `source` appears only in log messages.
///
/// Mirrors kbModel::load_ply + Renderer_Dx12::initialize_gaussian_splatting from
/// the blk reference: y is flipped, the quaternion is reordered, scale is taken
/// out of log space, opacity through a sigmoid, and the SH "rest" coefficients
/// are de-interleaved from [all R | all G | all B] into per-coefficient RGB.
/// Returns `Err` with a short human-readable reason (never panics) on malformed
/// input: the bytes may be a user-picked file, and the message is shown to them.
pub fn parse_splat_ply(
    bytes: &[u8],
    source: &str,
    max_splats: usize,
) -> Result<SplatParse, String> {
    let header_pos = find_subsequence(bytes, b"end_header")
        .ok_or_else(|| "not a .ply file (no end_header)".to_string())?;
    let header = std::str::from_utf8(&bytes[0..header_pos])
        .map_err(|_| "header is not valid utf-8".to_string())?;

    let mut vertex_count = 0usize;
    let mut stride = 0usize;
    let mut offsets: HashMap<String, usize> = HashMap::new();
    let mut format_ok = false;

    for line in header.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("format") => {
                let format = it.next().unwrap_or("");
                if format == "binary_little_endian" {
                    format_ok = true;
                } else {
                    return Err(format!(
                        "'{format}' .ply is unsupported (need binary_little_endian)"
                    ));
                }
            }
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

    if !format_ok {
        return Err("no 'format' line -- not a valid .ply".to_string());
    }

    // A 3DGS training export always carries these.  Compressed .plys (e.g.
    // SuperSplat's, which packs everything into chunk/packed_* properties) and
    // plain point clouds don't, and would otherwise "load" as garbage since
    // missing properties read as 0.
    for required in ["x", "y", "z", "scale_0", "rot_0", "opacity", "f_dc_0"] {
        if !offsets.contains_key(required) {
            return Err(format!(
                "no '{required}' property -- compressed or non-3DGS .ply? (unsupported)"
            ));
        }
    }

    // Binary data begins right after the newline following "end_header".
    let mut data_start = header_pos + "end_header".len();
    while data_start < bytes.len() && bytes[data_start] != b'\n' {
        data_start += 1;
    }
    data_start += 1;

    if vertex_count == 0 || stride == 0 {
        return Err("header lists no vertices".to_string());
    }

    let available = bytes.len().saturating_sub(data_start) / stride;
    if available < vertex_count {
        log!(
            "Warning: ply {source} header claims {vertex_count} verts but only {available} present"
        );
        vertex_count = available;
    }
    if vertex_count == 0 {
        return Err("vertex data is missing or truncated".to_string());
    }

    let mut clamped_from = None;
    if vertex_count > max_splats {
        log!("Warning: ply {source} has {vertex_count} splats; clamping to {max_splats}");
        clamped_from = Some(vertex_count);
        vertex_count = max_splats;
    }

    // Resolve every property offset up front: at multi-million splat counts,
    // per-splat HashMap lookups (and format!-built f_rest keys) dominate load
    // time.  Missing properties read as 0.0, same as before.
    let off = |name: &str| offsets.get(name).copied();
    let read = |base: usize, off: Option<usize>| -> f32 {
        match off {
            Some(off) => {
                let o = base + off;
                f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
            }
            None => 0.0,
        }
    };

    let xyz_off = [off("x"), off("y"), off("z")];
    let rot_off = [off("rot_0"), off("rot_1"), off("rot_2"), off("rot_3")];
    let scale_off = [off("scale_0"), off("scale_1"), off("scale_2")];
    let opacity_off = off("opacity");
    let dc_off = [off("f_dc_0"), off("f_dc_1"), off("f_dc_2")];

    // f_rest is packed channel-major: [R0..Rk-1 | G0..Gk-1 | B0..Bk-1] where k is
    // the per-channel coefficient count -- 15 for the standard degree-3 export,
    // 8 for a degree-2 one.  Derive k from the header so either layout
    // de-interleaves correctly, and take the first 8 coefficients (degrees 1+2)
    // of each channel, which is all the GPU struct carries.
    let num_rest = offsets.keys().filter(|k| k.starts_with("f_rest_")).count();
    let rest_per_channel = num_rest / 3;
    let mut rest_off = [[None; 3]; 8];
    for (n, coeff) in rest_off.iter_mut().enumerate() {
        for (c, chan) in coeff.iter_mut().enumerate() {
            *chan = off(&format!("f_rest_{}", n + c * rest_per_channel));
        }
    }

    let mut instances = Vec::<SplatInstance>::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let base = data_start + i * stride;

        // Position: rotate the 3DGS data frame 180 deg about Z (negate X and Y)
        // into this engine's right-handed (look_at_rh) frame.  Negating a SINGLE
        // axis is a reflection (det = -1) and mirrors the cloud; negating two is a
        // proper rotation (det = +1), so the scene keeps its handedness.
        let position = [
            -read(base, xyz_off[0]),
            -read(base, xyz_off[1]),
            read(base, xyz_off[2]),
            0.0,
        ];

        // Quaternion stored as rot_0=w, rot_1=x, rot_2=y, rot_3=z, so the
        // standard form is xyzw = (rot_1, rot_2, rot_3, rot_0).  To keep each
        // splat's orientation consistent with the 180-deg-about-Z position
        // rotation, conjugate the quaternion by that rotation, which for
        // (w,x,y,z) yields (w,-x,-y,z) -- i.e. negate x and y.
        let rotation = [
            -read(base, rot_off[1]),
            -read(base, rot_off[2]),
            read(base, rot_off[3]),
            read(base, rot_off[0]),
        ];

        // Scale out of log space, opacity through a sigmoid.
        let scale_opacity = [
            read(base, scale_off[0]).exp(),
            read(base, scale_off[1]).exp(),
            read(base, scale_off[2]).exp(),
            1.0 / (1.0 + (-read(base, opacity_off)).exp()),
        ];

        let sh0 = [
            read(base, dc_off[0]),
            read(base, dc_off[1]),
            read(base, dc_off[2]),
            0.0,
        ];

        // Assemble per-coefficient RGB triples from the channel-major offsets
        // resolved above.
        let mut sh_rest = [0.0f32; 24];
        for (n, coeff) in rest_off.iter().enumerate() {
            sh_rest[n * 3] = read(base, coeff[0]);
            sh_rest[n * 3 + 1] = read(base, coeff[1]);
            sh_rest[n * 3 + 2] = read(base, coeff[2]);
        }

        instances.push(SplatInstance {
            position,
            scale_opacity,
            rotation,
            sh0,
            sh_rest,
        });
    }

    log!("Loaded {} splats from {source}", instances.len());
    Ok(SplatParse {
        instances,
        clamped_from,
    })
}

// Splats past this fraction of the distance ordering are left outside the cull
// sphere.  See fit_cull_sphere.
const CULL_PERCENTILE: f32 = 0.99;

// Fits the world-space bounding sphere used for view-frustum culling.
//
// 3DGS captures almost always carry a scatter of "floater" splats far outside
// the real scene, and both a mean center and a max-distance radius get dragged
// out to them -- giving a sphere big enough to contain the camera, so the cull
// test passes no matter where you look.  The per-axis median and a percentile
// distance both ignore that tail, giving a sphere that tracks the actual cloud.
//
// The tradeoff: splats past CULL_PERCENTILE fall outside the sphere and wink out
// if the cloud leaves the frustum while they are still on screen.  Those are the
// same stray floaters the tight fit exists to ignore, and the cull only fires
// once the body of the cloud is already off screen.
fn fit_cull_sphere(instances: &[SplatInstance]) -> ([f32; 3], f32) {
    if instances.is_empty() {
        return ([0.0; 3], 0.0);
    }
    let median_axis = |axis: usize| {
        let mut v: Vec<f32> = instances.iter().map(|s| s.position[axis]).collect();
        let mid = v.len() / 2;
        v.select_nth_unstable_by(mid, f32::total_cmp);
        v[mid]
    };
    let center = [median_axis(0), median_axis(1), median_axis(2)];
    let mut dists: Vec<f32> = instances
        .iter()
        .map(|s| {
            let dx = s.position[0] - center[0];
            let dy = s.position[1] - center[1];
            let dz = s.position[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();
    let k = (((dists.len() - 1) as f32) * CULL_PERCENTILE).round() as usize;
    dists.select_nth_unstable_by(k, f32::total_cmp);
    (center, dists[k])
}

// The six world-space frustum planes (left, right, bottom, top, near, far) of a
// view-projection matrix, Gribb-Hartmann in cgmath's column-major convention.
// Normalized, so a plane dot a point is a true signed distance -- the splat
// vertex shader relies on that to test each splat's bounding sphere with six
// plain dot products (see gaussian_splat.wgsl).
fn frustum_planes(vp: cgmath::Matrix4<f32>) -> [[f32; 4]; 6] {
    let r0 = [vp.x.x, vp.y.x, vp.z.x, vp.w.x];
    let r1 = [vp.x.y, vp.y.y, vp.z.y, vp.w.y];
    let r2 = [vp.x.z, vp.y.z, vp.z.z, vp.w.z];
    let r3 = [vp.x.w, vp.y.w, vp.z.w, vp.w.w];
    let mut planes = [
        [r3[0]+r0[0], r3[1]+r0[1], r3[2]+r0[2], r3[3]+r0[3]], // left
        [r3[0]-r0[0], r3[1]-r0[1], r3[2]-r0[2], r3[3]-r0[3]], // right
        [r3[0]+r1[0], r3[1]+r1[1], r3[2]+r1[2], r3[3]+r1[3]], // bottom
        [r3[0]-r1[0], r3[1]-r1[1], r3[2]-r1[2], r3[3]-r1[3]], // top
        [r3[0]+r2[0], r3[1]+r2[1], r3[2]+r2[2], r3[3]+r2[3]], // near
        [r3[0]-r2[0], r3[1]-r2[1], r3[2]-r2[2], r3[3]-r2[3]], // far
    ];
    for p in planes.iter_mut() {
        let len = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
        if len > 0.0 {
            p[0] /= len;
            p[1] /= len;
            p[2] /= len;
            p[3] /= len;
        }
    }
    planes
}

// Returns false when the sphere (world-space center + radius) is entirely outside
// any of the six frustum planes.
fn sphere_in_frustum(vp: cgmath::Matrix4<f32>, center: [f32; 3], radius: f32) -> bool {
    let (cx, cy, cz) = (center[0], center[1], center[2]);
    frustum_planes(vp)
        .iter()
        .all(|p| p[0]*cx + p[1]*cy + p[2]*cz + p[3] + radius >= 0.0)
}

/// Diagnostic: which stage of the splat pass to skip, so frame time can be
/// attributed to the part that actually costs it.  Each setting removes exactly
/// one stage and leaves the rest running, so the fps delta against `Off` prices
/// that stage on its own.
///
/// Cycled with [B] in the splat demo; the active setting shows on the debug HUD.
/// Everything defaults to `Off`, so this costs nothing when unused.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SplatBisect {
    /// Normal rendering.
    #[default]
    Off,
    /// Skip the draw call, keep the sort and the (clearing) render pass.  The
    /// delta from `Off` is what the per-splat vertex invocations cost.
    NoDraw,
    /// Skip the 21 radix-sort dispatches; still draw, with a stale splat order.
    /// The delta from `Off` is what the sort costs.
    NoSort,
    /// Skip the whole pass, and with it the composite.  Should land near the
    /// no-splats-loaded baseline; anything short of that is cost elsewhere.
    NoPass,
}

impl SplatBisect {
    pub fn label(self) -> &'static str {
        match self {
            SplatBisect::Off => "off",
            SplatBisect::NoDraw => "no draw (sort + composite only)",
            SplatBisect::NoSort => "no sort (draw + composite only)",
            SplatBisect::NoPass => "no splat pass at all",
        }
    }

    pub fn next(self) -> Self {
        match self {
            SplatBisect::Off => SplatBisect::NoDraw,
            SplatBisect::NoDraw => SplatBisect::NoSort,
            SplatBisect::NoSort => SplatBisect::NoPass,
            SplatBisect::NoPass => SplatBisect::Off,
        }
    }
}

pub struct GaussianSplatPass {
    pipeline: wgpu::RenderPipeline,
    bisect: SplatBisect,
    uniform: SplatUniform,
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

    // All preloaded splat clouds; `active_model` indexes the one being rendered.
    // Cycling between them just changes the index (no async reload).
    models: Vec<SplatModel>,
    active_model: usize,
    params: SplatParams,
    model_transform: cgmath::Matrix4<f32>,
}

impl GaussianSplatPass {
    pub async fn new(
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating GaussianSplatPass");
        let device = &device_resources.device;

        let shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/gaussian_splat.wgsl", device_resources)
            .await;
        let shader = asset_manager.get_shader(&shader_handle);

        let uniform = SplatUniform::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Splat_uniform_buffer"),
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
                label: Some("Splat_uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Splat_uniform_bind_group"),
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
                label: Some("Splat_storage_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Splat_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout), Some(&storage_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Splat_pipeline"),
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
                    format: crate::resource::SCENE_COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // One 4-vertex strip per splat instance (see gaussian_splat.wgsl).
                // strip_index_format stays None: the draw is non-indexed.
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },

            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
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
                label: Some("Splat_sort_globals_layout"),
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
                    storage_entry(8, false), // indirect draw args (cull tally)
                ],
            });

        let sort_pass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Splat_sort_pass_layout"),
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
                label: Some("Splat_sort_pipeline_layout"),
                bind_group_layouts: &[Some(&sort_globals_layout), Some(&sort_pass_layout)],
                immediate_size: 0,
            });

        let make_pipeline = |label: &str, entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&sort_pipeline_layout),
                module: sort_shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let sort_pipeline_compute_keys = make_pipeline("Splat_sort_compute_keys", "cs_compute_keys");
        let sort_pipeline_histogram = make_pipeline("Splat_sort_histogram", "cs_histogram");
        let sort_pipeline_scan_reduce = make_pipeline("Splat_sort_scan_reduce", "cs_scan_reduce");
        let sort_pipeline_scan_spine = make_pipeline("Splat_sort_scan_spine", "cs_scan_spine");
        let sort_pipeline_scan_add = make_pipeline("Splat_sort_scan_add", "cs_scan_add");
        let sort_pipeline_scatter = make_pipeline("Splat_sort_scatter", "cs_scatter");

        GaussianSplatPass {
            pipeline,
            bisect: SplatBisect::default(),
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
            models: Vec::new(),
            active_model: 0,
            params: SplatParams::default(),
            model_transform: cgmath::Matrix4::from_scale(1.0),
        }
    }

    pub fn set_params(&mut self, params: &SplatParams) {
        self.params = *params;
    }

    pub fn set_transform(&mut self, transform: cgmath::Matrix4<f32>) {
        self.model_transform = transform;
    }

    /// Number of splat clouds currently loaded.
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Select which preloaded splat cloud to render.  Out-of-range indices are
    /// ignored.
    pub fn set_active_model(&mut self, index: usize) {
        if index < self.models.len() {
            self.active_model = index;
        }
    }

    /// Unloads every cloud (their GPU buffers drop with them).  Nothing renders
    /// until the next `load`/`load_from_bytes`.  Used by the editor's New Scene.
    pub fn clear_models(&mut self) {
        self.models.clear();
        self.active_model = 0;
    }

    /// Unloads a single cloud (its GPU buffers drop with it), keeping the
    /// others and renumbering `active_model` so it still points at the same
    /// logical cloud (or the next one, if the active cloud itself was
    /// removed).  Out-of-range indices are ignored.  Used by the editor's
    /// per-splat delete (and delete-undo).
    pub fn remove_model(&mut self, index: usize) {
        if index >= self.models.len() {
            return;
        }
        self.models.remove(index);
        if self.models.is_empty() {
            self.active_model = 0;
        } else if index < self.active_model {
            self.active_model -= 1;
        } else if index == self.active_model {
            self.active_model = self.active_model.min(self.models.len() - 1);
        }
    }

    /// Number of gaussian splats in the currently active cloud (0 if none).
    pub fn active_splat_count(&self) -> u32 {
        self.models.get(self.active_model).map_or(0, |m| m.num_splats)
    }

    /// Loads a splat .ply and appends it as a new model.  Missing/unreadable
    /// files are skipped (no model appended), so callers can preload an optional
    /// set and cycle over whatever loaded.  Returns true when a model was added.
    pub async fn load(&mut self, path: &str, device_resources: &DeviceResources<'_>) -> bool {
        let max_splats = self.device_max_splats(device_resources);
        let instances = match load_splat_ply(path, max_splats).await {
            Some(instances) => instances,
            None => return false,
        };
        self.add_model(instances, device_resources);
        true
    }

    /// Parses an in-memory .ply (e.g. one the user picked at runtime) and appends
    /// it as a new model.  Synchronous -- no file I/O -- so it is callable from a
    /// per-frame tick.  On success reports the loaded count and whether the cloud
    /// was clamped to the GPU budget; on failure returns a short user-displayable
    /// reason.  Never panics on malformed bytes.
    pub fn load_from_bytes(
        &mut self,
        bytes: &[u8],
        name: &str,
        device_resources: &DeviceResources<'_>,
    ) -> Result<SplatLoadInfo, String> {
        let max_splats = self.device_max_splats(device_resources);
        let parse = parse_splat_ply(bytes, name, max_splats)?;
        let info = SplatLoadInfo {
            num_splats: parse.instances.len() as u32,
            clamped_from: parse.clamped_from,
        };
        self.add_model(parse.instances, device_resources);
        Ok(info)
    }

    /// Clamps the platform splat budget to what this device can actually put in
    /// one storage-buffer binding, so oversized clouds degrade to a warning +
    /// clamp instead of a buffer-creation panic.
    fn device_max_splats(&self, device_resources: &DeviceResources<'_>) -> usize {
        let limits = device_resources.device.limits();
        let device_cap = limits.max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / size_of::<SplatInstance>() as u64;
        MAX_SPLATS.min(device_cap as usize)
    }

    /// Uploads parsed instances as GPU buffers + sort state and appends the model.
    fn add_model(
        &mut self,
        instances: Vec<SplatInstance>,
        device_resources: &DeviceResources<'_>,
    ) {
        let device = &device_resources.device;
        let num_splats = instances.len() as u32;

        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Splat_splat_buffer"),
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
        let keys_a = make_storage("Splat_keys_a");
        let keys_b = make_storage("Splat_keys_b");
        let vals_a = make_storage("Splat_vals_a");
        let vals_b = make_storage("Splat_vals_b");

        // Histogram is bucket-major: RADIX (256) buckets * num_tiles.  The scan
        // spine has one entry per tile (== one per 256-wide histogram block).
        let hist_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat_hist_buffer"),
            size: (256 * num_tiles) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat_block_sums_buffer"),
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
            label: Some("Splat_storage_bind_group"),
        });

        // Sort globals: counts are constant, the depth row and the cull inputs are
        // rewritten every time the sort runs.
        let sort_globals = SortGlobals {
            zc: [0.0; 4],
            num_elements: num_splats,
            num_tiles,
            _pad0: 0,
            _pad1: 0,
            ..Default::default()
        };
        let sort_globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Splat_sort_globals_buffer"),
            contents: bytemuck::cast_slice(&[sort_globals]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 4 strip vertices per splat; instance_count is refilled by the cull each
        // time the sort runs.  Starts at 0 so nothing draws before the first sort.
        let draw_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Splat_draw_args"),
            contents: bytemuck::cast_slice(&[4u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
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
                    wgpu::BindGroupEntry { binding: 8, resource: draw_args.as_entire_binding() },
                ],
            })
        };
        let sort_bind_group_a_to_b =
            make_sort_bg("Splat_sort_bg_a_to_b", &keys_a, &keys_b, &vals_a, &vals_b);
        let sort_bind_group_b_to_a =
            make_sort_bg("Splat_sort_bg_b_to_a", &keys_b, &keys_a, &vals_b, &vals_a);

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
            label: Some("Splat_pass_buffer"),
            contents: &pass_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pass_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Splat_pass_bind_group"),
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

        let (bounding_center, bounding_radius) = fit_cull_sphere(&instances);

        self.models.push(SplatModel {
            splat_buffer,
            num_splats,
            keys_a,
            keys_b,
            vals_a,
            vals_b,
            hist_buffer,
            block_sums_buffer,
            draw_args,
            storage_bind_group,
            num_tiles,
            sort_globals,
            sort_globals_buffer,
            sort_bind_group_a_to_b,
            sort_bind_group_b_to_a,
            pass_stride,
            pass_buffer,
            pass_bind_group,
            last_sort_mvp: [[f32::NAN; 4]; 4],
            last_sort_time: instant::Instant::now(),
            bounding_center,
            bounding_radius,
        });
    }

    pub fn has_model(&self) -> bool {
        !self.models.is_empty()
    }

    pub fn bisect(&self) -> SplatBisect {
        self.bisect
    }

    /// Advances the diagnostic bisect setting and returns the new one.
    pub fn cycle_bisect(&mut self) -> SplatBisect {
        self.bisect = self.bisect.next();
        self.bisect
    }

    /// Draws the active cloud into the splat scratch buffer.  Returns false when
    /// nothing was drawn (no cloud, or the whole cloud is off-screen), in which
    /// case the scratch buffer holds no fresh content and the caller must skip
    /// the composite -- compositing it anyway would blend the previous frame's
    /// splats back onto the scene.
    pub fn render(&mut self, ctx: &mut RenderContext) -> bool {
        if self.bisect == SplatBisect::NoPass {
            return false;
        }
        let bisect = self.bisect;
        let device_resources = &mut *ctx.device;
        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let model = match self.models.get_mut(self.active_model) {
            Some(m) => m,
            None => return false,
        };
        if model.num_splats == 0 {
            return false;
        }

        let (view_matrix, _, _) = game_camera.calculate_view_matrix();
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );
        let view_proj = proj_matrix * view_matrix;

        // The cloud's world transform (editor gizmo) offsets every splat, so the
        // frustum cull, depth-sort key and vertex shader all work through it.
        let model_mat = self.model_transform;
        let os = self.params.overall_scale;

        // Skip the sort and draw entirely if the (transformed) cloud is off-screen.
        let bc = model.bounding_center;
        let world_center =
            model_mat * cgmath::Vector4::new(bc[0] * os, bc[1] * os, bc[2] * os, 1.0);
        let col_len = |c: cgmath::Vector4<f32>| (c.x * c.x + c.y * c.y + c.z * c.z).sqrt();
        let max_scale = col_len(model_mat.x)
            .max(col_len(model_mat.y))
            .max(col_len(model_mat.z));
        let world_radius = model.bounding_radius * os * max_scale;
        if !sphere_in_frustum(
            view_proj,
            [world_center.x, world_center.y, world_center.z],
            world_radius,
        ) {
            return false;
        }

        // Depth ordering is keyed by the view-space depth of the transformed
        // splat -- the third row of (view * model).
        let vm = view_matrix * model_mat;
        let zc = [vm.x.z, vm.y.z, vm.z.z, vm.w.z];
        // cs_compute_keys both sorts and frustum-culls, and visibility depends on
        // where the camera is, not just where it points -- so the trigger is the
        // full MVP rather than the depth row alone.  In practice that means any
        // camera or cloud movement re-runs it.  The whole sort measures ~0.5 ms
        // for 700k splats, which is a fraction of what culling the draw saves.
        let mvp: [[f32; 4]; 4] = (view_proj * model_mat).into();
        let first_sort = model.last_sort_mvp[0][0].is_nan();
        let moved = model
            .last_sort_mvp
            .iter()
            .flatten()
            .zip(mvp.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        // NoSort leaves last_sort_mvp untouched, so turning the bisect back off
        // re-sorts immediately rather than leaving a stale order on screen.
        let needs_sort = bisect != SplatBisect::NoSort && (first_sort || moved);

        if needs_sort {
            model.sort_globals.zc = zc;
            model.sort_globals.model = model_mat.into();
            model.sort_globals.frustum_planes = frustum_planes(view_proj);
            let cloud_scale = col_len(model_mat.x);
            model.sort_globals.cull_params = [os, self.params.scale, cloud_scale, 0.0];
            device_resources.queue.write_buffer(
                &model.sort_globals_buffer,
                0,
                bytemuck::cast_slice(&[model.sort_globals]),
            );
            // Zero the survivor tally before the cull adds to it; vertex_count is
            // the fixed 4-vertex strip.
            device_resources.queue.write_buffer(
                &model.draw_args,
                0,
                bytemuck::cast_slice(&[4u32, 0u32, 0u32, 0u32]),
            );
            model.last_sort_mvp = mvp;
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
        self.uniform.model = model_mat.into();
        self.uniform.frustum_planes = frustum_planes(view_proj);
        device_resources.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );

        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GaussianSplatPass::render()"),
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
                // Renders into its own scratch buffer (render_textures[2]), not the
                // shared scene color: splats alpha-blend among themselves here in
                // display space (the reference 3DGS look), and SplatCompositePass
                // converts the finished composite into the linear HDR scene once,
                // right after this pass runs -- see splat_composite.wgsl for why
                // that has to happen on the composite rather than per fragment.
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &device_resources.render_textures[2].view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // Shared scene depth, written by the opaque model passes.  The
                // pipeline only tests against it (writes are disabled), so Load/
                // Store leaves it intact for later depth-tested passes.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &device_resources.render_textures[1].view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });

            // NoDraw still enters (and so still clears) the render pass, leaving
            // the composite a clean transparent buffer to blend: the only thing
            // removed is the per-splat vertex work.
            if bisect != SplatBisect::NoDraw {
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_bind_group(1, &model.storage_bind_group, &[]);
                // Indirect: the cull parked the off-screen splats past the end of
                // the sorted buffer and wrote the survivor count into
                // instance_count, so the draw covers exactly the visible prefix,
                // back-to-front, without a GPU->CPU readback.
                render_pass.draw_indirect(&model.draw_args, 0);
            }
        }

        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
        true
    }
}

// Keep the GPU struct sizes in lockstep with the WGSL layouts.
const _: () = assert!(size_of::<SplatInstance>() == 160);
const _: () = assert!(size_of::<SplatUniform>() == 336);

#[cfg(test)]
mod tests {
    use super::*;

    // Builds a minimal binary-little-endian 3DGS .ply with `n` splats, each with
    // position (1,2,3), identity-ish rotation, log-scale 0, and opacity logit 0.
    fn make_ply(n: usize) -> Vec<u8> {
        let mut header = String::from("ply\nformat binary_little_endian 1.0\n");
        header.push_str(&format!("element vertex {n}\n"));
        for p in [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ] {
            header.push_str(&format!("property float {p}\n"));
        }
        header.push_str("end_header\n");
        let mut bytes = header.into_bytes();
        for _ in 0..n {
            for v in [
                1.0f32, 2.0, 3.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    // A SplatInstance at `pos`; only the position matters to the sphere fit.
    fn at(pos: [f32; 3]) -> SplatInstance {
        SplatInstance {
            position: [pos[0], pos[1], pos[2], 1.0],
            scale_opacity: [1.0, 1.0, 1.0, 0.5],
            rotation: [1.0, 0.0, 0.0, 0.0],
            sh0: [0.5, 0.5, 0.5, 0.0],
            sh_rest: [0.0; 24],
        }
    }

    // The whole point of the percentile fit: a handful of far floaters must not
    // blow the cull sphere up to encompass them (which is what a mean centre and
    // a max-distance radius did, defeating the cull entirely).
    #[test]
    fn cull_sphere_ignores_floaters() {
        // 1000 splats in a unit-ish box around the origin, plus 5 floaters at 1000.
        let mut instances: Vec<SplatInstance> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                at([t, -t, t * 0.5])
            })
            .collect();
        for _ in 0..5 {
            instances.push(at([1000.0, 1000.0, 1000.0]));
        }

        let (center, radius) = fit_cull_sphere(&instances);

        // Centre stays with the body of the cloud, not dragged toward the floaters.
        assert!(center[0].abs() < 2.0, "center pulled by floaters: {center:?}");
        assert!(center[1].abs() < 2.0, "center pulled by floaters: {center:?}");
        assert!(center[2].abs() < 2.0, "center pulled by floaters: {center:?}");
        // Radius bounds the body, nowhere near the 1732 needed to reach a floater.
        assert!(radius < 10.0, "radius pulled by floaters: {radius}");
    }

    // cgmath's perspective looks down -Z, so an identity view puts the camera at
    // the origin facing -Z and the projection alone is a usable view-proj.
    #[test]
    fn frustum_test_accepts_and_rejects() {
        let vp = cgmath::perspective(cgmath::Deg(60.0f32), 1.0, 0.1, 100.0);

        // Straight ahead, well inside.
        assert!(sphere_in_frustum(vp, [0.0, 0.0, -10.0], 0.5));
        // Behind the camera.
        assert!(!sphere_in_frustum(vp, [0.0, 0.0, 10.0], 0.5));
        // Far off to the side.
        assert!(!sphere_in_frustum(vp, [100.0, 0.0, -10.0], 0.5));
        // Beyond the far plane.
        assert!(!sphere_in_frustum(vp, [0.0, 0.0, -500.0], 0.5));
        // Behind the camera, but big enough to reach into the frustum.  This is
        // the case that only passes when the planes are normalized: the radius
        // has to be comparable against a true signed distance.
        assert!(sphere_in_frustum(vp, [0.0, 0.0, 10.0], 50.0));
    }

    #[test]
    fn cull_sphere_handles_empty_and_single() {
        assert_eq!(fit_cull_sphere(&[]), ([0.0, 0.0, 0.0], 0.0));
        let (center, radius) = fit_cull_sphere(&[at([3.0, 4.0, 5.0])]);
        assert_eq!(center, [3.0, 4.0, 5.0]);
        assert_eq!(radius, 0.0);
    }

    #[test]
    fn parses_a_minimal_ply() {
        let parse = parse_splat_ply(&make_ply(3), "test", MAX_SPLATS).unwrap();
        assert_eq!(parse.instances.len(), 3);
        assert!(parse.clamped_from.is_none());
        // Position: X and Y are negated into the engine frame.
        assert_eq!(parse.instances[0].position[0..3], [-1.0, -2.0, 3.0]);
        // Log-scale 0 -> 1.0; opacity logit 0 -> sigmoid 0.5.
        assert_eq!(parse.instances[0].scale_opacity, [1.0, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn clamps_to_max_splats_and_reports_it() {
        let parse = parse_splat_ply(&make_ply(10), "test", 4).unwrap();
        assert_eq!(parse.instances.len(), 4);
        assert_eq!(parse.clamped_from, Some(10));
    }

    #[test]
    fn tolerates_truncated_data() {
        let mut ply = make_ply(5);
        ply.truncate(ply.len() - 2 * 14 * 4 - 1); // cut off the last two splats
        let parse = parse_splat_ply(&ply, "test", MAX_SPLATS).unwrap();
        assert_eq!(parse.instances.len(), 2);
        // Truncation is not the same as hitting the splat budget.
        assert!(parse.clamped_from.is_none());
    }

    #[test]
    fn rejects_garbage_without_panicking() {
        assert!(parse_splat_ply(b"not a ply at all", "test", MAX_SPLATS).is_err());
        assert!(parse_splat_ply(b"", "test", MAX_SPLATS).is_err());
        // Valid header but zero vertices.
        assert!(parse_splat_ply(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nend_header\n",
            "test",
            MAX_SPLATS
        )
        .is_err());
        // Header claims vertices but the data section is missing entirely.
        let no_data = String::from_utf8(make_ply(0)).unwrap().replace("vertex 0", "vertex 9");
        assert!(parse_splat_ply(no_data.as_bytes(), "test", MAX_SPLATS).is_err());
    }

    #[test]
    fn rejects_unsupported_formats_with_a_reason() {
        // ASCII .ply: would otherwise parse the text section as binary garbage.
        // (The format check fires before the vertex-count check, so an empty
        // data section is fine here.)
        let ascii = String::from_utf8(make_ply(0))
            .unwrap()
            .replace("binary_little_endian", "ascii");
        let err = parse_splat_ply(ascii.as_bytes(), "test", MAX_SPLATS).unwrap_err();
        assert!(err.contains("ascii"), "unexpected error: {err}");

        // Missing 3DGS properties (e.g. a compressed or plain point-cloud .ply).
        let plain = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
            property float x\nproperty float y\nproperty float z\nend_header\n\
            \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let err = parse_splat_ply(plain, "test", MAX_SPLATS).unwrap_err();
        assert!(err.contains("scale_0"), "unexpected error: {err}");

        // No format line at all.
        assert!(parse_splat_ply(
            b"ply\nelement vertex 1\nproperty float x\nend_header\n\x00\x00\x00\x00",
            "test",
            MAX_SPLATS
        )
        .is_err());
    }
}

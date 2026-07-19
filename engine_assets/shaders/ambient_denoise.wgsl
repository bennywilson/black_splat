// Cross-bilateral denoise for AmbientPass's AO + GI outputs
// (ambient_probe.wgsl): a depth+normal-edge-aware blur that smooths out the
// per-pixel sampling noise (kernel rotation, ray jitter) without a temporal
// history buffer. "Cross" because the blur weights come from a different
// pair of buffers (G-buffer depth/normal) than the ones being blurred
// (AO/GI) -- that's what keeps it from bleeding across silhouette edges and
// creases the way a plain Gaussian blur would.
//
// Run as two separable passes (horizontal then vertical, see
// AmbientPass::render) rather than one full 2D kernel -- approximating a
// bilateral filter as separable isn't exact (the edge-stopping weights
// technically break separability) but it's the standard practical
// approximation and is a fraction of the cost of a full NxN pass.

struct DenoiseUniform {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz world position, w unused
    params: vec4<f32>,       // xy render target size in pixels, zw texel step direction (1,0 or 0,1)
    // x: blur radius in taps (0..=RADIUS, AmbientSettings::denoise_radius),
    // y: edge-tolerance multiplier on DEPTH_SIGMA (AmbientSettings::denoise_strength), zw unused.
    blur_params: vec4<f32>,
};

@group(0) @binding(0)
var t_ao: texture_2d<f32>;
@group(0) @binding(1)
var t_gi: texture_2d<f32>;
@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var t_depth: texture_depth_2d;

@group(1) @binding(0)
var<uniform> denoise: DenoiseUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn reconstruct_world_pos(coords: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(coords) + vec2<f32>(0.5)) / denoise.params.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_w = denoise.inv_view_proj * ndc;
    return world_w.xyz / world_w.w;
}

// 9-tap Gaussian half-kernel (offsets 0..4), sigma ~= 2. denoise.blur_params.x
// (AmbientSettings::denoise_radius) picks how many of these 4 taps to
// actually use per side, clamped to this range on the Rust side.
const RADIUS: i32 = 4;
const GAUSSIAN: array<f32, 5> = array<f32, 5>(
    0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162,
);

// Edge-stopping tolerances: DEPTH_SIGMA is in world units (matches
// AO_RADIUS's scale in ambient_probe.wgsl), NORMAL_POWER sharply cuts off
// neighbors whose normal diverges from the center pixel's.
const DEPTH_SIGMA: f32 = 0.2;
const NORMAL_POWER: f32 = 16.0;

struct FsOut {
    @location(0) ao: f32,
    @location(1) gi: vec4<f32>,
};

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> FsOut {
    let coords = vec2<i32>(pos.xy);
    let depth_c = textureLoad(t_depth, coords, 0);
    if (depth_c >= 1.0) {
        // No geometry here -- ambient_probe.wgsl already wrote the neutral
        // values (AO 1.0, GI 0.0) for sky pixels, nothing to blur.
        return FsOut(textureLoad(t_ao, coords, 0).r, textureLoad(t_gi, coords, 0));
    }

    let normal_c = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let world_c = reconstruct_world_pos(coords, depth_c);
    let dir = vec2<i32>(i32(denoise.params.z), i32(denoise.params.w));
    let dims = vec2<i32>(denoise.params.xy);
    let radius = clamp(i32(denoise.blur_params.x), 0, RADIUS);
    let depth_sigma = DEPTH_SIGMA * max(denoise.blur_params.y, 0.001);

    var ao_sum = 0.0;
    var gi_sum = vec4<f32>(0.0);
    var w_sum = 0.0;
    for (var i = -radius; i <= radius; i = i + 1) {
        let sc = coords + dir * i;
        if (sc.x < 0 || sc.y < 0 || sc.x >= dims.x || sc.y >= dims.y) {
            continue;
        }
        let d = textureLoad(t_depth, sc, 0);
        if (d >= 1.0) {
            continue;
        }
        let n = normalize(textureLoad(t_normal, sc, 0).xyz * 2.0 - 1.0);
        let world_s = reconstruct_world_pos(sc, d);
        let dist2 = dot(world_s - world_c, world_s - world_c);

        let spatial_w = GAUSSIAN[u32(abs(i))];
        let depth_w = exp(-dist2 / (2.0 * depth_sigma * depth_sigma));
        let normal_w = pow(max(dot(n, normal_c), 0.0), NORMAL_POWER);
        let w = spatial_w * depth_w * normal_w;

        ao_sum += textureLoad(t_ao, sc, 0).r * w;
        gi_sum += textureLoad(t_gi, sc, 0) * w;
        w_sum += w;
    }

    if (w_sum < 0.0001) {
        return FsOut(textureLoad(t_ao, coords, 0).r, textureLoad(t_gi, coords, 0));
    }
    return FsOut(ao_sum / w_sum, gi_sum / w_sum);
}

// Screen-space ambient occlusion + a cheap single-bounce screen-space diffuse
// GI, both derived from ray-marching the existing G-buffer depth. Runs once
// per frame after GBufferPass and before LightingPass; LightingPass's
// skylight shader (light_skylight.wgsl) samples the two outputs here to
// darken/brighten its ambient diffuse term with something that reacts to
// nearby geometry, on top of the static baked skylight cubemap.
//
// GI is a same-frame approximation, not true relit radiance: a ray that hits
// geometry returns that surface's albedo scaled by facing, not what's
// actually lighting it. No temporal accumulation or history buffer. When a
// ray escapes the screen without hitting anything, it falls back to the
// baked skylight's SH irradiance (see light_skylight.wgsl's
// eval_sh_irradiance, duplicated here since this shader asset system has no
// cross-file includes -- skylight_sh_project.wgsl already established that
// precedent for the same reason).

struct AmbientUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz world position, w: 1.0 if a baked skylight exists
    target_dims: vec4<f32>, // xy render target size in pixels, z: ssao_enabled, w: ssgi_enabled
    gi_params: vec4<f32>,   // x: gi_intensity
};

@group(0) @binding(0)
var t_albedo: texture_2d<f32>;
@group(0) @binding(1)
var t_normal: texture_2d<f32>;
@group(0) @binding(2)
var t_spec: texture_2d<f32>;
@group(0) @binding(3)
var t_depth: texture_depth_2d;

@group(1) @binding(0)
var<uniform> ambient: AmbientUniform;
@group(1) @binding(1)
var t_env: texture_cube<f32>;
@group(1) @binding(2)
var s_env: sampler;
@group(1) @binding(3)
var<storage, read> sh_coeffs: array<vec4<f32>, 9>;

// Same real-SH irradiance evaluation as light_skylight.wgsl -- keep the
// constants identical to that copy if either ever changes.
fn eval_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let a0 = 3.141593;
    let a1 = 2.094395;
    let a2 = 0.785398;
    var res = sh_coeffs[0].rgb * (0.282095 * a0);
    res += sh_coeffs[1].rgb * (0.488603 * n.y * a1);
    res += sh_coeffs[2].rgb * (0.488603 * n.z * a1);
    res += sh_coeffs[3].rgb * (0.488603 * n.x * a1);
    res += sh_coeffs[4].rgb * (1.092548 * n.x * n.y * a2);
    res += sh_coeffs[5].rgb * (1.092548 * n.y * n.z * a2);
    res += sh_coeffs[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0) * a2);
    res += sh_coeffs[7].rgb * (1.092548 * n.x * n.z * a2);
    res += sh_coeffs[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y) * a2);
    return max(res / 3.141593, vec3<f32>(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn reconstruct_world_pos(coords: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(coords) + vec2<f32>(0.5)) / ambient.target_dims.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_w = ambient.inv_view_proj * ndc;
    return world_w.xyz / world_w.w;
}

// Projects a world-space point to pixel coords + screen-space depth; w<=0 or
// off-screen results are flagged via the returned bool.
fn project_to_screen(world_pos: vec3<f32>) -> vec3<f32> {
    let clip = ambient.view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    return vec3<f32>(uv, ndc.z);
}

fn hash13(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return mat3x3<f32>(tangent, bitangent, n);
}

const AO_SAMPLES: u32 = 12u;
const AO_RADIUS: f32 = 0.5;
const AO_BIAS: f32 = 0.02;
// Fixed hemisphere-biased kernel (unit sphere points, |k| in (0,1]). Without
// a per-pixel rotation this is sampled identically at every pixel, so flat
// surfaces get a coherent, structured occlusion pattern -- hard banded edges
// that read as shadows from several lights rather than soft contact AO.
// sample_ao rotates it around the surface normal (tbn's z-axis) per pixel to
// break that up into noise instead.
const AO_KERNEL: array<vec3<f32>, 12> = array<vec3<f32>, 12>(
    vec3<f32>(0.245, 0.056, 0.334), vec3<f32>(-0.152, 0.348, 0.209),
    vec3<f32>(0.398, -0.211, 0.113), vec3<f32>(-0.336, -0.284, 0.421),
    vec3<f32>(0.104, 0.462, 0.531), vec3<f32>(-0.482, 0.128, 0.267),
    vec3<f32>(0.221, -0.437, 0.618), vec3<f32>(0.512, 0.302, 0.089),
    vec3<f32>(-0.219, -0.098, 0.762), vec3<f32>(0.063, 0.571, 0.312),
    vec3<f32>(-0.402, 0.219, 0.874), vec3<f32>(0.331, -0.126, 0.945),
);

fn sample_ao(world_pos: vec3<f32>, tbn: mat3x3<f32>, rotation: f32) -> f32 {
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    var occlusion = 0.0;
    for (var i = 0u; i < AO_SAMPLES; i = i + 1u) {
        let k = AO_KERNEL[i];
        let rotated_k = vec3<f32>(k.x * cos_r - k.y * sin_r, k.x * sin_r + k.y * cos_r, k.z);
        let sample_world = world_pos + tbn * rotated_k * AO_RADIUS;
        let screen = project_to_screen(sample_world);
        if (screen.x < 0.0 || screen.x > 1.0 || screen.y < 0.0 || screen.y > 1.0) {
            continue;
        }
        let coords = vec2<i32>(screen.xy * ambient.target_dims.xy);
        let scene_depth = textureLoad(t_depth, coords, 0);
        if (scene_depth >= 1.0) {
            continue;
        }
        let scene_world = reconstruct_world_pos(coords, scene_depth);
        let scene_dist = length(scene_world - ambient.camera_pos.xyz);
        let sample_dist = length(sample_world - ambient.camera_pos.xyz);
        let range_check =
            clamp(AO_RADIUS / max(abs(scene_dist - sample_dist), 0.0001), 0.0, 1.0);
        if (scene_dist < sample_dist - AO_BIAS) {
            occlusion += range_check;
        }
    }
    return 1.0 - clamp(occlusion / f32(AO_SAMPLES), 0.0, 1.0);
}

const GI_STEPS: u32 = 16u;
const GI_STEP_SIZE: f32 = 0.15;
const GI_THICKNESS: f32 = 0.3;

fn trace_gi_ray(world_pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var pos = world_pos + dir * GI_STEP_SIZE * 0.5;
    for (var i = 0u; i < GI_STEPS; i = i + 1u) {
        let screen = project_to_screen(pos);
        if (screen.x < 0.0 || screen.x > 1.0 || screen.y < 0.0 || screen.y > 1.0) {
            break;
        }
        let coords = vec2<i32>(screen.xy * ambient.target_dims.xy);
        let scene_depth = textureLoad(t_depth, coords, 0);
        if (scene_depth < 1.0) {
            let scene_world = reconstruct_world_pos(coords, scene_depth);
            let scene_dist = distance(scene_world, ambient.camera_pos.xyz);
            let ray_dist = distance(pos, ambient.camera_pos.xyz);
            if (ray_dist > scene_dist && distance(scene_world, pos) < GI_THICKNESS) {
                let hit_albedo = textureLoad(t_albedo, coords, 0).rgb;
                let hit_normal =
                    normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
                let facing = max(dot(hit_normal, -dir), 0.0);
                return hit_albedo * facing;
            }
        }
        pos += dir * GI_STEP_SIZE;
    }
    if (ambient.camera_pos.w > 0.5) {
        return eval_sh_irradiance(dir);
    }
    return vec3<f32>(0.0);
}

const GI_RAYS: u32 = 4u;

struct FsOut {
    @location(0) ao: f32,
    @location(1) gi: vec4<f32>,
};

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> FsOut {
    let coords = vec2<i32>(pos.xy);
    let depth = textureLoad(t_depth, coords, 0);
    if (depth >= 1.0) {
        return FsOut(1.0, vec4<f32>(0.0));
    }

    let normal = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let world_pos = reconstruct_world_pos(coords, depth);
    let tbn = build_basis(normal);

    // See AmbientSettings in ambient.rs -- packed into the otherwise-unused
    // target_dims.zw rather than growing the uniform.
    var ao = 1.0;
    if (ambient.target_dims.z > 0.5) {
        // Offset seed from the GI rotation below so the two effects' noise
        // doesn't line up pixel-for-pixel.
        let ao_rotation = hash13(pos.xy + vec2<f32>(17.0, 31.0)) * 6.28318530718;
        ao = sample_ao(world_pos, tbn, ao_rotation);
    }

    var gi = vec3<f32>(0.0);
    if (ambient.target_dims.w > 0.5) {
        let rotation = hash13(pos.xy) * 6.28318530718;
        let cos_r = cos(rotation);
        let sin_r = sin(rotation);
        for (var i = 0u; i < GI_RAYS; i = i + 1u) {
            // Interleaved Fibonacci-ish hemisphere directions, rotated per-pixel.
            let fi = f32(i) + 0.5;
            let phi = fi * 2.39996323;
            let cos_theta = sqrt(1.0 - fi / f32(GI_RAYS));
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            let lx = cos(phi) * sin_theta;
            let ly = sin(phi) * sin_theta;
            let rotated = vec3<f32>(lx * cos_r - ly * sin_r, lx * sin_r + ly * cos_r, cos_theta);
            let dir = normalize(tbn * rotated);
            gi += trace_gi_ray(world_pos, dir);
        }
        gi /= f32(GI_RAYS);
        gi *= ambient.gi_params.x;
    }

    return FsOut(ao, vec4<f32>(gi, 1.0));
}

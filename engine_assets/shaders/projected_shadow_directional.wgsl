struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,
    direction_cone: vec4<f32>,
    color_cone: vec4<f32>,
    color2: vec4<f32>,
    camera_pos: vec4<f32>,
    target_dims: vec4<f32>,      // xy render target size, zw shadow map size
    shadow_matrices: array<mat4x4<f32>, 4>,
    shadow_rects: array<vec4<f32>, 4>,  // per tile: xy uv offset, zw uv scale
    shadow_params: vec4<f32>     // x num cascades, y depth bias
};

@group(0) @binding(0)
var t_depth: texture_depth_2d;
@group(0) @binding(1)
var t_shadow: texture_depth_2d;
@group(0) @binding(2)
var s_shadow: sampler_comparison;

@group(1) @binding(0)
var<uniform> light: LightUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// 3x3 PCF inside one atlas tile.  Samples are clamped a texel inside the tile
// so filtering never bleeds into a neighboring cascade.
fn sample_shadow_tile(tile: i32, tile_uv: vec2<f32>, depth_ref: f32) -> f32 {
    let rect = light.shadow_rects[tile];
    let texel = vec2<f32>(1.0, 1.0) / light.target_dims.zw;
    let uv_min = rect.xy + texel;
    let uv_max = rect.xy + rect.zw - texel;
    let center = rect.xy + tile_uv * rect.zw;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let uv = clamp(center + vec2<f32>(f32(x), f32(y)) * texel, uv_min, uv_max);
            sum = sum + textureSampleCompareLevel(t_shadow, s_shadow, uv, depth_ref);
        }
    }
    return sum / 9.0;
}

// 1 = fully lit, 0 = fully shadowed.  Walks the cascades near-to-far and uses
// the first whose light-space footprint contains the point.
fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let count = i32(light.shadow_params.x);
    for (var i = 0; i < count; i = i + 1) {
        let ls = light.shadow_matrices[i] * vec4<f32>(world_pos, 1.0);
        let ndc = ls.xyz / ls.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99
            || ndc.z <= 0.0 || ndc.z >= 1.0) {
            continue;  // Outside this cascade; try the next one out.
        }
        return sample_shadow_tile(i, uv, ndc.z - light.shadow_params.y);
    }
    return 1.0;  // Beyond the last cascade: unshadowed.
}

struct MaskOutput {
    @location(0) mask: vec4<f32>,       // Per-light shadow mask
    @location(1) accum: vec4<f32>       // Accumulation texture for shadow catchers
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> MaskOutput {
    let coords = vec2<i32>(pos.xy);
    let depth = textureLoad(t_depth, coords, 0);

    var factor = 1.0;
    if (depth < 1.0) {
        // World position rebuilt from the depth buffer.
        let uv = pos.xy / light.target_dims.xy;
        let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        let world_w = light.inv_view_proj * ndc;
        factor = shadow_factor(world_w.xyz / world_w.w);
    }

    var out: MaskOutput;
    out.mask = vec4<f32>(factor, factor, factor, 1.0);
    out.accum = vec4<f32>(factor, factor, factor, 1.0);
    return out;
}

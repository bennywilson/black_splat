struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,
    direction_cone: vec4<f32>,
    color_cone: vec4<f32>,
    color2: vec4<f32>,
    camera_pos: vec4<f32>,
    target_dims: vec4<f32>,      // xy render target size, zw shadow map size
    shadow_matrices: array<mat4x4<f32>, 4>,  // [0] is the spot's projection
    shadow_rects: array<vec4<f32>, 4>,       // [0]: xy uv offset, zw uv scale
    shadow_params: vec4<f32>     // x 1 = shadowed, y depth bias
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
    // Fullscreen triangle from the vertex index alone (no vertex buffer).
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// 1 = fully lit, 0 = fully shadowed: the spot's projected map with a 3x3 PCF,
// clamped a texel inside its rect so filtering never wraps at the edges.
fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let ls = light.shadow_matrices[0] * vec4<f32>(world_pos, 1.0);
    if (ls.w <= 0.0) {
        return 1.0;  // Behind the light.
    }
    let ndc = ls.xyz / ls.w;
    let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (tile_uv.x < 0.0 || tile_uv.x > 1.0 || tile_uv.y < 0.0 || tile_uv.y > 1.0
        || ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 1.0;  // Outside the projection: the cone falloff handles it.
    }
    let depth_ref = ndc.z - light.shadow_params.y;

    let rect = light.shadow_rects[0];
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

struct MaskOutput {
    @location(0) mask: vec4<f32>,
    @location(1) accum: vec4<f32>
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> MaskOutput {
    let coords = vec2<i32>(pos.xy);
    let depth = textureLoad(t_depth, coords, 0);

    var factor = 1.0;
    if (depth < 1.0 && light.shadow_params.x > 0.5) {
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

// Deferred skylight: an ambient hemisphere.  Every lit pixel gets albedo *
// mix(bottom, top) blended on how upward-facing its world normal is.  Drawn as
// a fullscreen triangle, additively accumulated with the other light passes.

struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,   // xyz world position, w range (unused here)
    direction_cone: vec4<f32>,   // xyz direction, w cos(outer angle) (unused here)
    color_cone: vec4<f32>,       // rgb top color * intensity, w cos(inner angle)
    color2: vec4<f32>,           // rgb bottom color * intensity
    camera_pos: vec4<f32>,
    target_dims: vec4<f32>       // xy render target size in pixels
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
var<uniform> light: LightUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle from the vertex index alone (no vertex buffer).
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(pos.xy);
    let depth = textureLoad(t_depth, coords, 0);
    if (depth >= 1.0) {
        // No world geometry here: leave the clear color untouched.
        discard;
    }

    let albedo = textureLoad(t_albedo, coords, 0).rgb;
    let normal = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let metallic = textureLoad(t_spec, coords, 0).x;

    let up_ness = normal.y * 0.5 + 0.5;
    let sky = mix(light.color2.rgb, light.color_cone.rgb, up_ness);

    // Metals have no diffuse; keep a little ambient so they don't go black
    // (no environment reflections yet).
    return vec4<f32>(albedo * sky * (1.0 - metallic * 0.7), 1.0);
}

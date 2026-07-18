// Deferred skylight: an ambient hemisphere, or -- once the skylight has a
// baked environment cubemap (see Renderer::bake_skylight_cubemap) -- that
// cubemap sampled by world normal, a cheap stand-in for a real irradiance
// convolution.

struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,   // xyz world position, w range (unused here)
    direction_cone: vec4<f32>,   // xyz direction, w cos(outer angle) (unused here)
    color_cone: vec4<f32>,       // rgb top color * intensity, w cos(inner angle)
    color2: vec4<f32>,           // rgb bottom color * intensity, w 1.0 if t_env is a real bake
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
@group(1) @binding(1)
var t_env: texture_cube<f32>;
@group(1) @binding(2)
var s_env: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(pos.xy);
    let depth = textureLoad(t_depth, coords, 0);
    if (depth >= 1.0) {
        // No world geometry here, bail
        discard;
    }

    let albedo = textureLoad(t_albedo, coords, 0).rgb;
    let normal = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let metallic = textureLoad(t_spec, coords, 0).x;

    var sky: vec3<f32>;
    if (light.color2.a > 0.5) {
        sky = textureSample(t_env, s_env, normal).rgb * light.color_cone.rgb;
    } else {
        let up_ness = normal.y * 0.5 + 0.5;
        sky = mix(light.color2.rgb, light.color_cone.rgb, up_ness);
    }

    // Metals have no diffuse; keep a little ambient so they don't go black
    // until we get environmental reflections
    return vec4<f32>(albedo * sky * (1.0 - metallic * 0.7), 1.0);
}

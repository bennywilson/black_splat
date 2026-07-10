// Deferred directional light: Lambert diffuse + Blinn-Phong specular from a
// light shining along direction_cone.xyz.  Fullscreen triangle, additively
// accumulated with the other light passes.

struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,   // xyz world position (unused), w range (unused)
    direction_cone: vec4<f32>,   // xyz direction the light points, w cos(outer)
    color_cone: vec4<f32>,       // rgb color * intensity, w cos(inner)
    color2: vec4<f32>,           // skylight bottom color (unused here)
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
        discard;
    }

    let albedo = textureLoad(t_albedo, coords, 0).rgb;
    let normal = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let spec = textureLoad(t_spec, coords, 0);

    // World position rebuilt from the depth buffer.
    let uv = pos.xy / light.target_dims.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_w = light.inv_view_proj * ndc;
    let world_pos = world_w.xyz / world_w.w;

    let to_light = normalize(-light.direction_cone.xyz);
    let n_dot_l = max(dot(normal, to_light), 0.0);

    // Blinn-Phong specular; gloss (spec.a) maps to the exponent.
    let view_dir = normalize(light.camera_pos.xyz - world_pos);
    let half_dir = normalize(to_light + view_dir);
    let shininess = mix(2.0, 128.0, spec.a);
    let spec_term = pow(max(dot(normal, half_dir), 0.0), shininess) * step(0.0001, n_dot_l);

    let lit = albedo * n_dot_l + spec.rgb * spec_term;
    return vec4<f32>(lit * light.color_cone.rgb, 1.0);
}

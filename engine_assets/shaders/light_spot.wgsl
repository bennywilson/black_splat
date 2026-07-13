struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,              // xyz world position, w range
    direction_cone: vec4<f32>,              // xyz direction the light points, w cos(outer)
    color_cone: vec4<f32>,                  // rgb color * intensity, w cos(inner)
    color2: vec4<f32>,                      // skylight bottom color (unused here)
    camera_pos: vec4<f32>,
    target_dims: vec4<f32>,                 // xy render target size, zw shadow map size
    shadow_matrices: array<mat4x4<f32>, 4>, // used by the mask pass
    shadow_rects: array<vec4<f32>, 4>,
    shadow_params: vec4<f32>                // x > 0 = sample the shadow mask
};

@group(0) @binding(0)
var t_albedo: texture_2d<f32>;
@group(0) @binding(1)
var t_normal: texture_2d<f32>;
@group(0) @binding(2)
var t_spec: texture_2d<f32>;
@group(0) @binding(3)
var t_depth: texture_depth_2d;
@group(0) @binding(4)
var t_mask: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> light: LightUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Cook-Torrance GGX with Schlick Fresnel and Smith-Schlick geometry.  The
// 1/PI diffuse normalization is folded into light intensity so an intensity
// of 1 stays intuitively bright.
fn pbr_brdf(albedo: vec3<f32>, metallic: f32, roughness: f32,
            n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let r = clamp(roughness, 0.045, 1.0);
    let a = r * r;
    let a2 = a * a;

    // GGX normal distribution.
    let d_denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d = a2 / (3.14159265 * d_denom * d_denom);

    // Smith-Schlick geometry term.
    let k = (r + 1.0) * (r + 1.0) / 8.0;
    let g = (n_dot_v / (n_dot_v * (1.0 - k) + k)) * (n_dot_l / (n_dot_l * (1.0 - k) + k));

    // Schlick Fresnel: dielectrics reflect ~4%, metals tint by albedo.
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    let f = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * pow(1.0 - h_dot_v, 5.0);

    let specular = d * g * f / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - f) * (1.0 - metallic);
    return (k_d * albedo + specular) * radiance * n_dot_l;
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
    let mr = textureLoad(t_spec, coords, 0);

    // World position rebuilt from the depth buffer.
    let uv = pos.xy / light.target_dims.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_w = light.inv_view_proj * ndc;
    let world_pos = world_w.xyz / world_w.w;

    let to_light_vec = light.position_range.xyz - world_pos;
    let dist = length(to_light_vec);
    let range = max(light.position_range.w, 0.001);
    var attenuation = saturate(1.0 - dist / range);
    attenuation = attenuation * attenuation;

    let to_light = to_light_vec / max(dist, 0.0001);

    // Cone falloff: 1 inside the inner cone, fading to 0 at the outer cone.
    let cos_angle = dot(normalize(light.direction_cone.xyz), -to_light);
    let cone = smoothstep(light.direction_cone.w, light.color_cone.w, cos_angle);
    attenuation *= cone;
    if (attenuation <= 0.0) {
        discard;
    }

    var shadow = 1.0;
    if (light.shadow_params.x > 0.5) {
        shadow = textureLoad(t_mask, coords, 0).x;
    }

    let view_dir = normalize(light.camera_pos.xyz - world_pos);
    let lit = pbr_brdf(albedo, mr.x, mr.y, normal, view_dir, to_light,
                       light.color_cone.rgb * attenuation);
    return vec4<f32>(lit * shadow, 1.0);
}

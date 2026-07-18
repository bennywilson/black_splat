// Deferred skylight: an ambient hemisphere, or -- once the skylight has a
// baked environment cubemap (see Renderer::bake_skylight_cubemap) -- that
// cubemap sampled by world normal, a cheap stand-in for a real irradiance
// convolution. Also contributes a mirror-sharp specular reflection term
// (Fresnel-weighted against the diffuse term), so metallic surfaces have
// something to show besides diffuse ambient.

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
        // No world geometry here. If we have a real bake, use it as a
        // skybox -- both a cheap background and the most direct way to see
        // what got captured. Otherwise leave the clear color alone.
        if (light.color2.a > 0.5) {
            let uv = pos.xy / light.target_dims.xy;
            let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
            let world_w = light.inv_view_proj * ndc;
            let world_pos = world_w.xyz / world_w.w;
            let view_dir = normalize(world_pos - light.camera_pos.xyz);
            let sky_bg = textureSample(t_env, s_env, view_dir).rgb * light.direction_cone.a;
            return vec4<f32>(sky_bg, 1.0);
        }
        discard;
    }

    let albedo = textureLoad(t_albedo, coords, 0).rgb;
    let normal = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let metallic = textureLoad(t_spec, coords, 0).x;

    // World position + view/reflection rays, needed for the specular
    // ambient term below (a fully metallic surface has no diffuse, so
    // without this it just renders black).
    let uv = pos.xy / light.target_dims.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_w = light.inv_view_proj * ndc;
    let world_pos = world_w.xyz / world_w.w;
    let view_dir = normalize(light.camera_pos.xyz - world_pos);
    let reflect_dir = reflect(-view_dir, normal);

    var sky: vec3<f32>;
    var refl: vec3<f32>;
    if (light.color2.a > 0.5) {
        // The captured cubemap is already full-color radiance -- scale by
        // plain intensity only, not the Color swatch (that's baked into the
        // capture's exposure, not a tint to reapply on top of real pixels).
        sky = textureSample(t_env, s_env, normal).rgb * light.direction_cone.a;
        // Mirror-sharp reflection -- no roughness blur yet, since the bake
        // has no mip chain to filter against (see Renderer::bake_skylight_cubemap).
        refl = textureSample(t_env, s_env, reflect_dir).rgb * light.direction_cone.a;
    } else {
        let up_ness = normal.y * 0.5 + 0.5;
        sky = mix(light.color2.rgb, light.color_cone.rgb, up_ness);
        let refl_up_ness = reflect_dir.y * 0.5 + 0.5;
        refl = mix(light.color2.rgb, light.color_cone.rgb, refl_up_ness);
    }

    // Schlick Fresnel split between diffuse ambient (rolls off with
    // metallic) and the specular reflection above -- otherwise fully
    // metallic, low-roughness surfaces would have nothing to show but a
    // near-black diffuse term.
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * pow(1.0 - n_dot_v, 5.0);

    let diffuse = albedo * sky * (1.0 - metallic) * (vec3<f32>(1.0, 1.0, 1.0) - fresnel);
    let specular = refl * fresnel;

    return vec4<f32>(diffuse + specular, 1.0);
}

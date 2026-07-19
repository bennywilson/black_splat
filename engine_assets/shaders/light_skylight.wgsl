// Deferred skylight: an ambient hemisphere, or -- once the skylight has a
// baked environment cubemap (see Renderer::bake_skylight_cubemap) -- diffuse
// irradiance from a 9-term spherical-harmonic projection of that cube plus a
// specular reflection sampled from its GGX-prefiltered mip chain (mip 0 =
// mirror, deeper mips = wider roughness lobe). Can additionally draw the bake
// as the background where nothing was rendered, to inspect it
// (Light::show_env_as_skybox).

struct LightUniform {
    inv_view_proj: mat4x4<f32>,
    position_range: vec4<f32>,   // xyz world position, w range (unused here)
    direction_cone: vec4<f32>,   // xyz direction, w cos(outer angle) (unused here)
    color_cone: vec4<f32>,       // rgb top color * intensity, w cos(inner angle)
    color2: vec4<f32>,           // rgb bottom color * intensity, w 1.0 if t_env is a real bake
    camera_pos: vec4<f32>,        // xyz camera world position, w 1.0 to draw t_env as the background
    target_dims: vec4<f32>       // xy render target size in pixels
};

// Mip chain's roughness range. See Renderer::bake_skylight_cubemap.
struct SkylightEnvUniform {
    mip_params: vec4<f32>,   // x: highest mip index (roughness 1.0 samples this level)
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
@group(1) @binding(3)
var<storage, read> sh_coeffs: array<vec4<f32>, 9>;
@group(1) @binding(4)
var<uniform> sky_env: SkylightEnvUniform;

// Standard real-SH irradiance evaluation (Ramamoorthi & Hanrahan): `sh_coeffs`
// holds raw radiance projection coefficients (written by
// skylight_sh_project.wgsl); the A0/A1/A2 constants below fold in the
// cosine-lobe convolution so this directly returns cosine-weighted
// irradiance, not raw radiance.
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
    // Divide the cosine-convolved projection by pi to turn irradiance into
    // the outgoing Lambertian radiance a diffuse albedo multiplies against.
    return max(res / 3.141593, vec3<f32>(0.0));
}

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
        // No world geometry here. With the skybox debug view on, draw the
        // bake as the background -- the most direct way to see what got
        // captured. Otherwise leave the clear color alone.
        if (light.camera_pos.w > 0.5) {
            let uv = pos.xy / light.target_dims.xy;
            let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
            let world_w = light.inv_view_proj * ndc;
            let world_pos = world_w.xyz / world_w.w;
            let view_dir = normalize(world_pos - light.camera_pos.xyz);
            // textureSampleLevel, not textureSample: this is inside branches on
            // per-fragment depth, and WGSL only permits implicit-derivative
            // sampling in uniform control flow.
            let sky_bg = textureSampleLevel(t_env, s_env, view_dir, 0.0).rgb * light.direction_cone.a;
            return vec4<f32>(sky_bg, 1.0);
        }
        discard;
    }

    let albedo = textureLoad(t_albedo, coords, 0).rgb;
    let normal = normalize(textureLoad(t_normal, coords, 0).xyz * 2.0 - 1.0);
    let spec = textureLoad(t_spec, coords, 0);
    let metallic = spec.x;
    let roughness = clamp(spec.y, 0.045, 1.0);

    // World position + view/reflection rays, needed for the specular
    // ambient term below (a fully metallic surface has no diffuse, so
    // without this it just renders black).
    let uv = pos.xy / light.target_dims.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_w = light.inv_view_proj * ndc;
    let world_pos = world_w.xyz / world_w.w;
    let view_dir = normalize(light.camera_pos.xyz - world_pos);
    let reflect_dir = normalize(reflect(-view_dir, normal));

    var sky: vec3<f32>;
    var refl: vec3<f32>;
    if (light.color2.a > 0.5) {
        // The captured cubemap is already full-color radiance -- scale by
        // plain intensity only, not the Color swatch (that's baked into the
        // capture's exposure, not a tint to reapply on top of real pixels).
        sky = eval_sh_irradiance(normal) * light.direction_cone.a;
        // Sample the GGX-prefiltered mip matching this surface's roughness
        // (mip 0 = mirror, sky_env.mip_params.x = roughest available level).
        let mip = roughness * sky_env.mip_params.x;
        refl = textureSampleLevel(t_env, s_env, reflect_dir, mip).rgb * light.direction_cone.a;
    } else {
        // No baked mip chain to blur against here -- bend the reflection ray
        // back toward the normal as roughness rises so the two-color gradient
        // still softens for rough surfaces instead of staying a hard mirror.
        let bent_reflect_dir = normalize(mix(reflect_dir, normal, roughness * roughness));
        let up_ness = normal.y * 0.5 + 0.5;
        sky = mix(light.color2.rgb, light.color_cone.rgb, up_ness);
        let refl_up_ness = bent_reflect_dir.y * 0.5 + 0.5;
        refl = mix(light.color2.rgb, light.color_cone.rgb, refl_up_ness);
    }

    // Schlick Fresnel split between diffuse ambient (rolls off with
    // metallic) and the specular reflection above -- otherwise fully
    // metallic, low-roughness surfaces would have nothing to show but a
    // near-black diffuse term.
    //
    // The grazing term is capped at `1 - roughness` rather than 1.0
    // (Fdez-Aguera): plain Schlick drives every surface to a full-strength
    // mirror at grazing angles, which turns matte dielectrics into chrome
    // around their silhouettes.
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let f_grazing = max(vec3<f32>(1.0 - roughness), f0);
    let fresnel = f0 + (f_grazing - f0) * pow(1.0 - n_dot_v, 5.0);

    // Split-sum env BRDF, analytic fit (Karis mobile approximation): the
    // energy a lobe this rough actually reflects, without a BRDF LUT. The
    // roughness-dependent terms matter -- pinning them to their roughness-0
    // values drives the weight to a flat 1.0 at grazing angles, so rough
    // surfaces come out mirror-bright right where they curve away.
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
          + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let env_brdf = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    let spec_weight = fresnel * env_brdf.x + env_brdf.y;

    let diffuse = albedo * sky * (1.0 - metallic) * (vec3<f32>(1.0, 1.0, 1.0) - fresnel);
    let specular = refl * spec_weight;

    return vec4<f32>(diffuse + specular, 1.0);
}

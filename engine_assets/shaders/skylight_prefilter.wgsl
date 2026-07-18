// GGX-prefilters a skylight's baked mip-0 environment cube into its roughness
// mip chain (see Renderer::bake_skylight_cubemap and
// LightingPass::prefilter_skylight_mips). One draw per (mip, face): the
// vertex shader is a fullscreen triangle; the fragment shader reconstructs
// this texel's world direction using the same 90-degree-FOV camera
// convention the original capture used (screen NDC -> inverse view-proj,
// camera at the origin so the unprojected point IS the direction), then
// averages a GGX-importance-sampled cone of directions around it, always
// reading the source cube's mip 0 (never the mip currently being written).

struct PrefilterUniform {
    inv_view_proj: mat4x4<f32>,
    params: vec4<f32>,   // x: roughness for this mip, y: this mip's face size in texels
};

@group(0) @binding(0)
var<uniform> u: PrefilterUniform;
@group(0) @binding(1)
var t_src: texture_cube<f32>;
@group(0) @binding(2)
var s_src: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Low-discrepancy Hammersley sequence (bit-reversal Van der Corput), the
// standard sample generator for GGX importance sampling (Karis 2013).
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// Maps a low-discrepancy 2D sample to a GGX half-vector around `n`, in world
// space (Karis 2013, "Real Shading in Unreal Engine 4").
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32, n: vec3<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * 3.14159265 * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let h_tangent = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * h_tangent.x + bitangent * h_tangent.y + n * h_tangent.z);
}

const SAMPLE_COUNT: u32 = 32u;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = u.params.y;
    let uv = pos.xy / size;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    let world = u.inv_view_proj * ndc;
    let n = normalize(world.xyz / world.w);

    let roughness = u.params.x;
    // Split-sum reference assumption: view direction, normal, and reflection
    // vector are all `n` (Karis 2013) -- fine for an ambient/IBL term where
    // there's no single "the" view direction until shading time.
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness, n);
        let l = normalize(2.0 * dot(n, h) * h - n);
        let n_dot_l = dot(n, l);
        if (n_dot_l > 0.0) {
            color += textureSampleLevel(t_src, s_src, l, 0.0).rgb * n_dot_l;
            weight_sum += n_dot_l;
        }
    }
    if (weight_sum > 0.0) {
        color = color / weight_sum;
    }
    return vec4<f32>(color, 1.0);
}

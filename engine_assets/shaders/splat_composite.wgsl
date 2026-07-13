
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from the vertex index (no vertex buffers).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var out: VSOut;
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.clip = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

@group(0) @binding(0) var t_splat: texture_2d<f32>;
@group(0) @binding(1) var s_splat: sampler;

struct TonemapUniform {
    // x A (HighlightScale), y B (MidtoneScale), z C (HighlightCurve), w D (MidtoneCurve)
    abcd: vec4<f32>,
    // x E (ShadowOffset), y exposure, z 1.0 when the postprocess tonemap is on
    e_exp_enabled: vec4<f32>,
};
@group(1) @binding(0) var<uniform> tm: TonemapUniform;

// sRGB EOTF (display -> linear).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lower = c / 12.92;
    let higher = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(higher, lower, c <= vec3<f32>(0.04045));
}

// Exact inverse of the postprocess tonemap, per channel: solves
// (A - yC)x^2 + (B - yD)x - yE = 0 for x >= 0.  The CPU keeps the curve
// invertible (A/C >= 1, A*D >= B*C) so qa >= 0 and disc >= 0 here.
fn tonemap_inverse(y: vec3<f32>) -> vec3<f32> {
    let a = tm.abcd.x;
    let b = tm.abcd.y;
    let c = tm.abcd.z;
    let d = tm.abcd.w;
    let e = tm.e_exp_enabled.x;

    let qa = a - y * c;
    let qb = b - y * d;
    let qc = -y * e;
    let disc = max(qb * qb - 4.0 * qa * qc, vec3<f32>(0.0));
    let two_a = 2.0 * max(qa, vec3<f32>(1e-6));
    let x = (-qb + sqrt(disc)) / two_a;
    return clamp(x, vec3<f32>(0.0), vec3<f32>(16.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let s = textureSample(t_splat, s_splat, in.uv);
    let a = s.a;
    if (a <= 0.0) {
        discard;
    }
    // The splat buffer holds a premultiplied over-composite; un-premultiply so
    // the nonlinear conversion runs on the actual splat color.
    let display = s.rgb / max(a, 1e-4);
    var linear = srgb_to_linear(display);
    if (tm.e_exp_enabled.z > 0.5) {
        let exposure = max(tm.e_exp_enabled.y, 1e-3);
        linear = tonemap_inverse(linear) / exposure;
    }
    // Non-premultiplied source: the pipeline blends this over the scene as
    // linear*a + scene*(1-a).
    return vec4<f32>(linear, a);
}

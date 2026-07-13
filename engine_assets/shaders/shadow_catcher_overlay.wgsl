// Screen-space shadow-catcher overlay.  An invisible catcher proxy has been
// rendered into its own depth buffer, and the frame's shadow-casting lights
// have been projected onto it into `t_catcher_shadow` (1 = lit, < 1 =
// shadowed by the inserted CG objects).  This pass multiply-blends that factor
// onto the already-composited scene color, darkening the Gaussian splats that
// sit behind the proxy so the CG objects read as grounded.

@group(0) @binding(0) var t_catcher_shadow: texture_2d<f32>;
@group(0) @binding(1) var t_catcher_depth: texture_depth_2d;
@group(0) @binding(2) var t_scene_depth: texture_depth_2d;

struct OverlayParams {
    // Artist-facing multiplier on how much the catcher darkens the scene.
    // 1 = the raw projected factor; > 1 deepens the shadow toward black; 0
    // disables it.  Applied as `1 - density * (1 - factor)`, clamped to [0, 1].
    // Padded to 16 bytes with plain scalars (vec3 would force 32-byte size).
    density: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(3) var<uniform> params: OverlayParams;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(pos.xy);
    let catcher_depth = textureLoad(t_catcher_depth, coords, 0);
    let scene_depth = textureLoad(t_scene_depth, coords, 0);

    var factor = 1.0;
    // catcher_depth < 1 => the proxy covers this pixel; catcher_depth <= scene
    // depth => the proxy (splat behind it) is what's visible here, not a nearer
    // CG object.  The small epsilon absorbs depth-buffer precision.
    if (catcher_depth < 0.9999 && catcher_depth <= scene_depth + 0.0005) {
        let raw = textureLoad(t_catcher_shadow, coords, 0).r;
        factor = clamp(1.0 - params.density * (1.0 - raw), 0.0, 1.0);
    }
    return vec4<f32>(factor, factor, factor, 1.0);
}

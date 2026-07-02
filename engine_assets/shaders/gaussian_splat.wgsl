// gaussian_splat.wgsl
//
// Ported from blk_engine's gaussian_splat_draw.shader (HLSL -> WGSL).
// Renders pre-sorted 3D gaussian splats as view-aligned billboards with
// spherical-harmonic view-dependent color and a gaussian falloff.

struct GlobalConstants {
    view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // x: falloff sharpness, y: splat scale, z: contrast, w: num_splats
    splat_params: vec4<f32>,
    // x: max sh degree, y: overall scale, z/w: unused
    splat_params_2: vec4<f32>,
};

struct Splat {
    position: vec4<f32>,        // xyz position, w unused
    scale_opacity: vec4<f32>,   // linear scale xyz, normalized opacity w
    rotation: vec4<f32>,        // quaternion x,y,z,w
    sh0: vec4<f32>,             // degree-0 SH (base color) rgb, w unused
    sh_rest: array<f32, 24>,    // 8 higher-order coeffs * 3 channels
};

@group(0) @binding(0) var<uniform> u: GlobalConstants;
@group(1) @binding(0) var<storage, read> g_splats: array<Splat>;
@group(1) @binding(1) var<storage, read> g_sorted_indices: array<u32>;

struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv_and_scale: vec4<f32>,
};

// Build the three rotation-matrix rows (local axes) from a quaternion.
// Matches the row layout used by the original HLSL implementation.
fn quat_axes(q_in: vec4<f32>) -> array<vec3<f32>, 3> {
    let q = normalize(q_in);
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    var axes: array<vec3<f32>, 3>;
    axes[0] = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    axes[1] = vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    axes[2] = vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
    return axes;
}

// sRGB -> linear.  The SH-evaluated splat color is in sRGB/gamma space, but the
// scene render target is an sRGB-format texture, so the GPU applies a linear->sRGB
// encode on store.  Converting to linear here cancels that encode (so colors
// aren't double-encoded / too bright) and lets alpha blending run in linear space.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lower = c / 12.92;
    let higher = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(higher, lower, c <= vec3<f32>(0.04045));
}

fn get_vertex_corner(corner_id: u32) -> vec2<f32> {
    var offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
    );
    return offsets[corner_id];
}

// RGB triple of higher-order SH coefficient `n` (0..7), read straight from storage.
fn sh_rest3(splat_id: u32, n: u32) -> vec3<f32> {
    return vec3<f32>(
        g_splats[splat_id].sh_rest[n * 3u],
        g_splats[splat_id].sh_rest[n * 3u + 1u],
        g_splats[splat_id].sh_rest[n * 3u + 2u],
    );
}

// Reads coefficients directly from storage (rather than from a whole-Splat copy)
// so the uniform degree branches control what actually gets fetched: degree 0
// touches 64 of each splat's 160 bytes, degree 2 all of them.  The struct's 8
// "rest" coefficients cover degrees 1+2, so 2 is the maximum meaningful degree.
fn evaluate_sh(n: vec3<f32>, splat_id: u32) -> vec3<f32> {
    let x = n.x;
    let y = n.y;
    let z = n.z;

    // Degree 0
    var result = 0.282095 * g_splats[splat_id].sh0.rgb;

    let degree = u.splat_params_2.x;

    // Degree 1
    if (degree >= 1.0) {
        result += (0.488603 * y) * sh_rest3(splat_id, 0u);
        result += (0.488603 * z) * sh_rest3(splat_id, 1u);
        result += (0.488603 * x) * sh_rest3(splat_id, 2u);
    }

    // Degree 2
    if (degree >= 2.0) {
        result += (1.092548 * x * y) * sh_rest3(splat_id, 3u);
        result += (1.092548 * y * z) * sh_rest3(splat_id, 4u);
        result += (0.315392 * (3.0 * z * z - 1.0)) * sh_rest3(splat_id, 5u);
        result += (1.092548 * x * z) * sh_rest3(splat_id, 6u);
        result += (0.546274 * (x * x - y * y)) * sh_rest3(splat_id, 7u);
    }

    return clamp(result + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> VSOutput {
    let overall_scale = u.splat_params_2.y;

    let sorted_index = vertex_id / 6u;
    let splat_id = g_sorted_indices[sorted_index];

    var output: VSOutput;

    // Skip padded / out-of-range entries by collapsing them to a degenerate point.
    if (f32(splat_id) >= u.splat_params.w) {
        output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.color = vec4<f32>(0.0);
        output.uv_and_scale = vec4<f32>(0.0);
        return output;
    }

    // Read fields individually (not `let splat = g_splats[splat_id]`): copying
    // the whole struct would fetch all 24 sh_rest floats before evaluate_sh's
    // degree branches get a chance to skip them.
    let splat_pos = g_splats[splat_id].position.xyz * overall_scale;
    // `var` (not `let`): naga/Vulkan only permit dynamic indexing (axes[long_axis_idx]
    // below) into a function-local variable, not into a value expression.
    var axes = quat_axes(g_splats[splat_id].rotation);
    let splat_scale = g_splats[splat_id].scale_opacity.xyz;
    let splat_opacity = g_splats[splat_id].scale_opacity.w;

    // Major/minor/intermediate axis selection.
    var long_axis_idx = 0;
    if (splat_scale.x > splat_scale.y) {
        if (splat_scale.x > splat_scale.z) { long_axis_idx = 0; } else { long_axis_idx = 2; }
    } else {
        if (splat_scale.y > splat_scale.z) { long_axis_idx = 1; } else { long_axis_idx = 2; }
    }
    var short_axis_idx = 0;
    if (splat_scale.x < splat_scale.y) {
        if (splat_scale.x < splat_scale.z) { short_axis_idx = 0; } else { short_axis_idx = 2; }
    } else {
        if (splat_scale.y < splat_scale.z) { short_axis_idx = 1; } else { short_axis_idx = 2; }
    }
    let mid_axis_idx = 3 - long_axis_idx - short_axis_idx;

    let long_axis = normalize(axes[long_axis_idx]);

    let long_scale = max(splat_scale.x, max(splat_scale.y, splat_scale.z));
    let short_scale = min(splat_scale.x, min(splat_scale.y, splat_scale.z));
    let mid_scale = splat_scale.x + splat_scale.y + splat_scale.z - short_scale - long_scale;

    // Billboard basis aligned to the long axis, facing the camera.
    let cam_forward = normalize(u.camera_pos.xyz - splat_pos);
    let cam_right = normalize(cross(cam_forward, long_axis));

    let mid_alignment = abs(dot(cam_forward, axes[mid_axis_idx]));
    let short_alignment = abs(dot(cam_forward, axes[short_axis_idx]));
    let t = clamp(short_alignment / (mid_alignment + short_alignment + 0.0001), 0.0, 1.0);
    let billboard_width = mix(short_scale, mid_scale, t);

    let corner = get_vertex_corner(vertex_id % 6u);
    let offset_x = corner.x * billboard_width;
    let offset_y = corner.y * long_scale;

    let vertex_offset = cam_right * offset_x + long_axis * offset_y;
    let world_pos = splat_pos + vertex_offset * u.splat_params.y * overall_scale;
    let clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);

    output.position = clip_pos;
    // Global opacity multiplier. A sharp falloff shrinks the visible footprint,
    // so the cloud reads more transparent -- raise this toward 1.0 for a solider
    // look, lower it for a more ethereal one.
    let opacity_scale = 0.85;
    output.color = vec4<f32>(evaluate_sh(-cam_forward, splat_id), clamp(splat_opacity, 0.0, 1.0) * opacity_scale);
    output.uv_and_scale = vec4<f32>(offset_x, offset_y, billboard_width, long_scale);
    return output;
}

@fragment
fn fs_main(input: VSOutput) -> @location(0) vec4<f32> {
    let sharpness = u.splat_params.x;
    let uv = input.uv_and_scale.xy;
    let scale = input.uv_and_scale.zw;
    let r = uv * uv / (scale * scale);
    let falloff = exp(-sharpness * (r.x + r.y));
    let output_alpha = clamp(input.color.a * falloff, 0.0, 1.0);
    // Contrast pivots around mid-gray (0.5): (x - 0.5) * contrast + 0.5.  Clamp to
    // [0,1] and return the straight (non-premultiplied) color: the ALPHA_BLENDING
    // pipeline already multiplies by src.a, so premultiplying here too would scale
    // color by alpha squared (too dark) and push RGB negative in the falloff skirt
    // (the black fringe at large scales).
    let contrasted = clamp(((input.color.rgb - 0.5) * u.splat_params.z) + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
    // Contrast pivots in gamma space (0.5 = mid-gray); convert to linear last so
    // the sRGB render target's encode-on-store gives back the correct color.
    let out_color = srgb_to_linear(contrasted);
    return vec4<f32>(out_color, output_alpha);
}

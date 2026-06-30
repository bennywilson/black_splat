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

fn evaluate_sh(n: vec3<f32>, splat: Splat) -> vec3<f32> {
    let x = n.x;
    let y = n.y;
    let z = n.z;

    // Degree 0
    var result = 0.282095 * splat.sh0.rgb;

    let degree = u.splat_params_2.x;

    // Degree 1
    if (degree >= 1.0) {
        let b1 = 0.488603 * y;
        let b2 = 0.488603 * z;
        let b3 = 0.488603 * x;
        result += b1 * vec3<f32>(splat.sh_rest[0], splat.sh_rest[1], splat.sh_rest[2]);
        result += b2 * vec3<f32>(splat.sh_rest[3], splat.sh_rest[4], splat.sh_rest[5]);
        result += b3 * vec3<f32>(splat.sh_rest[6], splat.sh_rest[7], splat.sh_rest[8]);
    }

    // Degree 2
    if (degree >= 2.0) {
        let b4 = 1.092548 * x * y;
        let b5 = 1.092548 * y * z;
        let b6 = 0.315392 * (3.0 * z * z - 1.0);
        let b7 = 1.092548 * x * z;
        let b8 = 0.546274 * (x * x - y * y);
        result += b4 * vec3<f32>(splat.sh_rest[9],  splat.sh_rest[10], splat.sh_rest[11]);
        result += b5 * vec3<f32>(splat.sh_rest[12], splat.sh_rest[13], splat.sh_rest[14]);
        result += b6 * vec3<f32>(splat.sh_rest[15], splat.sh_rest[16], splat.sh_rest[17]);
        result += b7 * vec3<f32>(splat.sh_rest[18], splat.sh_rest[19], splat.sh_rest[20]);
        result += b8 * vec3<f32>(splat.sh_rest[21], splat.sh_rest[22], splat.sh_rest[23]);
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

    let splat = g_splats[splat_id];

    let splat_pos = splat.position.xyz * overall_scale;
    // `var` (not `let`) so the axes can be indexed with a dynamically-chosen axis.
    var axes = quat_axes(splat.rotation);
    let splat_scale = splat.scale_opacity.xyz;

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
    output.color = vec4<f32>(evaluate_sh(-cam_forward, splat), clamp(splat.scale_opacity.w, 0.0, 1.0) * 0.24);
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
    let out_color = (((input.color.rgb * output_alpha) - 0.5) * u.splat_params.z) + 0.5;
    return vec4<f32>(out_color, output_alpha);
}

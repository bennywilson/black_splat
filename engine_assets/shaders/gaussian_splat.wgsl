// gaussian_splat.wgsl

struct GlobalConstants {
    view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // x: falloff sharpness, y: splat scale, z: contrast, w: num_splats
    splat_params: vec4<f32>,
    // x: max sh degree, y: overall scale, z/w: unused
    splat_params_2: vec4<f32>,
    model: mat4x4<f32>,
    // Normalized world-space frustum planes (left, right, bottom, top, near,
    // far); xyz = normal, w = offset.  Supplied by the host each frame.
    frustum_planes: array<vec4<f32>, 6>,
};

struct Splat {
    position: vec4<f32>,
    scale_opacity: vec4<f32>,
    rotation: vec4<f32>,
    sh0: vec4<f32>,
    sh_rest: array<f32, 24>,
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

// Quad corners in triangle-strip order: the strip's two triangles are
// (v0,v1,v2) and (v1,v2,v3), which together cover the quad.  Four vertices
// instead of the six a triangle list needs, so the vertex shader runs 4x per
// splat rather than 6x.  Winding alternates between the two triangles, which is
// fine here -- the splat pipeline sets cull_mode: None.
fn get_vertex_corner(corner_id: u32) -> vec2<f32> {
    var offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
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

// One instance per splat, four strip vertices each: the splat comes from the
// instance index and only the corner varies per vertex.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VSOutput {
    let overall_scale = u.splat_params_2.y;

    let sorted_index = instance_id;
    let splat_id = g_sorted_indices[sorted_index];

    var output: VSOutput;

    // Skip padded / out-of-range entries
    if (f32(splat_id) >= u.splat_params.w) {
        output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.color = vec4<f32>(0.0);
        output.uv_and_scale = vec4<f32>(0.0);
        return output;
    }

    // Cloud world transform (uniform scale only): pull the scalar scale off one
    // column and normalize the 3x3 into a pure rotation.  The rotation orients
    // the per-splat billboard, the scale grows it with the cloud, and positions
    // still use the full model matrix.
    let cloud_scale = length(u.model[0].xyz);

    let local_pos = g_splats[splat_id].position.xyz * overall_scale;
    let splat_pos = (u.model * vec4<f32>(local_pos, 1.0)).xyz;
    let splat_scale = g_splats[splat_id].scale_opacity.xyz;
    let splat_opacity = g_splats[splat_id].scale_opacity.w;
    let long_scale = max(splat_scale.x, max(splat_scale.y, splat_scale.z));

    // Frustum-reject before doing any real work.  The billboard offset below is
    // cam_right * offset_x + long_axis * offset_y on two perpendicular axes, and
    // neither term exceeds long_scale, so sqrt(2) * long_scale bounds the quad's
    // reach from its centre.  A splat whose bounding sphere falls entirely
    // outside a plane cannot produce a fragment, so collapse it to a degenerate
    // vertex (the same w = 0 trick the range check above uses) and skip the
    // quaternion decode, the axis selection and the SH evaluation -- the last of
    // which alone reads up to 96 bytes of sh_rest per vertex.
    let cull_radius = 1.41422 * long_scale * u.splat_params.y * overall_scale * cloud_scale;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = u.frustum_planes[i];
        if (dot(plane.xyz, splat_pos) + plane.w + cull_radius < 0.0) {
            output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            output.color = vec4<f32>(0.0);
            output.uv_and_scale = vec4<f32>(0.0);
            return output;
        }
    }

    let cloud_rot = mat3x3<f32>(u.model[0].xyz, u.model[1].xyz, u.model[2].xyz) * (1.0 / max(cloud_scale, 0.000001));

    // Note: Dynamic indexing into an array requires a var instead of let on naga/Vulkan
    var axes = quat_axes(g_splats[splat_id].rotation);
    axes[0] = cloud_rot * axes[0];
    axes[1] = cloud_rot * axes[1];
    axes[2] = cloud_rot * axes[2];

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

    let short_scale = min(splat_scale.x, min(splat_scale.y, splat_scale.z));
    let mid_scale = splat_scale.x + splat_scale.y + splat_scale.z - short_scale - long_scale;

    // Billboard basis aligned to the long axis, facing the camera.
    let cam_forward = normalize(u.camera_pos.xyz - splat_pos);
    let cam_right = normalize(cross(cam_forward, long_axis));

    let mid_alignment = abs(dot(cam_forward, axes[mid_axis_idx]));
    let short_alignment = abs(dot(cam_forward, axes[short_axis_idx]));
    let t = clamp(short_alignment / (mid_alignment + short_alignment + 0.0001), 0.0, 1.0);
    let billboard_width = mix(short_scale, mid_scale, t);

    let corner = get_vertex_corner(vertex_id);
    let offset_x = corner.x * billboard_width;
    let offset_y = corner.y * long_scale;

    let vertex_offset = cam_right * offset_x + long_axis * offset_y;
    let world_pos = splat_pos + vertex_offset * u.splat_params.y * overall_scale * cloud_scale;
    let clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);

    output.position = clip_pos;

    output.color = vec4<f32>(evaluate_sh(-cam_forward, splat_id), clamp(splat_opacity, 0.0, 1.0));
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
    let out_color = clamp(((input.color.rgb - 0.5) * u.splat_params.z) + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(out_color, output_alpha);
}

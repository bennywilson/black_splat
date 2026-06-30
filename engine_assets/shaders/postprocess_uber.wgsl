/**
 *  Vertex Shader
 */

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in_vertex: VertexInput) -> VertexOutput {
    var out_vertex: VertexOutput;

    out_vertex.tex_coords = in_vertex.tex_coords;
    out_vertex.clip_position = vec4<f32>(in_vertex.position.xyz, 1.0);

    return out_vertex;
}

/**
 *  Fragment Shader
 */

@group(0) @binding(0)
var t_post_process_filter: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_scene_color: texture_2d<f32>;

struct PostProcessUniform {
    time_mode_unused_unused: vec4<f32>,
};
@group(1) @binding(0)
var<uniform> postprocess_buffer: PostProcessUniform;

fn get_postprocess_mode(in_val: f32) -> i32 {
    if abs(in_val - 0.0) < 0.0001 {
        return 0;
    }
    if abs(in_val - 1.0) < 0.0001 {
        return 1;
    }
    if abs(in_val - 2.0) < 0.0001 {
        return 2;
    }
    return 3;
}

// Linear -> sRGB. Used only when the surface format is NOT sRGB (e.g. Chrome's
// WebGPU canvas), where the hardware won't do the encode for us.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lower = c * 12.92;
    let higher = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(higher, lower, c < vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var uv : vec2<f32> = in.tex_coords;
    var outColor: vec4<f32> = textureSample(t_scene_color, s_diffuse, uv);

    var postprocess_mode: i32 = get_postprocess_mode(postprocess_buffer.time_mode_unused_unused.y);

    if (postprocess_mode == 1) {
        outColor = textureSample(t_scene_color, s_diffuse, uv);
        outColor.x = dot(outColor.xyz, vec3<f32>(0.3,0.59,0.11));
        outColor.y = outColor.x;
        outColor.z = outColor.x;
    } else if (postprocess_mode == 2) {
        var uv_offset: vec2<f32> = vec2<f32>(0.0, postprocess_buffer.time_mode_unused_unused.x * -0.02f);
        var uv_scale: vec2<f32> = vec2<f32>(0.5, 0.5);
        var scanLine: f32 = textureSample(t_post_process_filter, s_diffuse, uv * uv_scale + uv_offset).x;
        outColor.r *= ((scanLine * 0.5) + 0.5);
        outColor.g *= ((scanLine * 0.5) + 0.5);
        outColor.b *= ((scanLine * 0.5) + 0.5);
    } else if (postprocess_mode == 3) {
        var uv_offset_1 = vec2<f32>(1.0, 1.0) * postprocess_buffer.time_mode_unused_unused.x * 0.03;
        var uv_offset_2 = vec2<f32>(-1.0, -.3) * postprocess_buffer.time_mode_unused_unused.x * 0.023;
        var uv_scale = vec2<f32>(1.0, 1.0);
        var uv_offset: vec2<f32> = textureSample(t_post_process_filter, s_diffuse, uv * uv_scale + uv_offset_1).gg;
        uv_offset.y = textureSample(t_post_process_filter, s_diffuse, uv * 0.5 * uv_scale + uv_offset_2).g;
        uv_offset = uv + uv_offset * 0.1;
        if (uv_offset.y > 0.9999) {
            uv_offset.y = 0.9999f;
        }
        outColor = textureSample(t_scene_color, s_diffuse, uv_offset);
    }

    // .z flags a non-sRGB surface: encode here so colors aren't displayed dark.
    if (postprocess_buffer.time_mode_unused_unused.z > 0.5) {
        outColor = vec4<f32>(linear_to_srgb(outColor.rgb), outColor.a);
    }
    return outColor;
}
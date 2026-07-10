// World-pass G-buffer shader: writes albedo, world normal and PBR
// metallic/roughness for the deferred lighting pass (see light_*.wgsl).
// model_color is the actor color multiplied by the material's color constant
// on the CPU; spec_color.xy are the material's metallic/roughness constants,
// multiplied by the material's metallic-roughness texture (glTF layout:
// G = roughness, B = metallic; the built-in white when none is assigned).

struct ModelUniform {
    world: mat4x4<f32>,
    inv_world: mat4x4<f32>,
    world_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_dir: vec4<f32>,
    target_dimensions: vec4<f32>,
    time_colorpow_: vec4<f32>,
    model_color: vec4<f32>,
    custom_data_1: vec4<f32>,
    sun_color: vec4<f32>,
    spec_color: vec4<f32>
};

@group(1) @binding(0)
var<uniform> model_uniform: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>
}

@vertex
fn vs_main(
    model: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    out.tex_coords = model.tex_coords;
    out.clip_position = model_uniform.world_view_proj * vec4<f32>(model.position.xyz, 1.0);
    // World-space normal (rotation + uniform scale; renormalized in the
    // fragment stage after interpolation).
    out.normal = (model_uniform.world * vec4<f32>(model.normal.xyz, 0.0)).xyz;

    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_color: texture_2d<f32>;
@group(0) @binding(1)
var s_color: sampler;
@group(0) @binding(2)
var t_spec: texture_2d<f32>;

struct GBufferOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) specular: vec4<f32>
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;

    let albedo = textureSample(t_color, s_color, in.tex_coords);
    out.color = vec4<f32>(albedo.rgb * model_uniform.model_color.rgb, 1.0);

    let normal = normalize(in.normal);
    out.normal = vec4<f32>(normal * 0.5 + 0.5, 1.0);

    let mr = textureSample(t_spec, s_color, in.tex_coords);
    out.specular = vec4<f32>(
        mr.b * model_uniform.spec_color.x,  // metallic
        mr.g * model_uniform.spec_color.y,  // roughness
        0.0,
        1.0
    );

    return out;
}

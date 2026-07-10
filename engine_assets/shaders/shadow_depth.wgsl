// Depth-only shadow-map shader: renders casters into a tile of the shadow
// atlas from the light's point of view.  No fragment stage -- the pipeline
// only writes depth.  One uniform slot per draw (dynamic offset).

struct ShadowDrawUniform {
    light_view_proj_world: mat4x4<f32>
};

@group(0) @binding(0)
var<uniform> draw_uniform: ShadowDrawUniform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return draw_uniform.light_view_proj_world * vec4<f32>(position, 1.0);
}

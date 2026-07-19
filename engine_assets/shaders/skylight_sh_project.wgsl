// GPU reduction of a baked skylight cubemap's mip 0 into a 9-term real-SH
// (band 0-2) RGB projection for diffuse irradiance. Runs as a single
// workgroup of 64 threads, each striding over a share of the 128*128*6
// texels, then a workgroup-shared tree reduction. Replaces a CPU readback
// that relied on a blocking map_async (native-only -- wasm can't
// synchronously block on GPU readback), so this now runs identically on
// both platforms and writes straight into a GPU buffer the lighting shader
// binds directly (see light_skylight.wgsl).
//
// Direction reconstruction and the solid-angle weight
// (4/(u^2+v^2+1)^1.5) mirror Renderer::prefilter_skylight_mips exactly:
// the same per-face inv_view_proj (Camera::from_look + 90-degree-FOV
// perspective, inverted) that wrote each texel is used to read its
// direction back out. The SH basis constants must match
// light_skylight.wgsl's eval_sh_irradiance exactly.

struct FaceMatrices {
    inv_view_proj: array<mat4x4<f32>, 6>,
};

struct ShResult {
    sh: array<vec4<f32>, 9>,
};

@group(0) @binding(0)
var<uniform> faces: FaceMatrices;
@group(0) @binding(1)
var t_src: texture_cube<f32>;
@group(0) @binding(2)
var s_src: sampler;
@group(0) @binding(3)
var<storage, read_write> out_sh: ShResult;

const FACE_SIZE: u32 = 128u;
const TEXELS_PER_FACE: u32 = FACE_SIZE * FACE_SIZE;
const TOTAL_TEXELS: u32 = TEXELS_PER_FACE * 6u;
const WORKGROUP_SIZE: u32 = 64u;
const PI: f32 = 3.141592653589793;

var<workgroup> shared_sh: array<array<vec4<f32>, 9>, WORKGROUP_SIZE>;
var<workgroup> shared_weight: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(64)
fn cs_project(@builtin(local_invocation_id) lid: vec3<u32>) {
    var sh: array<vec4<f32>, 9>;
    for (var i = 0u; i < 9u; i++) {
        sh[i] = vec4<f32>(0.0);
    }
    var weight_sum = 0.0;

    var idx = lid.x;
    loop {
        if (idx >= TOTAL_TEXELS) {
            break;
        }
        let face = idx / TEXELS_PER_FACE;
        let rem = idx % TEXELS_PER_FACE;
        let row = rem / FACE_SIZE;
        let col = rem % FACE_SIZE;

        let u = (f32(col) + 0.5) / f32(FACE_SIZE);
        let v = (f32(row) + 0.5) / f32(FACE_SIZE);
        let ndc = vec4<f32>(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
        let world = faces.inv_view_proj[face] * ndc;
        let n = normalize(world.xyz / world.w);

        let fu = u * 2.0 - 1.0;
        let fv = 1.0 - v * 2.0;
        let texel_area = (2.0 / f32(FACE_SIZE)) * (2.0 / f32(FACE_SIZE));
        let w = texel_area * 4.0 / pow(fu * fu + fv * fv + 1.0, 1.5);

        let radiance = textureSampleLevel(t_src, s_src, n, 0.0).rgb;

        // Real-SH basis, band 0-2 (must match light_skylight.wgsl's
        // eval_sh_irradiance constants).
        let basis0 = 0.282095;
        let basis1 = 0.488603 * n.y;
        let basis2 = 0.488603 * n.z;
        let basis3 = 0.488603 * n.x;
        let basis4 = 1.092548 * n.x * n.y;
        let basis5 = 1.092548 * n.y * n.z;
        let basis6 = 0.315392 * (3.0 * n.z * n.z - 1.0);
        let basis7 = 1.092548 * n.x * n.z;
        let basis8 = 0.546274 * (n.x * n.x - n.y * n.y);

        sh[0] += vec4<f32>(radiance * (basis0 * w), 0.0);
        sh[1] += vec4<f32>(radiance * (basis1 * w), 0.0);
        sh[2] += vec4<f32>(radiance * (basis2 * w), 0.0);
        sh[3] += vec4<f32>(radiance * (basis3 * w), 0.0);
        sh[4] += vec4<f32>(radiance * (basis4 * w), 0.0);
        sh[5] += vec4<f32>(radiance * (basis5 * w), 0.0);
        sh[6] += vec4<f32>(radiance * (basis6 * w), 0.0);
        sh[7] += vec4<f32>(radiance * (basis7 * w), 0.0);
        sh[8] += vec4<f32>(radiance * (basis8 * w), 0.0);
        weight_sum += w;

        idx += WORKGROUP_SIZE;
    }

    shared_sh[lid.x] = sh;
    shared_weight[lid.x] = weight_sum;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            for (var i = 0u; i < 9u; i++) {
                shared_sh[lid.x][i] = shared_sh[lid.x][i] + shared_sh[lid.x + stride][i];
            }
            shared_weight[lid.x] = shared_weight[lid.x] + shared_weight[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        let weight_sum_total = shared_weight[0];
        var norm = 0.0;
        if (weight_sum_total > 0.0) {
            norm = 4.0 * PI / weight_sum_total;
        }
        for (var i = 0u; i < 9u; i++) {
            out_sh.sh[i] = vec4<f32>(shared_sh[0][i].rgb * norm, 0.0);
        }
    }
}

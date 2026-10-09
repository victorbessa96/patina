//! WGSL shader sources for the viewport mesh pass.

/// Normal-shaded mesh pass with a directional light and a hemispherical
/// environment light (procedural sky/ground irradiance — the Wave-1
/// "bundled env map"; full HDR-texture IBL lands with Wave-2 image import).
/// Driven entirely by vertex normals (no textures yet — that lands with
/// the paint engine). Matches the `Vertex` and `CameraUniform` layouts in
/// `renderer.rs`.
pub const MESH_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    // xyz = normalized direction the light travels (surface -> fragment),
    // w unused (padding to satisfy uniform alignment).
    light_dir: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    return out;
}

const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.96, 0.90);
const SKY_COLOR: vec3<f32> = vec3<f32>(0.45, 0.55, 0.78);
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.30, 0.27, 0.24);
const BASE_COLOR: vec3<f32> = vec3<f32>(0.72, 0.72, 0.75);

/// Hemispherical environment irradiance: the procedural stand-in for an
/// environment map — sky above the horizon, ground below, blended by the
/// normal's vertical component.
fn environment_irradiance(n: vec3<f32>) -> vec3<f32> {
    let t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(GROUND_COLOR, SKY_COLOR, t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let to_light = normalize(-camera.light_dir.xyz);
    let diffuse = max(dot(n, to_light), 0.0);
    let env = environment_irradiance(n);
    // Environment acts as the ambient term; the sun adds directional
    // energy on top, weighted so lit surfaces approach full brightness.
    let color = BASE_COLOR * (env + diffuse * SUN_COLOR * 0.55);
    return vec4<f32>(color, 1.0);
}
"#;

/// Dab-splat compute pass: one workgroup per dab, rasterizing a filled,
/// radially-falloff'd circle into a storage texture with premultiplied
/// alpha-over compositing.
///
/// `Dab`'s field order/padding here must match `paint::Dab` byte-for-byte
/// (see that module's doc comment) — WGSL's `vec4<f32>` forces 16-byte
/// alignment, which Rust's `[f32; 4]` (4-byte aligned) does not provide for
/// free.
///
/// Workgroup-local contract (see `paint::PaintCompositor::splat_dabs`):
/// every invocation in a workgroup paints only pixels inside *its own*
/// dab's clamped bounding box, strided by `local_invocation_index` so no
/// two invocations in the same workgroup touch the same texel. There is no
/// such guarantee *across* workgroups in one dispatch — two dabs in the
/// same batch whose bounding boxes overlap would race on the shared
/// `read_write` storage texture (no atomics are used). The caller must
/// therefore keep spatially-overlapping dabs in separate `splat_dabs`
/// calls; wgpu's automatic hazard tracking orders successive compute
/// passes on the same texture correctly.
pub const PAINT_COMPUTE_SHADER: &str = r#"
struct Dab {
    pos: vec2<f32>,
    radius: f32,
    alpha: f32,
    color: vec4<f32>,
    hardness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> dabs: array<Dab>;
@group(0) @binding(1) var paint_tex: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;

const WORKGROUP_SIZE: u32 = 64u;

/// Radial falloff: 1.0 inside `hardness * radius`, smoothly down to 0.0 at
/// `radius`. `t` is the normalized distance from center (0 at center, 1 at
/// the edge). Written without WGSL's built-in `smoothstep` so the
/// `hardness == 1.0` (hard disc) case can't divide by zero.
fn radial_falloff(t: f32, hardness: f32) -> f32 {
    let denom = max(1.0 - hardness, 1e-4);
    let edge_t = clamp((t - hardness) / denom, 0.0, 1.0);
    return 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let dab = dabs[workgroup_id.x];
    let safe_radius = max(dab.radius, 1e-5);

    let min_xy = vec2<i32>(floor(dab.pos - safe_radius));
    let max_xy = vec2<i32>(ceil(dab.pos + safe_radius));
    let clamped_min = max(min_xy, vec2<i32>(0, 0));
    let clamped_max = min(max_xy, vec2<i32>(dims) - vec2<i32>(1, 1));
    if (clamped_max.x < clamped_min.x || clamped_max.y < clamped_min.y) {
        return;
    }

    let box_w = u32(clamped_max.x - clamped_min.x + 1);
    let box_h = u32(clamped_max.y - clamped_min.y + 1);
    let box_count = box_w * box_h;

    for (var i: u32 = local_index; i < box_count; i = i + WORKGROUP_SIZE) {
        let local_x = i32(i % box_w);
        let local_y = i32(i / box_w);
        let texel = clamped_min + vec2<i32>(local_x, local_y);

        let center = vec2<f32>(texel) + vec2<f32>(0.5, 0.5);
        let dist = distance(center, dab.pos);
        if (dist > safe_radius) {
            continue;
        }

        let coverage = dab.alpha * radial_falloff(dist / safe_radius, dab.hardness);
        let src = dab.color * coverage;
        let dst = textureLoad(paint_tex, texel);
        let out = src + dst * (1.0 - src.a);
        textureStore(paint_tex, texel, out);
    }
}
"#;

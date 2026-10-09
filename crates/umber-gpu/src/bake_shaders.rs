//! WGSL shader sources for the bake engine's compute-raycast pass.
//!
//! Wave 3 scope (docs/specs/requirements.md §3: "Compute-shader raycast/bvh
//! bake path (no RT hardware dependency)"). This first slice is the
//! raycast *core*: brute-force Möller–Trumbore against every triangle, no
//! acceleration structure — see [`AO_BAKE_SHADER`]'s own doc comment for
//! the perf budget and what a BVH slice would change.

/// Ambient-occlusion compute-raycast pass: one workgroup per output texel,
/// raycasting a hemisphere of rays from a parameter plane against a
/// brute-force triangle list.
///
/// # Why a parameter plane, not a UV-mapped position map
///
/// The eventual bake target is "one texel per UV location on the mesh
/// being baked," which needs a position map (UV → world position,
/// produced by rasterizing the mesh's own UV layout) as this pass's input.
/// Building that rasterizer is deferred to the next slice — see
/// `umber-bake/LANDING_NOTES_AO.md`. This slice instead raycasts from a
/// user-described plane (`PlaneUniform`: world origin + two axes + an
/// extent), so every texel's world position is a closed-form function of
/// its (u, v) grid coordinate instead of a mesh lookup. That proves the
/// raycast core — hemisphere sampling, ray-triangle intersection,
/// occlusion accumulation — independently of the position-map work, with
/// the two slices composing cleanly later (the position map will replace
/// only the "texel → world position" step below; everything from the
/// hemisphere sample onward is unchanged).
///
/// # Layout contracts
///
/// - `Tri` must match `umber_bake::ao`'s private `GpuTriangle` byte-for-byte
///   (four `vec4<f32>`s: three vertex positions, then the face normal, all
///   `w` components unused padding — the WGSL `vec4` 16-byte alignment is
///   free here since every field is already 16 bytes, unlike `Dab`'s mixed
///   layout in `shaders::PAINT_COMPUTE_SHADER`).
/// - `PlaneUniform`/`AoParams` must match `umber_bake::ao`'s private
///   `PlaneDesc`/`AoUniform` byte-for-byte; see those types' doc comments
///   for the explicit padding fields that make a `[f32; 3]` (Rust, 4-byte
///   aligned) line up with a WGSL `vec3<f32>` (16-byte aligned).
///
/// # Perf budget (this slice)
///
/// No BVH: every ray tests every triangle in the storage buffer, and every
/// texel's workgroup re-walks the same list. Cost is
/// `texels * rays * triangle_count`. At `rays = 16` this slice targets
/// meshes under ~5k triangles on a `512²` target (~4.1M ray-triangle tests
/// per full bake) — acceptable for a correctness-proving first slice, not
/// for production asset sizes. A BVH (or at minimum a uniform grid) over
/// the triangle list is the next slice's performance work; the shader's
/// outer-loop shape (`for t in 0..tri_count`) is exactly where that
/// acceleration structure would slot in without touching the hemisphere
/// sampling or occlusion-accumulation code around it.
///
/// # Sampling + reduction
///
/// Hemisphere directions are deterministic and stratified (uniform in
/// `cos(theta)`, golden-angle in `phi`) rather than randomized — see
/// `hemisphere_sample`'s doc comment for why a per-texel RNG was skipped.
/// Each workgroup (`@workgroup_size(64)`, matching
/// `shaders::PAINT_COMPUTE_SHADER`'s convention) owns exactly one texel;
/// invocations split the ray budget by `local_invocation_index` and
/// accumulate hits into a `var<workgroup> atomic<u32>` counter guarded by
/// `workgroupBarrier()`s, so the result doesn't depend on `rays` dividing
/// evenly into the workgroup size (a `rays < 64` config, e.g. the default
/// 16, simply leaves some invocations idle, not racing).
///
/// # Storage-texture access: `write`, not `read_write`
///
/// Unlike `shaders::PAINT_COMPUTE_SHADER` (which blends into existing
/// texel contents and therefore needs `read_write`, gated behind
/// `wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` on
/// `Rgba8Unorm` — see `paint::PaintError::MissingDeviceFeature`), this
/// pass writes each texel exactly once with no dependency on its prior
/// value. `write`-only storage-texture access on `Rgba8Unorm` is a core
/// WebGPU capability (unlike `read`/`read_write`, which are the
/// non-portable adapter extension for any format besides the `r32`
/// family), so this pass needs no extra device feature — confirmed by the
/// GPU test in `umber-bake::ao` requesting a plain default device.
pub const AO_BAKE_SHADER: &str = r#"
struct Tri {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    normal: vec4<f32>,
};

struct PlaneUniform {
    origin: vec3<f32>,
    _pad0: f32,
    u_axis: vec3<f32>,
    _pad1: f32,
    v_axis: vec3<f32>,
    _pad2: f32,
    extent: vec2<f32>,
    _pad3: vec2<f32>,
};

struct AoParams {
    plane: PlaneUniform,
    rays: u32,
    max_distance: f32,
    bias: f32,
    tri_count: u32,
};

@group(0) @binding(0) var<storage, read> triangles: array<Tri>;
@group(0) @binding(1) var ao_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: AoParams;
@group(0) @binding(3) var<uniform> dims: vec2<u32>;

var<workgroup> hit_count: atomic<u32>;

const WORKGROUP_SIZE: u32 = 64u;
const PI: f32 = 3.14159265358979;
const GOLDEN_CONJ: f32 = 0.6180339887498949;
const EPS: f32 = 1e-6;

/// Deterministic hemisphere direction for ray `i` of `n`, in a local frame
/// where +Z is the hemisphere pole: `cos(theta)` is stratified evenly
/// across `[0, 1)` (uniform over solid angle) and `phi` walks the golden
/// angle, giving low-discrepancy coverage without a per-texel random seed.
/// A fixed sample set (rather than per-texel jitter) keeps this slice's
/// output deterministic texel-to-texel, which is what the GPU test's
/// hand-derived occlusion bounds rely on.
fn hemisphere_sample(i: u32, n: u32) -> vec3<f32> {
    let nf = max(f32(n), 1.0);
    let cos_theta = 1.0 - (f32(i) + 0.5) / nf;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * fract(f32(i) * GOLDEN_CONJ);
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

/// Möller–Trumbore ray-triangle intersection. Double-sided (no backface
/// culling — requirements §3's "ignore backfaces" bake setting is deferred
/// to a later slice). Returns true iff the hit parameter `t` lands in the
/// open interval `(EPS, max_t)`.
fn hits_triangle(
    orig: vec3<f32>,
    dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    max_t: f32,
) -> bool {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(dir, e2);
    let a = dot(e1, h);
    if (abs(a) < EPS) {
        return false;
    }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    let t = f * dot(e2, q);
    return t > EPS && t < max_t;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    if (local_index == 0u) {
        atomicStore(&hit_count, 0u);
    }
    workgroupBarrier();

    let dims_f = vec2<f32>(dims);
    let texel_center = (vec2<f32>(workgroup_id.xy) + vec2<f32>(0.5, 0.5)) / dims_f;
    let local_uv = (texel_center - vec2<f32>(0.5, 0.5)) * params.plane.extent;

    let u_axis = params.plane.u_axis;
    let v_axis = params.plane.v_axis;
    let normal = normalize(cross(u_axis, v_axis));
    let surface_pos = params.plane.origin + local_uv.x * u_axis + local_uv.y * v_axis;
    let ray_origin = surface_pos + params.bias * normal;

    let n = params.rays;
    for (var i: u32 = local_index; i < n; i = i + WORKGROUP_SIZE) {
        let local_dir = hemisphere_sample(i, n);
        let dir = normalize(local_dir.x * u_axis + local_dir.y * v_axis + local_dir.z * normal);

        var hit = false;
        for (var t: u32 = 0u; t < params.tri_count; t = t + 1u) {
            let tri = triangles[t];
            if (hits_triangle(ray_origin, dir, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz, params.max_distance)) {
                hit = true;
                break;
            }
        }
        if (hit) {
            atomicAdd(&hit_count, 1u);
        }
    }
    workgroupBarrier();

    if (local_index == 0u) {
        let blocked = f32(atomicLoad(&hit_count)) / f32(max(n, 1u));
        let ao = clamp(1.0 - blocked, 0.0, 1.0);
        textureStore(ao_tex, vec2<i32>(workgroup_id.xy), vec4<f32>(ao, ao, ao, 1.0));
    }
}
"#;

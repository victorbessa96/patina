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
    // Bent-normal toggle (wave-4 item 5 slice 2): 0 = AO only, 1 = the
    // mesh-fed entry also accumulates unoccluded ray directions into
    // `bent_tex` (see `cs_main_from_position`'s doc comment for the frame
    // and encoding contract). Additive at the END so the pre-existing
    // prefix layout stays stable; the three trailing pads round the struct
    // to 96 bytes (WGSL uniform structs are 16-byte aligned), matching
    // `umber_bake::ao::AoUniform` — see that type's doc comment. `cs_main`
    // (the parameter-plane path) ignores these words entirely.
    bent_normals: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
};

@group(0) @binding(0) var<storage, read> triangles: array<Tri>;
@group(0) @binding(1) var ao_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: AoParams;
@group(0) @binding(3) var<uniform> dims: vec2<u32>;
@group(0) @binding(6) var bent_tex: texture_storage_2d<rgba8unorm, write>;

var<workgroup> hit_count: atomic<u32>;
// Bent-normal second accumulator (one partial sum per invocation —
// `atomic<u32>` works for AO's hit *count* but a direction *sum* is a
// `vec3<f32>` with no atomic add in core WGSL, so each invocation keeps its
// own partial sum here and invocation 0 reduces them serially). 64 entries
// of 16-byte-strided `vec3` is 1 KiB, well under the measured 16 KiB
// workgroup-storage budget (see POSITION_BAKE_SHADER's doc comment).
var<workgroup> bent_parts: array<vec3<f32>, WORKGROUP_SIZE>;

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

// --- Mesh-fed variant: reads the position-map pass's output instead of a
// parameter plane. See this file's module doc comment ("Mesh-fed AO entry
// point") for why this lives as a second entry point in the same module
// rather than a separate shader constant: it reuses `triangles`, `ao_tex`,
// `params`, `hit_count`, `hemisphere_sample`, and `hits_triangle` verbatim.

@group(0) @binding(4) var position_tex: texture_2d<f32>;
@group(0) @binding(5) var normal_tex: texture_2d<f32>;

struct Basis {
    b1: vec3<f32>,
    b2: vec3<f32>,
};

/// Branchless tangent-frame construction from a unit normal (Duff et al.,
/// "Building an Orthonormal Basis, Revisited", JCGT 2017). `cs_main`'s
/// plane hands the shader `u_axis`/`v_axis` directly; `cs_main_from_position`
/// only has a per-texel normal (the position pass's second output) and
/// needs *some* consistent tangent frame to steer `hemisphere_sample`'s
/// local directions into world space. Any orthonormal frame works — AO is
/// isotropic in `phi`, so the choice doesn't bias the result — this one was
/// picked for being branchless and numerically stable at the south-pole
/// case other common formulas (e.g. a naive `cross(n, vec3(0,0,1))`) fail.
fn orthonormal_basis(n: vec3<f32>) -> Basis {
    let sign_z = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign_z + n.z);
    let b = n.x * n.y * a;
    return Basis(
        vec3<f32>(1.0 + sign_z * n.x * n.x * a, sign_z * b, -sign_z * n.x),
        vec3<f32>(b, sign_z + n.y * n.y * a, -n.y),
    );
}

/// Mesh-fed ambient occlusion (+ optional bent normals): identical
/// hemisphere-raycast core to `cs_main`, but the per-texel ray origin and
/// tangent frame come from `position_tex`/`normal_tex` (the two outputs of
/// `POSITION_BAKE_SHADER`'s `cs_main`, see `bake_shaders::POSITION_BAKE_SHADER`)
/// rather than a closed-form plane. One workgroup per texel, matching `cs_main`'s
/// dispatch convention (`dispatch_workgroups(width, height, 1)`), so
/// `workgroup_id.xy` is directly the texel coordinate into both input
/// textures and `ao_tex`.
///
/// `position_tex`'s alpha channel is the position pass's coverage flag (see
/// that shader's doc comment): `0.0` means no UV triangle covered this
/// texel, so there is no surface to raycast from. Every invocation in a
/// workgroup loads the same texel's `covered` flag (it's a per-workgroup,
/// not per-invocation, value here, unlike `POSITION_BAKE_SHADER`'s 8x8-tile
/// layout where each invocation owns a *different* texel) — so branching on
/// it is workgroup-uniform and safe around the `workgroupBarrier()`s below.
/// Uncovered texels skip the raycast entirely and write `(0, 0, 0, 0)`,
/// distinguishable on readback from a fully-occluded-but-covered texel
/// (`(0, 0, 0, 1)`) by alpha alone — which is why `ao::bake_ao_mesh` returns
/// full RGBA8 instead of `ao::run`'s R-channel-only bytes.
///
/// # Bent normals (`params.bent_normals != 0`, wave-4 item 5 slice 2)
///
/// When enabled, the visibility loop doubles as a bent-normal accumulator
/// (Substance's shared-sample approach: the bent map reuses AO's rays, no
/// second dispatch): every ray that produces NO hit adds its world-space
/// direction to the invoking invocation's `bent_parts` slot, and invocation
/// 0 reduces the slots into the texel's bent vector.
///
/// Frame: the accumulated directions are the SAME world-space vectors the
/// hit test consumes (`local_dir` steered by the Duff-et-al.
/// `orthonormal_basis` frame — `dir = lx*b1 + ly*b2 + lz*normal`), so the
/// sum lives in world space, not tangent space. Output encoding follows the
/// other normal bakers' `* 0.5 + 0.5` convention (see
/// `umber_bake::normal_map`'s `tangent_normal * 0.5 + 0.5`, here applied to
/// the world-space bent vector): `rgb = bent * 0.5 + 0.5`,
/// `a = |sum| / ray_count` as confidence. A zero sum (fully occluded)
/// writes the geometric normal (`normal_tex`) encoded the same way with
/// `a = 0`; an uncovered texel writes `(0, 0, 0, 0)`, matching AO's
/// background convention. When the toggle is 0 the bent stores are skipped
/// and `ao_tex` output is bit-for-bit what the pre-bent shader wrote (the
/// AO path is untouched — integer hit counting, no shared state with the
/// accumulator).
@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main_from_position(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let coord = vec2<i32>(workgroup_id.xy);
    let sample = textureLoad(position_tex, coord, 0);
    let covered = sample.w > 0.5;

    if (local_index == 0u) {
        atomicStore(&hit_count, 0u);
    }
    workgroupBarrier();

    if (covered) {
        let surface_pos = sample.xyz;
        let normal = normalize(textureLoad(normal_tex, coord, 0).xyz);
        let basis = orthonormal_basis(normal);
        let ray_origin = surface_pos + params.bias * normal;

        let n = params.rays;
        // Per-invocation bent partial sum (world-space unoccluded
        // directions; reduced by invocation 0 below). Kept at zero when the
        // toggle is off so the gated add never executes.
        var bent_part = vec3<f32>(0.0, 0.0, 0.0);
        for (var i: u32 = local_index; i < n; i = i + WORKGROUP_SIZE) {
            let local_dir = hemisphere_sample(i, n);
            let dir = normalize(
                local_dir.x * basis.b1 + local_dir.y * basis.b2 + local_dir.z * normal
            );

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
            } else if (params.bent_normals != 0u) {
                bent_part = bent_part + dir;
            }
        }
        bent_parts[local_index] = bent_part;
    } else {
        bent_parts[local_index] = vec3<f32>(0.0, 0.0, 0.0);
    }
    workgroupBarrier();

    if (local_index == 0u) {
        if (covered) {
            let blocked = f32(atomicLoad(&hit_count)) / f32(max(params.rays, 1u));
            let ao = clamp(1.0 - blocked, 0.0, 1.0);
            textureStore(ao_tex, coord, vec4<f32>(ao, ao, ao, 1.0));
            if (params.bent_normals != 0u) {
                // Serial reduction over the invocation slots (fixed order,
                // so the bent sum is deterministic texel-to-texel like the
                // AO hit count; for `rays <= WORKGROUP_SIZE` each slot holds
                // exactly one ray's direction, matching a serial 0..n sum).
                var bent_sum = vec3<f32>(0.0, 0.0, 0.0);
                for (var k: u32 = 0u; k < WORKGROUP_SIZE; k = k + 1u) {
                    bent_sum = bent_sum + bent_parts[k];
                }
                let sum_len = length(bent_sum);
                if (sum_len > 0.0) {
                    let bent = bent_sum / sum_len;
                    let confidence = clamp(sum_len / f32(max(params.rays, 1u)), 0.0, 1.0);
                    textureStore(bent_tex, coord, vec4<f32>(bent * 0.5 + 0.5, confidence));
                } else {
                    // Fully occluded: no unoccluded direction exists, so
                    // fall back to the geometric normal with zero
                    // confidence (see this entry point's doc comment).
                    // (`normal` from the raycast block above is out of
                    // scope here — reload the same texel.)
                    let geom = normalize(textureLoad(normal_tex, coord, 0).xyz);
                    textureStore(bent_tex, coord, vec4<f32>(geom * 0.5 + 0.5, 0.0));
                }
            }
        } else {
            textureStore(ao_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            if (params.bent_normals != 0u) {
                textureStore(bent_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            }
        }
    }
}
"#;

/// UV→world-position rasterization: one compute pass that turns a mesh's
/// own UV layout into a world-space position map, the input the AO pass
/// (and every other mesh-map baker) ultimately needs instead of a
/// stand-in parameter plane. See `umber_bake::position`'s module doc for
/// the driver side (`PositionMapParams`, `bake_position_map`, the
/// `MAX_TRIS_PER_BAKE` mesh-size ceiling).
///
/// # Rasterization model
///
/// One workgroup per `8x8` texel tile (`@workgroup_size(8, 8, 1)`,
/// `dispatch_workgroups(ceil(width/8), ceil(height/8), 1)`); each
/// invocation owns exactly one texel (`global_invocation_id.xy`). For every
/// triangle, each invocation independently tests whether its own texel's
/// UV falls inside that triangle's UV footprint (a 2D analog of
/// `AO_BAKE_SHADER::hits_triangle`'s Möller–Trumbore inversion: the same
/// edge-vector/cross-product/`1/det` structure, collapsed from a 3D
/// ray-plane intersection to a 2D point-in-triangle barycentric solve —
/// see `cross2`/`cs_main`'s inner loop). A texel's UV is its texel-center
/// position remapped through `(1 - v)` on the v axis to match
/// `umber_app::paint_state`'s `texel = [u * texels_per_uv, (1 - v) *
/// texels_per_uv]` convention — without this flip, a mesh baked here and
/// painted there would disagree about which texture row is "up".
///
/// # No depth test — last-writer-wins on overlapping UV islands
///
/// This pass has no notion of "which UV island is on top": if two
/// triangles' UV footprints overlap the same texel (a seam/charting defect,
/// or deliberately overlapping islands), whichever triangle has the higher
/// index in the mesh's triangle list wins — not the nearest in any
/// meaningful sense, just last-tested. Fine for well-charted meshes (no UV
/// overlap by construction); produces silently-wrong results for
/// overlapping ones. A real fix needs either per-island depth/priority
/// metadata or non-overlapping UV charts guaranteed upstream — out of scope
/// for this slice (Wave-3+ seam/overlap work per the task brief).
///
/// # Shared-memory triangle batching — why not one array of `MAX_TRIS_PER_BAKE`
///
/// The task sketch asked for "the full triangle list, loaded once into
/// workgroup-shared memory, sized by `MAX_TRIS_PER_BAKE = 4096`." Checked
/// against this repo's actual GPU (see `umber-bake`'s landing notes for the
/// measured numbers): a plain `wgpu::DeviceDescriptor::default()` device —
/// the same one every other GPU test in this crate requests, and the only
/// kind of device umber-gpu's architecture rule lets a caller hand in
/// (`eframe`'s `wgpu_render_state`, not something this crate can raise
/// limits on after the fact) — reports `max_compute_workgroup_storage_size
/// = 16384` bytes. Even a minimal 32-byte-per-triangle UV-only record
/// (`ChunkEntry` below) at `4096` entries is `131072` bytes, 8x over
/// budget; a full per-triangle record with positions would be worse. So
/// this shader stages the triangle list through shared memory in
/// `TRI_CHUNK`-sized batches instead of one static array: `TRI_CHUNK = 256`
/// entries of `ChunkEntry` (`32` bytes each) is `8192` bytes, half the
/// measured budget, with headroom to spare. `MAX_TRIS_PER_BAKE` survives as
/// the Rust-side mesh-size ceiling (`position::validate` rejects anything
/// larger before it reaches the GPU at all) — it just isn't a shared-memory
/// array size anymore. The chunking loop (load a batch cooperatively →
/// barrier → every invocation tests its own texel against that batch →
/// barrier → next batch) still delivers the brief's actual goal (each
/// triangle's UV data is read from the global storage buffer once per
/// *workgroup*, not once per *invocation* — a 64x reduction in global
/// memory traffic for the containment test), just chunked to fit real
/// hardware instead of assuming an unbounded shared-memory budget.
///
/// World positions are deliberately kept *out* of the shared-memory batch
/// (`ChunkEntry` carries UVs only): a texel only needs a triangle's full
/// vertex positions once, for the single triangle that ends up winning it
/// (last-writer-wins, see above) — re-fetching that from the read-only
/// global `tris` buffer at the very end (one extra read per *covered*
/// texel, not per triangle-candidate) is cheaper than paying shared-memory
/// budget for data most candidates never need.
pub const POSITION_BAKE_SHADER: &str = r#"
struct PosTri {
    v0: vec4<f32>,    // xyz = world position of vertex 0, w = uv0.x
    v1: vec4<f32>,    // xyz = world position of vertex 1, w = uv1.x
    v2: vec4<f32>,    // xyz = world position of vertex 2, w = uv2.x
    uv_y: vec4<f32>,  // x = uv0.y, y = uv1.y, z = uv2.y, w unused
    normal: vec4<f32>, // xyz = face normal (consumed by cs_main_from_position), w unused
};

struct ChunkEntry {
    uv01: vec4<f32>, // x=uv0.x, y=uv0.y, z=uv1.x, w=uv1.y
    uv2: vec4<f32>,   // x=uv2.x, y=uv2.y, z/w unused
};

struct PositionParams {
    dims: vec2<u32>,
    tri_count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> tris: array<PosTri>;
@group(0) @binding(1) var pos_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var normal_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: PositionParams;

const TILE: u32 = 8u;
const TILE_AREA: u32 = 64u;
const TRI_CHUNK: u32 = 256u;
const DET_EPS: f32 = 1e-10;
const BARY_EPS: f32 = 1e-6;

var<workgroup> chunk: array<ChunkEntry, TRI_CHUNK>;

/// 2D cross product (the scalar "perp-dot"): `a.x*b.y - a.y*b.x`.
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

@compute @workgroup_size(TILE, TILE, 1)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let in_bounds = gid.x < params.dims.x && gid.y < params.dims.y;
    // Texel-center UV, v-flipped to match umber_app::paint_state's
    // `(1 - v) * texels_per_uv` convention (see this shader's doc comment).
    // Computed unconditionally (not gated on `in_bounds`) because every
    // invocation in the tile must stay in lock-step through the
    // `workgroupBarrier()`s below regardless of whether its own texel is
    // inside the target — barriers require uniform control flow across the
    // whole workgroup, so only the final `textureStore` is guarded.
    let dims_f = vec2<f32>(params.dims);
    let texel_uv = vec2<f32>(
        (f32(gid.x) + 0.5) / max(dims_f.x, 1.0),
        1.0 - (f32(gid.y) + 0.5) / max(dims_f.y, 1.0),
    );

    var best_tri: i32 = -1;
    var best_w: f32 = 0.0;
    var best_u: f32 = 0.0;
    var best_v: f32 = 0.0;

    for (var base: u32 = 0u; base < params.tri_count; base = base + TRI_CHUNK) {
        let count = min(params.tri_count - base, TRI_CHUNK);

        // Cooperative load: the TILE_AREA invocations in this workgroup
        // split up to `count` triangles between them, so each triangle's
        // UV data is read from `tris` once per workgroup, not once per
        // invocation (see this shader's "shared-memory triangle batching"
        // doc comment).
        for (var i: u32 = local_index; i < count; i = i + TILE_AREA) {
            let t = tris[base + i];
            chunk[i] = ChunkEntry(
                vec4<f32>(t.v0.w, t.uv_y.x, t.v1.w, t.uv_y.y),
                vec4<f32>(t.v2.w, t.uv_y.z, 0.0, 0.0),
            );
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < count; k = k + 1u) {
            let e = chunk[k];
            let uv0 = e.uv01.xy;
            let uv1 = e.uv01.zw;
            let uv2 = e.uv2.xy;
            let e1 = uv1 - uv0;
            let e2 = uv2 - uv0;
            let denom = cross2(e1, e2);
            // Degenerate (zero-UV-area) triangle: never claims a texel.
            // `ao::bake_ao_mesh`'s GPU test relies on this to keep an
            // occluder triangle (added to the mesh purely for AO raycasting,
            // with all-identical UVs) out of the position map entirely.
            if (abs(denom) >= DET_EPS) {
                let inv_denom = 1.0 / denom;
                let s = texel_uv - uv0;
                let bu = cross2(s, e2) * inv_denom;
                let bv = cross2(e1, s) * inv_denom;
                let bw = 1.0 - bu - bv;
                // Inclusive bounds (not strict >= 0): texels exactly on a
                // shared edge between two adjoining triangles (e.g. the
                // unit quad's diagonal) must be claimed by at least one of
                // them, not fall through a seam gap.
                if (bu >= -BARY_EPS && bv >= -BARY_EPS && bw >= -BARY_EPS) {
                    best_tri = i32(base + k);
                    best_w = bw;
                    best_u = bu;
                    best_v = bv;
                }
            }
        }
        workgroupBarrier();
    }

    if (in_bounds) {
        let coord = vec2<i32>(i32(gid.x), i32(gid.y));
        if (best_tri >= 0) {
            let tri = tris[u32(best_tri)];
            let pos = tri.v0.xyz * best_w + tri.v1.xyz * best_u + tri.v2.xyz * best_v;
            textureStore(pos_tex, coord, vec4<f32>(pos, 1.0));
            textureStore(normal_tex, coord, vec4<f32>(normalize(tri.normal.xyz), 1.0));
        } else {
            textureStore(pos_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            textureStore(normal_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        }
    }
}
"#;

/// Screen-space curvature estimation from a position/normal map: one
/// workgroup per output texel, reading the 4-neighborhood's normals and
/// positions to estimate signed directional curvature.
///
/// # Estimator
///
/// Per covered texel, over the in-bounds, covered 4-neighbors (`+x`, `-x`,
/// `+y`, `-y`):
///
/// ```text
/// k_i = dot(n_i - n_0, normalize(p_i - p_0)) / max(length(p_i - p_0), 1e-6)
/// curv = clamp(-strength * mean(k_i), -1, 1)
/// ```
///
/// This is the discrete directional-curvature estimator used by screen-space
/// bakers (e.g. Blender's pointiness-adjacent cavity approximations and
/// various "curvature from normal buffer" post passes): where the surface
/// bends, neighboring normals differ along the direction of travel, and
/// dividing by the travel distance turns that difference into a
/// curvature-scale quantity. Averaging the four axis directions makes it
/// rotation-tolerant without a full multi-ring fit (see
/// `umber-bake/LANDING_NOTES_CURVATURE.md` for sources and limits).
///
/// # Sign convention (MeshLab: convex = negative = darker)
///
/// The raw `dot(dn, dir) / len` term is *positive* on an outward bulge
/// (sphere check: `n(p) = p/R`, so `n_i - n_0 = (p_i - p_0)/R` points along
/// the travel direction and the dot product is `|dp|/R > 0`). The leading
/// minus flips it so convex (outward-bulge) regions read *negative* and
/// bake *darker*, concave crevices positive/brighter — MeshLab's
/// convention, where mean curvature is negative on convex parts. Documented
/// here (not just in the landing notes) because the minus is otherwise an
/// inviting "simplification" for a future reader to delete.
///
/// # Layout contracts
///
/// - `CurvatureParams` must match `umber_bake::curvature`'s private
///   `CurvatureUniform` byte-for-byte (`width`, `height`, `strength`, one
///   `f32` pad — 16 bytes, already a multiple of WGSL's 16-byte
///   uniform-struct alignment, so no further padding is needed).
/// - `position_tex`/`normal_tex` are the two `Rgba32Float` outputs of
///   `POSITION_BAKE_SHADER`'s `cs_main`, bound read-only and sampled with
///   `textureLoad` at integer texel coordinates (no sampler, no filtering
///   — filtering would smear normals across UV seams; see the landing
///   notes). `position_tex`'s alpha is the position pass's coverage flag:
///   `0.0` means no UV triangle covered that texel.
/// - `out_tex` is a write-only `Rgba8Unorm` storage texture (core WebGPU,
///   no device feature — the same reason `AO_BAKE_SHADER` uses `write`,
///   not `read_write`): `rgb` is `(curv + 1) / 2` grayscale, `a` is
///   coverage (`1.0` covered, `0.0` uncovered).
///
/// # Dispatch + edge behavior
///
/// One workgroup (`@workgroup_size(1)`) per texel, dispatched as
/// `dispatch_workgroups(width, height, 1)`, so `workgroup_id.xy` is
/// directly the texel coordinate into all three textures — the same
/// "one workgroup per texel" shape as `AO_BAKE_SHADER`'s
/// `cs_main_from_position`, minus the ray loop (each texel's work here is
/// four neighbor loads, so the 64-invocation workgroup would sit idle).
/// Neighbor coordinates are clamped explicitly against `params.width` /
/// `params.height` (never sampled out of bounds); out-of-bounds and
/// uncovered neighbors are *skipped, not zero-filled* (a hole must not
/// flatten a neighboring ridge), and a covered texel with no valid
/// neighbors at all (isolated single-texel island) bakes mid-gray
/// (`curv = 0`). Uncovered texels write `(0, 0, 0, 0)`, distinguishable
/// from a flat-but-covered texel (`(~0.5, ~0.5, ~0.5, 1)`) by alpha alone
/// — the same alpha convention `cs_main_from_position` uses, which is why
/// `curvature::bake_curvature_mesh` returns full RGBA8.
pub const CURVATURE_BAKE_SHADER: &str = r#"
struct CurvatureParams {
    width: u32,
    height: u32,
    strength: f32,
    _pad: f32,
};

@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: CurvatureParams;
@group(0) @binding(2) var position_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;

const CURV_EPS: f32 = 1e-6;

@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(workgroup_id.xy);
    let center = textureLoad(position_tex, coord, 0);
    if (center.w <= 0.5) {
        textureStore(out_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let p0 = center.xyz;
    let n0 = normalize(textureLoad(normal_tex, coord, 0).xyz);
    let dims = vec2<i32>(i32(params.width), i32(params.height));

    var sum = 0.0;
    var count = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        var off = vec2<i32>(1, 0);
        if (k == 1u) {
            off = vec2<i32>(-1, 0);
        } else if (k == 2u) {
            off = vec2<i32>(0, 1);
        } else if (k == 3u) {
            off = vec2<i32>(0, -1);
        }
        let nc = coord + off;
        if (nc.x < 0 || nc.y < 0 || nc.x >= dims.x || nc.y >= dims.y) {
            continue;
        }
        let s = textureLoad(position_tex, nc, 0);
        if (s.w <= 0.5) {
            continue;
        }
        let dp = s.xyz - p0;
        let len = length(dp);
        let dir = dp / max(len, CURV_EPS);
        let dn = textureLoad(normal_tex, nc, 0).xyz - n0;
        sum = sum + dot(dn, dir) / max(len, CURV_EPS);
        count = count + 1u;
    }

    var curv = 0.0;
    if (count > 0u) {
        // Negated: the raw estimator is positive on outward bulges (see
        // this constant's doc comment); MeshLab convention wants convex
        // negative (= darker).
        curv = -sum / f32(count) * params.strength;
    }
    curv = clamp(curv, -1.0, 1.0);
    let g = (curv + 1.0) * 0.5;
    textureStore(out_tex, coord, vec4<f32>(g, g, g, 1.0));
}
"#;

/// Thickness (local solid depth) estimation by inward hemisphere raycasting:
/// one workgroup per output texel, casting rays *into* the mesh and recording
/// the nearest opposite-face hit.
///
/// # Estimator
///
/// Per covered texel, over `params.rays` deterministic stratified hemisphere
/// directions (the same `hemisphere_sample` as [`AO_BAKE_SHADER`], steered by
/// the same Duff-et-al. `orthonormal_basis` tangent frame from the position
/// pass's per-texel normal), pointed into the surface — i.e. around the
/// *negative* normal instead of [`AO_BAKE_SHADER`]'s outward `+normal`
/// hemisphere:
///
/// ```text
/// min_dist = min over rays, over triangles of hit t in (EPS, max_distance)
/// thickness = clamp(min_dist / max_distance, 0, 1)
/// ```
///
/// A per-ray miss contributes nothing (that ray simply never lowers
/// `min_dist`, which starts at `max_distance`); a texel whose *every* ray
/// misses therefore bakes `1.0`. That is a deliberate convention, not a
/// fallback: a miss means the ray escaped through open geometry, so the
/// surface reads as "no opposite face within range" — i.e. maximally thick
/// (see `umber-bake/LANDING_NOTES_THICKNESS.md` for the open-mesh artifact
/// this implies). Output is grayscale `rgb = thickness`, `a = coverage
/// (`1.0` covered, `0.0` uncovered).
///
/// Uncovered texels (the position pass's coverage flag `<= 0.5`) skip the
/// raycast entirely and write `(0, 0, 0, 0)`, distinguishable on readback
/// from a covered-but-maximally-thick texel (`(1, 1, 1, 1)`) by alpha alone
/// — the same alpha convention `AO_BAKE_SHADER::cs_main_from_position` uses.
///
/// # Why into the mesh, and where the ray starts
///
/// `ray_origin = surface_pos - bias * normal` (nudged *inside* along the
/// negative normal, the mirror of AO's outward `+ bias * normal`). Starting
/// outside and casting inward would cross the originating surface itself at
/// `t ~= bias` and bake a false near-zero thickness; starting just inside
/// puts the origin triangle behind every inward ray so only genuine
/// opposite faces lower `min_dist`. With `bias == 0` the origin sits exactly
/// on the surface and the `(EPS, max_t)` open-interval guard in
/// `ray_hit_distance` still rejects the self-hit at `t ~= 0`.
///
/// # Layout contracts
///
/// - `Tri` must match `umber_bake::thickness`'s private `GpuTriangle`
///   byte-for-byte (four `vec4<f32>`s, identical to `AO_BAKE_SHADER`'s `Tri`
///   — the same brute-force triangle list, no BVH, same perf budget of
///   `texels * rays * triangle_count`; see [`AO_BAKE_SHADER`]'s doc comment).
/// - `ThicknessParams` must match `umber_bake::thickness`'s private
///   `ThicknessUniform` byte-for-byte (`rays`, `max_distance`, `bias`,
///   `tri_count` — four 4-byte scalars, 16 bytes total, already a multiple
///   of WGSL's 16-byte uniform-struct alignment, so no padding field is
///   needed).
/// - `position_tex`/`normal_tex` are the two `Rgba32Float` outputs of
///   `POSITION_BAKE_SHADER`'s `cs_main`, bound read-only and sampled with
///   `textureLoad` at integer texel coordinates (no sampler, no filtering).
///   Binding numbers (`0/1/2` plus `4/5`) deliberately mirror
///   `AO_BAKE_SHADER::cs_main_from_position`'s set (which skips `3`, the
///   plane-path `dims` uniform this mesh-fed pass has no use for) so the
///   two raycast passes stay grep-comparable.
/// - `thickness_tex` is a write-only `Rgba8Unorm` storage texture (core
///   WebGPU, no device feature — the same reason [`AO_BAKE_SHADER`] uses
///   `write`, not `read_write`).
///
/// # Dispatch + workgroup shape
///
/// One workgroup (`@workgroup_size(1)`) per texel, dispatched as
/// `dispatch_workgroups(width, height, 1)`, so `workgroup_id.xy` is directly
/// the texel coordinate into all three textures — the same "one workgroup
/// per texel" shape as `AO_BAKE_SHADER::cs_main_from_position`, minus the
/// 64-invocation fan-out. A 64-wide workgroup with an atomic counter works
/// for AO's hit *count* (`atomic<u32>`), but a minimum *distance* is a float
/// with no atomic-min in core WGSL; fanning out would need a shared-memory
/// array plus a manual reduction for no real win (per-texel work here is
/// `rays * tri_count` brute-force tests, trivially serial at this slice's
/// mesh-size budget). Each texel's ray loop is therefore serial, and the
/// output stays deterministic texel-to-texel like AO's fixed sample set.
pub const THICKNESS_BAKE_SHADER: &str = r#"
struct Tri {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    normal: vec4<f32>,
};

struct ThicknessParams {
    rays: u32,
    max_distance: f32,
    bias: f32,
    tri_count: u32,
};

@group(0) @binding(0) var<storage, read> triangles: array<Tri>;
@group(0) @binding(1) var thickness_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ThicknessParams;
@group(0) @binding(4) var position_tex: texture_2d<f32>;
@group(0) @binding(5) var normal_tex: texture_2d<f32>;

const PI: f32 = 3.14159265358979;
const GOLDEN_CONJ: f32 = 0.6180339887498949;
const EPS: f32 = 1e-6;

/// Deterministic hemisphere direction for ray `i` of `n`, in a local frame
/// where +Z is the hemisphere pole: `cos(theta)` is stratified evenly
/// across `[0, 1)` (uniform over solid angle) and `phi` walks the golden
/// angle. Verbatim the same sample set as `AO_BAKE_SHADER::hemisphere_sample`
/// (fixed, not per-texel jittered) so thickness inherits AO's determinism;
/// the caller negates the pole (`- local_dir.z * normal`) to point the
/// hemisphere into the mesh.
fn hemisphere_sample(i: u32, n: u32) -> vec3<f32> {
    let nf = max(f32(n), 1.0);
    let cos_theta = 1.0 - (f32(i) + 0.5) / nf;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * fract(f32(i) * GOLDEN_CONJ);
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

/// Möller–Trumbore ray-triangle intersection returning the hit distance.
/// Double-sided (no backface culling, matching `AO_BAKE_SHADER`). Returns the
/// hit parameter `t` iff it lands in the open interval `(EPS, max_t`,
/// otherwise `-1.0` — the distance-returning analog of AO's boolean
/// `hits_triangle`, so the caller can keep the minimum across rays.
fn ray_hit_distance(
    orig: vec3<f32>,
    dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    max_t: f32,
) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(dir, e2);
    let a = dot(e1, h);
    if (abs(a) < EPS) {
        return -1.0;
    }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t = f * dot(e2, q);
    if (t > EPS && t < max_t) {
        return t;
    }
    return -1.0;
}

struct Basis {
    b1: vec3<f32>,
    b2: vec3<f32>,
};

/// Branchless tangent-frame construction from a unit normal (Duff et al.,
/// "Building an Orthonormal Basis, Revisited", JCGT 2017). Verbatim the same
/// frame as `AO_BAKE_SHADER::orthonormal_basis`: any orthonormal frame works
/// (thickness, like AO, is isotropic in `phi`), this one was picked for being
/// branchless and stable at the south pole.
fn orthonormal_basis(n: vec3<f32>) -> Basis {
    let sign_z = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign_z + n.z);
    let b = n.x * n.y * a;
    return Basis(
        vec3<f32>(1.0 + sign_z * n.x * n.x * a, sign_z * b, -sign_z * n.x),
        vec3<f32>(b, sign_z + n.y * n.y * a, -n.y),
    );
}

/// Mesh-fed thickness: per-texel ray origin and tangent frame come from
/// `position_tex`/`normal_tex` (the two outputs of `POSITION_BAKE_SHADER`'s
/// `cs_main`). `position_tex`'s alpha channel is the position pass's coverage
/// flag: `0.0` means no UV triangle covered this texel, so there is no surface
/// to cast inward from — the texel writes `(0, 0, 0, 0)` with no raycast.
/// Otherwise every ray is cast around the *negative* normal, the minimum hit
/// distance across the whole ray set is normalized by `max_distance`, and the
/// result is written grayscale with full coverage alpha. A texel whose rays
/// all miss writes `1.0` (no opposite face within range reads as maximally
/// thick — see this constant's doc comment).
@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(workgroup_id.xy);
    let sample = textureLoad(position_tex, coord, 0);
    if (sample.w <= 0.5) {
        textureStore(thickness_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let surface_pos = sample.xyz;
    let normal = normalize(textureLoad(normal_tex, coord, 0).xyz);
    let basis = orthonormal_basis(normal);
    let ray_origin = surface_pos - params.bias * normal;

    var min_dist = params.max_distance;
    let n = params.rays;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let local_dir = hemisphere_sample(i, n);
        let dir = normalize(
            local_dir.x * basis.b1 + local_dir.y * basis.b2 - local_dir.z * normal
        );
        for (var t: u32 = 0u; t < params.tri_count; t = t + 1u) {
            let tri = triangles[t];
            let hit = ray_hit_distance(
                ray_origin, dir, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz, params.max_distance
            );
            if (hit > 0.0 && hit < min_dist) {
                min_dist = hit;
            }
        }
    }

    let range = max(params.max_distance, EPS);
    let thickness = clamp(min_dist / range, 0.0, 1.0);
    textureStore(thickness_tex, coord, vec4<f32>(thickness, thickness, thickness, 1.0));
}
"#;

/// UV-padding dilation (post-bake island-margin expansion): one workgroup
/// per output texel, spreading island-edge colors outward into uncovered
/// texels so exported maps have no transparent seams.
///
/// # Algorithm (one dilation step per dispatch)
///
/// Per texel, from the read-only input map `src_tex` into the write-only
/// output `dst_tex`:
///
/// ```text
/// if src.a > 0: copy through unchanged
/// else: scan the 8 neighbors; among covered neighbors pick the one with
///       the HIGHEST coverage (alpha) value, ties keep the first found;
///       write its rgb with alpha 1.0 — or (0,0,0,0) if no neighbor is covered.
/// ```
///
/// `N` steps of padding = `N` ping-pong dispatches driven from Rust (see
/// `umber_bake::dilation::dilate_map`): each step reads the previous step's
/// output, so one ring of texels is claimed per dispatch and the covered
/// front advances exactly one texel per iteration.
///
/// # Why the dilated alpha must normalize to 1.0 (chain propagation)
///
/// A newly-dilated texel must be indistinguishable from an originally
/// covered texel on the *next* step (`a = 1.0`, i.e. byte `255`, both) or
/// propagation stalls after one ring: the next pass's "covered" test is
/// `a > 0`, and its donor ranking compares coverage values, so writing
/// anything less than full coverage would make second-ring texels rank
/// below (or read as uncovered next to) the front and the wave would die
/// out. The donor's rgb is copied verbatim — no blending, no falloff — so
/// color propagates unchanged no matter how many rings it travels.
///
/// # Layout contracts
///
/// - `src_tex` is a read-only `texture_2d<f32>` over an `Rgba8Unorm`
///   texture, sampled with `textureLoad` at integer texel coordinates (no
///   sampler, no filtering — filtering would smear island colors across UV
///   seams, the same reason `CURVATURE_BAKE_SHADER` uses `textureLoad`). The
///   Rust side must create the source texture with
///   `TEXTURE_BINDING` usage; the `BakeTarget` textures it ping-pongs
///   between already carry that flag via `PaintTarget`.
/// - `dst_tex` is a write-only `Rgba8Unorm` storage texture (core WebGPU,
///   no device feature — the same reason [`AO_BAKE_SHADER`] uses `write`,
///   not `read_write`).
/// - `dims` is a `vec2<u32>` uniform holding the target width/height, used
///   only to clamp the 8-neighborhood against the texture edges (no wrap).
/// - No other state: no triangle buffer, no position/normal inputs.
///
/// # Dispatch + workgroup shape
///
/// One workgroup (`@workgroup_size(1)`) per texel, dispatched as
/// `dispatch_workgroups(width, height, 1)`, so `workgroup_id.xy` is directly
/// the texel coordinate into both textures — the same "one workgroup per
/// texel" shape as `AO_BAKE_SHADER::cs_main_from_position` and the
/// curvature/thickness passes, minus any ray loop (per-texel work here is
/// nine `textureLoad`s, trivially serial).
///
/// # Known artifact: thin diagonal streaks from first-found tie-breaking
///
/// With binary coverage (alpha `0` or `1` — everything the bake passes and
/// this pass itself produce), every covered neighbor ties at `1.0` and the
/// strict-`>` comparison keeps the first in scan order (top-left row-major:
/// `(-1,-1)` first, `(1,1)` last). Along diagonal fronts this biases donor
/// choice toward the top-left, leaving faint diagonal streaks in the padded
/// margin where two fronts meet. Acceptable for this slice — Substance's
/// dilator shows the same streaking — and confined to the margin (covered
/// texels copy through untouched, so source pixels are never altered).
pub const DILATE_BAKE_SHADER: &str = r#"
struct DilateDims {
    dims: vec2<u32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> dilate_dims: DilateDims;

@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(workgroup_id.xy);
    let center = textureLoad(src_tex, coord, 0);
    if (center.w > 0.0) {
        textureStore(dst_tex, coord, center);
        return;
    }

    let dims = vec2<i32>(dilate_dims.dims);
    var best_rgb = vec3<f32>(0.0, 0.0, 0.0);
    var best_a = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nc = coord + vec2<i32>(dx, dy);
            if (nc.x < 0 || nc.y < 0 || nc.x >= dims.x || nc.y >= dims.y) {
                continue;
            }
            let s = textureLoad(src_tex, nc, 0);
            // Strict `>`: ties keep the first donor found in scan order
            // (top-left row-major), giving deterministic output — see this
            // constant's "diagonal streaks" doc comment.
            if (s.w > 0.0 && s.w > best_a) {
                best_a = s.w;
                best_rgb = s.xyz;
            }
        }
    }

    if (best_a > 0.0) {
        // Full coverage alpha: the next ping-pong step must read this
        // texel as covered as if it were original island (see this
        // constant's "chain propagation" doc comment).
        textureStore(dst_tex, coord, vec4<f32>(best_rgb, 1.0));
    } else {
        textureStore(dst_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
"#;

/// Tangent-space normal baking from a position/normal map: one workgroup
/// per output texel, transforming each covered texel's world normal into a
/// screen-space tangent frame built from the position map's own
/// x-derivative.
///
/// # Screen-space tangent frame (no explicit UVs)
///
/// Per covered texel:
///
/// ```text
/// dpdx = central (both x-neighbors covered) or one-sided (island border)
///        position difference; uncovered/out-of-bounds neighbors are skipped,
///        never zero-filled (a hole must not tilt a neighboring frame)
/// T = normalize(dpdx - N * dot(N, dpdx))   // Gram-Schmidt against N
/// B = cross(N, T)
/// tangent_normal = (dot(T, n), dot(B, n), dot(N, n))  // transpose(TBN) * n
/// ```
///
/// where `N`/`n` is the position pass's per-texel face normal
/// (`normal_tex`, normalized) — the only normal this slice has, so the
/// world normal being transformed and the frame's axis coincide (a flat
/// facet correctly bakes `(0, 0, 1)`; smooth/interpolated normals plug into
/// `n` without touching the frame code — see "Wave-4 refinement" below).
/// A texel with no covered x-neighbor (isolated single-texel island) or a
/// near-zero gradient falls back to the world axis least aligned with `N`,
/// orthogonalized the same way, so `T` is never a normalized zero vector.
///
/// # UV-alignment assumption (matches Substance's default when UVs are axis-aligned)
///
/// Texel `+x` is `+u`: the position pass maps texel centers through
/// `u = (x + 0.5) / width` (see `POSITION_BAKE_SHADER`'s doc comment), so
/// `dP/dx` points along the surface's `+u` direction and `T` is the
/// `+u` tangent — exactly Substance Painter's default tangent frame on
/// meshes whose UV islands are axis-aligned to the baked surface. `B =
/// cross(N, T)` then points along `+v` on such meshes (not along texel
/// `+y`, which is `-v` under the position pass's `(1 - v)` flip), keeping
/// green "up" in UV space per the OpenGL convention below. On rotated UV
/// islands the frame twists with the screen axes instead of the UVs —
/// the known screen-space limitation this slice accepts (same class of
/// artifact as `CURVATURE_BAKE_SHADER`'s UV-seam blindness).
///
/// # Wave-4 refinement plan (per-texel UV-derivative TBN)
///
/// Once the position pass exports per-texel UVs in a channel, replace the
/// `dpdx`-only construction with the standard UV-derivative frame: solve
/// `dP/du`, `dP/dv` from neighbor differences (`dp = dP/du * du + dP/dv *
/// dv` over two covered neighbors), `T = normalize(dP/du - N * dot(N,
/// dP/du))`, `B` from `dP/dv` with a `cross(N, T)`-handedness check
/// against the UV winding. The `transpose(TBN) * n` transform and
/// the encoding below stay unchanged; only the `T`/`B` derivation moves.
///
/// # Encoding + convention (OpenGL default, DirectX flip)
///
/// `rgb = tangent_normal * 0.5 + 0.5`, `a` = coverage (`1.0` covered,
/// `0.0` uncovered). Default is OpenGL (`+Y` up = green up). A nonzero
/// `flip_y` inverts the green channel *after* encoding (`g = 1 - g`) for
/// DirectX (`-Y` up). Uncovered texels write `(0, 0, 0, 0)`,
/// distinguishable from a flat-but-covered texel (`(~0.5, ~0.5, 1, 1)`)
/// by alpha alone — the same alpha convention
/// `AO_BAKE_SHADER::cs_main_from_position` uses, which is why
/// `normal_map::bake_tangent_normal_mesh` returns full RGBA8.
///
/// # Layout contracts
///
/// - `TangentNormalParams` must match `umber_bake::normal_map`'s private
///   `TangentNormalUniform` byte-for-byte (`width`, `height`, `flip_y`,
///   one `u32` pad — 16 bytes, already a multiple of WGSL's 16-byte
///   uniform-struct alignment, so no further padding is needed).
/// - `position_tex`/`normal_tex` are the two `Rgba32Float` outputs of
///   `POSITION_BAKE_SHADER`'s `cs_main`, bound read-only and sampled with
///   `textureLoad` at integer texel coordinates (no sampler, no filtering
///   — filtering would smear normals across UV seams; see
///   `CURVATURE_BAKE_SHADER`'s doc comment). `position_tex`'s alpha is
///   the position pass's coverage flag: `0.0` means no UV triangle covered
///   that texel.
/// - `out_tex` is a write-only `Rgba8Unorm` storage texture (core WebGPU,
///   no device feature — the same reason [`AO_BAKE_SHADER`] uses `write`,
///   not `read_write`).
///
/// # Dispatch + edge behavior
///
/// One workgroup (`@workgroup_size(1)`) per texel, dispatched as
/// `dispatch_workgroups(width, height, 1)`, so `workgroup_id.xy` is
/// directly the texel coordinate into all three textures — the same
/// "one workgroup per texel" shape as `CURVATURE_BAKE_SHADER`, minus any
/// ray loop (per-texel work here is a handful of `textureLoad`s,
/// trivially serial).
pub const TANGENT_NORMAL_BAKE_SHADER: &str = r#"
struct TangentNormalParams {
    width: u32,
    height: u32,
    flip_y: u32,
    _pad: u32,
};

@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: TangentNormalParams;
@group(0) @binding(2) var position_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;

const TN_EPS: f32 = 1e-6;

@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(workgroup_id.xy);
    let center = textureLoad(position_tex, coord, 0);
    if (center.w <= 0.5) {
        textureStore(out_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let p0 = center.xyz;
    let n_axis = normalize(textureLoad(normal_tex, coord, 0).xyz);
    let dims = vec2<i32>(i32(params.width), i32(params.height));

    // Screen-space dP/dx from the covered x-neighbors: central difference
    // when both sides are covered, one-sided at island borders. Out-of-
    // bounds and uncovered neighbors are skipped, not zero-filled (a hole
    // must not tilt a neighboring frame) — the same rule
    // CURVATURE_BAKE_SHADER uses for its 4-neighborhood.
    var has_px = false;
    var has_nx = false;
    var p_px = vec3<f32>(0.0, 0.0, 0.0);
    var p_nx = vec3<f32>(0.0, 0.0, 0.0);
    if (coord.x + 1 < dims.x) {
        let s = textureLoad(position_tex, coord + vec2<i32>(1, 0), 0);
        if (s.w > 0.5) {
            p_px = s.xyz;
            has_px = true;
        }
    }
    if (coord.x - 1 >= 0) {
        let s = textureLoad(position_tex, coord + vec2<i32>(-1, 0), 0);
        if (s.w > 0.5) {
            p_nx = s.xyz;
            has_nx = true;
        }
    }
    var dpdx = vec3<f32>(0.0, 0.0, 0.0);
    var has_dpdx = false;
    if (has_px && has_nx) {
        dpdx = (p_px - p_nx) * 0.5;
        has_dpdx = true;
    } else if (has_px) {
        dpdx = p_px - p0;
        has_dpdx = true;
    } else if (has_nx) {
        dpdx = p0 - p_nx;
        has_dpdx = true;
    }

    // Gram-Schmidt orthogonalization of the +u tangent against N. An
    // isolated texel (no covered x-neighbor) or a near-zero gradient falls
    // back to the world axis least aligned with N, orthogonalized the same
    // way, so T is never a normalized zero vector.
    var t = vec3<f32>(1.0, 0.0, 0.0);
    if (has_dpdx) {
        t = dpdx - n_axis * dot(n_axis, dpdx);
    }
    var t_len = length(t);
    if (!has_dpdx || t_len < TN_EPS) {
        var axis = vec3<f32>(1.0, 0.0, 0.0);
        if (abs(n_axis.x) > 0.9) {
            axis = vec3<f32>(0.0, 1.0, 0.0);
        }
        t = axis - n_axis * dot(n_axis, axis);
        t_len = length(t);
    }
    let T = t / max(t_len, TN_EPS);
    let B = cross(n_axis, T);

    // World normal into the frame: transpose(TBN) * n. This slice's only
    // normal source is the position pass's face normal, so n == n_axis and
    // a flat facet bakes (0, 0, 1) by construction; a wave-4 smooth-normal
    // input plugs into `n` here without touching the frame above.
    let n = n_axis;
    var tn = vec3<f32>(dot(T, n), dot(B, n), dot(n_axis, n));
    tn = clamp(tn, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
    var rgb = tn * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
    if (params.flip_y != 0u) {
        rgb.y = 1.0 - rgb.y;
    }
    rgb = clamp(rgb, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    textureStore(out_tex, coord, vec4<f32>(rgb, 1.0));
}
"#;

/// Deterministic per-texel region-ID coloring (ID bake, wave-4 item 5 slice 1):
/// one workgroup per output texel, hashing the winning triangle's index with
/// FNV-1a and encoding the hash as the texel color.
///
/// # Why the triangle index is re-derived here instead of read from the position map
///
/// `POSITION_BAKE_SHADER` writes per-texel world position (xyz) + coverage (w)
/// and face normal — there is deliberately no triangle-index channel (adding
/// one would cost a third `Rgba32Float` target for data only this pass needs).
/// This shader therefore re-binds the mesh's triangle buffer (the same `PosTri`
/// layout the position pass consumes, carrying each triangle's UV footprint)
/// and re-runs the *identical* point-in-triangle containment test
/// (`cross2`/`1/det` barycentric solve, same `ID_DET_EPS`/`ID_BARY_EPS`
/// epsilons, same edge-inclusive bounds, same last-writer-wins on overlap) to
/// recover the winning triangle index deterministically. The position map is
/// still the coverage authority: its alpha (`<= 0.5` means uncovered — the
/// same convention `CURVATURE_BAKE_SHADER` documents) gates the hash path, so
/// this pass never disagrees with the position pass about *whether* a texel is
/// covered, only re-derives *which* triangle covers it.
///
/// A per-texel position hash was considered (hash the covered texel's world-pos
/// bits) and rejected: it yields a different color per texel — gradient noise,
/// useless for region masking — where the design-doc contract
/// (`docs/specs/id-bake-bent-normals-design.md`) wants region-constant colors
/// whose value IS the FNV-1a hash of the region identity. The material flavor
/// rides the high-to-low pass (item 4), which binds the source mesh's parts;
/// see `umber_bake::id`'s module doc for the honest v1 flavor gate.
///
/// # Encoding
///
/// `hash = fnv1a_32(triangle_index as 4 little-endian bytes)`,
/// `rgb = [h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF] / 255`, `a = 1.0`.
/// Each hash byte is an exact integer multiple of `1/255`, so the
/// `Rgba8Unorm` store round-trips it byte-exactly (the float closest to
/// `b/255` scales back to within `1e-6` of `b`). Uncovered texels write
/// `(0, 0, 0, 0)`, distinguishable from any covered texel (whose alpha is
/// always `1.0`, even if its hash bytes happen to be `(0, 0, 0)`) by alpha
/// alone — the same alpha convention
/// `AO_BAKE_SHADER::cs_main_from_position` uses.
///
/// # Layout contracts
///
/// - `PosTri` must match `umber_bake::position`'s private `GpuPosTri`
///   byte-for-byte (three world-space positions with UV x in each `w`, the
///   three UV y components packed in `uv_y`, the face normal — 80 bytes).
///   Only the UV fields are consumed here (world positions/normal ignored).
/// - `IdParams` must match `umber_bake::id`'s private `IdUniform`
///   byte-for-byte (`dims`, `tri_count`, one `u32` pad — 16 bytes, already a
///   multiple of WGSL's 16-byte uniform-struct alignment).
/// - `position_tex` is the position pass's `Rgba32Float` output, bound
///   read-only and sampled with `textureLoad` (no sampler, no filtering).
/// - `out_tex` is a write-only `Rgba8Unorm` storage texture (core WebGPU,
///   no device feature — the same reason [`AO_BAKE_SHADER`] uses `write`,
///   not `read_write`).
///
/// # Dispatch + edge behavior
///
/// One workgroup (`@workgroup_size(1)`) per texel, dispatched as
/// `dispatch_workgroups(width, height, 1)`, so `workgroup_id.xy` is directly
/// the texel coordinate — the same shape as `CURVATURE_BAKE_SHADER` (serial
/// per-texel triangle loop, no shared-memory chunking: the containment test
/// is arithmetic-cheap and this pass runs once per bake, not per ray).
/// Degenerate (zero-UV-area) triangles never claim a texel, matching the
/// position pass. If a covered texel's re-test finds no triangle (unreachable
/// when both passes share identical containment arithmetic — a logic bug, not
/// a data case), the texel writes background `(0, 0, 0, 0)` rather than a
/// plausible-but-wrong color, so the failure is visible on review.
pub const ID_BAKE_SHADER: &str = r#"
struct PosTri {
    v0: vec4<f32>,    // xyz = world position of vertex 0 (ignored here), w = uv0.x
    v1: vec4<f32>,    // xyz = world position of vertex 1 (ignored here), w = uv1.x
    v2: vec4<f32>,    // xyz = world position of vertex 2 (ignored here), w = uv2.x
    uv_y: vec4<f32>,  // x = uv0.y, y = uv1.y, z = uv2.y, w unused
    normal: vec4<f32>, // face normal (ignored here), w unused
};

struct IdParams {
    dims: vec2<u32>,
    tri_count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> tris: array<PosTri>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: IdParams;
@group(0) @binding(3) var position_tex: texture_2d<f32>;

const ID_DET_EPS: f32 = 1e-10;
const ID_BARY_EPS: f32 = 1e-6;
const FNV_OFFSET_BASIS: u32 = 2166136261u;
const FNV_PRIME: u32 = 16777619u;

/// 2D cross product (the scalar "perp-dot"): `a.x*b.y - a.y*b.x`.
/// Verbatim the same helper as `POSITION_BAKE_SHADER::cross2`.
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

/// One FNV-1a step over a single byte: `(h ^ b) * prime` with wrapping
/// `u32` arithmetic (WGSL unsigned overflow wraps, matching Rust's
/// `wrapping_mul` in the `umber_bake::id` test mirror).
fn fnv1a_step(h: u32, b: u32) -> u32 {
    return (h ^ b) * FNV_PRIME;
}

/// FNV-1a over the triangle index's 4 little-endian bytes.
fn hash_triangle(tri: u32) -> u32 {
    var h = FNV_OFFSET_BASIS;
    h = fnv1a_step(h, tri & 0xFFu);
    h = fnv1a_step(h, (tri >> 8u) & 0xFFu);
    h = fnv1a_step(h, (tri >> 16u) & 0xFFu);
    h = fnv1a_step(h, (tri >> 24u) & 0xFFu);
    return h;
}

@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(workgroup_id.xy);
    if (textureLoad(position_tex, coord, 0).w <= 0.5) {
        textureStore(out_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Texel-center UV, v-flipped to match umber_app::paint_state's
    // `(1 - v) * texels_per_uv` convention — verbatim the same formula as
    // POSITION_BAKE_SHADER so both passes agree texel-for-texel.
    let dims_f = vec2<f32>(params.dims);
    let texel_uv = vec2<f32>(
        (f32(workgroup_id.x) + 0.5) / max(dims_f.x, 1.0),
        1.0 - (f32(workgroup_id.y) + 0.5) / max(dims_f.y, 1.0),
    );

    // Serial containment loop with last-writer-wins: triangle order, edge-
    // inclusive bounds, and degenerate-triangle rejection all mirror
    // POSITION_BAKE_SHADER's chunked loop exactly, so the recovered index is
    // the same triangle the position pass's `best_tri` held.
    var best_tri = 0u;
    var found = false;
    for (var t: u32 = 0u; t < params.tri_count; t = t + 1u) {
        let tri = tris[t];
        let uv0 = vec2<f32>(tri.v0.w, tri.uv_y.x);
        let uv1 = vec2<f32>(tri.v1.w, tri.uv_y.y);
        let uv2 = vec2<f32>(tri.v2.w, tri.uv_y.z);
        let e1 = uv1 - uv0;
        let e2 = uv2 - uv0;
        let denom = cross2(e1, e2);
        if (abs(denom) >= ID_DET_EPS) {
            let inv_denom = 1.0 / denom;
            let s = texel_uv - uv0;
            let bu = cross2(s, e2) * inv_denom;
            let bv = cross2(e1, s) * inv_denom;
            let bw = 1.0 - bu - bv;
            if (bu >= -ID_BARY_EPS && bv >= -ID_BARY_EPS && bw >= -ID_BARY_EPS) {
                best_tri = t;
                found = true;
            }
        }
    }
    if (!found) {
        textureStore(out_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let h = hash_triangle(best_tri);
    let rgb = vec3<f32>(
        f32(h & 0xFFu),
        f32((h >> 8u) & 0xFFu),
        f32((h >> 16u) & 0xFFu),
    ) / 255.0;
    textureStore(out_tex, coord, vec4<f32>(rgb, 1.0));
}
"#;

/// High→low bake transfer: per-texel rays from the LOW mesh's UV map into
/// the HIGH mesh's triangle soup (wave-4 item 4, final slice —
/// `docs/specs/high-to-low-transfer-design.md` §"The architecture" point 3).
///
/// # Per-texel ray configuration
///
/// The LOW surface data comes from the EXISTING position pass, bound
/// read-only exactly like the curvature/thickness passes consume it
/// (`POSITION_BAKE_SHADER`'s output contract — `bake_shaders.rs` docs and
/// `umber_bake::position::bake_position_map` agree):
///
/// - `low_pos_tex` (`Rgba32Float`): `rgb` = world-space position, `a` =
///   coverage (`1.0` iff some LOW triangle's UV footprint covers this texel).
/// - `low_normal_tex` (`Rgba32Float`): `xyz` = the LOW face normal (unit),
///   `a` = coverage. The position pass emits this companion texture from the
///   same dispatch (`position::bake_position_and_normal`) — the ray direction
///   reads it, not a re-derived gradient.
///
/// Per covered texel:
///
/// ```text
/// origin    = low_pos + low_normal * front_offset
/// dir       = -low_normal                       // into the surface, toward the high
/// t_surface = front_offset                      // the LOW surface along the ray (dir is unit)
/// ```
///
/// Uncovered texels (`a <= 0.5`) skip the raycast and write background
/// `(0, 0, 0, 0)` — the same alpha convention
/// `AO_BAKE_SHADER::cs_main_from_position` uses.
///
/// # Intersection: brute force, no GPU BVH
///
/// The HIGH triangle buffer is walked flat, one Möller–Trumbore test per
/// triangle, keeping the nearest hit — the `PlaneDesc` bakes' flat-brute-force
/// precedent the design doc contracts (a GPU BVH is the deferred perf row;
/// the CPU BVH in `umber_mesh::bvh` stays CPU-side for callers). The test
/// mirrors `umber_mesh::raycast::ray_intersect`'s ground-truth semantics
/// (see also `bvh::tri_hit`): double-sided (no culling), `1e-8` parallel
/// epsilon, `u in 0..=1` / `v >= 0` / `u + v <= 1` barycentric bounds, `t >
/// 1e-5` near-clip. Ties (two HIGH triangles reporting equal `t`, e.g. a
/// shared edge) resolve to the FIRST triangle in buffer order (strict
/// `<` replacement) — deterministic, documented here because the CPU BVH
/// instead tie-breaks to the smallest triangle index.
///
/// The loop is bounded above by `t_surface + back_distance`: hits past the
/// far clamp can never validate, so they are culled inside the intersection
/// test itself (the "bounded early-out via the clamp distances").
///
/// # Clamp gate (mirrors `umber_mesh::bake_support::TransferClamps`)
///
/// A hit at ray parameter `hit_t` is valid iff
/// `t_surface - front_distance <= hit_t <= t_surface + back_distance`
/// (`TransferClamps::clamps_hit`). One nuance: the far bound inside the
/// intersection loop is STRICT (`t < max_t`, the Möller–Trumbore open
/// interval), while `clamps_hit` is inclusive on both sides — a hit landing
/// bit-exactly on `t_surface + back_distance` is culled here but kept by the
/// CPU gate. Only exact-boundary hits can observe the difference; every
/// tested configuration keeps hits well clear of the bounds.
///
/// # Outputs (one shader, uniform-selected via `map_mode`)
///
/// - `0 = height`: `h = (hit_t - t_surface + front_distance) /
///   (front_distance + back_distance)`, written grayscale with `a = 1`.
///   Note the centering: a hit exactly ON the low surface reads
///   `front_distance / (front_distance + back_distance)` — `0.5` only when
///   `front_distance == back_distance`. (An early brief draft pinned `0.5`
///   for asymmetric clamps; the shader implements the design doc's
///   "normalized by `front_distance + back_distance`" formula, and the
///   `umber_bake::transfer` tests derive their exact bytes from it.)
/// - `1 = world normal`: the HIT triangle's face normal (`tri.normal`,
///   normalized), encoded `rgb = n * 0.5 + 0.5`, `a = 1`. This is the
///   "object/world space" normal-space option requirements §3 names.
/// - `2 = tangent normal`: the HIT normal transformed into the LOW's
///   per-texel UV-derivative TBN frame (`transfer_tbn` below), encoded
///   `rgb = transpose(TBN) * n * 0.5 + 0.5`, `a = 1` — the Mikktspace-style
///   tangent-space fold (green = +Y up, OpenGL).
///   WORLD-space v1: the HIGH normal is written as-is, NOT transformed into
///   the LOW's tangent frame — per-texel UV-derivative TBN is the deferred
///   fold (see `TANGENT_NORMAL_BAKE_SHADER`'s screen-space-frame discussion
///   for why a correct tangent frame needs per-texel UV derivatives this
///   pass does not bind). Interpolated vertex normals are likewise deferred:
///   like every other baker here, the triangle buffer carries the recomputed
///   FACE normal (always available, even for UV-less IMPORTS without vertex
///   normals), so a faceted HIGH bakes faceted normals.
///
/// A validated hit outside `map_mode ∈ {0, 1, 2}` cannot occur (the Rust side
/// only ever writes `0`/`1`/`2`); misses write background `(0, 0, 0, 0)`.
///
/// # The cage does NOT enter v1's GPU path
///
/// `Cage::lerp_at` needs per-vertex offsets interpolated mid-shader, which
/// needs a cage buffer this binding set does not carry. Cage support lands
/// when the app-wiring slice adds that buffer; the FRONT/BACK clamps DO
/// enter v1 (see above).
///
/// # Layout contracts
///
/// - `Tri` must match `umber_bake::transfer`'s private `GpuTriangle`
///   byte-for-byte (four `vec4<f32>`s: three HIGH vertex positions, then the
///   HIGH face normal — byte-identical to `AO_BAKE_SHADER`'s/`THICKNESS_BAKE_SHADER`'s
///   `Tri`, NOT `POSITION_BAKE_SHADER`'s 80-byte `PosTri`: the HIGH mesh
///   needs no UVs per the design doc ("no UVs needed"), and the `PosTri`
///   builder ERRORS on missing UVs, so the UV-free AO-style layout is the
///   fitting reuse — only positions/normal are consumed here anyway).
/// - `TransferParams` must match `umber_bake::transfer`'s private
///   `TransferUniform` byte-for-byte (`front_distance`, `back_distance`,
///   `front_offset`, `high_tri_count`, `map_mode`, `width`, `height`, one
///   `u32` pad — 32 bytes, already a multiple of WGSL's 16-byte
///   uniform-struct alignment). `width`/`height` document the target the
///   dispatch was sized for; the shader itself needs no dims uniform
///   (`workgroup_id.xy` IS the texel coordinate — the
///   `cs_main_from_position` convention).
/// - `low_pos_tex`/`low_normal_tex` are the two `Rgba32Float` outputs of
///   `POSITION_BAKE_SHADER`'s `cs_main`, bound read-only and sampled with
///   `textureLoad` (no sampler, no filtering).
/// - `out_tex` is a write-only `Rgba8Unorm` storage texture (core WebGPU,
///   no device feature — the same reason [`AO_BAKE_SHADER`] uses `write`,
///   not `read_write`).
///
/// # Dispatch + workgroup shape
///
/// One workgroup (`@workgroup_size(1)`) per texel, dispatched as
/// `dispatch_workgroups(width, height, 1)` — the same shape as
/// `CURVATURE_BAKE_SHADER`, minus the neighborhood (per-texel work here is
/// one serial brute-force ray loop; a minimum-*distance* plus nearest-hit
/// record has no atomic form in core WGSL, so — like thickness — the loop
/// stays serial and the output stays deterministic texel-to-texel).
pub const TRANSFER_BAKE_SHADER: &str = r#"
struct Tri {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    normal: vec4<f32>,
};

struct TransferParams {
    front_distance: f32,
    back_distance: f32,
    front_offset: f32,
    high_tri_count: u32,
    map_mode: u32,
    width: u32,
    height: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> high_tris: array<Tri>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: TransferParams;
@group(0) @binding(3) var low_pos_tex: texture_2d<f32>;
@group(0) @binding(4) var low_normal_tex: texture_2d<f32>;

// Möller–Trumbore constants mirroring `umber_mesh::raycast::ray_intersect`
// (see also `umber_mesh::bvh::tri_hit`): 1e-8 parallel epsilon, 1e-5
// near-clip. (The AO/thickness shaders use a looser 1e-6 single EPS for
// both roles; this pass tracks the raycast ground truth instead, since the
// transfer tests derive exact hit parameters from it.)
const MT_PARALLEL_EPS: f32 = 1e-8;
const MT_T_EPS: f32 = 1e-5;

/// Möller–Trumbore ray-triangle intersection returning the hit distance.
/// Double-sided (no backface culling, matching `AO_BAKE_SHADER`). Returns the
/// hit parameter `t` iff it lands in the open interval `(MT_T_EPS, max_t)`,
/// otherwise `-1.0` — the distance-returning shape of
/// `THICKNESS_BAKE_SHADER::ray_hit_distance`, so the caller can keep the
/// nearest hit across the HIGH triangle list.
fn ray_hit_distance(
    orig: vec3<f32>,
    dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    max_t: f32,
) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(dir, e2);
    let a = dot(e1, h);
    if (abs(a) < MT_PARALLEL_EPS) {
        return -1.0;
    }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t = f * dot(e2, q);
    if (t > MT_T_EPS && t < max_t) {
        return t;
    }
    return -1.0;
}

/// Mikktspace-style per-texel UV-derivative frame; green = +Y up (OpenGL/Mikktspace).
struct TbnFrame {
    T: vec3<f32>,
    B: vec3<f32>,
};

fn transfer_tbn(coord: vec2<i32>, p0: vec3<f32>, N: vec3<f32>) -> TbnFrame {
    let dims = vec2<i32>(i32(params.width), i32(params.height));
    var p_px = vec3<f32>(0.0, 0.0, 0.0);
    var p_nx = vec3<f32>(0.0, 0.0, 0.0);
    var p_py = vec3<f32>(0.0, 0.0, 0.0);
    var p_ny = vec3<f32>(0.0, 0.0, 0.0);
    var has_px = false;
    var has_nx = false;
    var has_py = false;
    var has_ny = false;
    var len_px = 0.0;
    var len_nx = 0.0;
    var len_py = 0.0;
    var len_ny = 0.0;
    if (coord.x + 1 < dims.x) {
        let s = textureLoad(low_pos_tex, coord + vec2<i32>(1, 0), 0);
        if (s.w > 0.5) { p_px = s.xyz; has_px = true; len_px = length(s.xyz - p0); }
    }
    if (coord.x - 1 >= 0) {
        let s = textureLoad(low_pos_tex, coord + vec2<i32>(-1, 0), 0);
        if (s.w > 0.5) { p_nx = s.xyz; has_nx = true; len_nx = length(s.xyz - p0); }
    }
    if (coord.y + 1 < dims.y) {
        let s = textureLoad(low_pos_tex, coord + vec2<i32>(0, 1), 0);
        if (s.w > 0.5) { p_py = s.xyz; has_py = true; len_py = length(s.xyz - p0); }
    }
    if (coord.y - 1 >= 0) {
        let s = textureLoad(low_pos_tex, coord + vec2<i32>(0, -1), 0);
        if (s.w > 0.5) { p_ny = s.xyz; has_ny = true; len_ny = length(s.xyz - p0); }
    }
    // Local texel scale = smallest valid neighbor jump; a jump > 4x it
    // crossed a UV seam, so that neighbor is skipped (opposite side used).
    var scale = 1e30;
    if (has_px) { scale = min(scale, len_px); }
    if (has_nx) { scale = min(scale, len_nx); }
    if (has_py) { scale = min(scale, len_py); }
    if (has_ny) { scale = min(scale, len_ny); }
    scale = max(scale, 1e-6);
    if (has_px && len_px > 4.0 * scale) { has_px = false; }
    if (has_nx && len_nx > 4.0 * scale) { has_nx = false; }
    if (has_py && len_py > 4.0 * scale) { has_py = false; }
    if (has_ny && len_ny > 4.0 * scale) { has_ny = false; }
    // Texel +x is +u; texel +y is -v (position pass's `(1 - v)` flip), so
    // the v-derivative raw vector is (ny - py), not (py - ny).
    var has_du = false;
    var has_dv = false;
    var du = vec3<f32>(0.0, 0.0, 0.0);
    var dv = vec3<f32>(0.0, 0.0, 0.0);
    if (has_px && has_nx) { du = (p_px - p_nx) * 0.5; has_du = true; }
    else if (has_px) { du = p_px - p0; has_du = true; }
    else if (has_nx) { du = p0 - p_nx; has_du = true; }
    if (has_py && has_ny) { dv = (p_ny - p_py) * 0.5; has_dv = true; }
    else if (has_ny) { dv = p_ny - p0; has_dv = true; }
    else if (has_py) { dv = p0 - p_py; has_dv = true; }
    // Zero-derivative texel: fall back to whichever derivative axis is
    // nonzero, else the (1,0,0) Gram-Schmidt fallback below.
    var t_raw = vec3<f32>(1.0, 0.0, 0.0);
    if (has_du && length(du) > 1e-6) { t_raw = du; }
    else if (has_dv && length(dv) > 1e-6) { t_raw = dv; }
    var t = t_raw - N * dot(N, t_raw);
    var tl = length(t);
    if (!(tl > 1e-6)) {
        t = vec3<f32>(1.0, 0.0, 0.0) - N * dot(N, vec3<f32>(1.0, 0.0, 0.0));
        tl = length(t);
    }
    let T = t / max(tl, 1e-6);
    return TbnFrame(T, cross(N, T));
}

/// Mesh-fed high→low transfer: per-texel ray origin and frame come from
/// `low_pos_tex`/`low_normal_tex` (the two outputs of `POSITION_BAKE_SHADER`'s
/// `cs_main`, baked from the LOW mesh). `low_pos_tex`'s alpha is the position
/// pass's coverage flag: `<= 0.5` means no LOW UV triangle covered this texel,
/// so there is no surface to cast from — the texel writes `(0, 0, 0, 0)`.
/// Otherwise one ray (`origin = low_pos + n * front_offset` along `-n`) is
/// brute-forced against the HIGH triangle list, gated by the front/back
/// clamps, and the uniform-selected map is written (see this constant's doc
/// comment for the ray math, the gate, and both encodings).
@compute @workgroup_size(1)
fn cs_main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(workgroup_id.xy);
    let sample = textureLoad(low_pos_tex, coord, 0);
    if (sample.w <= 0.5) {
        textureStore(out_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let low_pos = sample.xyz;
    let n = normalize(textureLoad(low_normal_tex, coord, 0).xyz);
    let origin = low_pos + n * params.front_offset;
    let dir = -n;
    // The LOW surface sits `front_offset` behind the origin along the unit
    // ray (origin = surface + n * front_offset, dir = -n), so the surface
    // parameter is exactly `front_offset`.
    let t_surface = params.front_offset;
    // Far-clamp early-out: hits at or past `t_surface + back_distance` can
    // never validate (see this constant's "Clamp gate" doc comment for the
    // strict-vs-inclusive nuance at the exact boundary).
    let max_t = t_surface + params.back_distance;

    var best_t = max_t;
    var best_n = vec3<f32>(0.0, 0.0, 1.0);
    var found = false;
    for (var t: u32 = 0u; t < params.high_tri_count; t = t + 1u) {
        let tri = high_tris[t];
        let hit = ray_hit_distance(
            origin, dir, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz, max_t
        );
        // Strict `<`: ties keep the FIRST triangle in buffer order (see this
        // constant's doc comment — deterministic, unlike an index-agnostic
        // `<=` that would depend on traversal order).
        if (hit > 0.0 && (!found || hit < best_t)) {
            best_t = hit;
            best_n = normalize(tri.normal.xyz);
            found = true;
        }
    }
    // Near-clamp gate (`TransferClamps::clamps_hit`'s lower half; the upper
    // half was already enforced by `max_t` inside the loop).
    if (!found || best_t < t_surface - params.front_distance) {
        textureStore(out_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    if (params.map_mode == 0u) {
        let span = max(params.front_distance + params.back_distance, 1e-6);
        let h = clamp(
            (best_t - t_surface + params.front_distance) / span, 0.0, 1.0
        );
        textureStore(out_tex, coord, vec4<f32>(h, h, h, 1.0));
    } else if (params.map_mode == 1u) {
        let enc = clamp(
            best_n * 0.5 + vec3<f32>(0.5, 0.5, 0.5),
            vec3<f32>(0.0, 0.0, 0.0),
            vec3<f32>(1.0, 1.0, 1.0),
        );
        textureStore(out_tex, coord, vec4<f32>(enc, 1.0));
    } else {
        // Tangent-space hit normal: transpose(TBN) * n, +Y-up green.
        var axis = textureLoad(low_normal_tex, coord, 0).xyz;
        let axis_len = length(axis);
        if (!(axis_len > 1e-6)) { axis = vec3<f32>(0.0, 0.0, 1.0); }
        else { axis = axis / axis_len; }
        let frame = transfer_tbn(coord, low_pos, axis);
        var hn = best_n;
        let hn_len = length(hn);
        if (!(hn_len > 1e-6)) { hn = axis; }
        else { hn = hn / hn_len; }
        var tn = vec3<f32>(dot(frame.T, hn), dot(frame.B, hn), dot(axis, hn));
        tn = clamp(tn, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
        var rgb = clamp(
            tn * 0.5 + vec3<f32>(0.5, 0.5, 0.5),
            vec3<f32>(0.0, 0.0, 0.0),
            vec3<f32>(1.0, 1.0, 1.0),
        );
        if (!(abs(rgb.x) <= 1.0)) { rgb.x = 0.5; }
        if (!(abs(rgb.y) <= 1.0)) { rgb.y = 0.5; }
        if (!(abs(rgb.z) <= 1.0)) { rgb.z = 0.5; }
        textureStore(out_tex, coord, vec4<f32>(rgb, 1.0));
    }
}
"#;

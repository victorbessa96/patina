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

/// Mesh-fed ambient occlusion: identical hemisphere-raycast core to
/// `cs_main`, but the per-texel ray origin and tangent frame come from
/// `position_tex`/`normal_tex` (the two outputs of `POSITION_BAKE_SHADER`'s
/// `cs_main`, see `bake_shaders::POSITION_BAKE_SHADER`) rather than a
/// closed-form plane. One workgroup per texel, matching `cs_main`'s
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
            }
        }
    }
    workgroupBarrier();

    if (local_index == 0u) {
        if (covered) {
            let blocked = f32(atomicLoad(&hit_count)) / f32(max(params.rays, 1u));
            let ao = clamp(1.0 - blocked, 0.0, 1.0);
            textureStore(ao_tex, coord, vec4<f32>(ao, ao, ao, 1.0));
        } else {
            textureStore(ao_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
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

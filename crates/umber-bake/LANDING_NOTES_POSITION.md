# umber-bake position-map + mesh-fed AO — landing notes

This slice replaces `ao::run`'s parameter-plane stand-in (see
`LANDING_NOTES_AO.md`) with the real thing: `umber-bake/src/position.rs`
rasterizes a mesh's own UV layout into a world-position + face-normal
texture pair, and `ao::bake_ao_mesh` raycasts AO against a mesh using
that pair instead of a closed-form plane. `ao::run` (plane) is untouched
and still works — both bake paths coexist.

## A note on how this landed: a concurrent-draft collision

Partway through this slice, `crates/umber-bake/src/position.rs` and the
`POSITION_BAKE_SHADER` section of `crates/umber-gpu/src/bake_shaders.rs`
were each overwritten on disk by a second, independently-written draft —
not something this session produced. That draft used a vertex+index-buffer
GPU layout, `@workgroup_size(1, 1, 1)` (no shared memory at all), no
normal-texture output, first-hit-wins instead of last-writer-wins on
overlapping UV triangles, and no `v`-flip on the UV convention (so it
would have painted mirrored relative to `umber_app::paint_state`). It was
**not** adopted: it conflicted with the shader this session had already
built and empirically verified against this sandbox's adapter (see
"Shared-memory budget" below), and it skipped several of the task's
explicit requirements (the triangle-count ceiling, the normal output AO
needs, shared-memory staging). The conflicting block was deleted from
both files before the final build/test pass; everything described below
is what survived that and is what's on disk now. Flagging this for the
record, not as blame — multiple independent attempts at the same slice
racing on one working tree is an unusual failure mode worth a future
reviewer knowing about if the history here ever looks confusing.

## Why a shared-memory *batching* loop, not a `4096`-entry array

The task sketch asked for "the full triangle list, loaded once into
workgroup-shared memory, sized by a const `MAX_TRIS_PER_BAKE = 4096`."
Before writing the shader, this session probed the actual sandbox adapter
(a throwaway GPU test, removed before landing):

```
adapter max_compute_workgroup_storage_size: 65536
device(default) max_compute_workgroup_storage_size: 16384
```

`wgpu::DeviceDescriptor::default()` is the device every other GPU test in
this crate requests, and — per `umber-gpu`'s architecture rule — the only
kind of device this crate can ever actually get in the real app (eframe's
`wgpu_render_state`, constructed outside this crate's control; there is no
hook to raise its limits after the fact). A `4096`-entry array of even a
minimal 32-byte-per-triangle UV-only record is `131072` bytes — 8x over
the *default* budget, and still 2x over this adapter's own reported
*maximum*. A full per-triangle record with positions would be worse. So
the literal ask is not implementable against any device this crate will
actually run on, not just "slower than hoped."

`POSITION_BAKE_SHADER::cs_main` instead streams triangles through shared
memory in `TRI_CHUNK = 256`-sized batches (`ChunkEntry`, 32 bytes each —
UVs only, `8192` bytes total, half the measured 16 KiB budget, with
headroom to spare): cooperative load → `workgroupBarrier()` → every
invocation in the 8x8 tile tests its own texel against the batch →
`workgroupBarrier()` → next batch. `MAX_TRIS_PER_BAKE` (`4096`, exported
from `position.rs`) survives as `position::validate`'s Rust-side mesh-size
ceiling — meshes larger are rejected before any GPU work happens — it just
isn't a shared-memory array size anymore. The chunking still delivers the
brief's actual underlying goal (each triangle's UV data is read from the
global storage buffer once per *workgroup*, not once per *invocation* — a
64x reduction in global memory traffic for the containment test), just
sized to fit real hardware instead of an unbounded budget.

World positions are deliberately kept **out** of the shared batch —
`ChunkEntry` carries only the two UVs needed for containment. A texel
only needs a triangle's full vertex positions once, for whichever
triangle ends up winning it; that's one extra read from the read-only
global `tris` buffer per *covered texel* at the very end of the batch
loop, not one per triangle-candidate. See `POSITION_BAKE_SHADER`'s own doc
comment in `bake_shaders.rs` for the full byte-budget derivation.

## No depth test — last-writer-wins on overlapping UV islands

Per the task brief: this pass has no notion of "which UV island is on
top." If two triangles' UV footprints cover the same texel (a seam/charting
defect, or deliberately overlapping islands), whichever has the higher
index in the mesh's triangle list wins — not nearest-anything, just
last-tested in the batch loop (`best_tri`/`best_u`/`best_v`/`best_w` are
simply overwritten on every subsequent containing triangle, never
`break`-shortcut on the first hit). This is correct and sufficient for
well-charted meshes (no UV overlap by construction) and silently wrong for
badly-charted ones. A real fix needs either per-island depth/priority
metadata or a guarantee of non-overlapping charts upstream of this pass —
explicitly out of scope here (Wave-3+ seam/overlap work per the brief).

Inclusive barycentric bounds (`>= -BARY_EPS`, not strict `>= 0`) exist
specifically so a texel sitting exactly on a shared edge between two
adjoining triangles (e.g. the test quad's diagonal) is claimed by *at
least one* of them rather than falling through a seam gap — both
triangles agree on the interpolated position along a shared edge, so it
doesn't matter which one "wins" there.

## The UV→texel convention matches the paint path, not an arbitrary choice

A texel's UV is derived from its texel-center grid coordinate with the `v`
axis flipped: `v = 1 - (y + 0.5) / height`. This isn't an arbitrary
pick — it matches `umber_app::paint_state::push_event`'s existing
`texel = [u * texels_per_uv, (1 - v) * texels_per_uv]` convention (the
live viewport-painting path landed in 708ae8a). Without this flip, a mesh
baked by this pass and a mesh painted by the existing paint path would
disagree about which texture row is "up" — the bake would come out
mirrored relative to anything already painted. Checked by reading
`paint_state.rs` directly rather than assumed.

## Why the position pass also outputs a normal map (not in the task sketch)

The task brief's `bake_position_map` returns only positions. But
`ao::bake_ao_mesh` needs more than a ray *origin* per texel — the plane
path's hemisphere pole came from `cross(plane.u_axis, plane.v_axis)`
(user-supplied); a position map alone has no equivalent "which way is
up" for a mesh-fed texel. So `POSITION_BAKE_SHADER::cs_main` writes a
second `Rgba32Float` storage texture holding the winning triangle's face
normal (already computed in `GpuPosTri.normal`, carried through
unchanged from `position::build_pos_triangles`, which recomputes it from
triangle positions rather than trusting `mesh.normals` — the OBJ loader
can leave that array short, see `umber_mesh::load_obj`'s doc comment, the
same reasoning `ao::build_triangles` already uses). `AO_BAKE_SHADER`'s new
`cs_main_from_position` entry point reads it and builds a tangent frame
via Duff et al.'s branchless orthonormal-basis construction ("Building an
Orthonormal Basis, Revisited," JCGT 2017) — picked specifically for being
branchless and numerically stable at the south-pole case naive formulas
(e.g. `cross(n, vec3(0,0,1))`) fail on. AO is isotropic in `phi`, so any
consistent orthonormal frame gives the same result; which one doesn't
matter for correctness, only that it's well-defined for every normal.

## No CPU round-trip between the position and AO passes

The task brief's literal phrasing ("`bake_ao_mesh` ... runs
`bake_position_map` first, then feeds the position texture into the AO
pass") could be read as: call the public, CPU-readback-returning
`bake_position_map`, then re-upload its `Vec<f32>` into a fresh GPU
texture for the AO pass. That would be a wasteful and fragile round trip
(two readback/upload cycles, and the normal map would need rederiving a
second time to boot). Instead, `position::bake_position_and_normal`
(`pub(crate)`) runs the rasterization pass and returns the GPU-resident
`PositionMapGpu` (both textures, never copied to the CPU) directly;
`bake_position_map` (the public, brief-literal entry point) calls it and
then reads back only the position texture; `ao::bake_ao_mesh` calls the
*same* `bake_position_and_normal` and binds both textures straight into
`cs_main_from_position`. Both public entry points share one
implementation; neither mesh touches the CPU between the two passes in
the AO path.

## `bake_ao_mesh` returns full RGBA8, not `run`'s R-channel-only bytes

`ao::run` returns just the R channel because its parameter plane has no
concept of "uncovered" — every plane texel is raycast. `bake_ao_mesh`'s
texels can be genuinely uncovered (no UV triangle claims them), and an
uncovered texel (`cs_main_from_position` writes `(0,0,0,0)`) is
bit-for-bit identical to a *covered-but-fully-occluded* texel
(`(0,0,0,1)` in Rgba8Unorm, i.e. `rgb=0, a=255`) if only the R channel
is kept — the two meanings would be silently conflated. Returning full
RGBA8 keeps alpha as the coverage flag, distinguishable from occlusion by
construction.

## Single-mesh self-occlusion: how the GPU test gets an occluder without a second mesh

`bake_ao_mesh` takes one `MeshData` for both the UV-rasterization target
and the AO occlusion geometry — which is exactly how a real self-occlusion
AO bake should work (a mesh occludes itself; a separate "occluder mesh"
parameter was never part of the brief's signature). The GPU test needs a
floating occluder triangle that must *not* claim any position-map texel
itself (it isn't part of the surface being baked) but must still appear
in the raycast triangle list. Solved by giving the occluder vertices three
identical UVs (`[0,0]` repeated): its UV-space area is exactly zero, so
`cs_main`'s `abs(denom) >= DET_EPS` degeneracy check skips it during
rasterization — it never wins a position-map texel — while
`ao::build_triangles` (unchanged, reused as-is) still includes it in the
occlusion-raycast buffer `cs_main_from_position` tests every ray against.

## Geometry re-derivation for `bake_ao_mesh`'s GPU test

`quad_with_floating_occluder` (in `ao.rs`'s GPU test module) rotates
`ao::run`'s existing hand-derived scenario (see `LANDING_NOTES_AO.md`,
"Picking the GPU test's numbers") onto a mesh's own `+Z` face normal
instead of a plane's arbitrary `u_axis`/`v_axis`:

- The quad (`±30` world units, full `[0, 1]` UVs) has face normal `(0, 0,
  1)` for this vertex winding (`build_triangles`' cross-product
  computation, unchanged). Baked at `33x33` (not `32x32`): `(33 - 1) / 2 =
  16`, and `(16 + 0.5) / 33 = 0.5` exactly, so the center texel (16, 16)
  sits at UV `(0.5, 0.5)` *exactly*, mapping to the quad's exact world
  center `(0, 0, 0)` — not merely "close to it," which a `32x32` target
  (no texel centered on `0.5`) can't give. This also exercises the 8x8
  tile's partial-tail path (`33` isn't a multiple of `8`).
- The occluder is equilateral, circumradius `18`, centered at `(0, 0, 2)`
  — `2` world units above the quad along its own normal, directly over
  the quad's center. The inradius/circumradius hit/miss bound from
  `ao::run`'s original derivation (`H·tan(theta) <= 9` → guaranteed hit
  for *any* azimuth; `> 18` → guaranteed miss) is rotation-invariant, so
  it holds unchanged here even though the occluder's vertices now vary
  over `x, y` instead of `x, z`.
- `Duff`'s basis for `n = (0, 0, 1)` happens to come out as the identity
  (`b1 = (1, 0, 0)`, `b2 = (0, 1, 0)`) for this specific normal, so the
  local-to-world ray mapping is literally `dir = local_dir` here — but
  the derivation doesn't depend on that coincidence; the bound holds for
  any `phi` offset.
- Measured (same adapter as `ao::run`'s original test, Intel/Mesa/Vulkan):
  **center = 48, far_corner = 255** — identical to `ao::run`'s own
  measured values, as expected since it's the same ray geometry under a
  rotation.

## Testing

Non-GPU: `position::tests` (11) — `GpuPosTri`/`PositionUniform` layout
size-asserts, `build_pos_triangles`' normal computation + UV packing,
out-of-range position/UV index rejection (`MalformedMesh`, not a panic),
and four `validate` cases. `ao::tests` unchanged (10).

GPU (`--features gpu`): `position::tests::gpu` (5) —
`full_uv_quad_center_and_corner_match_analytic_position` (the task's test
(a): center exact, corner against the quad's own analytic affine map, not
the raw mesh vertex — a texel center never lands exactly on a UV edge at
finite resolution, documented in the test), `half_uv_quad_leaves_the_other_half_uncovered`
(test (c): a covered and an uncovered texel in the same bake, paired per
this slice's own review — asserting only "alpha 0" without a positive
"alpha 1" control elsewhere proves less than it looks like), plus
empty-mesh and over-budget rejection without touching the GPU pipeline.
`ao::tests::gpu` (+4) — `bake_ao_mesh_center_occluded_far_corner_clear`
(test (b)), zero-rays and position-map-error passthrough.

Full suite: `cargo fmt -p umber-bake -p umber-gpu -- --check`, `cargo
clippy -p umber-bake -p umber-gpu --all-targets -- -D warnings`, `cargo
clippy -p umber-bake --all-targets --features gpu -- -D warnings` all
clean. `cargo test -p umber-bake`: 20 (up from 11). `cargo test -p
umber-bake --features gpu`: 29 (up from 13). `cargo test -p umber-gpu`
and `--features gpu`: 37/50, unchanged — this slice touches no existing
`umber-gpu` test. `cargo build --workspace` green.

`cargo clippy -p umber-gpu --all-targets --features gpu -- -D warnings`
still fails on the same pre-existing `renderer.rs:883` issue
`LANDING_NOTES_AO.md`/`LANDING_NOTES_PAINT.md`/`LANDING_NOTES_TILE_POOL.md`
already flagged — confirmed unrelated: `renderer.rs` has no changes from
this slice. Not fixed here (outside this task's ownership).

## Not done in this slice (explicit non-goals, not gaps)

- **UV-overlap depth compare.** See "No depth test" above.
- **Acceleration structure.** `cs_main_from_position`'s occlusion raycast
  is still `ao::run`'s brute-force Möller–Trumbore over every triangle —
  unchanged from the plane slice, same perf budget caveat applies.
- **Cosine-weighted hemisphere sampling, cage controls, dilation,
  ignore-backfaces, match-by-name.** Unchanged non-goals from
  `LANDING_NOTES_AO.md`; this slice didn't touch `hemisphere_sample` or
  `hits_triangle`.
- **`R32Float`-only normal map, or a smaller normal format.** The normal
  texture is `Rgba32Float` like the position texture, for the same
  "no extra device feature" reasoning — not format-optimized (it's
  never read back to the CPU, only sampled by `cs_main_from_position`,
  so the bandwidth cost is paid once per bake, not per readback).
- **Dynamic shared-memory sizing based on actual device limits.** `TRI_CHUNK
  = 256` is a fixed compile-time constant sized against this session's
  measured `16384`-byte default-device floor; a device reporting a smaller
  limit than the WebGPU spec's own floor would be out of spec. No runtime
  adjustment is implemented.

## Reviewer checklist

- [ ] **Re-read "A note on how this landed"** — a second, unused draft
  briefly occupied `position.rs`/part of `bake_shaders.rs` during this
  session. It was deleted, not merged; confirm (e.g. via `grep
  "last-writer-wins\|break; // First hit"` in `bake_shaders.rs`) that
  what's on disk is this session's design, not a silent hybrid.
- [ ] **Re-derive `quad_with_floating_occluder`'s geometry independently**
  before trusting `bake_ao_mesh`'s GPU test — same highest-leverage check
  `LANDING_NOTES_AO.md` already asked for on the plane version, now also
  needing the rotation argument ("Duff's basis happens to be the
  identity for `n=(0,0,1)`") checked for sign errors.
  Measured values (`center=48`, `far_corner=255`) are recorded above.
- [ ] **`MAX_TRIS_PER_BAKE` is a Rust-side ceiling, not a shared-memory
  array size** — see "Why a shared-memory batching loop" above. Don't
  read the constant's name as a promise that `4096` triangles' worth of
  *anything* lives in workgroup storage at once; `TRI_CHUNK = 256` is the
  actual batch size.
- [ ] **The position pass writes a normal map the task brief didn't ask
  for.** Necessary for `bake_ao_mesh` to have a hemisphere pole per
  texel; flag if the design intended something else to supply it (e.g.
  per-vertex `mesh.normals`, which this deliberately avoids trusting).
- [ ] **`bake_ao_mesh` returns full RGBA8, `ao::run` still returns
  R-only** — an intentional asymmetry (see that section above), not an
  inconsistency to "fix" by making one match the other.
- [ ] **Last-writer-wins, no depth test, brute-force occlusion raycast,
  no acceleration structure** — all explicit non-goals, carried over
  from or consistent with `LANDING_NOTES_AO.md`.
- [ ] **Write-only `Rgba32Float` storage needing no device feature** was
  probe-verified on this sandbox's adapter (Intel/Mesa/Vulkan) before
  either shader was written. Re-confirm on the CI lavapipe/WARP path
  `LANDING_NOTES_AO.md` already flagged this concern for.

# umber-bake AO pass — landing notes

Wave-3 claw: the first real slice of the bake engine —
`crates/umber-bake/src/ao.rs` (new module) plus `AO_BAKE_SHADER`, a new
WGSL compute-raycast shader in `crates/umber-gpu/src/bake_shaders.rs`
(new file). Scope per docs/specs/requirements.md §3: "Compute-shader
raycast/bvh bake path (no RT hardware dependency)," AO listed first among
the P0 bakers. This lands the raycast core — hemisphere sampling,
brute-force Möller–Trumbore, occlusion accumulation — proven against a
flat parameter plane; it does **not** land UV→world position mapping or
any acceleration structure. Both are flagged below as explicit next-slice
work, not oversights.

## Why a parameter plane, not a UV-mapped position map

A real AO bake needs "world position of the texel at UV `(u,v)` on the
mesh being painted," which requires rasterizing the mesh's own UV layout
into a position map first — a whole separate piece of work (triangle
rasterization into texture space, interpolating world position across
each UV triangle). The task brief for this slice explicitly named this
simplification and asked for the raycast core to be proven independently
first: `AO_BAKE_SHADER` instead takes a `PlaneUniform` (world origin, two
axes, an extent) and computes each texel's world position as a
closed-form function of its grid coordinate — no mesh-UV lookup at all.

This composes cleanly with the next slice: the position-map work only
ever needs to replace "texel → world position" (currently the plane math
in `cs_main`); the hemisphere sampling, `hits_triangle`, and the
atomic-counter occlusion reduction around it are unchanged. Nothing here
is a dead end.

## What was built

**`crates/umber-gpu/src/bake_shaders.rs`** (new file, per the task's
scope split — I own only this new file in `umber-gpu`, not any existing
module)
- `AO_BAKE_SHADER` — the WGSL compute pass. One workgroup
  (`@workgroup_size(64)`, matching `shaders::PAINT_COMPUTE_SHADER`'s
  convention) per output texel (`workgroup_id.xy` is the texel
  coordinate; dispatched as `dispatch_workgroups(width, height, 1)`).
  Each invocation strides the ray budget by `local_invocation_index`,
  running `hits_triangle` (Möller–Trumbore, double-sided, brute force
  over every triangle in the storage buffer) and accumulating hits into
  a `var<workgroup> atomic<u32>` counter guarded by `workgroupBarrier()`;
  invocation 0 computes `ao = 1 - hits/rays` and writes it to all three
  color channels (alpha = 1).
- `hemisphere_sample(i, n)` — deterministic, stratified-in-`cos(theta)`,
  golden-angle-in-`phi` hemisphere direction. See "Sampling" below for
  why this isn't random or cosine-weighted.
- `hits_triangle` — standard Möller–Trumbore, `t ∈ (EPS, max_t)`.

**`crates/umber-gpu/src/lib.rs`** — one line added, `pub mod
bake_shaders;` (alphabetically before `camera`). No other line touched;
`paint.rs`/`paint_thread.rs`/`tile_pool.rs` are untouched, consumed only
through their existing public API (`PaintTarget`).

**`crates/umber-bake/src/ao.rs`** (new module)
- `PlaneDesc` — `#[repr(C)]`, `Pod`/`Zeroable`, 64 bytes, mirroring
  `paint::Dab`'s private-padding convention: `origin`/`u_axis`/`v_axis`
  are `[f32; 3]` each followed by a private `f32` pad (rounding each up
  to the WGSL `vec3<f32>`'s 16-byte alignment), then `extent: [f32; 2]`
  plus a trailing `[f32; 2]` pad. Build via `PlaneDesc::new`.
- `AoBakeParams { rays, max_distance, bias, plane }` —
  `AoBakeParams::new(max_distance, bias, plane)` defaults `rays` to
  `AoBakeParams::DEFAULT_RAYS` (16); override via struct-update syntax for
  a different count.
- `AoBakeError` (`thiserror`) — `EmptyMesh`, `InvalidRayCount`,
  `EmptyTarget { width, height }`, `Readback(String)`.
- `BakeTarget` — thin wrapper over `umber_gpu::PaintTarget` (see "Why
  `BakeTarget` wraps `PaintTarget`, not `TilePool`" below).
- `validate(mesh, params, width, height)` (private) — the four error
  checks above, factored out so they're unit-testable without a GPU
  device (see "Testing" below).
- `run(device, queue, mesh, params, width, height) -> Result<Vec<u8>,
  AoBakeError>` — validates, builds the GPU triangle buffer + uniforms,
  builds a fresh pipeline + bind group, dispatches, reads back, and
  returns the target's R channel as `width * height` bytes.

**`crates/umber-bake/Cargo.toml`** — added `wgpu`, `bytemuck`, `glam`,
`umber-gpu`, `umber-mesh` (all already pinned at the workspace level, no
version choices made here), plus the same `pollster`-behind-`gpu`-feature
pattern `umber-gpu`'s own `Cargo.toml` already uses.

**`crates/umber-bake/src/lib.rs`** — added `pub mod ao;` and `pub use
ao::{AoBakeError, AoBakeParams, BakeTarget, PlaneDesc};`. `run` is **not**
re-exported at the crate root — call it as `umber_bake::ao::run(..)`; a
bare `run` felt too generic a name to put at the crate root next to
`BakeMap` (existing `lib.rs` content, untouched).

## Signature deviation from the task sketch (and why)

The task brief's literal signature is `run(device, queue, mesh:
&MeshData, params) -> Result<Vec<u8>>` — no resolution parameter. But a
bake unavoidably needs an output size, and the task's own GPU-test
requirement ("bake 32x32 with 16 rays") only makes sense if the caller
can choose it. `run` here takes two additional parameters, `width: u32,
height: u32`, inserted after `params`. This is the same kind of
documented deviation `LANDING_NOTES_PAINT.md`/`LANDING_NOTES_TILE_POOL.md`
flagged for their own tasks — correctness-driven, not a stylistic choice.

## Why `BakeTarget` wraps `PaintTarget`, not `TilePool`

The task brief says "`BakeTarget` wraps the TilePool single-tile case."
Read literally, that would mean routing every bake through
`TilePool::get_or_create`, but `TilePool`'s tiles are hard-fixed at
`TILE_SIZE` (512×512) — a bake target's resolution is whatever the
caller asks for (the GPU test below bakes 32×32; a real asset might want
2K or 4K, neither a clean multiple of 512 necessarily). `tile_pool.rs`'s
own module docs describe its `HashMap<TileId, PaintTarget>` as wrapping
"each entry a full-size `PaintTarget`" — i.e. `PaintTarget` *is* "the
tile-pool's single-tile case," conceptually, independent of the pool's
512-grid bookkeeping on top of it. `BakeTarget` reuses that single-tile
building block directly (`PaintTarget::new(device, width, height)`,
arbitrary dimensions) rather than forcing bake output through the pool's
fixed-512 addressing, which would be actively wrong for a 32×32 test
target. Flagging this explicitly since it's a interpretation call on
ambiguous brief wording, not a literal implementation of "wraps TilePool."

## Storage-texture access: `write`, not `read_write` — no device feature needed

`paint::PaintCompositor` needs `read_write` storage-texture access on
`Rgba8Unorm` because it blends into existing texel contents
(premultiplied-alpha-over), which requires `wgpu::Features::
TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` on this adapter (documented in
`LANDING_NOTES_PAINT.md`'s "the landmine" section). This AO pass writes
every texel exactly once with no dependency on its prior value, so it
only needs `write`-only storage-texture access
(`wgpu::StorageTextureAccess::WriteOnly`) — which is a **core** WebGPU
capability on `Rgba8Unorm`, not the non-portable adapter extension.
`ao::tests::gpu::try_request_device` requests a plain
`wgpu::DeviceDescriptor::default()` (no required features at all) and the
GPU test passes — confirming this empirically, not just per spec text.
This is a real simplification over the paint pass, not an oversight:
there was no reason to inherit paint's device-feature requirement for an
operation that never needs to read the texture it's writing.

## `R8`: still `Rgba8Unorm`, not a real single-channel format

Per the task brief: `BakeTarget` is `Rgba8Unorm` under the hood (reusing
`PaintTarget` verbatim, see above); `run` writes the AO value into all
three color channels (plus alpha = 1) and extracts just the R channel on
readback (`rgba.chunks_exact(4).map(|px| px[0])`), returning `width *
height` bytes. A real `R8Unorm` storage texture would halve the memory
and bandwidth this pass uses — not done here since `PaintTarget` doesn't
expose a format choice, and changing that is `paint.rs`'s call, not this
task's (see "ownership" in the task brief). Flagged as a follow-up
candidate if `PaintTarget`/`BakeTarget` diverge enough to be worth a
format parameter.

## Sampling: deterministic stratified hemisphere, not cosine-weighted, not random

`hemisphere_sample(i, n)` stratifies `cos(theta)` uniformly across `[0,
1)` (`cos_theta = 1 - (i + 0.5) / n`) — uniform over solid angle, not the
cosine-weighted (Lambertian-correct) distribution a production AO baker
would eventually want. `phi` walks the golden angle
(`2π · fract(i · φ⁻¹)`), giving low-discrepancy azimuthal coverage with
no RNG state. Two reasons for this choice, not just "simplest thing":

1. **No RNG needed at all** — a per-invocation seeded RNG (e.g. PCG) is
   extra shader complexity this slice doesn't need to prove the raycast
   core works. Deterministic sampling is also what makes the GPU test
   below hand-verifiable: the exact ray set for `rays = 16` is known at
   write time, so the test's occlusion bounds are derived from closed-form
   geometry, not "run it and see."
2. **Uniform-over-solid-angle (not cosine-weighted) was chosen so the
   *inscribed-circle* argument in "Picking the GPU test's numbers" below
   holds for every ray index via `cos(theta)` alone**, independent of
   `phi`. Switching to cosine-weighted sampling later is a one-line
   change to `hemisphere_sample`'s `cos_theta` formula; the rest of the
   pipeline doesn't care which hemisphere distribution feeds it.

## Perf budget (read before pointing this at a real mesh)

No BVH, no spatial grid: every ray tests every triangle
(`params.tri_count` iterations per ray, inside the per-texel ray loop).
Total work is `texels × rays × triangle_count`. At the default 16 rays on
a 512² target, that's `512² × 16 × N` ray-triangle tests — budgeted for
`N < 5,000` triangles (the task brief's own stated target), i.e. roughly
2×10¹⁰ ray-triangle tests worst case, which is already GPU-compute-bound
territory and will not scale to production asset triangle counts (tens
of thousands to millions). The shader's inner loop
(`for (var t: u32 = 0u; t < params.tri_count; t = t + 1u)`) is exactly
where a BVH traversal would replace the brute-force scan in the next
slice — the hemisphere sampling, atomic reduction, and texel addressing
around it would not need to change.

## Picking the GPU test's numbers (read this before trusting the assertions)

The task asked for: "mesh = one big triangle floating above the plane
pointing down at it; bake 32x32 with 16 rays; assert center texel heavily
occluded (value < 64), far corner texel unoccluded (>= 200)." Hitting
both bounds reliably — not just "probably" — needed the geometry to be
derived, not guessed, because a single triangle's hit/miss pattern
against 16 fixed (not random) ray directions depends on both `cos(theta)`
*and* `phi` per ray, and naive round numbers (e.g. a triangle just
"somewhere above" the plane) left too much margin risk.

**The derivation:**
- Triangle: equilateral, lying flat at height `H = 2` above the plane,
  centered on the plane's origin (so the ray origin for the center texel
  and the triangle's centroid coincide in the horizontal plane).
  Circumradius `18`, hence inradius `9` (inradius = circumradius / 2 for
  an equilateral triangle). Vertices: `(0, 2, 18)`, `(-15.588457, 2, -9)`,
  `(15.588457, 2, -9)` (120° apart on the circumradius-18 circle).
- For a ray at polar angle `theta` from vertical, its horizontal
  displacement when it reaches height `H` is `H · tan(theta)`,
  **independent of its azimuth `phi`**. So:
  - Any ray with `H · tan(theta) ≤ 9` (the inradius) is **guaranteed** to
    land inside the triangle's inscribed circle — a hit, for *every*
    `phi`.
  - Any ray with `H · tan(theta) > 18` (the circumradius) is
    **guaranteed** to land outside the triangle entirely — a miss, for
    *every* `phi`.
  - In between is azimuth-dependent (near a vertex bulge vs. near an
    edge gap).
- For the 16 stratified rays (`cos_theta_i = 1 - (i + 0.5)/16`), computing
  `H · tan(theta_i)` for each `i` shows indices `0..=12` (13 rays) satisfy
  the inradius bound (`≤ 9`) — **guaranteed hits** regardless of `phi`.
  Index `13` falls in the uncertain band; indices `14, 15` exceed the
  circumradius bound entirely (and are additionally clipped by
  `max_distance`, see below) — guaranteed misses.
- Worst case (index 13 misses): `13/16` hit fraction → `ao = 3/16 ≈
  0.1875` → byte `≈ 48`. Best case (index 13 hits): `14/16` → byte `≈
  32`. Either way, **comfortably under the 64 threshold** — confirmed:
  the actual measured byte value is **48** (see "Live-verified" below).
- `max_distance = 15`: chosen so every guaranteed-hit ray (`i ≤ 12`,
  worst-case travel distance `t = H / cos(theta_12) ≈ 9.14`) is well
  within budget, while rays beyond the circumradius bound (`i = 14, 15`,
  `t ≈ 42.6` and `t ≈ 128`) get clipped before they'd matter anyway.
- Far-corner texel: the plane's extent is `[60, 60]`, so the texel at
  grid index `(0, 0)` on a 32×32 target sits roughly `41` world units
  (horizontal) from the triangle's centroid — comfortably more than the
  triangle's circumradius (`18`) plus `max_distance` (`15`) = `33`. By the
  triangle inequality, the *nearest possible point* of the triangle to
  that corner is at least `41 - 18 = 23` units away — which exceeds
  `max_distance = 15`. So **no ray from that corner can geometrically
  reach the triangle at all**, regardless of direction: this isn't a
  probabilistic "probably clear," it's a proof of zero hits. Confirmed:
  the measured byte value is **255**.

This geometry is only correct for *this* triangle/plane/ray-count
combination — it is not a general-purpose test fixture. If a future
reviewer changes `rays`, the plane's `extent`, the triangle's height, or
its circumradius, this derivation needs to be redone (the comment block
directly above the GPU test in `ao.rs` repeats the key numbers so a
future edit sees the dependency without having to find this file).

## Live-verified in this sandbox

Same adapter as the other Wave-2/3 claws in this repo (a real Intel
integrated GPU via Mesa/Vulkan, not a software rasterizer — see
`LANDING_NOTES_PAINT.md` for the `adapter.get_info()` details). Measured
via a temporary `eprintln!` (removed before landing):
`center = 48, far_corner = 255` — exactly the worst-case bound derived
above for `center` and exactly the proven value for `far_corner`, not
merely "within tolerance."

## Testing

Non-GPU (`ao::tests`, 11 tests): three layout-size assertions
(`PlaneDesc` 64 bytes, `AoUniform` 80 bytes, `GpuTriangle` 64 bytes —
mirroring `paint::Dab`'s `size_of` const-assert convention),
`PlaneDesc::new` field round-trip, `AoBakeParams::new`'s default-rays
value, `build_triangles`' face-normal computation against a known
right-angle triangle, and four `validate` cases (empty mesh, zero rays,
zero-sized target, and the accepting/happy path) — made possible by
factoring `validate` out of `run` specifically so these don't need a GPU
device.

GPU (`ao::tests::gpu`, feature `gpu`, 2 tests, skip-if-no-adapter —
matching `paint.rs`'s more permissive convention over
`paint_thread.rs`/`tile_pool.rs`'s stricter "panic if no adapter" one,
since this is a new crate without an established convention of its own
yet):
- `ao_bake_center_occluded_far_corner_clear` — the hand-derived scenario
  above.
- `ao_bake_rejects_empty_mesh_without_touching_the_gpu_pipeline` — `run`
  on an empty `MeshData` returns `AoBakeError::EmptyMesh` without
  panicking inside pipeline/bind-group construction (confirms `validate`
  is actually wired into `run`, not just unit-tested in isolation).

All green: `cargo fmt -p umber-bake -p umber-gpu -- --check`, `cargo
clippy -p umber-bake -p umber-gpu --all-targets -- -D warnings`, `cargo
clippy -p umber-bake --all-targets --features gpu -- -D warnings`,
`cargo test -p umber-gpu` (37, unchanged from before this slice), `cargo
test -p umber-gpu --features gpu` (50, unchanged), `cargo test -p
umber-bake` (11, up from the pre-existing 1), `cargo test -p umber-bake
--features gpu` (13), `cargo build --workspace`.

`cargo clippy -p umber-gpu --all-targets --features gpu -- -D warnings`
(umber-gpu's *own* test target under its *own* `gpu` feature, as opposed
to umber-bake depending on it) still fails on the same pre-existing
`renderer.rs:883` issue `LANDING_NOTES_PAINT.md`/`LANDING_NOTES_TILE_POOL.md`
already flagged — confirmed via `git diff HEAD -- crates/umber-gpu/src/renderer.rs`
showing no changes to that file from this slice. Not fixed here
(`renderer.rs` is outside this task's ownership).

## Not done in this slice (explicit non-goals, not gaps)

- **UV→world position map.** The single biggest piece of follow-up work
  — see "Why a parameter plane" above. Without it, this bake path cannot
  yet produce a texture a paint layer actually uses.
- **Acceleration structure (BVH/grid).** See "Perf budget" above — the
  brute-force inner loop is a known, intentional limit for this slice's
  mesh-size budget.
- **Cage controls, dilation, ignore-backfaces, "low poly as high,"
  match-by-name** — all named in requirements §3's "common settings" row
  but out of scope for proving the raycast core.
- **Backface culling.** `GpuTriangle.normal` is computed and carried in
  the storage buffer for exactly this future use, but `hits_triangle` is
  currently double-sided (any `|a| ≥ EPS` hit counts, regardless of which
  face was struck).
- **Golden-image testing.** Like `LANDING_NOTES_PAINT.md` flagged for the
  paint pass, this is hand-picked-pixel testing (two specific texels),
  not a `golden::compare_rgba8` reference-image comparison. The same
  `golden.rs` harness this flagged as the right long-term home for paint
  applies here too.
- **Pipeline caching across calls.** `run` builds a fresh
  `wgpu::ComputePipeline`/`BindGroupLayout` every call (see `run`'s doc
  comment) rather than an `AoBaker`-style persistent struct
  (`PaintCompositor`'s pattern). Fine for a one-shot bake entry point;
  revisit if a caller ends up invoking `run` in a hot loop (e.g.
  auto-rebake-on-parameter-change, requirements §3's P1 row).

## Reviewer checklist

- [ ] **Re-derive "Picking the GPU test's numbers" independently** before
  trusting the GPU test — this is the highest-leverage thing to check by
  hand, since the whole test's reliability rests on it, not on tolerance
  margins. The measured values (`center = 48`, `far_corner = 255`) are
  recorded above for a sanity cross-check against your own re-derivation.
- [ ] **`BakeTarget` wraps `PaintTarget`, not `TilePool`** — a judgment
  call on ambiguous brief wording (see that section above). Confirm this
  reading before building the next slice on top of `BakeTarget`, since a
  `TilePool`-routed version would behave very differently (512-quantized
  resolution) from what's landed here.
- [ ] **`run` gained `width: u32, height: u32` parameters** not in the
  task's literal signature — necessary (there was no other way to choose
  a bake resolution), but flag if some other part of the design was
  supposed to supply resolution a different way (e.g. a `BakeTarget`
  passed in by the caller instead of width/height, which `run` would
  construct internally).
- [ ] **No backface culling, no acceleration structure, no UV-mapped
  position map** — all explicit non-goals above, not gaps to silently
  paper over in review. Don't point this `run` at a real multi-thousand-
  triangle mesh and expect it to be fast; it isn't, by design, yet.
- [ ] **Hemisphere sampling is uniform-over-solid-angle, not
  cosine-weighted** — correct for proving the raycast core, not
  physically correct for a production AO result (which should weight
  grazing rays less). Flag if a reviewer expects the output values
  themselves (not just hit/miss structure) to already be
  production-quality AO.
- [ ] **`write`-only storage-texture access needing no device feature**
  (see that section above) was verified empirically on this sandbox's
  adapter (Intel/Mesa/Vulkan), the same one `LANDING_NOTES_PAINT.md`
  flagged as a real GPU, not lavapipe. Re-confirm on the CI
  lavapipe/WARP path requirements §10 specs before relying on this
  working everywhere.
- [ ] **`run` is not re-exported at the `umber-bake` crate root** — call
  it as `umber_bake::ao::run`. Reconsider if the next slice's API wants a
  flatter surface.

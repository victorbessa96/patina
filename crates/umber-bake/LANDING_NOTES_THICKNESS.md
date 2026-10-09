# umber-bake thickness pass — landing notes

Wave-3 claw: inward-raycast thickness (local solid depth) baking —
`crates/umber-bake/src/thickness.rs` (new module) plus
`THICKNESS_BAKE_SHADER`, a new WGSL compute pass appended to
`crates/umber-gpu/src/bake_shaders.rs` (append-only; no existing shader
touched). Scope per docs/specs/requirements.md §3 (thickness is a P0
baker). Composition mirrors `ao::bake_ao_mesh` exactly: rasterize the
mesh's own UV layout into a position/normal map via
`position::bake_position_and_normal`, then bind both textures read-only
into the second pass — only the second pass differs (minimum inward hit
distance instead of outward occlusion fraction, so the 64-invocation
workgroup fan-out collapses to a serial per-texel loop; see "Why
`@workgroup_size(1)`" below).

## What was built

**`crates/umber-gpu/src/bake_shaders.rs`** — appended
`THICKNESS_BAKE_SHADER` (see that constant's doc comment for the full
contract). One workgroup (`@workgroup_size(1)`) per texel, dispatched as
`dispatch_workgroups(width, height, 1)`; `workgroup_id.xy` is the texel
coordinate into all three textures. Binding numbers (`0/1/2` plus `4/5`)
deliberately mirror `AO_BAKE_SHADER::cs_main_from_position`'s set
(including the skipped `3`, the plane-path `dims` uniform this mesh-fed
pass has no use for) so the two raycast passes stay grep-comparable.

**`crates/umber-bake/src/thickness.rs`** (new module, shaped like
`curvature.rs`)
- `ThicknessParams { max_distance, bias, rays }` —
  `ThicknessParams::DEFAULT_RAYS` (`16`, matching
  `AoBakeParams::DEFAULT_RAYS`), `DEFAULT_MAX_DISTANCE` (`10.0`),
  `DEFAULT_BIAS` (`0.01`), `new(max_distance, bias)` (rays default to 16,
  struct-update to override — the `AoBakeParams::new` convention), and
  `Default`.
- `ThicknessBakeError` (`thiserror`) — `InvalidRayCount`,
  `EmptyTarget { width, height }`, `PositionMap(#[from] PositionMapError)`,
  `Readback(String)`.
- `bake_thickness_mesh(device, queue, mesh, width, height, params)
  -> Result<Vec<u8>, ThicknessBakeError>` — validates rays + dims, runs
  `bake_position_and_normal`, builds the triangle buffer + fresh pipeline
  + bind group, dispatches, and returns full `width * height * 4` RGBA8
  bytes (`rgb = clamp(min_hit / max_distance, 0, 1)` grayscale, `a` =
  coverage). Reuses `ao::BakeTarget` for the output texture +
  256-byte-row-pitch readback rather than duplicating that machinery, the
  same reuse `curvature` makes over `ao`.

**`crates/umber-bake/src/lib.rs`** — two lines added, `pub mod thickness;`
plus its re-export. No other line touched.

**`crates/umber-bake/src/position.rs`** — untouched (same as the
curvature slice: `TEXTURE_BINDING` was already on both targets, so no
change was needed).

## Estimator + sources

Per covered texel, over `rays` deterministic stratified inward hemisphere
directions (the same `hemisphere_sample` + Duff-et-al. tangent frame as
AO, with the pole negated to `-normal`):

```text
min_dist = min over rays, over triangles of hit t in (EPS, max_distance)
thickness = clamp(min_dist / max_distance, 0, 1)
```

This is the standard raycast-thickness estimator behind mesh-map bakers:
where AO asks "how much of the sky is visible from here," thickness asks
"how far before the ray exits the solid again." Taking the *minimum*
across the ray set (rather than the mean) approximates the most-direct
through-path — for parallel faces the near-vertical ray wins and the
result is the face gap divided by `cos(tilt)`, i.e. essentially the gap
itself. Points of comparison, not code sources (no vendored code; the
kernel is `hemisphere_sample` + Möller–Trumbore + a scalar min, all
written from the math):

- Möller–Trumbore ("Fast, Minimum Storage Ray-Triangle Intersection,"
  1997) is the intersection routine, shared verbatim in structure with
  `AO_BAKE_SHADER::hits_triangle` (this pass's `ray_hit_distance` is its
  distance-returning analog).
- Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT 2017) is
  the tangent frame, shared verbatim with AO (any orthonormal frame
  works; thickness, like AO, is isotropic in `phi`).
- Substance Painter / Marmoset Toolbag thickness bakers solve the same
  artist need (thickness-driven subsurface/translucency masks) with the
  same "cast inside, nearest opposite hit" structure; the miss-means-thick
  convention below matches what those tools' white-means-thick ramps imply.

Deliberate simplifications vs. production bakers: brute-force triangle
scan (no BVH — same `texels * rays * tri_count` budget as AO, same
few-thousand-triangle ceiling), double-sided hits (no backface culling,
matching AO), uniform-over-solid-angle sampling (not cosine-weighted —
inherited from AO's fixed sample set for determinism), no cage support.

## The miss-means-thick convention and its artifact (read before using this on open meshes)

A per-ray miss contributes nothing — it never lowers `min_dist`, which
starts at `max_distance`. A texel whose *every* ray misses therefore bakes
`1.0`. This is a deliberate convention, documented in the shader's own doc
comment (not just here): a miss means the ray escaped through open
geometry, so the surface reads as "no opposite face within range," i.e.
maximally thick. The alternative (bake `0.0`, "paper thin") would be worse:
a single-sided quad with no backside would then read as zero-thickness
everywhere and drive subsurface/translucency masks to exactly the wrong
extreme.

The artifact this accepts: **open meshes read thick.** Any surface with no
opposite face within `max_distance` — a single-sided plane, an unclosed
rim, the outside of a thin-walled vessel scanned from the exterior —
bakes white (`255`), indistinguishable from genuinely deep solid. This is
inherent to inward raycasting without a closed-solid prior, not a bug in
the min-reduction; reviewers should treat "white" as "no return within
range" rather than "measured deep." Mitigations, all deferred: cage
controls (requirements §3's "common settings" row) to bound the search,
two-sided (enter+exit) thickness for walled geometry, and documenting the
ramp direction in the UI so artists expect white-means-open.

A second, smaller artifact: the minimum over a hemisphere *underestimates*
true normal-direction thickness on curved interiors (a tilted ray can clip
a nearby curl at shorter `t` than the straight-through path). Averaging
would overestimate instead; the min was chosen because underestimation
fails safe for the primary use (subsurface radius) and matches the
"nearest opposite face" mental model. Flagged, not fixed.

## Ray count / bias tradeoffs

- **Rays (default 16):** more rays tighten the min toward the true
  nearest-opposite distance (denser hemisphere coverage finds the most
  direct path) at linear cost (`rays * tri_count` tests per texel). Fewer
  rays risk overestimation where the only direct path falls between
  samples — but for near-parallel faces even 4 rays suffice (ray 0 alone
  carries the min; see "Picking the GPU test's numbers"). 16 inherits AO's
  default so both bakers share one perf story.
- **Bias (default 0.01, applied *inward*):** the mirror of AO's outward
  bias, and the sign matters — see the shader doc comment's "Why into the
  mesh" section. Starting outside and casting inward would cross the
  originating surface at `t ~= bias` and bake false near-zero thickness;
  starting just inside puts the origin triangle behind every inward ray.
  Too-large bias eats real thin geometry (a `0.1` bias on a `0.15` wall
  reports `0.05`); too-small bias (or `0.0`) still works via the `(EPS,
  max_t)` open-interval self-hit guard, but leaves adjacent-triangle
  grazing hits at tiny `t` un-suppressed. `0.01` matches AO's magnitude.
- **`max_distance` is the ramp, not just a cutoff:** it both bounds the
  search (perf) and normalizes the output, so doubling it halves every
  gray value. Callers should set it to the thickest feature they care
  about, not "something huge for safety" — anything beyond it saturates
  to white by design.

## Why `@workgroup_size(1)` (not AO's 64-wide fan-out)

AO's 64-invocation workgroup with an `atomic<u32>` hit counter works
because a hit *count* is an integer with an atomic add. A minimum
*distance* is a float with no atomic-min in core WGSL; fanning out would
need a 64-entry shared-memory array plus a manual barrier-guarded
reduction, for per-texel work (`16 * tri_count` brute-force tests at this
slice's mesh budget) that is trivially serial. So each texel's ray loop
is serial in one invocation — the same dispatch shape as the curvature
pass, with AO's sampling/intersection/basis machinery carried over
unchanged. If a BVH slice later makes per-ray work dominant, revisiting
the fan-out is fair game; nothing in the uniform or binding layout
precludes it.

## Picking the GPU test's numbers (read this before trusting the assertions)

- **Slab:** front `[-1, 1]` quad at `z = 0` (full `[0, 1]` UVs, `+Z` face
  normal) + back `[-2, 2]` quad at `z = -2` (gap `2.0` behind the front
  face along its inward `-Z`), all back-quad UVs collapsed to one point
  so the position pass skips it (zero-UV-area, `abs(denom) < DET_EPS`).
  `max_distance = 10.0`, `bias = 0.01`, `rays = 16`.
- The min across the set is ray 0's hit (largest `cos(theta) = 1 - 0.5/16
  = 0.96875`): `t = (2.0 - 0.01) / 0.96875 ~= 2.054`, thickness `~=
  0.2054`, byte `~= 52`. Parallel planes make this position-independent.
- The back face is oversized (`±2` vs front `±1`) so ray 0 *always* hits:
  max sideways drift `~0.51` from any front texel (`|x| <= 1`) lands
  within `±1.51 < ±2`. Every covered texel is therefore guaranteed the
  same min regardless of azimuth — no edge falloff, no probabilistic
  argument.
- The `[20, 90]` band (`0.08..0.35`) admits tilt/quantization slack while
  pinning the result away from both `0` and `255`, and far below the open
  quad's `255`.
- **Open quad:** identical front quad, no back face — every inward ray
  misses, every covered texel must read exactly `255`. Same params/size
  as the slab, so together they prove the baker measures the gap rather
  than returning a constant.

## Live-verified in this sandbox

Same adapter as the other Wave-2/3 claws in this repo. Measured via a
temporary `eprintln!` (removed before landing): slab center texel
`[52, 52, 52, 255]`, min `52`, max `52` — *every* texel exactly `52`,
matching the hand derivation (`2.054 / 10.0 * 255 ~= 52.4`) to the byte,
not merely "within tolerance." Open quad: all `255` as predicted.

## Testing

- `slab_bakes_gap_over_max_distance_everywhere` (gpu): 32×32, *every*
  texel asserted `a == 255`, `r ∈ [20, 90]`, grayscale (`r == g == b`).
- `open_quad_bakes_full_thickness_everywhere` (gpu): 32×32, *every* texel
  asserted `a == 255`, `r == 255`.
- `..._rejects_zero_rays_without_touching_the_gpu_pipeline`,
  `..._rejects_zero_sized_target`, `..._propagates_position_map_errors`
  (gpu): validation before any GPU work + error passthrough, mirroring
  the AO/curvature mesh tests.
- Unit (no GPU): both layout-size asserts (`GpuTriangle` 64,
  `ThicknessUniform` 16), `new`'s 16-ray default, `Default` matching
  `new` with the default constants, `build_triangles` face normal,
  three `validate` cases.

## Reviewer checklist

- [ ] `bake_shaders.rs` diff is append-only (existing shaders byte-identical).
- [ ] `position.rs` untouched — confirm `TEXTURE_BINDING` was already on
      both targets rather than trusting this note.
- [ ] Inward hemisphere: negate-pole (`- local_dir.z * normal`) intact;
      flipping the sign turns thickness into outward-AO-with-min and the
      slab test (which would then read all-miss `255`) fails.
- [ ] Inward bias (`- bias * normal`) intact; flipping it bakes false
      near-zero and the slab band fails from below.
- [ ] Miss-means-thick (`min_dist` init at `max_distance`) intact; the
      open-quad test pins `255`, so an init-at-zero regression fails it.
- [ ] No `unwrap`/`expect` outside `#[cfg(test)]` modules.
- [ ] `cargo fmt --check`, `cargo clippy -D warnings` clean under both
      default and `--features gpu` for `umber-bake`; `umber-gpu` default
      clean. (`umber-gpu --features gpu` still fails on the same
      pre-existing `renderer.rs:883` useless-comparison lints
      `LANDING_NOTES_AO.md` already flagged — confirmed via `git diff`
      showing no changes to that file from this slice. Not fixed here.)
- [ ] Full suites green: `umber-bake` 32/32 default (24 pre-existing + 8
      new) and 50/50 gpu (37 pre-existing + 13 new: 8 unit + 5 gpu);
      `umber-gpu` 50/50 gpu unchanged.
- [ ] The open-mesh-reads-thick artifact above is acceptable as a
      documented convention (with cage/two-sided follow-ups), not a
      blocker.

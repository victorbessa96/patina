# umber-bake curvature pass — landing notes

Wave-3 claw: signed screen-space curvature baking —
`crates/umber-bake/src/curvature.rs` (new module) plus
`CURVATURE_BAKE_SHADER`, a new WGSL compute pass appended to
`crates/umber-gpu/src/bake_shaders.rs` (append-only; no existing shader
touched). Scope per docs/specs/requirements.md §3 (curvature is a P0
baker). Composition mirrors `ao::bake_ao_mesh` exactly: rasterize the
mesh's own UV layout into a position/normal map via
`position::bake_position_and_normal`, then bind both textures read-only
into the second pass — only the second pass differs (4-neighborhood
curvature estimator instead of hemisphere raycasting, so no triangle
buffer, no raycount uniform, no atomics).

## What was built

**`crates/umber-gpu/src/bake_shaders.rs`** — appended
`CURVATURE_BAKE_SHADER` (see that constant's doc comment for the full
contract). One workgroup (`@workgroup_size(1)`) per texel, dispatched as
`dispatch_workgroups(width, height, 1)`; `workgroup_id.xy` is the texel
coordinate into all three textures. Bindings pack contiguously
(`0` = write-only `Rgba8Unorm` output, `1` = `CurvatureParams` uniform,
`2/3` = read-only `texture_2d<f32>` position/normal) rather than
inheriting the AO pass's numbering, since there is no triangle storage
buffer on this pass.

**`crates/umber-bake/src/curvature.rs`** (new module)
- `CurvatureParams { strength }` — `CurvatureParams::DEFAULT_STRENGTH`
  (`1.0`), `new`, and `Default`.
- `CurvatureBakeError` (`thiserror`) — `EmptyTarget { width, height }`,
  `PositionMap(#[from] PositionMapError)`, `Readback(String)`.
- `bake_curvature_mesh(device, queue, mesh, width, height, params)
  -> Result<Vec<u8>, CurvatureBakeError>` — validates, runs
  `bake_position_and_normal`, builds a fresh pipeline + bind group,
  dispatches, and returns full `width * height * 4` RGBA8 bytes
  (`rgb = (curv + 1) / 2` grayscale, `a` = coverage). Reuses
  `ao::BakeTarget` for the output texture + 256-byte-row-pitch readback
  rather than duplicating that machinery.

**`crates/umber-bake/src/lib.rs`** — two lines added, `pub mod curvature;`
plus its re-export. No other line touched.

**`crates/umber-bake/src/position.rs`** — untouched. The task brief
allowed adding `TEXTURE_BINDING` if sampled `textureLoad` needed it, but
the flag is already there: both targets are created with
`STORAGE_BINDING | TEXTURE_BINDING | extra` (position.rs's `make_target`),
with the comment explicitly naming the AO second pass as the consumer.
This pass consumes the same views through the same usage flag, so no
change was needed.

## Estimator + sources

Per covered texel, over the in-bounds, *covered* 4-neighbors:

```text
k_i = dot(n_i - n_0, normalize(p_i - p_0)) / max(length(p_i - p_0), 1e-6)
curv = clamp(-strength * mean(k_i), -1, 1)
```

This is the standard discrete directional-curvature estimator behind
screen-space "curvature from normal/depth buffer" passes: where the
surface bends, neighboring normals differ along the direction of travel,
and dividing by the travel distance turns that difference into a
curvature-scale quantity. Averaging four axis directions (rather than
fitting a full curvature tensor over a multi-ring neighborhood) trades
angular resolution for a 4-`textureLoad` kernel with no sampler and no
filtering — appropriate for a first slice whose job is crevice/edge
masking, not metrology. Points of comparison, not code sources (no
vendored code; the ~15-line kernel is written from the math):

- Rusinkiewicz's per-vertex tensor fitting ("Estimating Curvatures and
  Their Derivatives on Triangle Meshes," 2004) is the object-space
  reference for what "curvature from normals + positions" means; this
  pass is its screen-space 4-sample degenerate cousin.
- Blender's "Pointiness" (Cycles OSL) and "Cavity" (bake type) outputs
  solve the adjacent user need (edge/crevice masks) with different
  machinery (ambient-occlusion-like raycasts); the sign convention below
  is chosen to match what artists expect from those tools' grayscale
  ramps (recesses bright, bulges dark).
- "Curvature from screenspace normal buffer" post-process passes (e.g.
  the LWJGL / LearnOpenGL-style edge-detection lineage) use the same
  `dot(dn, dir)` structure on a normal G-buffer; this pass additionally
  divides by distance so the response is scale-aware rather than
  texel-size-aware.

Deliberate simplifications vs. those references: single ring (no
multi-scale response — wide bevels read weaker than tight creases at the
same angle), axis-aligned 4-tap (diagonal features respond ~`1/√2` as
strongly), face normals from the position pass (faceted input —
per-vertex smooth normals would need a second position-pass output).

## Sign convention (MeshLab: convex = negative = darker)

The raw `dot(dn, dir) / len` term is *positive* on an outward bulge —
sphere check: `n(p) = p/R`, so `n_i - n_0 = (p_i - p_0)/R` points along
the travel direction and the dot product is `|dp|/R > 0`. The shader's
leading minus flips it so convex (outward-bulge) regions read *negative*
and bake *darker*, concave crevices positive/brighter. This is MeshLab's
convention (mean curvature negative on convex parts with outward
normals), and it matches the artist expectation that curvature maps read
like cavity maps (dark = exposed bump, bright = recess). The minus is
documented in the shader's own doc comment, not just here, because it is
otherwise an inviting "simplification" for a future reader to delete —
deleting it inverts every downstream mask (edge-wear becomes
crevice-dirt). The tent GPU test pins the convention: it asserts the
convex ridge bakes *darker* than the flat slopes, so a sign flip fails
the suite, not just a review.

## UV-seam artifact note (known, accepted for this slice)

UV seams are cuts in the position/normal map: two texels adjacent on the
mesh can be far apart in UV, and — worse — a texel on one side of a seam
can sit UV-adjacent to a texel from a *different* part of the mesh.
Wherever that happens, this estimator reads a bogus neighbor (wrong
position, wrong normal) and bakes a dark/bright fringe along the seam
that has no geometric meaning. Mitigations, all deferred, not oversights:

- The tent test's ridge is deliberately an *interior* UV line (`u = 0.5`,
  shared edge, no seam) — the suite proves the estimator correct where
  UV adjacency equals mesh adjacency, and says nothing about seams.
- Real fixes: bake with dilated UV island margins and run a post-bake
  dilation/pull-through pass over uncovered texels (on the roadmap per
  the crate docs: "dilation" is listed Wave-3 scope), or detect seam
  texels (normal/position discontinuity vs. mesh adjacency) and fall back
  to fewer taps there. Both need the dilation pass first.
- Smoothing the input (per-vertex normals out of the position pass)
  would shrink but not remove the fringe; don't mistake it for a fix.

A second, smaller artifact: single-texel-wide UV islands have no valid
neighbors and bake flat mid-gray by construction (`count == 0 → curv =
0`), silently dropping real high-frequency curvature. Same dilation
roadmap item addresses it.

## Testing

- `flat_quad_bakes_mid_gray_everywhere` (gpu): 32×32 full-UV quad, *every*
  texel asserted `rgb ∈ [120, 136]`, `a == 255`. On a flat quad every
  neighbor pair has bitwise-identical face normals, so `dn = 0` exactly
  and the only tolerance needed is `Rgba8Unorm` rounding of `0.5`
  (`127` vs `128`) plus float slack.
- `tent_crease_bakes_darker_than_flat_slopes` (gpu): roof mesh (two quads,
  convex ridge at interior `u = 0.5`, outward windings verified by hand —
  left `(-0.707, 0, 0.707)`, right `(0.707, 0, 0.707)`). Ridge falls on
  the column 15/16 boundary at 32×32; both columns sample `|dn| ≈ 1.41`
  over a `≈0.06` step (`k ≈ 22`, saturates the clamp even averaged 1:3
  with flat neighbors), so crease texels assert `< 100` across rows
  8/16/24 while slope columns assert the mid-gray band — plus a direct
  `flat - crease ≥ 28` difference assertion. Bounds throughout, no exact
  values.
- `..._rejects_zero_sized_target` + `..._propagates_position_map_errors`
  (gpu): validation happens before any GPU work, mirroring the AO mesh
  tests.
- Unit (no GPU): uniform size assert (16), `DEFAULT_STRENGTH == 1.0`,
  validate accept/reject.

## Reviewer checklist

- [ ] `bake_shaders.rs` diff is append-only (existing shaders byte-identical).
- [ ] `position.rs` untouched — confirm `TEXTURE_BINDING` was already on
      both targets rather than trusting this note.
- [ ] Shader minus sign intact (convex-negative); tent test fails if flipped.
- [ ] No `unwrap`/`expect` outside `#[cfg(test)]` modules.
- [ ] `cargo fmt --check`, `cargo clippy -D warnings` clean under both
      `--no-default-features` (default) and `--features gpu`.
- [ ] Full `umber-bake` suites green in both configs (29 gpu / 20 default
      pre-existing, plus the 4 new unit + 4 new gpu tests here).
- [ ] Seam-fringe + single-texel-island limits above are acceptable as
      documented follow-ups behind the dilation pass, not blockers.

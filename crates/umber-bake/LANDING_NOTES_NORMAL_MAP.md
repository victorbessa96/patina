# umber-bake tangent-space normal map — landing notes

Wave-3 claw: tangent-space normal baking —
`crates/umber-bake/src/normal_map.rs` (new module) plus
`TANGENT_NORMAL_BAKE_SHADER`, a new WGSL compute pass appended to
`crates/umber-gpu/src/bake_shaders.rs` (append-only; no existing shader
touched). Scope per docs/specs/requirements.md §3 (normal is a P0
baker). Composition mirrors `ao::bake_ao_mesh` exactly: rasterize the
mesh's own UV layout into a position/normal map via
`position::bake_position_and_normal`, then bind both textures read-only
into the second pass — only the second pass differs (a per-texel
screen-space TBN transform instead of hemisphere raycasting, so no
triangle buffer, no raycount uniform, no atomics).

## What was built

**`crates/umber-gpu/src/bake_shaders.rs`** — appended
`TANGENT_NORMAL_BAKE_SHADER` (see that constant's doc comment for the
full contract). One workgroup (`@workgroup_size(1)`) per texel,
dispatched as `dispatch_workgroups(width, height, 1)`; `workgroup_id.xy`
is the texel coordinate into all three textures. Bindings pack
contiguously (`0` = write-only `Rgba8Unorm` output, `1` =
`TangentNormalParams` uniform, `2/3` = read-only `texture_2d<f32>`
position/normal) rather than inheriting the AO pass's numbering, since
there is no triangle storage buffer on this pass — the same packing
`CURVATURE_BAKE_SHADER` uses.

**`crates/umber-bake/src/normal_map.rs`** (new module)
- `TangentNormalParams { directx_y_flip: bool }` — `Default` is `false`
  (OpenGL, green up); `new(bool)` for the explicit choice.
- `TangentNormalBakeError` (`thiserror`) — `EmptyTarget { width, height }`,
  `PositionMap(#[from] PositionMapError)`, `Readback(String)`.
- `bake_tangent_normal_mesh(device, queue, mesh, width, height, params)
  -> Result<Vec<u8>, TangentNormalBakeError>` — validates, runs
  `bake_position_and_normal`, builds a fresh pipeline + bind group,
  dispatches, and returns full `width * height * 4` RGBA8 bytes
  (`rgb = tangent_normal * 0.5 + 0.5`, `a` = coverage). Reuses
  `ao::BakeTarget` for the output texture + 256-byte-row-pitch readback
  rather than duplicating that machinery.
- `uniform_for(width, height, params)` — the bool-to-`u32`-word mapping
  factored out so it is unit-testable without a device (WGSL uniforms
  have no bools).

**`crates/umber-bake/src/lib.rs`** — two lines added, `pub mod normal_map;`
plus its re-export. No other line touched.

**`crates/umber-bake/src/position.rs`** — untouched. The readback
position/normal textures already carry `TEXTURE_BINDING` (position.rs's
`make_target`), so binding them `texture_2d<f32>` here needs no change.

## Screen-space TBN derivation + its UV-alignment assumption

Per covered texel, over the in-bounds, *covered* x-neighbors (uncovered
or out-of-bounds neighbors are skipped, not zero-filled — the same rule
`CURVATURE_BAKE_SHADER` uses, so a hole never tilts a neighboring
frame):

```text
dpdx = central difference (both x-neighbors covered) or one-sided (island border)
T = normalize(dpdx - N * dot(N, dpdx))   // Gram-Schmidt against N
B = cross(N, T)
tangent_normal = transpose(TBN) * n = (dot(T, n), dot(B, n), dot(N, n))
```

`N`/`n` is the position pass's per-texel face normal (`normal_tex`,
normalized) — the only normal this slice has, so the world normal being
transformed and the frame's axis coincide, and a flat facet bakes
`(0, 0, 1)` by construction. A texel with no covered x-neighbor
(isolated single-texel island) or a near-zero gradient falls back to the
world axis least aligned with `N`, orthogonalized the same way, so `T`
is never a normalized zero vector. This is the standard baker approach
(Mikkelsen's tangent-frame lineage, minus the UV-derivative solve —
that is the wave-4 item below, not an oversight).

Texel `+x` is `+u`: the position pass maps texel centers through
`u = (x + 0.5) / width`, so `dP/dx` points along the surface's `+u`
direction and `T` is the `+u` tangent — exactly Substance Painter's
default tangent frame on meshes whose UV islands are axis-aligned to the
baked surface. `B = cross(N, T)` then points along `+v` on such meshes
(not along texel `+y`, which is `-v` under the position pass's
`(1 - v)` flip), keeping green "up" in UV space per the OpenGL
convention. On rotated UV islands the frame twists with the screen axes
instead of the UVs — the known screen-space limitation this slice
accepts (same class of artifact as `CURVATURE_BAKE_SHADER`'s UV-seam
blindness, and confined the same way: the suite proves the transform
correct where UV adjacency equals mesh adjacency).

## Wave-4 per-texel UV-derivative plan

Once the position pass exports per-texel UVs in a channel, replace the
`dpdx`-only construction with the standard UV-derivative frame: solve
`dP/du`, `dP/dv` from neighbor differences (`dp = dP/du * du + dP/dv *
dv` over two covered neighbors), `T = normalize(dP/du - N * dot(N,
dP/du))`, `B` from `dP/dv` with a `cross(N, T)`-handedness check against
the UV winding. The `transpose(TBN) * n` transform and the RGBA8
encoding stay unchanged; only the `T`/`B` derivation moves. That same
pass is also where a smooth (interpolated vertex-normal) input plugs
into `n` — today `n == N`, so every bake is flat-shaded in tangent
space, which is why the flip test below is near-identity rather than a
real green inversion (a nonzero tangent `y` needs smooth normals first).

## DirectX/OpenGL flip

Default is OpenGL (`+Y` up = green up). A nonzero `flip_y` uniform word
(from `TangentNormalParams::directx_y_flip`) inverts the green channel
*after* encoding (`g = 1 - g`) for DirectX (`-Y` up). Uncovered texels
write `(0, 0, 0, 0)`, distinguishable from flat-but-covered
(`(~0.5, ~0.5, 1, 1)`) by alpha alone — the same alpha convention
`cs_main_from_position` uses.

Deliberate deviation from the task sketch to record: the sketch
expected the flipped flat-quad green to read `255 - 128 = 127`. On the
flat quad the tangent `y` is `0 ± 1e-7`, i.e. encoded `0.5` — a fixed
point of `g = 1 - g` up to `Rgba8Unorm` quantization — so the
analytically-correct flipped output is byte-identical (within one
quantization step) to the unflipped output, and a test asserting
`!= 128` would be red on a correct implementation. The suite therefore
asserts near-identity (`|flipped - plain| <= 1` per channel with
identical blue/alpha) and documents this; the byte-inversion form of
the test becomes meaningful once wave-4 smooth normals give tangent
`y != 0` something to invert.

## Testing

- `flat_quad_bakes_plus_z_tangent_everywhere` (gpu): 32×32 full-UV quad,
  *every* texel asserted `r, g ∈ {127, 128}`, `b == 255`, `a == 255`.
  The dots are `0 ± 1e-7` in `f32`, so the two-value band is exactly the
  `Rgba8Unorm` rounding of `0.5` (`127.5` → either side by driver), not
  slack — anything else (a sign flip, a dropped transform) fails.
- `directx_flip_is_near_identity_on_flat_quad` (gpu): same mesh baked
  with `directx_y_flip` false and true; asserts identical coverage and
  blue with red/green within one quantization step (see the flip section
  above for why near-identity, not inversion, is the correct
  expectation here).
- `slanted_quad_stays_blue_dominant_in_tangent_space` (gpu): horizontal
  quad (`pos = (2u - 1, 0, 2v - 1)`, face normal `(0, -1, 0)` from the
  documented winding). Tangent bake asserts blue `> 190` (near-flat in
  tangent space) with full coverage on *every* texel, while
  `bake_world_normal_map` on the same mesh asserts `(0, -1, 0)` at the
  center texel — the two halves together prove the TBN transform ran
  (a world-normal passthrough would read green-dominant, not blue).
- `..._rejects_zero_sized_target` + `..._propagates_position_map_errors`
  (gpu): validation happens before any GPU work, mirroring the AO mesh
  tests.
- Unit (no GPU): uniform size assert (16), params default/new,
  `uniform_for` bool→word mapping (`false → 0`, `true → 1`) with
  dimension passthrough, validate accept/reject.

## Reviewer checklist

- [ ] `bake_shaders.rs` diff is append-only (existing shaders byte-identical).
- [ ] `position.rs` untouched — confirm `TEXTURE_BINDING` was already on
      both targets rather than trusting this note.
- [ ] Screen-space frame math matches the derivation above; fallback axis
      can't degenerate (`|N·axis| ≤ 0.9` by construction).
- [ ] Flip section's `127`-vs-near-identity deviation agreed (don't
      "fix" the test to assert `127` — it would fail on correct code).
- [ ] No `unwrap`/`expect` outside `#[cfg(test)]` modules.
- [ ] `cargo fmt --check`, `cargo clippy -D warnings` clean under both
      `--no-default-features` (default) and `--features gpu`.
- [ ] Full `umber-bake` suites green in both configs (61 gpu / 36 default
      pre-existing, plus the 5 new unit + 5 new gpu tests here).
- [ ] Rotated-island frame twist + flat-shaded (`n == N`) limits above
      are acceptable as documented wave-4 follow-ups, not blockers.

# umber-bake UV-padding dilation — landing notes

Wave-3 claw: post-bake UV padding — `crates/umber-bake/src/dilation.rs`
(new module) plus `DILATE_BAKE_SHADER`, a new WGSL compute pass appended to
`crates/umber-gpu/src/bake_shaders.rs` (append-only; no existing shader
touched). Scope per docs/specs/requirements.md §3 (dilation is listed Wave-3
scope, and the curvature notes already name it as the follow-up behind the
seam-fringe artifact). Unlike the mesh-fed bakers, this pass is *not*
mesh-fed: its input is a CPU-side RGBA8 map (typically a bake readback),
uploaded once and ping-ponged on the GPU.

## What was built

**`crates/umber-gpu/src/bake_shaders.rs`** — appended
`DILATE_BAKE_SHADER` (see that constant's doc comment for the full
contract). One workgroup (`@workgroup_size(1)`) per texel, dispatched as
`dispatch_workgroups(width, height, 1)`; `workgroup_id.xy` is the texel
coordinate into both textures. Bindings pack contiguously (`0` = read-only
`texture_2d<f32>` input over `Rgba8Unorm`, `1` = write-only `Rgba8Unorm`
output, `2` = `vec2<u32>` dims uniform) — no triangle buffer, no
position/normal inputs, no other state.

**`crates/umber-bake/src/dilation.rs`** (new module)
- `DilateParams { iterations }` — `DilateParams::DEFAULT_ITERATIONS`
  (`16`), `new`, and `Default`.
- `DilateError` (`thiserror`) — `EmptyTarget { width, height }`,
  `SizeMismatch { len, width, height, expected }`, `Readback(String)`.
  (`SizeMismatch` has no analog in the sibling bakers — they take a mesh
  and dimensions, not a byte buffer, so there is nothing to mismatch. A
  wrong-length slice would otherwise upload garbage or panic the row math,
  hence the explicit variant.)
- `dilate_map(device, queue, map, width, height, params)
  -> Result<Vec<u8>, DilateError>` — validates, uploads into a
  `TEXTURE_BINDING | COPY_DST` staging texture, runs `N` ping-pong passes
  between two `ao::BakeTarget`s, reads the final target back as full
  `width * height * 4` RGBA8. One entry point only, per the brief (no
  in-place convenience). Reuses `ao::BakeTarget` for the ping-pong pair +
  256-byte-row-pitch readback rather than duplicating that machinery.

**`crates/umber-bake/src/lib.rs`** — two lines added, `pub mod dilation;`
plus its re-export. No other line touched.

## Algorithm (8-neighbor nearest-donor propagation)

One dilation step, per texel, from the read-only input into the write-only
output:

```text
if src.a > 0: copy through unchanged
else: scan the 8 neighbors; among covered neighbors pick the one with the
      HIGHEST coverage (alpha), ties keep the first found in top-left
      row-major scan order; write its rgb with alpha 1.0 —
      or (0,0,0,0) if no neighbor is covered.
```

`N` steps = `N` ping-pong dispatches driven from Rust (two `BakeTarget`s,
roles swapped each iteration; pass 0 reads the upload texture, every later
pass reads the target the previous pass wrote). Each pass reads the
previous pass's output, so the covered front advances exactly one texel per
dispatch and `iterations` is the padding width in texels. Cost is linear:
`N` full-map dispatches of nine `textureLoad`s + one `textureStore` per
texel — no raycast, no triangle buffer, trivially bandwidth-bound.

### Why the dilated alpha must normalize to 1.0 for chain propagation

A newly-dilated texel must be indistinguishable from an originally covered
texel on the *next* step (`a = 1.0`, byte `255`, both) or propagation
stalls after one ring: the next pass's covered test is `a > 0` and its
donor ranking compares coverage values, so anything less than full coverage
would make second-ring texels rank below (or read as uncovered next to) the
front and the wave would die out. The donor's rgb is copied verbatim — no
blending, no falloff — so color propagates unchanged no matter how many
rings it travels, and covered source pixels copy through every pass
untouched (interior pixels survive any iteration count byte-identical,
pinned by the round-trip GPU test).

### A note on "diamond" vs. square (brief wording vs. shader contract)

The brief's test sketch says a single texel dilated 4 steps yields "a
diamond of radius 4". With the mandated 8-neighborhood that is a
*square*: king-move expansion claims one Chebyshev ring per pass, so after
4 iterations the covered set is Chebyshev radius 4 (a 9×9 square), which
*contains* the Manhattan diamond — every diamond texel is covered with the
donor color, but so are the diamond's outside corners (e.g. `(±4, ±4)`).
The suite asserts the exact square the shader contract guarantees (and the
diamond-covered half explicitly), rather than the sketch's "outside diamond
uncovered" half, which is mathematically unreachable under the normative
per-texel rule. If a future reader wants true diamond (Manhattan) fronts,
that needs a 4-neighborhood kernel — a deliberate shader change, not a test
fix.

## Iteration cost

`N` dispatches, each `O(texels)`. Default 16 covers a full mip chain's
bleed radius on 1k–2k maps. All `N` passes are recorded into one command
encoder and submitted once; wgpu's hazard tracking orders the successive
passes on the shared textures. `iterations == 0` skips the GPU entirely and
returns the input cloned.

## Known artifact: thin diagonal streaks from first-found tie-breaking

With binary coverage (everything the bake passes and this pass produce is
alpha `0` or `1`), every covered neighbor ties at `1.0` and the strict-`>`
comparison keeps the first donor in scan order (top-left row-major).
Where two fronts meet along a diagonal, donor choice biases top-left,
leaving faint diagonal streaks in the padded margin. Acceptable for this
slice — Substance's dilator streaks the same way — and confined to the
margin: source pixels are never altered, only the padding between islands.

## Odd/even ping-pong landing

Pass `i` writes to B when its parity matches the last pass's (`i % 2 ==
(iterations - 1) % 2`), else to A — so the final pass always lands in B,
the target read back, for both odd and even `iterations` (pinned by the
round-trip test running both 3 and 4). A fresh `BakeTarget` cannot serve as
the upload source (its texture lacks `COPY_DST`, so neither
`queue.write_texture` nor a staging copy could land bytes in it) — hence
the dedicated `TEXTURE_BINDING | COPY_DST` upload texture feeding pass 0.

## Testing

- `single_texel_dilates_to_chebyshev_square_containing_diamond` (gpu):
  15×15, one covered center texel, 4 iterations — every texel at Chebyshev
  ≤ 4 asserts the center's exact color, every texel beyond asserts
  `(0,0,0,0)`, plus the diamond-covered half stated directly. Exact
  values throughout: dilation is a discrete cellular process, so unlike
  the raycast/estimator tests there is no quantization slack to bound.
- `zero_iterations_returns_input_unchanged` (gpu): mixed
  covered/uncovered 8×6 map asserts byte-identical output at `N = 0`.
- `square_front_advances_exactly_one_texel_per_iteration` (gpu): 8×8
  square in 32×32, 5 iterations — texels 5 out from two island edges assert
  the island color, texels 6+ out assert uncovered. Pins "N steps = N
  rings" on both axes.
- `fully_covered_map_round_trips_byte_identical` (gpu): varied colors
  incl. mid-alpha 128 (which must also copy through verbatim — the
  `Rgba8Unorm` load/store round trip is exact), at both 3 (odd) and 4
  (even) iterations, pinning copy-through and the parity handling.
- `..._rejects_zero_sized_target` +
  `..._rejects_size_mismatch_without_touching_the_gpu_pipeline` (gpu):
  validation before any GPU work, mirroring the sibling bakers.
- Unit (no GPU): `DEFAULT_ITERATIONS == 16`, validate accept/reject
  (empty target, size mismatch).

## Reviewer checklist

- [ ] `bake_shaders.rs` diff is append-only (existing shaders byte-identical).
- [ ] `umber-cli` untouched (dragon's parallel lane).
- [ ] Dilated alpha is full coverage (`1.0`), not a partial/front-distance
      value — propagation stalls otherwise; single-texel test fails.
- [ ] Square-vs-diamond note above accepted (8-neighborhood ⇒ Chebyshev
      fronts); test asserts the square, not the sketch's diamond.
- [ ] No `unwrap`/`expect` outside `#[cfg(test)]` modules.
- [ ] `cargo fmt --check`, `cargo clippy -D warnings` clean under both
      `--no-default-features` (default) and `--features gpu`.
- [ ] Full `umber-bake` suites green in both configs (50 gpu / 32 default
      pre-existing, plus the 4 new unit + 6 new gpu tests here); full
      `umber-gpu` suite green (50 pre-existing gpu).
- [ ] Diagonal-streak limit above is acceptable as documented behavior, not
      a blocker.

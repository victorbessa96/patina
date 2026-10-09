# Landing notes — umber-brush stroke conditioning (Wave 2 start)

Pure-CPU stroke conditioning pipeline: one-euro smoothing → lazy-mouse pull
→ MyPaint-style residual dab spacing. No GPU, no `umber-gpu` dependency;
`glam` + `thiserror` only. `StrokeEvent` / `OneEuroParams` untouched.

## What was built

- `src/one_euro.rs` — `OneEuroFilter` (2D position) + `OneEuroScalar`
  (pressure channel, per report §4: "pressure through a separate one-euro
  instance"). Per-axis adaptive cutoff `min_cutoff + beta*|dx|`, derivative
  estimated through a fixed `d_cutoff` low-pass (Casiez CHI'12 reference
  structure). `new`/`configure` sanitize out-of-range fields to defaults;
  `try_new`/`try_configure` reject with `OneEuroError`. First sample seeds
  and passes through; duplicate/backwards timestamps hold the estimate
  (`saturating_sub`, last-accepted time kept); non-finite inputs hold.
- `src/lazy_mouse.rs` — `LazyMouse { radius_px }` with
  `apply(target, current)`: snap when within the radius, else advance exactly
  one radius along `current → target`. Non-finite inputs hold `current`.
- `src/spacing.rs` — `SpacingAccumulator { dabs_per_radius, residual }`
  (`residual` = banked sub-dab travel in **pixels**) and
  `plan_dabs(from, to, radius_px, pressure: [from_p, to_p]) -> Vec<DabPlan>`.
  Dab step = `radius_px / dabs_per_radius`; each `DabPlan { pos,
  alpha_multiplier }` linearly interpolates the endpoint-pressure span at
  the dab's fractional distance. Zero-length / degenerate inputs emit
  nothing and leave `residual` untouched. `plan_dabs_uniform` covers the
  constant-pressure case.
- `src/wiring.rs` — `StrokeConditioner` chaining the three stages per
  `StrokeEvent` (first event seeds, later events emit). Anchor advances to
  the last emitted dab (position *and* interpolated pressure) so pressure
  stays continuous across event boundaries. `reset()` between strokes so no
  partial dab leaks into the next stroke.
- `src/lib.rs` — `pub mod` wiring + re-exports only.

## Residual-carry identity test design

Core property (in `spacing.rs` tests): planning `[0,0]→[10,0]` in one call
must equal planning `[0,0]→[5,0]` + `[5,0]→[10,0]` concatenated — same dab
count, positions, `alpha_multiplier`s, and final `residual`.

Why it holds: the next dab fires at `step − residual` from the segment
start, so after the first half banks `residual₁ = 5 − k₁·step`, the second
half's first dab lands at `5 + (step − residual₁) = (k₁+1)·step` — exactly
the unsplit grid. Density `1.5` with radius `4.0` (step `2.666…`) was chosen
so the split falls mid-step and the test exercises carry rather than
alignment luck. Comparison uses `1e-4` epsilon, not bitwise equality, since
`lerp` from different segment origins can differ in the last ulp.
Companion tests: ten 1 px hops bank exactly one 10 px-step dab (slow strokes
don't over-stamp); zero-length segments don't touch `residual`.

## What needs live-tablet validation

- **Tuning**: defaults (`min_cutoff 1.0 Hz`, `beta 0.007`, `d_cutoff 1.0 Hz`)
  are starting points. Follow the Casiez recipe (beta = 0, raise min_cutoff
  to kill jitter, then raise beta to kill lag) against a real pen at 125+
  Hz; verify end-to-end latency stays under ~20 ms.
- **Timestamps**: pipeline assumes `time_ns` is monotonic per stroke with
  ~ms resolution. Confirm the stylus crate's clock never jumps (mode
  changes, pen re-proximity); stale stamps are held, not fixed.
- **Radius policy**: spacing divides by `radius_px` — callers must clamp to
  something like MyPaint's `[0.2, 1000]` px; near-zero radii would shrink
  the dab step toward the `MAX_DABS_PER_CALL` guard (65 536/call).
- **Pressure curve**: `alpha_multiplier` is linear in raw pressure; the
  pressure→opacity response curve is a later slice.
- **Lazy radius UX**: leash length vs. one-euro strength interaction needs
  artist testing (report: one-euro default, pulled-string opt-in).

## Reviewer checklist

- [ ] `cargo fmt -p umber-brush` clean (scoped `-p`; do not reformat the
      whole workspace — a parallel agent owns `umber-core`).
- [ ] `cargo clippy -p umber-brush --all-targets -- -D warnings` passes.
- [ ] `cargo test -p umber-brush` passes (19 tests: filter convergence /
      beta-lag / monotonic-hold / scalar / lazy converge+clamp+degenerate /
      spacing evenness+identity+accumulation+zero-length+pressure+degenerate /
      wiring smoke+determinism+reset+rejection).
- [ ] No `unwrap()` outside tests; no new deps beyond `glam`/`thiserror`;
      no file outside `crates/umber-brush` touched; no git operations run.
- [ ] Determinism: no wall-clock reads, no RNG; float comparisons in tests
      are epsilon-based (see identity-test note above).

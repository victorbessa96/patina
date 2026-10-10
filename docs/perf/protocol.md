# Umber — Performance Protocol (§12)

The measurement contract for the requirements §12 budgets. Written
2026-10-09 (full-project audit finding: the budget targets existed
with NOTHING measuring them). This doc defines what we measure, how,
and where the numbers land — instrumentation lands with the wave-4
painting work so every new perf-relevant change ships with a number.

## Budgets (from requirements §12 — unchanged, now enforced)

| Budget | Target | How measured |
|---|---|---|
| Input-to-photon | < 20 ms @ 60Hz | `frame_timestamps` tracing span: stylus/pointer event receive → frame present callback. P50/P95/P99 per session, reported in the perf HUD + tracy. |
| Stroke throughput | 4K set, no dropped dabs @ 200Hz input | Dab-count instrumentation: dabs staged vs dabs rendered per frame. A dropped dab is a staged dab absent from the frame's upload. Assert zero in the stroke soak test. |
| Viewport | 60 fps sustained, P95 frame-time < 16.7ms, 1M-tri + 4K set | The reference-scene benchmark: a 1M-tri procedural mesh (subdivided cube grid, checked in) rendered headless via lavapipe in CI + live on dev GPUs. |
| Undo RAM | 100 steps @ 4K within ~2GB | TilePool accounting: `total_bytes()` sampled after each undo push in the soak test; assert monotonic-under-cap. |
| Installer | < 150 MB tracked per release | `cargo dist` artifact size logged at release CI; tracked in this doc's table, not gated. |

## Instrumentation plan (wave-4, lands with the painting work)

1. **`profiling` facade + tracy-client** behind a `perf` feature
   (default off — zero cost when off, the crate is already the §10
   P0 plan). Spans: `frame`, `stroke_eval`, `dab_upload`, `bake_pass`,
   `display_pass`, `undo_push`.
2. **The perf HUD** (egui overlay, `perf` feature): frame ms, dab
   throughput, tile-pool bytes, undo depth. Live numbers beat after-
   the-fact reports for the artist-facing floor.
3. **CI soak tests** (`perf` feature, nightly job — NOT per-commit;
   the load directive): the 200Hz stroke soak, the undo soak, the
   reference-scene render. Numbers land in the CI summary + a
   `perf/baseline.json` checked in; a regression >10% on any budget
   fails the job.

## The reference-scene bench (viewport row)

`crates/umber-gpu/src/ref_scene.rs`, behind umber-gpu's `perf` feature:

```
UMBER_REF_SCENE_FRAMES=1000 UMBER_REF_SCENE_JSON=ref-scene.json \
  cargo test -p umber-gpu --features perf --release ref_scene -- --nocapture
```

- Scene: 289-subdivision cube grid (1,002,252 tris, generator checked
  in, deterministic) + four solid 4096² maps, drawn through the
  viewport's own mesh-pass `draw` into a 1920×1080 sRGB + Depth32Float
  offscreen target. 3 warm-up frames are not counted; the default is 120
  measured frames, and recorded baselines use ≥ 1000 (the rule below).
- The 4K set is **resident, unsampled**: the mesh pass has no texture
  binding yet. Until it gains one, the number covers the 1M-tri half of
  the budget plus the set's VRAM pressure, not texture sampling.
- The test asserts only that the harness is correct: N frames timed, all
  times non-zero, p50 ≤ p95 ≤ p99, and the mesh covers more than 2% of
  the target (an empty render covers about 0%). It prints the P95 < 16.7ms verdict without failing on it.
  The nightly job's >10% regression comparison is the gate.
- Output: one `ref-scene baseline:` line, plus one `ref-scene json:` line
  (`frames, mean_ms, p50_ms, p95_ms, p99_ms, tris` + adapter, device
  type, backend). Set `UMBER_REF_SCENE_JSON` to also write the JSON to a
  file. Record the adapter with every row, since lavapipe and the dev GPU
  are separate baselines.

## Baselines (fill as instrumentation lands)

| Date | Commit | Input→photon P95 | Dab drop @200Hz | Viewport P95 (ref scene) | Undo RAM @100 steps | Notes |
|---|---|---|---|---|---|---|
| 2026-10-10 | 19e71c5 | — (tracy session pending) | **0** (2914/2914 staged=composited; inter-dispatch mean 8.56ms, p50 10ms, p95 15ms; 4096² target, 3070) | — (ref-scene bench not built) | 19016 B final, sub-linear (first-10 Δ 1746B > last-10 Δ 980B), empirical ceiling 38032B | First real row. Undo numbers are the metadata-only model (no GPU-tile bytes in Document yet — the ceiling pins shape, NOT the §12 ~2GB@4K budget). Stroke soak ran the `perf` feature suite remotely (single-threaded, ICD pinned). |

## Rules

- No perf-relevant PR merges without a number next to the claim.
- Baselines are per-hardware-class: lavapipe (CI floor), the named
  mid-range reference GPU (dev machine), recorded separately.
- "Feels fine" is not a measurement. Neither is a single run — every
  budget number is P95 over ≥ 1000 frames or ≥ 1000 dabs.

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

## Baselines (fill as instrumentation lands)

| Date | Commit | Input→photon P95 | Dab drop @200Hz | Viewport P95 (ref scene) | Undo RAM @100 steps | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | instrumentation pending (wave-4) |

## Rules

- No perf-relevant PR merges without a number next to the claim.
- Baselines are per-hardware-class: lavapipe (CI floor), the named
  mid-range reference GPU (dev machine), recorded separately.
- "Feels fine" is not a measurement. Neither is a single run — every
  budget number is P95 over ≥ 1000 frames or ≥ 1000 dabs.

# Perf Instrumentation — Wave-4 Design (§12, protocol implementation)

Wave-4 item 8. The protocol doc (docs/perf/protocol.md, 68176b7)
defines WHAT to measure; this note contracts the instrumentation
itself. Written 2026-10-09 against the tree at `5ba0af4`.

## The facade

The `profiling` crate (facade pattern, zero-cost when no backend
feature) + `tracy-client` behind a `perf` feature — root Cargo.toml
workspace deps. No crate gains tracy unconditionally; the feature
wires at umber-app only (`--features perf` on the dev run). The
profiling spans the protocol names, nothing else — span set:

| Span | Where | Fires |
|---|---|---|
| `frame` | umber-app update loop top → end | once per egui frame |
| `stroke_eval` | PaintState::push_event_inner (incl. mirror expansion) | per pointer event |
| `dab_upload` | PaintThread command processing, upload path | per Stage batch |
| `bake_pass` | each umber-bake bake fn | per bake |
| `display_pass` | viewport paint callback | per frame with paint |
| `undo_push` | UndoStack::push | per command |

`profiling::span!` macros at those six points, one line each — the
diff is small and reviewable. The `puffin`/`superluminal` features
stay off.

## The HUD

umber-app, behind `perf`: an egui overlay window (default off, View
menu toggle `Show Perf HUD`) reading:
- frame ms (from the `frame` span's wall time — egui's `input.time`
  delta between frames, simplest correct clock)
- `FrameStats` deltas per frame (dabs_composited delta — the stroke
  throughput line; drops are computable from the staged-vs-
  composited relationship: staged dabs in, composited out, any
  shortfall is a drop)
- tile-pool/paint-target bytes (PaintTarget size — v1 is the one
  TARGET_SIZE² texture; the pool accounting lands with the tile-pool
  slice)
- undo depth (UndoStack len — v1 proxy for the RAM budget until
  tile accounting lands)

v1 is read-only display — the numbers exist, budgets get enforced
when the soak tests land (below).

## The soak tests

`#[cfg(feature = "perf")] #[test]` in umber-app — the two the
protocol names, headless-legal (the wiring claw proved the real
adapter works headless; lavapipe covers CI):

1. **Stroke soak**: 200Hz synthetic stroke (5,000 events over 25s
   of simulated time — the logical clock makes this fast: advance
   time_ns by 5ms per event, no wall-clock sleeping), a 4K-target
   PaintState, assert `drops == 0` (staged == composited at drain)
   and record P95 inter-dispatch interval into the test output.
2. **Undo soak**: 100 undo pushes of full-canvas dabs on a Document,
   sample the doc's in-memory size proxy (layers + tile bytes)
   after each push, assert monotonic-under-cap: the growth curve
   stays bounded (the §12 target is ~2GB @ 4K; the CI machine
   assert is the curve's shape — sub-linear in push count — plus a
   hard byte ceiling scaled to the CI's 4K target, not the full
   one).

The baseline table in protocol.md fills from these tests' first
green run on lavapipe; the dev-GPU numbers land when `perf` runs
locally.

## What this is NOT

- No nightly CI job yet — the load directive holds; the job spec
  lands when the wave-4 gate work completes.
- No display-LUT/display-transform measurement — §9's territory.
- No bake-farm throughput — single-pass spans only.

## Build order (one claw slice + app wiring)

1. Deps + spans (root Cargo.toml feature, six span lines, umber-app
   feature gate) — claw-shaped, small, verifiable via
   `cargo build -p umber-app --features perf` + span presence grep.
2. HUD overlay (umber-app; egui window; read-only) — rides the
   same slice.
3. The two soak tests (feature-gated) — the slice's tests ARE its
   verification; they must be able to fail (assert drop-free, assert
   bounded growth — both real assertions, not smoke).

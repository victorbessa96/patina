# Input→Photon — the Tracy Session Design (the last unmeasured §12 column)

Written 2026-10-10 15:14 against `909ee0c`. The baseline table's
first column has instrumentation (main.rs's `frame` span +
`stroke_eval`/`dab_upload` spans) but NO recorded numbers — the
session-capture path is missing. This design closes it.

## What exists

- The `profiling` facade + `profile-with-tracy` behind the app's
  `perf` feature (main.rs: 8 span sites).
- The §12 budget: `< 20ms @ 60Hz`, measured stylus/pointer event
  receive → frame present, P50/P95/P99 per session.
- tracy-client captures LIVE sessions — a running app + the
  Tracy GUI/profiler connect over the local socket.

## The gap

No headless/replayable capture: the numbers the baseline table
wants need a DETERMINISTIC event source + a capture sink, not a
human driving a stylus while a profiler watches.

## The design: the synthetic session harness

A `perf`-gated test/binary that:

1. **Synthesizes the input stream**: the same stroke-soak event
   pattern (the 200Hz logical clock, the begin/extend/end
   sequence) fed through the app's real event path — the same
   `process_pending` drains, the real span instrumentation
   firing per event (no GUI: the event pipeline minus the
   winit/egui surface, exactly what perf_soak already drives;
   the `frame` span fires around the drain+render-encode step).
   The frame-present side: the PaintThread's completion stats
   stand in for the present callback (the honest proxy — the
   winit present isn't reachable headless; the doc says so).
2. **Captures**: the tracy-client in capture mode (`--capture`
   flag on the tracy binary writes a .tracy file) OR — the
   simpler v1 — the span-timing path compiled to a stdout
   reporter (the profiling crate's second backend:
   `profile-with-stdout`? CHECK availability in the vendored
   version — if absent, the harness reads the same Instants the
   spans wrap and computes the percentiles directly, printing
   the `input-to-photon baseline:` line like perf_soak does).
3. **Computes**: event-receive → frame-complete delta per event
   window; P50/P95/P99 across the session; the budget verdict
   printed, not asserted (the nightly comparison gates, per the
   protocol's own rule).

**The honest v1**: option (b) — the harness measures the real
span boundaries (the same Instant deltas the spans would feed
tracy) and prints/JSONs them; the full .tracy capture is the
follow-up when a live session is worth profiling interactively.
No new deps for v1.

## Tests

1. The synthesized session runs the full event path (the staged
   dabs > 0, the drains issued — the soak's own conditioners).
2. The percentile math on a synthetic known-timing stream
   (hand-crafted deltas → the expected P50/P95/P99).
3. The budget line prints with all three percentiles + the
   session's event count.
4. The JSON sidecar shape (frames, p50/p95/p99_ms, events).

## Build

One claw slice, code-only (no cargo — the machine directive):
the harness in umber-app behind `perf` (a sibling of
perf_soak.rs), the four tests, the protocol doc updated with the
run command. The dragon reviews the code statically; the gate
runs when the build machine returns.

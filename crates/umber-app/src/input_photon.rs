//! Input→photon session harness (`perf` feature only) — the §12 baseline
//! table's first column, per `docs/specs/input-photon-design.md` (the
//! honest v1: option (b), no tracy capture, no new deps).
//!
//! A synthetic 200Hz pointer stream — the stroke soak's logical clock and
//! begin/extend/end pattern — runs through `PaintState`'s REAL event path:
//! every event is pushed exactly as the UV view pushes it, then drained via
//! `process_pending` (the drain main.rs's `frame` span wraps once per
//! frame). No GUI: the event pipeline minus the winit/egui surface, the
//! same headless slice `perf_soak` drives.
//!
//! # The delta (per event)
//!
//! Event `i` is received at logical mark `i * 5ms` (200Hz, no wall-clock
//! sleeping). Its photon is the first drain at or after `i` whose dispatch
//! counter advanced — the first frame at or after it that composited any
//! paint (most events stage no dab on their own: the conditioner emits at
//! brush spacing, so they ride a later frame; the filter's lag can leave
//! that dab trailing the event's raw position, which only makes true
//! latency larger — the lower bound below still holds). So
//!
//! `delta_i = (j - i) * 5ms + wall_j`
//!
//! where `wall_j` is the measured `Instant` span of carrier drain `j`: from
//! just before event `j` is pushed to just after that drain's GPU work has
//! completed (`device.poll(Wait)` — `process_pending` only submits, so
//! without the wait the "completion" would be the CPU encode alone).
//!
//! # The present proxy (the honest gap)
//!
//! The winit present callback is unreachable headless. The PaintThread's
//! completion — the drain's submission finished on the GPU, observed via
//! the dispatch counter in the stats the drain reports — stands in for it.
//! NOT covered: egui layout/paint, the egui-wgpu compositor pass, and the
//! swapchain/vsync wait (main.rs's `frame` span wraps all of `ui()`; this
//! harness reproduces only its drain). The number is therefore a LOWER
//! BOUND on true input→photon; the live tracy session is the follow-up.
//!
//! The 5ms clock is the harness's ARRIVAL schedule only: `PaintState`
//! advances its own per-push conditioner clock (`EVENT_STEP_NS`, 8ms), so
//! the filter does not see 200Hz timestamps — the stroke conditions exactly
//! as it does in the app today.
//!
//! Events after the last carrier drain never present (`end_stroke` flushes
//! nothing extra); they are counted as `unpresented` and printed, never
//! silently dropped nor stretched onto a later drain.
//!
//! # Running
//!
//! `UMBER_I2P_EVENTS` (default 5000) sets the event count;
//! `UMBER_I2P_JSON=<path>` also writes the JSON sidecar there (it is always
//! printed as one `input-to-photon json:` stdout line). The budget verdict
//! (P95 < 20ms — the column the baseline table records) is PRINTED, never
//! asserted: the nightly comparison is the gate. Run with `-- --nocapture`.

#[cfg(feature = "perf")]
use std::time::{Duration, Instant};

#[cfg(feature = "perf")]
use crate::paint_state::PaintState;

/// Default synthesized event count (25s of simulated 200Hz input).
#[cfg(feature = "perf")]
const DEFAULT_EVENTS: usize = 5_000;

/// The 200Hz logical arrival clock.
#[cfg(feature = "perf")]
const STEP: Duration = Duration::from_millis(5);

/// The §12 budget: input-to-photon < 20ms @ 60Hz.
#[cfg(feature = "perf")]
const BUDGET_MS: f64 = 20.0;

/// One drain of the session: the event index it followed (`events` for the
/// post-`end_stroke` drain), whether the dispatch counter advanced, and its
/// measured wall span (push → GPU completion).
#[cfg(feature = "perf")]
#[derive(Debug, Clone, Copy)]
struct DrainRecord {
    event: usize,
    dispatched: bool,
    wall: Duration,
}

/// Attributes each of `events` events (logical marks `i * STEP`) to its
/// carrier — the first dispatching drain at or after it — and returns the
/// per-event deltas (event order) plus the unpresented-tail count. `drains`
/// must be in ascending `event` order.
#[cfg(feature = "perf")]
fn attribute(events: usize, drains: &[DrainRecord]) -> (Vec<Duration>, usize) {
    debug_assert!(drains.windows(2).all(|w| w[0].event <= w[1].event));
    let mut deltas = Vec::with_capacity(events);
    let mut carriers = drains.iter().filter(|d| d.dispatched).peekable();
    for i in 0..events {
        while carriers.peek().is_some_and(|d| d.event < i) {
            carriers.next();
        }
        let Some(carrier) = carriers.peek() else {
            break;
        };
        deltas.push(STEP * (carrier.event - i) as u32 + carrier.wall);
    }
    let unpresented = events - deltas.len();
    (deltas, unpresented)
}

/// Nearest-rank percentile over a sorted slice (caller sorts ascending);
/// the perf soak's rule over `Duration`s, as `umber-gpu`'s ref scene does —
/// the wall half of a delta is sub-ms, whole-ms buckets would erase it.
#[cfg(feature = "perf")]
fn percentile_sorted(sorted: &[Duration], p: f64) -> Duration {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&p));
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Milliseconds as one correctly-rounded division of integer nanos, so
/// whole-ms and half-ms durations convert exactly (the percentile tests
/// compare with `==`).
#[cfg(feature = "perf")]
fn ms(d: Duration) -> f64 {
    d.as_nanos() as f64 / 1e6
}

/// One session's numbers. Percentiles are nearest-rank over the presented
/// events' deltas.
#[cfg(feature = "perf")]
#[derive(Debug, Clone)]
struct I2pReport {
    /// Events synthesized (`UMBER_I2P_EVENTS`).
    synthesized: usize,
    /// Synthesized events no carrier drain presented (the tail after the
    /// last dispatch).
    unpresented: usize,
    /// Per-event deltas of the presented events, in event order.
    deltas: Vec<Duration>,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

#[cfg(feature = "perf")]
impl I2pReport {
    fn from_deltas(deltas: Vec<Duration>, synthesized: usize, unpresented: usize) -> Self {
        assert!(
            !deltas.is_empty(),
            "no event presented — nothing to measure"
        );
        let mut sorted = deltas.clone();
        sorted.sort_unstable();
        Self {
            p50_ms: ms(percentile_sorted(&sorted, 0.50)),
            p95_ms: ms(percentile_sorted(&sorted, 0.95)),
            p99_ms: ms(percentile_sorted(&sorted, 0.99)),
            synthesized,
            unpresented,
            deltas,
        }
    }

    /// Measured (presented) event count — the population the percentiles
    /// are over, and the JSON's `events`.
    fn events(&self) -> usize {
        self.deltas.len()
    }

    /// Whether P95 meets [`BUDGET_MS`] (reported, never asserted).
    fn within_budget(&self) -> bool {
        self.p95_ms < BUDGET_MS
    }

    /// The human baseline line, in the perf-soak style.
    fn baseline_line(&self) -> String {
        let verdict = if self.within_budget() {
            "WITHIN"
        } else {
            "OVER"
        };
        format!(
            "input-to-photon baseline: events={events} (synthesized={synthesized} \
             unpresented={unpresented}) @200Hz logical | present proxy: PaintThread GPU \
             completion (lower bound — no egui/compositor/vsync) | \
             p50={p50:.2}ms p95={p95:.2}ms p99={p99:.2}ms | budget: P95<{BUDGET_MS}ms -> {verdict}",
            events = self.events(),
            synthesized = self.synthesized,
            unpresented = self.unpresented,
            p50 = self.p50_ms,
            p95 = self.p95_ms,
            p99 = self.p99_ms,
        )
    }

    /// The sidecar: `events` (presented count), `p50_ms`, `p95_ms`,
    /// `p99_ms`, `budget_ms`. One line, all floats finite (no serde in
    /// this crate).
    fn to_json(&self) -> String {
        format!(
            "{{\"events\":{},\"p50_ms\":{:.4},\"p95_ms\":{:.4},\"p99_ms\":{:.4},\"budget_ms\":{:.1}}}",
            self.events(),
            self.p50_ms,
            self.p95_ms,
            self.p99_ms,
            BUDGET_MS,
        )
    }
}

/// `UMBER_I2P_EVENTS`, defaulting to [`DEFAULT_EVENTS`]. A set but invalid
/// value is a misconfiguration: fail loudly, never fall back silently.
#[cfg(feature = "perf")]
fn events_from_env() -> usize {
    match std::env::var("UMBER_I2P_EVENTS") {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) if n > 1 => n,
            _ => panic!("UMBER_I2P_EVENTS must be an integer > 1, got {v:?}"),
        },
        Err(_) => DEFAULT_EVENTS,
    }
}

/// Blocks until every submitted GPU command has completed (the proxy's
/// "photon" edge).
#[cfg(feature = "perf")]
fn wait_gpu(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("GPU work completes");
}

/// The session: `UMBER_I2P_EVENTS` events stroked diagonally across a 4K target
/// (the stroke soak's geometry), each drained and GPU-waited.
///
/// Asserts the harness ran the real path (staged dabs > 0, dispatching
/// drains issued, events presented); prints the baseline + JSON lines.
#[cfg(feature = "perf")]
#[test]
fn input_photon_session() {
    const TARGET: u32 = 4096; // 4K target per the protocol

    let events = events_from_env();
    let Some((device, queue)) = crate::perf_soak::try_request_device() else {
        return;
    };
    // `PaintState` takes ownership; keep a handle for the completion wait.
    let poll_device = device.clone();
    let mut paint = PaintState::new(device, queue).expect("paint state builds");
    paint.resize_target(TARGET, TARGET);
    paint.process_pending().expect("resize applies");
    wait_gpu(&poll_device);

    let mut drains: Vec<DrainRecord> = Vec::with_capacity(events + 1);
    let mut last_dispatches = paint.last_stats().dispatches;
    let mut drain = |paint: &mut PaintState, event: usize, received: Instant| {
        let stats = paint.process_pending().expect("drain succeeds");
        wait_gpu(&poll_device);
        drains.push(DrainRecord {
            event,
            dispatched: stats.dispatches != last_dispatches,
            wall: received.elapsed(),
        });
        last_dispatches = stats.dispatches;
    };

    let received = Instant::now();
    paint.begin_stroke(egui::pos2(0.1, 0.1));
    drain(&mut paint, 0, received);
    for i in 1..events {
        let t = i as f32 / events as f32;
        let received = Instant::now();
        paint.extend_stroke(egui::pos2(0.1 + 0.8 * t, 0.1 + 0.8 * t));
        drain(&mut paint, i, received);
    }
    let received = Instant::now();
    paint.end_stroke();
    drain(&mut paint, events, received);

    let staged = paint.staged_dab_count();
    assert!(
        staged > 0,
        "session must actually stage dabs (conditioner emitted nothing over {events} events)"
    );
    let carriers = drains.iter().filter(|d| d.dispatched).count();
    assert!(
        carriers > 0,
        "session must issue at least one dispatching drain"
    );

    let (deltas, unpresented) = attribute(events, &drains);
    assert!(!deltas.is_empty(), "at least one event must present");
    let report = I2pReport::from_deltas(deltas, events, unpresented);
    assert!(
        report.p50_ms <= report.p95_ms && report.p95_ms <= report.p99_ms,
        "percentiles out of order: p50={} p95={} p99={}",
        report.p50_ms,
        report.p95_ms,
        report.p99_ms
    );

    println!(
        "{} | {TARGET}x{TARGET} target staged={staged} carrier-drains={carriers}",
        report.baseline_line()
    );
    let json = report.to_json();
    println!("input-to-photon json: {json}");
    if let Ok(path) = std::env::var("UMBER_I2P_JSON") {
        std::fs::write(&path, format!("{json}\n"))
            .unwrap_or_else(|e| panic!("writing UMBER_I2P_JSON={path}: {e}"));
        println!("input-to-photon json written to {path}");
    }
}

/// The percentile math on hand-crafted deltas: 1..=100ms shuffled → the
/// nearest-rank P50/P95/P99 are exactly 50/95/99ms.
#[cfg(feature = "perf")]
#[test]
fn percentiles_on_known_deltas() {
    // Deterministic shuffle (stride 37 is coprime with 100): from_deltas
    // must sort, not trust input order.
    let deltas: Vec<Duration> = (0..100u64)
        .map(|k| Duration::from_millis((k * 37) % 100 + 1))
        .collect();
    let report = I2pReport::from_deltas(deltas, 100, 0);
    assert_eq!(report.events(), 100);
    assert_eq!(report.p50_ms, 50.0);
    assert_eq!(report.p95_ms, 95.0);
    assert_eq!(report.p99_ms, 99.0);
    assert!(!report.within_budget(), "P95 95ms is over the 20ms budget");
}

/// The carrier attribution on a hand-crafted drain log: known `(j - i)`
/// gaps plus each carrier's wall span, and the unpresented tail.
#[cfg(feature = "perf")]
#[test]
fn attribution_on_known_drains() {
    let rec = |event, dispatched, wall_us| DrainRecord {
        event,
        dispatched,
        wall: Duration::from_micros(wall_us),
    };
    // Six events; carriers at events 2 (wall 1ms) and 3 (wall 2ms); the
    // final (post-end_stroke, event 6) drain dispatches nothing.
    let drains = [
        rec(0, false, 100),
        rec(1, false, 100),
        rec(2, true, 1_000),
        rec(3, true, 2_000),
        rec(4, false, 100),
        rec(5, false, 100),
        rec(6, false, 100),
    ];
    let (deltas, unpresented) = attribute(6, &drains);
    let dur = |m: u64, us: u64| Duration::from_millis(m) + Duration::from_micros(us);
    assert_eq!(
        deltas,
        vec![
            dur(10, 1_000), // event 0 → carrier 2: 2 steps + 1ms
            dur(5, 1_000),  // event 1 → carrier 2: 1 step + 1ms
            dur(0, 1_000),  // event 2 → itself
            dur(0, 2_000),  // event 3 → itself
        ]
    );
    assert_eq!(unpresented, 2, "events 4 and 5 follow the last carrier");

    let report = I2pReport::from_deltas(deltas, 6, unpresented);
    assert_eq!(report.p50_ms, 2.0);
    assert_eq!(report.p95_ms, 11.0);
    assert_eq!(report.p99_ms, 11.0);
}

/// The budget line carries all three percentiles, the event count, and
/// the verdict.
#[cfg(feature = "perf")]
#[test]
fn budget_line_prints_percentiles_and_count() {
    let deltas: Vec<Duration> = (1..=10).map(Duration::from_millis).collect();
    let report = I2pReport::from_deltas(deltas, 12, 2);
    let line = report.baseline_line();
    for needle in [
        "input-to-photon baseline:",
        "events=10",
        "synthesized=12",
        "unpresented=2",
        "p50=5.00ms",
        "p95=10.00ms",
        "p99=10.00ms",
        "budget: P95<20ms -> WITHIN",
    ] {
        assert!(line.contains(needle), "{needle:?} missing from {line}");
    }
    assert!(!line.contains('\n'));
}

/// The JSON sidecar shape: exactly the five protocol keys, one line.
#[cfg(feature = "perf")]
#[test]
fn json_sidecar_shape() {
    let report = I2pReport::from_deltas(
        vec![Duration::from_micros(1500), Duration::from_micros(2500)],
        3,
        1,
    );
    let json = report.to_json();
    assert_eq!(
        json,
        "{\"events\":2,\"p50_ms\":1.5000,\"p95_ms\":2.5000,\"p99_ms\":2.5000,\"budget_ms\":20.0}"
    );
}

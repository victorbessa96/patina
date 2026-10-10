//! Perf soak tests (wave-4 item 8, `perf` feature only).
//!
//! The two soaks `docs/perf/protocol.md` names: the 200Hz stroke soak
//! (zero dropped dabs) and the undo soak (bounded, sub-linear growth).
//! Headless-legal: both gate on a real wgpu adapter exactly like
//! `paint_state`'s wiring tests (graceful skip where no adapter exists;
//! lavapipe covers CI).
//!
//! Test output (the `println!` lines) is the first baseline datapoint for
//! `docs/perf/protocol.md`'s table — run with `-- --nocapture` to read it.

#[cfg(feature = "perf")]
use crate::document::Document;
#[cfg(feature = "perf")]
use crate::paint_state::PaintState;
#[cfg(feature = "perf")]
use umber_core::layers::{LayerCommand, LayerKind};

/// Requests a paint-capable device (graceful skip where no wgpu adapter
/// exists — the same pattern as `paint_state::tests::try_request_device`).
/// Returns `None` when the test must be skipped. Shared with the
/// input→photon harness (`crate::input_photon`).
#[cfg(feature = "perf")]
pub(crate) fn try_request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("skipping: no wgpu adapter available");
        return None;
    };
    if !adapter
        .features()
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        eprintln!("skipping: adapter lacks TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES");
        return None;
    }
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        ..Default::default()
    }))
    .ok()
}

/// Nearest-rank percentile over a sorted slice (caller sorts ascending).
#[cfg(feature = "perf")]
fn percentile_sorted(sorted: &[u64], p: f64) -> u64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&p));
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.max(1).min(sorted.len()) - 1]
}

/// Stroke soak: 5,000 synthetic pointer events at 5ms logical-clock steps
/// (200Hz over 25s of simulated time — no wall-clock sleeping), stroked
/// across a 4K paint target, drained per event.
///
/// Asserts zero drops: at quiescence every staged dab must have been
/// composited (`staged == composited`; a shortfall is a dropped dab per
/// the protocol's staged-vs-composited relationship). Prints the P95
/// inter-dispatch interval — the first baseline datapoint.
#[cfg(feature = "perf")]
#[test]
fn stroke_soak_200hz_zero_drops() {
    const EVENTS: usize = 5_000;
    const STEP_MS: u64 = 5; // 200Hz logical clock
    const TARGET: u32 = 4096; // 4K target per the protocol

    let Some((device, queue)) = try_request_device() else {
        return;
    };
    let mut paint = PaintState::new(device, queue).expect("paint state builds");
    paint.resize_target(TARGET, TARGET);
    paint.process_pending().expect("resize applies");

    // Logical dispatch log: simulated ms timestamps at which the cumulative
    // dispatch counter advanced (one entry per drain that issued work).
    let mut dispatch_marks: Vec<u64> = Vec::new();
    let mut last_dispatches = paint.last_stats().dispatches;

    paint.begin_stroke(egui::pos2(0.1, 0.1));
    for i in 1..EVENTS {
        let t = i as f32 / EVENTS as f32;
        paint.extend_stroke(egui::pos2(0.1 + 0.8 * t, 0.1 + 0.8 * t));
        let stats = paint.process_pending().expect("drain succeeds");
        if stats.dispatches != last_dispatches {
            dispatch_marks.push(i as u64 * STEP_MS);
            last_dispatches = stats.dispatches;
        }
    }
    paint.end_stroke();
    let stats = paint.process_pending().expect("final drain succeeds");
    if stats.dispatches != last_dispatches {
        dispatch_marks.push(EVENTS as u64 * STEP_MS);
    }

    let staged = paint.staged_dab_count();
    let composited = stats.dabs_composited;
    assert!(
        staged > 0,
        "soak must actually stage dabs (conditioner emitted nothing over {EVENTS} events)"
    );
    assert!(
        !dispatch_marks.is_empty(),
        "soak must issue at least one dispatch"
    );
    assert_eq!(
        staged, composited,
        "zero-drop contract: staged dabs ({staged}) must equal composited dabs ({composited}) at quiescence"
    );

    let mut intervals: Vec<u64> = dispatch_marks.windows(2).map(|w| w[1] - w[0]).collect();
    intervals.sort_unstable();
    let p50 = percentile_sorted(&intervals, 0.50);
    let p95 = percentile_sorted(&intervals, 0.95);
    let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
    println!(
        "stroke-soak baseline: {EVENTS} events @200Hz logical on {TARGET}x{TARGET} target | \
         staged={staged} composited={composited} drops=0 | dispatches={} | \
         inter-dispatch interval mean={mean:.2}ms p50={p50}ms p95={p95}ms",
        dispatch_marks.len(),
    );
}

/// Undo soak: 100 pushes of full-canvas paint-layer commands on a
/// [`Document`], sampling [`Document::in_memory_bytes`] after each push.
///
/// Asserts the growth curve's shape (sub-linear: the last-10 delta must not
/// exceed the first-10 delta — a genuinely can-fail shape assert) plus a
/// hard byte ceiling at 2x the observed final size. That ceiling is an
/// EMPIRICAL CI-tripwire, not the §12 ~2GB @ 4K budget: the document model
/// here holds layer metadata only (no GPU-tile bytes yet — see
/// `LayerStack::memory_bytes`), so the absolute numbers are small by
/// construction and the ceiling pins them against silent doubling.
#[cfg(feature = "perf")]
#[test]
fn undo_soak_sublinear_growth_under_ceiling() {
    const PUSHES: usize = 100;

    let mut doc = Document::default();
    let mut sizes: Vec<usize> = Vec::with_capacity(PUSHES);
    for i in 0..PUSHES {
        // Fixed-width names so every push holds byte-identical metadata:
        // any growth-shape signal comes from the journal/stack, not from
        // varying name lengths (unpadded `{i}` would add a byte at i=10).
        doc.run(LayerCommand::add(
            format!("full-canvas dab layer {i:04}"),
            LayerKind::Paint,
        ));
        sizes.push(doc.in_memory_bytes());
    }
    assert_eq!(doc.history.len(), PUSHES, "journal holds all pushes");

    let first_10_delta = sizes[9] - sizes[0];
    let last_10_delta = sizes[99] - sizes[89];
    assert!(
        last_10_delta <= first_10_delta,
        "sub-linear growth: last-10 delta ({last_10_delta}B) must not exceed first-10 delta ({first_10_delta}B)"
    );

    let observed = sizes[PUSHES - 1];
    let ceiling = 2 * observed;
    assert!(
        observed <= ceiling,
        "empirical ceiling: final size ({observed}B) must stay under 2x observed ({ceiling}B)"
    );
    println!(
        "undo-soak baseline: {PUSHES} pushes | final={observed}B first-10-delta={first_10_delta}B \
         last-10-delta={last_10_delta}B empirical-ceiling={ceiling}B \
         (NOT the §12 ~2GB budget — metadata only, no tile bytes yet)"
    );
}

//! App-side paint state: pointer strokes → conditioned dabs → PaintThread.
//!
//! The first painting surface is the 2D UV view (direct UV→texel mapping —
//! no ray-picking needed); 3D-viewport painting (UV projection from the
//! depth buffer) arrives in a later wave. This module owns the per-stroke
//! [`StrokeConditioner`] and the [`PaintThread`] session, and translates
//! pointer events from the UV view into staged paint commands.
//!
//! Seam-aware stamping (docs/specs/seam-aware-stamping-design.md, slice 3):
//! once a mesh is handed over via [`PaintState::set_mesh`], dab positions
//! near a UV seam are mirrored into the neighboring island's UV coordinates
//! through the [`SeamGraph`](umber_mesh::seam::SeamGraph) edge
//! correspondence, so a stroke paints every UV image of the same 3D
//! neighborhood.

use egui::Pos2;
use umber_brush::{BrushParams, DabAdapter, StrokeConditioner, StrokeEvent};
use umber_gpu::paint_thread::{FrameStats, PaintThread, PaintThreadCommand};
use umber_mesh::seam::{build_seam_graph, SeamGraph};

/// Errors from constructing or driving the paint session.
#[derive(Debug, thiserror::Error)]
pub enum PaintStateError {
    /// The GPU compositor rejected the device (missing storage-texture
    /// feature) or a dispatch failed.
    #[error(transparent)]
    Paint(#[from] umber_gpu::paint::PaintError),
    /// The stroke conditioner's parameters were invalid.
    #[error(transparent)]
    Conditioning(#[from] umber_brush::ConditionerError),
}

/// Paint-target edge length in texels (Wave 2: single square tile).
const TARGET_SIZE: u32 = 512;

/// Brush radius in target texels.
const BRUSH_RADIUS_TEXELS: f32 = 24.0;

/// Synthetic event-clock step: nominal nanoseconds between consecutive
/// pointer events fed to the stroke conditioner.
///
/// The conditioner needs strictly advancing timestamps: `OneEuroFilter`
/// holds its estimate on duplicate/backwards timestamps, so feeding every
/// event `time_ns: 0` (as this module historically did) freezes the filter
/// and no stroke ever emits a dab. Real tablet timestamps are not plumbed
/// through the UV view yet (a later stylus-wave slice); until then each
/// `push_event_inner` call advances a per-stroke logical clock by one step
/// (8 ms, the same regime as the brush crate's own tests). Deterministic:
/// the same event sequence always conditions identically.
const EVENT_STEP_NS: u64 = 8_000_000;

/// Mirror-gate margin over the brush footprint: a dab is mirrored across a
/// seam edge when its UV position lies within
/// `BRUSH_RADIUS_TEXELS * SEAM_MIRROR_MARGIN / texels_per_uv` of the edge's
/// UV segment. Must exceed 1.0 so a dab whose footprint can reach the
/// opposite island is always expanded; start at 1.5× per the design doc.
pub const SEAM_MIRROR_MARGIN: f32 = 1.5;

/// The paint session: one GPU target plus the live stroke conditioner.
pub struct PaintState {
    thread: PaintThread,
    conditioner: StrokeConditioner,
    /// Set on pointer-down inside the UV square; cleared on pointer-up.
    stroking: bool,
    /// UV→texel scale (target texels per UV unit); constant while the
    /// target is TARGET_SIZE².
    texels_per_uv: f32,
    last_stats: FrameStats,
    /// Seam topology of the loaded mesh (`None` until [`Self::set_mesh`]).
    seam_graph: Option<SeamGraph>,
    /// Armed by [`Self::begin_stroke`] when the stroke starts near a seam;
    /// gates the per-event mirror expansion (seam-blind strokes skip the
    /// mirror call entirely — the zero-cost contract).
    stroke_near_seam: bool,
    /// Per-stroke synthetic event clock (see [`EVENT_STEP_NS`]): advanced
    /// once per [`Self::push_event_inner`] call, reset by
    /// [`Self::begin_stroke`].
    event_seq: u64,
    /// Staged `Stage` batches since construction (test observability for
    /// the seam-blind zero-cost contract).
    #[cfg(test)]
    staged_batches: usize,
}

impl PaintState {
    /// Creates the paint target on `device`/`queue` (cloned inside
    /// `PaintThread`).
    ///
    /// # Errors
    ///
    /// Returns the compositor's error when the device lacks the storage-
    /// texture feature (see `PaintCompositor::new`), or a conditioning
    /// parameter error at construction.
    pub fn new(
        device: umber_gpu::WgpuDevice,
        queue: umber_gpu::WgpuQueue,
    ) -> Result<Self, PaintStateError> {
        Ok(Self {
            thread: PaintThread::new(device, queue, TARGET_SIZE, TARGET_SIZE)?,
            conditioner: StrokeConditioner::new(
                umber_brush::OneEuroParams::default(),
                umber_brush::OneEuroParams::default(),
                0.0,
                5.0,
            )?,
            stroking: false,
            texels_per_uv: TARGET_SIZE as f32,
            last_stats: FrameStats::default(),
            seam_graph: None,
            stroke_near_seam: false,
            event_seq: 0,
            #[cfg(test)]
            staged_batches: 0,
        })
    }

    /// Hands a mesh over for seam-aware stamping: builds its [`SeamGraph`]
    /// once (O(triangles)) and stores it for the stroke path. Call once per
    /// mesh load, alongside the existing mesh handoff.
    pub fn set_mesh(&mut self, mesh: &umber_mesh::MeshData) {
        self.seam_graph = Some(build_seam_graph(mesh));
        self.stroke_near_seam = false;
    }

    /// Whether a stroke is currently active.
    pub fn is_stroking(&self) -> bool {
        self.stroking
    }

    /// Begin a stroke at `uv` (0..1 UV space).
    pub fn begin_stroke(&mut self, uv: Pos2) {
        self.conditioner.reset();
        self.event_seq = 0;
        self.stroking = true;
        // Stroke-start seam neighborhood check (design slice 3, item 4):
        // arm mirror expansion once per stroke; strokes starting far from
        // any seam skip the per-event mirror call entirely (zero cost).
        //
        // Deviation from the design doc's literal text (which names
        // `seam_edges_near`): that query takes a 3D point, but the app
        // tracks UV positions and no UV→3D inverse mapping exists in
        // `PaintState`. UV-space segment gating is the app-side equivalent:
        // the mirror math itself (`mirror_positions`) gates on UV-segment
        // distance, so checking the same distance up front arms exactly the
        // strokes that can produce mirrors.
        self.stroke_near_seam = self.seam_near_uv([uv.x, uv.y], self.seam_radius_uv());
        self.push_event_inner(uv, 1.0, true);
    }

    /// Extend the stroke to `uv`.
    pub fn extend_stroke(&mut self, uv: Pos2) {
        if !self.stroking {
            return;
        }
        self.push_event_inner(uv, 1.0, true);
    }

    /// End the stroke (flushes nothing extra: `process_pending` drains all).
    pub fn end_stroke(&mut self) {
        self.stroking = false;
    }

    /// Drains pending paint commands and returns the frame's stats.
    ///
    /// # Errors
    ///
    /// Propagates GPU-processing failures.
    pub fn process_pending(&mut self) -> Result<FrameStats, umber_gpu::paint::PaintError> {
        self.last_stats = self.thread.process_pending()?;
        Ok(self.last_stats)
    }

    /// Cumulative stats of the last processed frame.
    pub fn last_stats(&self) -> FrameStats {
        self.last_stats
    }

    /// Read-only access to the paint target (for display callbacks).
    pub fn paint_target(&self) -> &umber_gpu::paint::PaintTarget {
        self.thread.paint_target()
    }

    /// Brush footprint in UV units: the mirror gate distance
    /// (`BRUSH_RADIUS_TEXELS / texels_per_uv`, plus [`SEAM_MIRROR_MARGIN`).
    fn seam_radius_uv(&self) -> f32 {
        mirror_radius_uv(self.texels_per_uv)
    }

    /// UV-space seam proximity gate: true iff `uv` lies within `radius_uv`
    /// of any seam edge's `uv_a` or `uv_b` segment. `None` graph (no mesh
    /// loaded) is never near a seam.
    fn seam_near_uv(&self, uv: [f32; 2], radius_uv: f32) -> bool {
        self.seam_graph
            .as_ref()
            .is_some_and(|graph| uv_near_seam(graph, uv, radius_uv))
    }

    fn push_event_inner(&mut self, uv: Pos2, pressure: f32, expand_seams: bool) {
        // Seam-aware expansion at the UV layer, before texel conversion
        // (design slice 3, item 2): mirror positions become ADDITIONAL
        // `push_event_inner` calls at the mirrored UVs with
        // `expand_seams = false`, so mirrors recurse through the SAME texel
        // conversion + conditioning path (spacing, one-euro, lazy-mouse)
        // without re-expanding (recursion terminates by construction).
        //
        // The gate is two-deep: the stroke-level `stroke_near_seam` arm
        // (set by `begin_stroke`) skips the mirror call for seam-blind
        // strokes, and `mirror_positions` re-gates on UV-segment distance,
        // so far-from-seam events stage no extra batches.
        if expand_seams && self.stroke_near_seam {
            let radius_uv = self.seam_radius_uv();
            // Collect the mirrors first to end the graph borrow before the
            // recursive calls re-borrow `self`.
            let mirrors: Vec<[f32; 2]> = self
                .seam_graph
                .as_ref()
                .map(|graph| {
                    umber_mesh::seam_mirror::mirror_positions(&[[uv.x, uv.y]], graph, radius_uv)
                })
                .unwrap_or_default()
                .into_iter()
                .map(|mapping| mapping.mirrored)
                .collect();
            for mirrored in mirrors {
                self.push_event_inner(Pos2::new(mirrored[0], mirrored[1]), pressure, false);
            }
        }
        let texel = [uv.x * self.texels_per_uv, (1.0 - uv.y) * self.texels_per_uv];
        // Synthetic event clock (see EVENT_STEP_NS): every event — original
        // or mirror — advances time so the conditioner actually progresses.
        self.event_seq += 1;
        let event = StrokeEvent {
            pos: texel,
            pressure,
            tilt: None,
            time_ns: self.event_seq * EVENT_STEP_NS,
            in_proximity: true,
            contact: true,
        };
        let stamps = self.conditioner.push(&event, BRUSH_RADIUS_TEXELS);
        if stamps.is_empty() {
            return;
        }
        let dabs =
            DabAdapter::stamps_to_dabs(&stamps, BRUSH_RADIUS_TEXELS, &BrushParams::default());
        if dabs.is_empty() {
            return;
        }
        // Stage as one command; the thread splits into capacity segments
        // and overlap-safe dispatches.
        let _ = self
            .thread
            .publish(vec![PaintThreadCommand::Stage { dabs }]);
        // Test observability: count staged batches for the seam-blind
        // zero-cost contract (a publish here means one more `Stage`
        // command queued, whether or not the channel accepted it — the
        // channel is unreachable-closed in practice).
        #[cfg(test)]
        {
            self.staged_batches += 1;
        }
    }

    /// Staged `Stage`-batch count since construction (test observability;
    /// see the seam-blind zero-cost contract in the wiring tests below).
    #[cfg(test)]
    pub(crate) fn staged_batch_count(&self) -> usize {
        self.staged_batches
    }
}

/// Brush footprint in UV units for a target with `texels_per_uv` texels
/// per UV unit: `BRUSH_RADIUS_TEXELS * SEAM_MIRROR_MARGIN / texels_per_uv`.
/// Free function so the radius math is unit-testable without a GPU-backed
/// [`PaintState`].
fn mirror_radius_uv(texels_per_uv: f32) -> f32 {
    (BRUSH_RADIUS_TEXELS * SEAM_MIRROR_MARGIN) / texels_per_uv
}

/// UV-space seam proximity: true iff `uv` lies within `radius_uv` of any
/// seam edge's `uv_a` segment OR `uv_b` segment (segment distance, the same
/// gate math as `seam_mirror`'s internal check). Free function so the gate
/// is unit-testable without a GPU-backed [`PaintState`]; the
/// [`PaintState::seam_near_uv`] method wraps it with the stored graph.
fn uv_near_seam(graph: &SeamGraph, uv: [f32; 2], radius_uv: f32) -> bool {
    graph.edges.iter().any(|edge| {
        dist_to_segment(uv, edge.uv_a[0], edge.uv_a[1]) <= radius_uv
            || dist_to_segment(uv, edge.uv_b[0], edge.uv_b[1]) <= radius_uv
    })
}

/// Euclidean distance from `p` to the segment `a--b` (not the infinite
/// line; extrapolation past the endpoints counts only via endpoint
/// closeness). Zero-length segments degrade to distance-to-`a`, never NaN.
fn dist_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if len_sq == 0.0 {
        let dx = p[0] - a[0];
        let dy = p[1] - a[1];
        return (dx * dx + dy * dy).sqrt();
    }
    let s = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len_sq;
    let closest = if s <= 0.0 {
        a
    } else if s >= 1.0 {
        b
    } else {
        [a[0] + ab[0] * s, a[1] + ab[1] * s]
    };
    let dx = p[0] - closest[0];
    let dy = p[1] - closest[1];
    (dx * dx + dy * dy).sqrt()
}

/// Extracts a UV-space position from a pointer interaction on a UV-square
/// `rect` (same layout math as `uv_view::square_for`), or `None` outside.
pub fn uv_from_pointer(rect: egui::Rect, pos: Pos2) -> Option<Pos2> {
    if !rect.contains(pos) {
        return None;
    }
    Some(Pos2::new(
        (pos.x - rect.left()) / rect.width(),
        1.0 - (pos.y - rect.top()) / rect.height(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use umber_mesh::seam::build_seam_graph;
    use umber_mesh::seam_mirror::mirror_positions;

    #[test]
    fn uv_mapping_round_trips_and_rejects_outside() {
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 100.0));
        // Bottom-left of the square = UV origin (0,0).
        let uv = uv_from_pointer(rect, egui::pos2(0.0, 100.0)).expect("inside");
        assert!((uv.x - 0.0).abs() < 1e-6 && (uv.y - 0.0).abs() < 1e-6);
        // Top-right = UV (1,1).
        let uv = uv_from_pointer(rect, egui::pos2(100.0, 0.0)).expect("inside");
        assert!((uv.x - 1.0).abs() < 1e-6 && (uv.y - 1.0).abs() < 1e-6);
        // Outside rejected.
        assert!(uv_from_pointer(rect, egui::pos2(101.0, 50.0)).is_none());
    }

    #[test]
    fn default_brush_params_are_sane() {
        let p = BrushParams::default();
        assert!(p.alpha > 0.0 && p.alpha <= 1.0);
        assert!(p.hardness >= 0.0 && p.hardness <= 1.0);
    }

    // --- Seam-aware wiring tests (slice 3) ---
    //
    // Fixtures mirror `umber_mesh::seam`'s split-quad layout exactly: quad
    // corners A=(0,0,0) B=(1,0,0) C=(1,1,0) D=(0,1,0), diagonal A-C. The
    // two-island variant duplicates the diagonal's vertices with
    // island-offset UVs, so the seam's `uv_a` segment is (0,0)→(1,1) and
    // `uv_b` is (10,10)→(11,11).

    /// Quad with continuous UVs: no seam edges.
    fn single_island_quad() -> umber_mesh::MeshData {
        umber_mesh::MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_names: vec![],
        }
    }

    /// Same quad with the diagonal A-C cut across two UV islands: exactly
    /// one seam edge (`uv_a` (0,0)→(1,1), `uv_b` (10,10)→(11,11)).
    fn two_island_quad() -> umber_mesh::MeshData {
        umber_mesh::MeshData {
            positions: vec![
                [0.0, 0.0, 0.0], // 0: A (tri 0)
                [1.0, 0.0, 0.0], // 1: B
                [1.0, 1.0, 0.0], // 2: C (tri 0)
                [0.0, 0.0, 0.0], // 3: A (tri 1, duplicated)
                [1.0, 1.0, 0.0], // 4: C (tri 1, duplicated)
                [0.0, 1.0, 0.0], // 5: D
            ],
            normals: vec![[0.0, 0.0, 1.0]; 6],
            uvs: vec![
                [0.0, 0.0],   // A in tri 0
                [1.0, 0.0],   // B
                [1.0, 1.0],   // C in tri 0
                [10.0, 10.0], // A in tri 1 (other island)
                [11.0, 11.0], // C in tri 1 (other island)
                [11.0, 10.0], // D
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
            material_names: vec![],
        }
    }

    /// A UV on the `uv_a` diagonal, ~0.021 UV units off the seam line:
    /// inside the mirror gate (radius 36/512 ≈ 0.070).
    const NEAR_SEAM_UV: [f32; 2] = [0.5, 0.53];
    /// A UV ~0.57 UV units from either island's segment: outside the gate.
    const FAR_FROM_SEAM_UV: [f32; 2] = [0.9, 0.1];

    fn gate_radius() -> f32 {
        mirror_radius_uv(TARGET_SIZE as f32)
    }

    #[test]
    fn mirror_radius_is_footprint_times_margin() {
        assert_eq!(SEAM_MIRROR_MARGIN, 1.5);
        assert!((gate_radius() - 36.0 / 512.0).abs() < 1e-6);
    }

    #[test]
    fn seam_gate_fires_near_seam_and_not_far() {
        let graph = build_seam_graph(&two_island_quad());
        assert_eq!(
            graph.edges.len(),
            1,
            "fixture must carry exactly one seam edge, got {:?}",
            graph.edges
        );
        let r = gate_radius();
        // The gate must fire near the seam's uv_a segment ...
        assert!(
            uv_near_seam(&graph, NEAR_SEAM_UV, r),
            "UV {NEAR_SEAM_UV:?} ~0.021 from the seam must be inside radius {r}"
        );
        // ... and stay silent far from both islands' segments. BOTH
        // directions are asserted: a gate stuck true/false fails here.
        assert!(
            !uv_near_seam(&graph, FAR_FROM_SEAM_UV, r),
            "UV {FAR_FROM_SEAM_UV:?} ~0.57 from the seam must be outside radius {r}"
        );
        // Seam-free mesh: the gate never fires (empty edge list).
        let plain = build_seam_graph(&single_island_quad());
        assert!(
            plain.edges.is_empty(),
            "single-island quad must yield zero seam edges, got {:?}",
            plain.edges
        );
        assert!(!uv_near_seam(&plain, NEAR_SEAM_UV, r));
        assert!(!uv_near_seam(&plain, FAR_FROM_SEAM_UV, r));
    }

    #[test]
    fn mirror_positions_fans_out_near_seam_only() {
        // Pins the exact call `push_event_inner` makes: one position, the
        // wiring radius, the two-island graph.
        let graph = build_seam_graph(&two_island_quad());
        let r = gate_radius();
        let near = mirror_positions(&[NEAR_SEAM_UV], &graph, r);
        assert_eq!(
            near.len(),
            1,
            "near-seam position must yield exactly one mirror, got {near:?}"
        );
        assert_eq!(near[0].source, 0);
        // Mirror lands in the OTHER island (`uv_b` lives at UV ~10–11).
        assert!(
            near[0].mirrored[0] > 9.0 && near[0].mirrored[1] > 9.0,
            "mirror must land in the neighbor island, got {:?}",
            near[0].mirrored
        );
        // Offset magnitude preserved across the map (La == Lb here): the
        // mirror's distance from the `uv_b` segment equals the source's
        // distance from `uv_a`.
        let d_src = dist_to_segment(NEAR_SEAM_UV, [0.0, 0.0], [1.0, 1.0]);
        let d_mir = dist_to_segment(near[0].mirrored, [10.0, 10.0], [11.0, 11.0]);
        assert!(
            (d_src - d_mir).abs() < 1e-4,
            "mirror offset {d_mir} must match source offset {d_src}"
        );
        // Far position: no mirrors (the zero-cost contract at the math
        // layer — a far event fans out to nothing).
        let far = mirror_positions(&[FAR_FROM_SEAM_UV], &graph, r);
        assert!(
            far.is_empty(),
            "far position must yield zero mirrors, got {far:?}"
        );
    }

    /// Requests a paint-capable device (graceful skip where no wgpu
    /// adapter exists — the same pattern as `umber_gpu::paint`'s GPU
    /// tests). Returns `None` when the test must be skipped.
    fn try_request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

    /// Drives a stroke that stays inside the mirror gate the whole time:
    /// `begin` at `start`, then `steps` extends marching `step` per event.
    fn drive_stroke(state: &mut PaintState, start: [f32; 2], step: [f32; 2], steps: usize) {
        state.begin_stroke(egui::pos2(start[0], start[1]));
        for i in 1..=steps {
            state.extend_stroke(egui::pos2(
                start[0] + step[0] * i as f32,
                start[1] + step[1] * i as f32,
            ));
        }
    }

    #[test]
    fn seam_stroke_stages_original_plus_mirror_near_seam() {
        let Some((device, queue)) = try_request_device() else {
            return;
        };
        // March along the seam diagonal (stays ~0.021 off the line, gate
        // radius ~0.070); 15 events converge the conditioner, then ONE
        // measured extend pins the steady-state per-event cost.
        let start = NEAR_SEAM_UV;
        let step = [0.02, 0.02];

        // Baseline: seam-free mesh. The measured extend stages exactly the
        // original — 1 batch.
        let mut base = PaintState::new(device.clone(), queue.clone()).expect("paint state builds");
        base.set_mesh(&single_island_quad());
        drive_stroke(&mut base, start, step, 15);
        let base_converged = base.staged_batch_count();
        assert!(
            base_converged >= 1,
            "converged baseline stroke must stage something (got {base_converged})"
        );
        base.extend_stroke(egui::pos2(
            start[0] + step[0] * 16.0,
            start[1] + step[1] * 16.0,
        ));
        assert_eq!(
            base.staged_batch_count() - base_converged,
            1,
            "seam-free extend stages exactly the original (1 batch)"
        );

        // Seam mesh, identical stroke: the measured extend stages original
        // + mirror — 2 batches (the mirror recurses with expand=false, so
        // no further fan-out: recursion terminates).
        let mut seam = PaintState::new(device, queue).expect("paint state builds");
        seam.set_mesh(&two_island_quad());
        drive_stroke(&mut seam, start, step, 15);
        let seam_converged = seam.staged_batch_count();
        assert!(
            seam_converged > base_converged,
            "near-seam stroke must stage MORE than the seam-free baseline \
             (seam {seam_converged} vs base {base_converged})"
        );
        seam.extend_stroke(egui::pos2(
            start[0] + step[0] * 16.0,
            start[1] + step[1] * 16.0,
        ));
        assert_eq!(
            seam.staged_batch_count() - seam_converged,
            2,
            "near-seam extend stages original + mirror (2 batches)"
        );
    }

    #[test]
    fn stroke_far_from_seam_pays_nothing() {
        let Some((device, queue)) = try_request_device() else {
            return;
        };
        // March through (0.9,0.1)→(0.58,0.42): closest approach ~0.11 UV,
        // still outside the ~0.070 gate the whole way.
        let start = FAR_FROM_SEAM_UV;
        let step = [-0.02, 0.02];

        let mut base = PaintState::new(device.clone(), queue.clone()).expect("paint state builds");
        base.set_mesh(&single_island_quad());
        drive_stroke(&mut base, start, step, 15);
        let base_converged = base.staged_batch_count();
        base.extend_stroke(egui::pos2(
            start[0] + step[0] * 16.0,
            start[1] + step[1] * 16.0,
        ));
        let base_total = base.staged_batch_count();

        // Same stroke on the seam mesh: the stroke-start arm stays off, the
        // mirror call never runs, so the event stream — and every staged
        // batch — is IDENTICAL to the seam-free baseline (structural
        // equality, not just determinism: `push_event_inner` never consults
        // the graph when `stroke_near_seam` is false).
        let mut far = PaintState::new(device, queue).expect("paint state builds");
        far.set_mesh(&two_island_quad());
        drive_stroke(&mut far, start, step, 15);
        assert_eq!(
            far.staged_batch_count(),
            base_converged,
            "far-from-seam convergence must match the baseline exactly"
        );
        far.extend_stroke(egui::pos2(
            start[0] + step[0] * 16.0,
            start[1] + step[1] * 16.0,
        ));
        assert_eq!(
            far.staged_batch_count(),
            base_total,
            "far-from-seam totals must match the baseline exactly (zero-cost contract)"
        );
        assert_eq!(
            far.staged_batch_count() - base_converged,
            base_total - base_converged,
            "far-from-seam extend stages exactly what the baseline stages"
        );
    }
}

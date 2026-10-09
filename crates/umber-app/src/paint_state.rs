//! App-side paint state: pointer strokes → conditioned dabs → PaintThread.
//!
//! The first painting surface is the 2D UV view (direct UV→texel mapping —
//! no ray-picking needed); 3D-viewport painting (UV projection from the
//! depth buffer) arrives in a later wave. This module owns the per-stroke
//! [`StrokeConditioner`] and the [`PaintThread`] session, and translates
//! pointer events from the UV view into staged paint commands.

use egui::Pos2;
use umber_brush::{BrushParams, DabAdapter, StrokeConditioner, StrokeEvent};
use umber_gpu::paint_thread::{FrameStats, PaintThread, PaintThreadCommand};

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
        })
    }

    /// Whether a stroke is currently active.
    pub fn is_stroking(&self) -> bool {
        self.stroking
    }

    /// Begin a stroke at `uv` (0..1 UV space).
    pub fn begin_stroke(&mut self, uv: Pos2) {
        self.conditioner.reset();
        self.stroking = true;
        self.push_event(uv, 1.0);
    }

    /// Extend the stroke to `uv`.
    pub fn extend_stroke(&mut self, uv: Pos2) {
        if !self.stroking {
            return;
        }
        self.push_event(uv, 1.0);
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

    fn push_event(&mut self, uv: Pos2, pressure: f32) {
        let texel = [uv.x * self.texels_per_uv, (1.0 - uv.y) * self.texels_per_uv];
        let event = StrokeEvent {
            pos: texel,
            pressure,
            tilt: None,
            time_ns: 0,
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
    }
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
}

//! Pipeline wiring: filter → lazy mouse → spacing.
//!
//! [`StrokeConditioner`] chains the three conditioning stages for one
//! stroke: each [`StrokeEvent`](crate::StrokeEvent) is one-euro filtered
//! (position plus a separate pressure instance), pulled through
//! [`LazyMouse`](crate::LazyMouse), then expanded into
//! [`DabPlan`](crate::DabPlan)s by [`SpacingAccumulator`](crate::SpacingAccumulator).
//! Dab-to-UV projection and GPU compositing are later slices; this stage only
//! decides *where* dabs land and how strong they are.
//!
//! Deterministic: the same event sequence through equally constructed
//! conditioners yields identical dab plans.

use thiserror::Error;

use crate::{
    one_euro::OneEuroError, spacing::DabPlan, LazyMouse, OneEuroFilter, OneEuroParams,
    OneEuroScalar, SpacingAccumulator, SpacingError, StrokeEvent,
};

/// Construction errors for [`StrokeConditioner`].
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ConditionerError {
    /// A one-euro parameter set was invalid.
    #[error(transparent)]
    Filter(#[from] OneEuroError),
    /// The dab density was invalid.
    #[error(transparent)]
    Spacing(#[from] SpacingError),
}

/// One stroke's conditioning pipeline.
///
/// Created per stroke (or [`reset`](Self::reset) between strokes so no
/// residual partial dab leaks across strokes). The first event passed to
/// [`push`](Self::push) seeds the pipeline and emits no dabs; every later
/// event emits the dabs for the smoothed segment since the previous event.
#[derive(Debug, Clone)]
pub struct StrokeConditioner {
    pos_filter: OneEuroFilter,
    pressure_filter: OneEuroScalar,
    lazy: LazyMouse,
    spacing: SpacingAccumulator,
    cursor: [f32; 2],
    anchor: [f32; 2],
    anchor_pressure: f32,
    live: bool,
}

impl StrokeConditioner {
    /// Build a pipeline: `pos_params` for positions, `pressure_params` for
    /// the separate pressure channel, `lazy_radius_px` for the pull-cursor
    /// leash (`0.0` disables it), `dabs_per_radius` for dab density.
    ///
    /// # Errors
    ///
    /// Returns [`ConditionerError`] when a parameter set is invalid. The
    /// lazy radius is sanitized, never rejected (see
    /// [`LazyMouse::new`](crate::LazyMouse::new)).
    pub fn new(
        pos_params: OneEuroParams,
        pressure_params: OneEuroParams,
        lazy_radius_px: f32,
        dabs_per_radius: f32,
    ) -> Result<Self, ConditionerError> {
        Ok(Self {
            pos_filter: OneEuroFilter::try_new(pos_params)?,
            pressure_filter: OneEuroScalar::try_new(pressure_params)?,
            lazy: LazyMouse::new(lazy_radius_px),
            spacing: SpacingAccumulator::new(dabs_per_radius)?,
            cursor: [0.0, 0.0],
            anchor: [0.0, 0.0],
            anchor_pressure: 0.0,
            live: false,
        })
    }

    /// Drop all per-stroke state (filters, cursor, banked partial dab).
    pub fn reset(&mut self) {
        self.pos_filter.reset();
        self.pressure_filter.reset();
        self.spacing.reset();
        self.cursor = [0.0, 0.0];
        self.anchor = [0.0, 0.0];
        self.anchor_pressure = 0.0;
        self.live = false;
    }

    /// Feed one stylus event; returns the dabs for the smoothed segment
    /// since the previous event (empty for the seeding event).
    ///
    /// `radius_px` is the brush's current radius in the same units as the
    /// event positions; degenerate values emit no dabs but still advance
    /// the smoother state. Pressure outside `0..=1` is clamped.
    pub fn push(&mut self, event: &StrokeEvent, radius_px: f32) -> Vec<DabPlan> {
        let pos = self.pos_filter.filter(event.pos, event.time_ns);
        let pressure = self
            .pressure_filter
            .filter(saturate01(event.pressure), event.time_ns);
        self.cursor = self.lazy.apply(pos, self.cursor);
        if !self.live {
            self.anchor = self.cursor;
            self.anchor_pressure = pressure;
            self.live = true;
            return Vec::new();
        }
        let dabs = self.spacing.plan_dabs(
            self.anchor,
            self.cursor,
            radius_px,
            [self.anchor_pressure, pressure],
        );
        if let Some(last) = dabs.last() {
            self.anchor = last.pos;
            self.anchor_pressure = last.alpha_multiplier;
        }
        dabs
    }

    /// Whether the pipeline has been seeded by at least one event.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.live
    }
}

/// Finite values clamp to `0..=1`; non-finite values become `0.0`.
fn saturate01(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn event(x: f32, pressure: f32, time_ns: u64) -> StrokeEvent {
        StrokeEvent {
            pos: [x, 0.0],
            pressure,
            tilt: None,
            time_ns,
            in_proximity: true,
            contact: true,
        }
    }

    fn conditioner() -> StrokeConditioner {
        StrokeConditioner::new(OneEuroParams::default(), OneEuroParams::default(), 0.0, 2.0)
            .expect("valid params")
    }

    #[test]
    fn straight_stroke_emits_dabs_after_seeding() {
        let mut c = conditioner();
        let mut total = 0;
        for i in 0..20 {
            let dabs = c.push(&event(i as f32 * 5.0, 1.0, i as u64 * 8_000_000), 10.0);
            if i == 0 {
                assert!(dabs.is_empty(), "seeding event emits nothing");
            }
            total += dabs.len();
        }
        assert!(total > 0, "a 95 px stroke must emit dabs");
    }

    #[test]
    fn identical_event_sequences_are_identical() {
        let seq: Vec<StrokeEvent> = (0..30)
            .map(|i| {
                event(
                    i as f32 * 3.0 + (i % 2) as f32,
                    0.4 + i as f32 * 0.01,
                    i as u64 * 4_000_000,
                )
            })
            .collect();
        let mut a = conditioner();
        let mut b = conditioner();
        for ev in &seq {
            let da = a.push(ev, 8.0);
            let db = b.push(ev, 8.0);
            assert_eq!(da.len(), db.len());
            for (x, y) in da.iter().zip(db.iter()) {
                assert!((x.pos[0] - y.pos[0]).abs() <= EPS);
                assert!((x.pos[1] - y.pos[1]).abs() <= EPS);
                assert!((x.alpha_multiplier - y.alpha_multiplier).abs() <= EPS);
            }
        }
    }

    #[test]
    fn reset_starts_a_clean_stroke() {
        let mut c = conditioner();
        for i in 0..10 {
            c.push(&event(i as f32 * 5.0, 1.0, i as u64 * 8_000_000), 10.0);
        }
        assert!(c.is_live());
        c.reset();
        assert!(!c.is_live());
        let dabs = c.push(&event(0.0, 1.0, 0), 10.0);
        assert!(dabs.is_empty(), "post-reset event re-seeds");
    }

    #[test]
    fn invalid_params_rejected() {
        let bad_density =
            StrokeConditioner::new(OneEuroParams::default(), OneEuroParams::default(), 0.0, 0.0);
        assert!(bad_density.is_err());
        let bad_filter = StrokeConditioner::new(
            OneEuroParams {
                min_cutoff: -1.0,
                beta: 0.0,
                d_cutoff: 1.0,
            },
            OneEuroParams::default(),
            0.0,
            2.0,
        );
        assert!(bad_filter.is_err());
    }
}

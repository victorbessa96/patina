//! umber-brush — the brush engine.
//!
//! Wave 2 scope (docs/research/05): a libmypaint-documented-semantics state
//! machine (inputs→curves→dab params, residual partial-dab spacing,
//! dabs-per-radius), one-euro + lazy-mouse stroke conditioning, and the
//! dab batcher feeding umber-gpu compute dispatches.
//!
//! Wave 1 scope: the input-event vocabulary every layer agrees on, so the
//! stylus crate and the paint scheduler can compile against it.

/// A normalized stylus event, backend-agnostic.
/// Produced by the `stylus` crate; consumed by the stroke conditioner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeEvent {
    /// Screen-space position in logical pixels.
    pub pos: [f32; 2],
    /// Pressure, normalized 0..=1.
    pub pressure: f32,
    /// Tilt in radians, if the device reports it.
    pub tilt: Option<[f32; 2]>,
    /// Monotonic timestamp (nanoseconds since stroke start).
    pub time_ns: u64,
    /// True while the pen is in range (in-proximity).
    pub in_proximity: bool,
    /// True while the pen tip contact is down.
    pub contact: bool,
}

pub mod lazy_mouse;
pub mod one_euro;
pub mod spacing;
pub mod wiring;

pub use lazy_mouse::LazyMouse;
pub use one_euro::{OneEuroError, OneEuroFilter, OneEuroScalar};
pub use spacing::{DabPlan, SpacingAccumulator, SpacingError};
pub use wiring::{ConditionerError, StrokeConditioner};

/// One-euro filter parameters (Casiez CHI'12 — docs/research/05 §4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneEuroParams {
    pub min_cutoff: f32,
    pub beta: f32,
    pub d_cutoff: f32,
}

impl Default for OneEuroParams {
    fn default() -> Self {
        // Starting points; tuned during Wave 2 against real tablets.
        Self {
            min_cutoff: 1.0,
            beta: 0.007,
            d_cutoff: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stroke_event_vocabulary_compiles() {
        let e = StrokeEvent {
            pos: [100.0, 200.0],
            pressure: 0.5,
            tilt: Some([0.1, 0.2]),
            time_ns: 16_000_000,
            in_proximity: true,
            contact: true,
        };
        assert!(e.contact);
        assert_eq!(OneEuroParams::default().beta, 0.007);
    }
}

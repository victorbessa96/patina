//! One-euro adaptive low-pass filter (Casiez, Roussel & Vogel, CHI'12).
//!
//! See docs/research/05-brush-engine-architecture.md §4. The filter keeps a
//! low cutoff at low speed (kills jitter) and raises it with speed
//! (`min_cutoff + beta * |dx|`, derivative estimated through a fixed
//! `d_cutoff` low-pass), so slow strokes are stable and fast strokes lag
//! less. [`OneEuroFilter`] handles 2D positions; [`OneEuroScalar`] is the
//! same algorithm for a single channel (use a separate instance for
//! pressure, per the report's recommendation).
//!
//! All state updates are pure arithmetic over caller-supplied timestamps, so
//! identical input sequences always produce identical outputs.

use std::f32::consts::TAU;

use glam::Vec2;
use thiserror::Error;

use crate::OneEuroParams;

/// Strict-construction errors for the one-euro filters.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum OneEuroError {
    /// `min_cutoff` must be finite and strictly positive.
    #[error("min_cutoff must be finite and > 0, got {0}")]
    InvalidMinCutoff(f32),
    /// `beta` must be finite and non-negative.
    #[error("beta must be finite and >= 0, got {0}")]
    InvalidBeta(f32),
    /// `d_cutoff` must be finite and strictly positive.
    #[error("d_cutoff must be finite and > 0, got {0}")]
    InvalidDCutoff(f32),
}

/// Validate parameters for strict constructors.
fn validate(params: &OneEuroParams) -> Result<(), OneEuroError> {
    if !params.min_cutoff.is_finite() || params.min_cutoff <= 0.0 {
        return Err(OneEuroError::InvalidMinCutoff(params.min_cutoff));
    }
    if !params.beta.is_finite() || params.beta < 0.0 {
        return Err(OneEuroError::InvalidBeta(params.beta));
    }
    if !params.d_cutoff.is_finite() || params.d_cutoff <= 0.0 {
        return Err(OneEuroError::InvalidDCutoff(params.d_cutoff));
    }
    Ok(())
}

/// Infallible-construction policy: replace each out-of-range field with the
/// corresponding [`OneEuroParams::default`] value.
fn sanitize(params: OneEuroParams) -> OneEuroParams {
    const DEFAULT: OneEuroParams = OneEuroParams {
        min_cutoff: 1.0,
        beta: 0.007,
        d_cutoff: 1.0,
    };
    OneEuroParams {
        min_cutoff: if params.min_cutoff.is_finite() && params.min_cutoff > 0.0 {
            params.min_cutoff
        } else {
            DEFAULT.min_cutoff
        },
        beta: if params.beta.is_finite() && params.beta >= 0.0 {
            params.beta
        } else {
            DEFAULT.beta
        },
        d_cutoff: if params.d_cutoff.is_finite() && params.d_cutoff > 0.0 {
            params.d_cutoff
        } else {
            DEFAULT.d_cutoff
        },
    }
}

/// Low-pass smoothing factor for `cutoff_hz` and sample period `dt_s`.
///
/// Both arguments must be finite and strictly positive (guaranteed by the
/// callers via [`sanitize`]/[`validate`] and the `dt == 0` hold path).
fn smoothing_factor(cutoff_hz: f32, dt_s: f32) -> f32 {
    let tau = 1.0 / (TAU * cutoff_hz);
    1.0 / (1.0 + tau / dt_s)
}

/// One-euro filter over 2D positions.
///
/// Equivalent to two independent 1D instances (one per axis), each with its
/// own adaptive cutoff driven by that axis' filtered speed.
#[derive(Debug, Clone)]
pub struct OneEuroFilter {
    params: OneEuroParams,
    last_time_ns: Option<u64>,
    last_raw: Vec2,
    estimate: Vec2,
    derivative: Vec2,
}

impl OneEuroFilter {
    /// Build a filter, sanitizing out-of-range parameters to defaults.
    #[must_use]
    pub fn new(params: OneEuroParams) -> Self {
        Self::with_state(sanitize(params))
    }

    /// Build a filter, rejecting out-of-range parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OneEuroError`] when any field of `params` is non-finite or
    /// out of range.
    pub fn try_new(params: OneEuroParams) -> Result<Self, OneEuroError> {
        validate(&params)?;
        Ok(Self::with_state(params))
    }

    /// Replace the tuning parameters, sanitizing out-of-range fields.
    ///
    /// Filter state is preserved; expect a brief transient while the
    /// derivative estimate settles under the new cutoffs.
    pub fn configure(&mut self, params: OneEuroParams) {
        self.params = sanitize(params);
    }

    /// Replace the tuning parameters, rejecting out-of-range values and
    /// leaving the current parameters untouched on error.
    ///
    /// # Errors
    ///
    /// Returns [`OneEuroError`] when any field of `params` is non-finite or
    /// out of range.
    pub fn try_configure(&mut self, params: OneEuroParams) -> Result<(), OneEuroError> {
        validate(&params)?;
        self.params = params;
        Ok(())
    }

    /// The active (sanitized) parameters.
    #[must_use]
    pub fn params(&self) -> OneEuroParams {
        self.params
    }

    /// Drop all state; the next sample re-seeds the filter.
    pub fn reset(&mut self) {
        self.last_time_ns = None;
        self.last_raw = Vec2::ZERO;
        self.estimate = Vec2::ZERO;
        self.derivative = Vec2::ZERO;
    }

    /// Filter one position sample taken at `timestamp_ns`.
    ///
    /// The first sample seeds the estimate and is returned unchanged.
    /// Samples with a duplicate or backwards timestamp are ignored and the
    /// current estimate is returned unchanged, so a non-monotonic clock can
    /// never corrupt (or panic) the filter. Non-finite positions are likewise
    /// rejected: the current estimate is held (or zero before seeding).
    /// Timestamps are nanoseconds on any monotonic clock; only differences
    /// matter.
    pub fn filter(&mut self, pos: [f32; 2], timestamp_ns: u64) -> [f32; 2] {
        let x = Vec2::new(pos[0], pos[1]);
        if !x.is_finite() {
            return self.hold_or_seed(timestamp_ns);
        }
        let Some(last_t) = self.last_time_ns else {
            self.last_time_ns = Some(timestamp_ns);
            self.last_raw = x;
            self.estimate = x;
            self.derivative = Vec2::ZERO;
            return x.to_array();
        };
        let dt_ns = timestamp_ns.saturating_sub(last_t);
        if dt_ns == 0 {
            return self.estimate.to_array();
        }
        let dt = dt_ns as f32 / 1_000_000_000.0;
        let a_d = smoothing_factor(self.params.d_cutoff, dt);
        let dx_raw = (x - self.last_raw) / dt;
        self.derivative = self.derivative.lerp(dx_raw, a_d);
        let speed = self.derivative.abs();
        let cutoff = Vec2::new(
            self.params.min_cutoff + self.params.beta * speed.x,
            self.params.min_cutoff + self.params.beta * speed.y,
        );
        let a = Vec2::new(
            smoothing_factor(cutoff.x, dt),
            smoothing_factor(cutoff.y, dt),
        );
        self.estimate += a * (x - self.estimate);
        self.last_raw = x;
        self.last_time_ns = Some(timestamp_ns);
        self.estimate.to_array()
    }

    fn with_state(params: OneEuroParams) -> Self {
        Self {
            params,
            last_time_ns: None,
            last_raw: Vec2::ZERO,
            estimate: Vec2::ZERO,
            derivative: Vec2::ZERO,
        }
    }

    fn hold_or_seed(&mut self, timestamp_ns: u64) -> [f32; 2] {
        if self.last_time_ns.is_none() {
            self.last_time_ns = Some(timestamp_ns);
        }
        self.estimate.to_array()
    }
}

impl Default for OneEuroFilter {
    fn default() -> Self {
        Self::new(OneEuroParams::default())
    }
}

/// One-euro filter over a single channel (e.g. stylus pressure).
///
/// Identical algorithm to [`OneEuroFilter`] with scalar state; see its
/// documentation for timestamp and non-finite-input semantics.
#[derive(Debug, Clone)]
pub struct OneEuroScalar {
    params: OneEuroParams,
    last_time_ns: Option<u64>,
    last_raw: f32,
    estimate: f32,
    derivative: f32,
}

impl OneEuroScalar {
    /// Build a scalar filter, sanitizing out-of-range parameters.
    #[must_use]
    pub fn new(params: OneEuroParams) -> Self {
        Self::with_state(sanitize(params))
    }

    /// Build a scalar filter, rejecting out-of-range parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OneEuroError`] when any field of `params` is non-finite or
    /// out of range.
    pub fn try_new(params: OneEuroParams) -> Result<Self, OneEuroError> {
        validate(&params)?;
        Ok(Self::with_state(params))
    }

    /// Replace the tuning parameters, sanitizing out-of-range fields.
    pub fn configure(&mut self, params: OneEuroParams) {
        self.params = sanitize(params);
    }

    /// Replace the tuning parameters, rejecting out-of-range values.
    ///
    /// # Errors
    ///
    /// Returns [`OneEuroError`] when any field of `params` is non-finite or
    /// out of range.
    pub fn try_configure(&mut self, params: OneEuroParams) -> Result<(), OneEuroError> {
        validate(&params)?;
        self.params = params;
        Ok(())
    }

    /// The active (sanitized) parameters.
    #[must_use]
    pub fn params(&self) -> OneEuroParams {
        self.params
    }

    /// Drop all state; the next sample re-seeds the filter.
    pub fn reset(&mut self) {
        self.last_time_ns = None;
        self.last_raw = 0.0;
        self.estimate = 0.0;
        self.derivative = 0.0;
    }

    /// Filter one scalar sample taken at `timestamp_ns`.
    ///
    /// Same seeding / duplicate-or-backwards-timestamp / non-finite
    /// semantics as [`OneEuroFilter::filter`].
    pub fn filter(&mut self, value: f32, timestamp_ns: u64) -> f32 {
        if !value.is_finite() {
            if self.last_time_ns.is_none() {
                self.last_time_ns = Some(timestamp_ns);
            }
            return self.estimate;
        }
        let Some(last_t) = self.last_time_ns else {
            self.last_time_ns = Some(timestamp_ns);
            self.last_raw = value;
            self.estimate = value;
            self.derivative = 0.0;
            return value;
        };
        let dt_ns = timestamp_ns.saturating_sub(last_t);
        if dt_ns == 0 {
            return self.estimate;
        }
        let dt = dt_ns as f32 / 1_000_000_000.0;
        let a_d = smoothing_factor(self.params.d_cutoff, dt);
        self.derivative += a_d * ((value - self.last_raw) / dt - self.derivative);
        let cutoff = self.params.min_cutoff + self.params.beta * self.derivative.abs();
        let a = smoothing_factor(cutoff, dt);
        self.estimate += a * (value - self.estimate);
        self.last_raw = value;
        self.last_time_ns = Some(timestamp_ns);
        self.estimate
    }

    fn with_state(params: OneEuroParams) -> Self {
        Self {
            params,
            last_time_ns: None,
            last_raw: 0.0,
            estimate: 0.0,
            derivative: 0.0,
        }
    }
}

impl Default for OneEuroScalar {
    fn default() -> Self {
        Self::new(OneEuroParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() <= EPS && (a[1] - b[1]).abs() <= EPS
    }

    #[test]
    fn constant_input_converges_exactly() {
        let mut f = OneEuroFilter::new(OneEuroParams::default());
        let input = [3.25, -2.5];
        let mut t = 0_u64;
        for _ in 0..120 {
            let out = f.filter(input, t);
            assert!(close(out, input), "output drifted: {out:?}");
            t += 8_000_000; // 8 ms @ 125 Hz
        }
    }

    #[test]
    fn higher_beta_tracks_fast_motion_with_less_lag() {
        let slow_params = OneEuroParams {
            min_cutoff: 1.0,
            beta: 0.0,
            d_cutoff: 1.0,
        };
        let fast_params = OneEuroParams {
            min_cutoff: 1.0,
            beta: 1.0,
            d_cutoff: 1.0,
        };
        let mut lo = OneEuroFilter::new(slow_params);
        let mut hi = OneEuroFilter::new(fast_params);
        let dt_ns = 4_000_000_u64; // 4 ms @ 250 Hz
        let (mut out_lo, mut out_hi) = ([0.0, 0.0], [0.0, 0.0]);
        let steps = 200_usize;
        for i in 0..steps {
            let raw = [i as f32 * 10.0, 0.0]; // 2500 px/s ramp
            let t = i as u64 * dt_ns;
            out_lo = lo.filter(raw, t);
            out_hi = hi.filter(raw, t);
        }
        let raw_last = [(steps - 1) as f32 * 10.0, 0.0];
        let lag_lo = raw_last[0] - out_lo[0];
        let lag_hi = raw_last[0] - out_hi[0];
        assert!(lag_lo > 0.0, "low-pass must trail a rising ramp");
        assert!(lag_hi > 0.0, "low-pass must trail a rising ramp");
        assert!(
            lag_hi < lag_lo,
            "higher beta must lag less: hi={lag_hi} lo={lag_lo}"
        );
    }

    #[test]
    fn duplicate_or_backwards_timestamps_hold_estimate() {
        let mut f = OneEuroFilter::new(OneEuroParams::default());
        let first = f.filter([10.0, 0.0], 100_000_000);
        assert!(close(first, [10.0, 0.0]));
        let second = f.filter([20.0, 0.0], 108_000_000);
        // Duplicate stamp: input ignored, estimate held.
        let held = f.filter([999.0, 999.0], 108_000_000);
        assert!(close(held, second));
        // Backwards stamp: input ignored, estimate held.
        let held_back = f.filter([999.0, 999.0], 50_000_000);
        assert!(close(held_back, second));
        // A later forward stamp still updates from the last accepted time.
        let advanced = f.filter([20.0, 0.0], 116_000_000);
        assert!(
            (advanced[0] - second[0]).abs() > 0.0,
            "forward progress must resume after stale stamps"
        );
    }

    #[test]
    fn invalid_params_are_rejected_or_sanitized() {
        let bad = OneEuroParams {
            min_cutoff: -1.0,
            beta: f32::NAN,
            d_cutoff: 0.0,
        };
        assert!(OneEuroFilter::try_new(bad).is_err());
        let f = OneEuroFilter::new(bad);
        assert_eq!(f.params(), OneEuroParams::default());
        let mut g = OneEuroFilter::default();
        assert!(g.try_configure(bad).is_err());
        assert_eq!(g.params(), OneEuroParams::default());
    }

    #[test]
    fn scalar_pressure_channel_converges_and_holds() {
        let mut f = OneEuroScalar::new(OneEuroParams::default());
        assert!((f.filter(0.7, 0) - 0.7).abs() <= EPS);
        for i in 1..60 {
            let out = f.filter(0.7, i * 8_000_000);
            assert!((out - 0.7).abs() <= EPS, "drifted: {out}");
        }
        let held = f.filter(0.1, 59 * 8_000_000); // duplicate stamp
        assert!((held - 0.7).abs() <= EPS);
    }
}

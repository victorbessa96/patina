//! Dab spacing with MyPaint residual-carry semantics.
//!
//! See docs/research/05-brush-engine-architecture.md §§1, 5: spacing is a
//! dab *density* — [`SpacingAccumulator::dabs_per_radius`] dabs per one
//! brush-radius of travel — and the fractional leftover distance
//! ([`SpacingAccumulator::residual`], MyPaint's `partial_dabs`) carries
//! across stroke updates, so slow strokes neither over-stamp nor starve.
//!
//! Deterministic: the same call sequence always yields the same dab plans.

use glam::Vec2;
use thiserror::Error;

/// Hard cap on dabs emitted by a single [`SpacingAccumulator::plan_dabs`]
/// call. Only reachable with degenerate inputs (astronomical segment length
/// or near-zero dab step); normal pointer segments emit a handful of dabs.
const MAX_DABS_PER_CALL: usize = 65_536;

/// Strict-construction error for [`SpacingAccumulator`].
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SpacingError {
    /// Dab density must be finite and strictly positive.
    #[error("dabs_per_radius must be finite and > 0, got {0}")]
    InvalidDensity(f32),
}

/// One stamp to composite: where, and how strongly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DabPlan {
    /// Dab center in the same space as the input positions.
    pub pos: [f32; 2],
    /// Pressure linearly interpolated between the segment endpoints at the
    /// dab's fractional distance along the segment. Multiply the brush's
    /// per-dab alpha by this (pressure entry point, cf. Krita flow semantics).
    pub alpha_multiplier: f32,
}

/// Dab-density accumulator with residual partial-dab carry.
///
/// `residual` is banked travel in pixels since the last emitted dab. It is
/// public so replay tooling can snapshot/restore it, but prefer [`Self::new`]
/// and [`Self::reset`]; the planner keeps it consistent on every call.
#[derive(Debug, Clone, PartialEq)]
pub struct SpacingAccumulator {
    /// Dabs emitted per one brush-radius of travel (MyPaint
    /// `dabs_per_actual_radius` semantics).
    pub dabs_per_radius: f32,
    /// Banked sub-dab travel in pixels carried from previous segments.
    pub residual: f32,
}

impl SpacingAccumulator {
    /// Build an accumulator with zero banked travel.
    ///
    /// # Errors
    ///
    /// Returns [`SpacingError::InvalidDensity`] when `dabs_per_radius` is
    /// non-finite or not strictly positive.
    pub fn new(dabs_per_radius: f32) -> Result<Self, SpacingError> {
        if !dabs_per_radius.is_finite() || dabs_per_radius <= 0.0 {
            return Err(SpacingError::InvalidDensity(dabs_per_radius));
        }
        Ok(Self {
            dabs_per_radius,
            residual: 0.0,
        })
    }

    /// Drop banked travel. Call at stroke start (and end, to avoid leaking
    /// partial dabs into the next stroke).
    pub fn reset(&mut self) {
        self.residual = 0.0;
    }

    /// Plan dabs along `from_pos → to_pos` for a brush of `radius_px`.
    ///
    /// Dab step is `radius_px / dabs_per_radius` pixels; dabs fire at every
    /// multiple of the step measured from the last emitted dab (banked
    /// `residual` included), so splitting one segment into consecutive calls
    /// yields the same dab pattern as a single call.
    ///
    /// `pressure` is the `[from, to]` endpoint-pressure span; each dab's `alpha_multiplier`
    /// linearly interpolates it at the dab's fractional distance `t` along
    /// the segment (`t = 0` at `from_pos`, `t = 1` at `to_pos`). When one
    /// logical segment is split across calls, subdivide the span at the
    /// split point so interpolation stays continuous (see the
    /// residual-carry identity test).
    ///
    /// Degenerate inputs (zero-length segment, non-positive or non-finite
    /// radius/density, non-finite positions) emit nothing and leave
    /// `residual` untouched. Non-finite pressures are treated as `0.0`;
    /// finite pressures are clamped to `0..=1`.
    pub fn plan_dabs(
        &mut self,
        from_pos: [f32; 2],
        to_pos: [f32; 2],
        radius_px: f32,
        pressure: [f32; 2],
    ) -> Vec<DabPlan> {
        let mut out = Vec::new();
        if !self.dabs_per_radius.is_finite() || self.dabs_per_radius <= 0.0 {
            return out;
        }
        if !radius_px.is_finite() || radius_px <= 0.0 {
            return out;
        }
        let from = Vec2::new(from_pos[0], from_pos[1]);
        let to = Vec2::new(to_pos[0], to_pos[1]);
        if !from.is_finite() || !to.is_finite() {
            return out;
        }
        let segment = to - from;
        let dist = segment.length();
        if !dist.is_finite() || dist <= 0.0 {
            return out;
        }
        let step = radius_px / self.dabs_per_radius;
        if !step.is_finite() || step <= 0.0 {
            return out;
        }
        if !self.residual.is_finite() || self.residual < 0.0 {
            self.residual = 0.0;
        }

        let p0 = saturate01(pressure[0]);
        let p1 = saturate01(pressure[1]);

        // Distance from `from_pos` to the next dab boundary. When banked
        // travel already covers a full step (`d <= 0`, e.g. after the radius
        // shrank), the dab fires at the segment start (`t = 0`).
        let mut d = step - self.residual;
        let mut last_emit: Option<f32> = None;
        while d <= dist && out.len() < MAX_DABS_PER_CALL {
            let t = (d / dist).clamp(0.0, 1.0);
            out.push(DabPlan {
                pos: (from + segment * t).to_array(),
                alpha_multiplier: p0 + (p1 - p0) * t,
            });
            last_emit = Some(d.max(0.0));
            d += step;
        }
        self.residual = match last_emit {
            Some(last) => dist - last,
            None => self.residual + dist,
        };
        out
    }

    /// Plan dabs for a segment held at constant `pressure`.
    pub fn plan_dabs_uniform(
        &mut self,
        from_pos: [f32; 2],
        to_pos: [f32; 2],
        radius_px: f32,
        pressure: f32,
    ) -> Vec<DabPlan> {
        self.plan_dabs(from_pos, to_pos, radius_px, [pressure, pressure])
    }
}

/// Finite values clamp to `0..=1`; non-finite values become `0.0`
/// (`f32::clamp` maps NaN to an endpoint, which would invent pressure).
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

    fn positions(dabs: &[DabPlan]) -> Vec<[f32; 2]> {
        dabs.iter().map(|d| d.pos).collect()
    }

    #[test]
    fn straight_segment_emits_evenly_spaced_dabs() {
        let mut acc = SpacingAccumulator::new(2.0).expect("valid density");
        // step = 10 / 2 = 5 px over 100 px -> 20 dabs at 5, 10, ..., 100.
        let dabs = acc.plan_dabs([0.0, 0.0], [100.0, 0.0], 10.0, [1.0, 1.0]);
        assert_eq!(dabs.len(), 20);
        for (i, dab) in dabs.iter().enumerate() {
            let want_x = 5.0 + i as f32 * 5.0;
            assert!(
                (dab.pos[0] - want_x).abs() <= EPS && dab.pos[1].abs() <= EPS,
                "dab {i} at {:?}, want x={want_x}",
                dab.pos
            );
            assert!((dab.alpha_multiplier - 1.0).abs() <= EPS);
        }
        assert!(acc.residual.abs() <= EPS, "exact fit banks nothing");
    }

    #[test]
    fn residual_carry_matches_unsplit_segment() {
        let radius = 4.0;
        let density = 1.5; // step = 2.666... px: splits fall mid-step
        let pressure = [0.2, 0.9];

        let mut whole = SpacingAccumulator::new(density).expect("valid density");
        let expected = whole.plan_dabs([0.0, 0.0], [10.0, 0.0], radius, pressure);

        // Callers splitting a segment must subdivide the pressure span at
        // the split point, just as they split the geometry: the span is
        // parameterized per call (`t = 0` at each call's start).
        let mid_pressure = pressure[0] + (pressure[1] - pressure[0]) * 0.5;
        let mut split = SpacingAccumulator::new(density).expect("valid density");
        let mut got = split.plan_dabs([0.0, 0.0], [5.0, 0.0], radius, [pressure[0], mid_pressure]);
        got.extend(split.plan_dabs([5.0, 0.0], [10.0, 0.0], radius, [mid_pressure, pressure[1]]));

        assert_eq!(got.len(), expected.len(), "split changed the dab count");
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a.pos[0] - b.pos[0]).abs() <= EPS && (a.pos[1] - b.pos[1]).abs() <= EPS,
                "dab {i} moved: {:?} vs {:?}",
                a.pos,
                b.pos
            );
            assert!(
                (a.alpha_multiplier - b.alpha_multiplier).abs() <= EPS,
                "dab {i} alpha changed: {} vs {}",
                a.alpha_multiplier,
                b.alpha_multiplier
            );
        }
        assert!(
            (split.residual - whole.residual).abs() <= EPS,
            "residual diverged: {} vs {}",
            split.residual,
            whole.residual
        );
        // And the leftover really is partial: a fresh accumulator banks it.
        assert!(split.residual > 0.0 && split.residual < radius / density);
        let _ = positions(&got);
    }

    #[test]
    fn residual_accumulates_across_many_short_segments() {
        let mut acc = SpacingAccumulator::new(1.0).expect("valid density");
        // 1 px steps with a 10 px dab step: first dab after 10 segments.
        let mut total = 0;
        for i in 0..10 {
            let dabs = acc.plan_dabs_uniform([i as f32, 0.0], [i as f32 + 1.0, 0.0], 10.0, 1.0);
            total += dabs.len();
            if i < 9 {
                assert!(dabs.is_empty(), "segment {i} must not fire yet");
            }
        }
        assert_eq!(total, 1, "ten 1 px hops must bank exactly one dab");
    }

    #[test]
    fn zero_length_segment_emits_nothing_and_banks_nothing() {
        let mut acc = SpacingAccumulator::new(2.0).expect("valid density");
        acc.plan_dabs([0.0, 0.0], [3.0, 0.0], 10.0, [1.0, 1.0]);
        let before = acc.residual;
        let dabs = acc.plan_dabs([7.0, 7.0], [7.0, 7.0], 10.0, [1.0, 1.0]);
        assert!(dabs.is_empty());
        assert!(
            (acc.residual - before).abs() <= EPS,
            "zero-length segment must not touch residual"
        );
    }

    #[test]
    fn pressure_interpolates_between_endpoints() {
        let mut acc = SpacingAccumulator::new(1.0).expect("valid density");
        // step = 5 px: dabs at t = 0.5 and t = 1.0.
        let dabs = acc.plan_dabs([0.0, 0.0], [10.0, 0.0], 5.0, [0.0, 1.0]);
        assert_eq!(dabs.len(), 2);
        assert!((dabs[0].alpha_multiplier - 0.5).abs() <= EPS);
        assert!((dabs[1].alpha_multiplier - 1.0).abs() <= EPS);
    }

    #[test]
    fn degenerate_inputs_emit_nothing_without_panicking() {
        assert!(SpacingAccumulator::new(0.0).is_err());
        assert!(SpacingAccumulator::new(-2.0).is_err());
        assert!(SpacingAccumulator::new(f32::NAN).is_err());

        let mut acc = SpacingAccumulator::new(2.0).expect("valid density");
        assert!(acc
            .plan_dabs([0.0, 0.0], [10.0, 0.0], 0.0, [1.0, 1.0])
            .is_empty());
        assert!(acc
            .plan_dabs([0.0, 0.0], [10.0, 0.0], -5.0, [1.0, 1.0])
            .is_empty());
        assert!((acc.residual - 0.0).abs() <= EPS);
        // Corrupted density state (e.g. restored from a bad snapshot) also holds.
        acc.dabs_per_radius = 0.0;
        assert!(acc
            .plan_dabs([0.0, 0.0], [10.0, 0.0], 5.0, [1.0, 1.0])
            .is_empty());
    }
}

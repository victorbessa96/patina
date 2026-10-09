//! Translation from brush-space stamps to paint-space dabs.
//!
//! The bridge between `umber_brush::spacing::DabPlan` stamps (pointer
//! geometry with pressure already interpolated into
//! `alpha_multiplier`) and `umber_gpu::paint::Dab` (premultiplied paint
//! stamps for the compute splat pass). Pure geometry+color math — no
//! GPU objects, fully testable on any platform.
//!
//! Coordinate contract: stamps carry positions in the same space the
//! caller planned them in. For Wave-2 direct painting that space is the
//! paint-target texel grid (the app maps UV → texel before spacing);
//! the adapter neither knows nor scales by texture size.

use crate::DabPlan;
use umber_gpu::paint::Dab;

/// Static brush parameters for one stroke segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushParams {
    /// Premultiplied RGBA stroke color, 0..=1 per channel.
    pub color: [f32; 4],
    /// Base per-dab opacity, 0..=1 (multiplied by each stamp's
    /// `alpha_multiplier`, which carries the pressure curve).
    pub alpha: f32,
    /// Stamp hardness 0..=1 (1 = hard edge, 0 = softest falloff).
    pub hardness: f32,
    /// Nonlinear response applied to the pressure multiplier
    /// (1.0 = linear; >1 favors light pressure falloff).
    pub pressure_gamma: f32,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            color: [0.85, 0.2, 0.1, 1.0],
            alpha: 0.8,
            hardness: 0.5,
            pressure_gamma: 1.0,
        }
    }
}

/// Converts brush-space stamps into paint-space dabs.
#[derive(Debug)]
pub struct DabAdapter;

impl DabAdapter {
    /// Maps one segment's `DabPlan` stamps to paint `Dab`s.
    ///
    /// `radius_px` is the brush radius the stamps were spaced with
    /// (`SpacingAccumulator::plan_dabs` takes it per segment; the
    /// adapter carries it onto each dab unchanged). Each stamp's
    /// `alpha_multiplier` (pressure-interpolated) is shaped by
    /// `pressure_gamma`, multiplied by the base `alpha`, and clamped.
    pub fn stamps_to_dabs(stamps: &[DabPlan], radius_px: f32, params: &BrushParams) -> Vec<Dab> {
        let mut out = Vec::with_capacity(stamps.len());
        for stamp in stamps {
            let shaped = stamp.alpha_multiplier.powf(params.pressure_gamma);
            let alpha = (params.alpha * shaped).clamp(0.0, 1.0);
            out.push(Dab::new(
                stamp.pos,
                radius_px,
                params.color,
                alpha,
                params.hardness,
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(pos: [f32; 2], alpha_multiplier: f32) -> DabPlan {
        DabPlan {
            pos,
            alpha_multiplier,
        }
    }

    #[test]
    fn empty_stamps_yield_no_dabs() {
        assert!(DabAdapter::stamps_to_dabs(&[], 8.0, &BrushParams::default()).is_empty());
    }

    #[test]
    fn multiplier_carries_pressure_into_alpha() {
        let dabs = DabAdapter::stamps_to_dabs(
            &[stamp([10.0, 12.0], 0.5)],
            8.0,
            &BrushParams {
                alpha: 0.8,
                ..BrushParams::default()
            },
        );
        assert_eq!(dabs.len(), 1);
        // alpha = base 0.8 * multiplier 0.5, linear gamma
        assert!((dabs[0].alpha - 0.4).abs() < 1e-6);
        assert_eq!(dabs[0].pos, [10.0, 12.0]);
        assert!((dabs[0].radius - 8.0).abs() < 1e-6);
    }

    #[test]
    fn gamma_shapes_light_pressure() {
        let params = BrushParams {
            pressure_gamma: 2.0,
            ..BrushParams::default()
        };
        // multiplier 0.5, gamma 2 -> 0.25; base 0.8 -> 0.2
        let dabs = DabAdapter::stamps_to_dabs(&[stamp([0.0, 0.0], 0.5)], 8.0, &params);
        assert!((dabs[0].alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn alpha_clamps_at_one() {
        let dabs = DabAdapter::stamps_to_dabs(
            &[stamp([0.0, 0.0], 1.0)],
            8.0,
            &BrushParams {
                alpha: 1.0,
                ..BrushParams::default()
            },
        );
        assert!((dabs[0].alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn color_and_hardness_pass_through() {
        let params = BrushParams {
            color: [0.1, 0.2, 0.3, 1.0],
            hardness: 0.9,
            ..BrushParams::default()
        };
        let dabs = DabAdapter::stamps_to_dabs(&[stamp([1.0, 1.0], 1.0)], 4.0, &params);
        assert_eq!(dabs[0].color, [0.1, 0.2, 0.3, 1.0]);
        assert!((dabs[0].hardness - 0.9).abs() < 1e-6);
    }

    #[test]
    fn integration_spacing_to_dabs() {
        // End-to-end: a spacing-planned segment adapts into dabs with
        // monotonically interpolated pressure between the endpoints.
        let mut acc = crate::SpacingAccumulator::new(5.0).expect("valid density");
        let stamps = acc.plan_dabs([0.0, 0.0], [40.0, 0.0], 8.0, [1.0, 0.0]);
        assert!(!stamps.is_empty());
        let dabs = DabAdapter::stamps_to_dabs(&stamps, 8.0, &BrushParams::default());
        assert_eq!(dabs.len(), stamps.len());
        // First dab carries near-start pressure, last near-end: alphas descend.
        assert!(dabs[0].alpha > dabs[dabs.len() - 1].alpha);
        // Positions stay inside the segment.
        for dab in &dabs {
            assert!(dab.pos[0] >= 0.0 && dab.pos[0] <= 40.0);
            assert_eq!(dab.pos[1], 0.0);
        }
    }
}

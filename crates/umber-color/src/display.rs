//! Display color management: linear working space → display transform.
//!
//! Wave 2 scope (docs/specs/requirements.md §9): scene-linear working
//! space, per-channel CM flags, display transform. The Wave-2 transform
//! is the fixed sRGB display (the same IEC 61966-2-1 curve
//! `umber_export::png` encodes with); OCIO v2 config loading (via the
//! pure-Rust `ocio` crate — smoke-tested, Windows-CI-safe) arrives with
//! Wave 3's display-transform panel, when arbitrary configs and looks
//! become user-selectable.
//!
//! Architecture note: this module owns the CPU reference path (software
//! transforms + tests). The GPU display LUT path lands in `umber-gpu`
//! with the viewport HDR work (Wave 3) — the CPU path is the spec both
//! the shader and the exporter are validated against.

/// The active display transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayTransform {
    /// Linear scene values shown as-is (no transfer) — raw inspection.
    None,
    /// IEC 61966-2-1 sRGB encode (the default artist view).
    #[default]
    Srgb,
}

/// CPU reference: applies the display transform to a single linear rgb
/// triple (0..=1 per channel; values outside clamp).
pub fn apply_display(linear: [f32; 3], transform: DisplayTransform) -> [f32; 3] {
    match transform {
        DisplayTransform::None => linear,
        DisplayTransform::Srgb => [
            linear_to_srgb(linear[0].clamp(0.0, 1.0)),
            linear_to_srgb(linear[1].clamp(0.0, 1.0)),
            linear_to_srgb(linear[2].clamp(0.0, 1.0)),
        ],
    }
}

/// CPU reference: inverts the display transform (display → linear),
/// e.g. for sampling color pickers in the working space.
pub fn invert_display(display: [f32; 3], transform: DisplayTransform) -> [f32; 3] {
    match transform {
        DisplayTransform::None => display,
        DisplayTransform::Srgb => [
            srgb_to_linear(display[0].clamp(0.0, 1.0)),
            srgb_to_linear(display[1].clamp(0.0, 1.0)),
            srgb_to_linear(display[2].clamp(0.0, 1.0)),
        ],
    }
}

/// IEC 61966-2-1: linear → sRGB transfer (f32 in 0..=1).
fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// IEC 61966-2-1 inverse: sRGB → linear (f32 in 0..=1).
fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.040_45 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_anchors_match_spec() {
        // Same anchors the exporter pins: 0, 1, mid-gray, low linear.
        assert_eq!(linear_to_srgb(0.0), 0.0);
        // f32: 1.055 - 0.055 lands one ulp below 1.0 — epsilon, not eq.
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        assert!((linear_to_srgb(0.5) - 0.735_357).abs() < 1e-4);
        // 1/255 sits just above the linear-segment threshold: the power
        // branch gives 0.049840 (not the naive 12.92*val = 0.0506).
        assert!((linear_to_srgb(1.0 / 255.0) - 0.049_840).abs() < 1e-4);
    }

    #[test]
    fn display_roundtrip_recovers_linear() {
        for &l in &[0.0f32, 0.05, 0.18, 0.5, 0.9, 1.0] {
            let displayed = apply_display([l, l, l], DisplayTransform::Srgb);
            let back = invert_display(displayed, DisplayTransform::Srgb);
            for c in back {
                assert!(
                    (c - l).abs() < 1e-5,
                    "roundtrip failed for {l}: got {back:?}"
                );
            }
        }
    }

    #[test]
    fn none_transform_is_identity() {
        let v = [0.25, 0.5, 0.75];
        assert_eq!(apply_display(v, DisplayTransform::None), v);
        assert_eq!(invert_display(v, DisplayTransform::None), v);
    }

    #[test]
    fn out_of_range_clamps_before_transform() {
        let out = apply_display([2.0, -1.0, 0.5], DisplayTransform::Srgb);
        assert!((out[0] - 1.0).abs() < 1e-6); // f32 curve lands 1 ulp low
        assert_eq!(out[1], 0.0);
        assert!((out[2] - 0.735_357).abs() < 1e-4);
    }

    #[test]
    fn default_display_is_srgb() {
        // The artist default: managed sRGB view.
        assert_eq!(DisplayTransform::default(), DisplayTransform::Srgb);
    }
}

//! Display color management: linear working space → display transform.
//!
//! Wave 2 scope (docs/specs/requirements.md §9): scene-linear working
//! space, per-channel CM flags, display transform. The Wave-2 transform
//! is the fixed sRGB display (the same IEC 61966-2-1 curve
//! `umber_export::png` encodes with); OCIO v2 config loading stays
//! behind the stub-gated `ocio` bridge (see `ocio.rs`).
//!
//! Wave 5 (the display-transform panel): [`DisplayTransform`] gains the
//! BT.709 view, and [`DisplaySettings`] + [`apply_display_chain`] add
//! the viewer chain — exposure, then the view transform, then display
//! gamma — that the app's Display panel edits and the `.umber` project
//! persists.
//!
//! The GPU display LUT: [`build_display_lut`] tabulates the chain into
//! 256 RGBA8 entries (the chain on each `i/255` linear sample, quantized
//! to the display byte). The app uploads those bytes into
//! `umber_gpu::DisplayLut`, which the viewport mesh pass and the UV
//! view's texture display sample as their last step — `umber-gpu` never
//! sees [`DisplaySettings`], only the bytes.
//!
//! Architecture note: this module owns the CPU reference path (software
//! transforms + tests). The LUT is that path, tabulated — the CPU chain
//! is the spec both the shader and the exporter are validated against.

use serde::{Deserialize, Serialize};

/// The active display (view) transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DisplayTransform {
    /// Linear scene values shown as-is (no transfer) — raw inspection.
    /// Labelled "Raw" in the UI and on disk.
    #[serde(rename = "Raw")]
    None,
    /// IEC 61966-2-1 sRGB encode (the default artist view).
    #[default]
    Srgb,
    /// ITU-R BT.709 OETF (the broadcast/video view).
    Rec709,
}

impl DisplayTransform {
    /// Every view, in UI order.
    pub const ALL: [Self; 3] = [Self::None, Self::Srgb, Self::Rec709];

    /// The UI label (combo box + status line).
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "Raw",
            Self::Srgb => "sRGB",
            Self::Rec709 => "Rec.709",
        }
    }
}

/// CPU reference: applies the display transform to a single linear rgb
/// triple (0..=1 per channel; values outside clamp — except `None`,
/// which passes values through untouched).
pub fn apply_display(linear: [f32; 3], transform: DisplayTransform) -> [f32; 3] {
    match transform {
        DisplayTransform::None => linear,
        DisplayTransform::Srgb => linear.map(|c| linear_to_srgb(c.clamp(0.0, 1.0))),
        DisplayTransform::Rec709 => linear.map(|c| linear_to_rec709(c.clamp(0.0, 1.0))),
    }
}

/// CPU reference: inverts the display transform (display → linear),
/// e.g. for sampling color pickers in the working space.
pub fn invert_display(display: [f32; 3], transform: DisplayTransform) -> [f32; 3] {
    match transform {
        DisplayTransform::None => display,
        DisplayTransform::Srgb => display.map(|c| srgb_to_linear(c.clamp(0.0, 1.0))),
        DisplayTransform::Rec709 => display.map(|c| rec709_to_linear(c.clamp(0.0, 1.0))),
    }
}

/// The viewer chain the Display panel edits: exposure (EV stops), the
/// view transform, then display gamma. Persisted in the `.umber`
/// project (`ProjectSettings::display`); `serde(default)` fills any
/// missing key, so pre-display projects load as [`Self::default`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// The view transform.
    pub view: DisplayTransform,
    /// Exposure in EV stops: linear values multiply by `2^exposure`
    /// before the view. UI range [`Self::EXPOSURE_RANGE`].
    pub exposure: f32,
    /// Display gamma, applied after the view as `v^(1/gamma)` (the
    /// viewer convention: > 1 brightens). UI range [`Self::GAMMA_RANGE`].
    pub gamma: f32,
}

impl Default for DisplaySettings {
    /// The identity chain: Raw / 0 EV / gamma 1. Deliberately NOT
    /// `DisplayTransform::default()` (sRGB) — a project with no display
    /// key loads as "no viewer adjustment", the additive rule.
    fn default() -> Self {
        Self {
            view: DisplayTransform::None,
            exposure: 0.0,
            gamma: 1.0,
        }
    }
}

impl DisplaySettings {
    /// The exposure slider's range (EV).
    pub const EXPOSURE_RANGE: std::ops::RangeInclusive<f32> = -5.0..=5.0;
    /// The gamma slider's range.
    pub const GAMMA_RANGE: std::ops::RangeInclusive<f32> = 0.5..=2.5;

    /// Clamps exposure/gamma into the UI ranges; a non-finite value
    /// (hand-edited project file) falls back to its default.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let fix = |v: f32, range: &std::ops::RangeInclusive<f32>, fallback: f32| {
            if v.is_finite() {
                v.clamp(*range.start(), *range.end())
            } else {
                fallback
            }
        };
        Self {
            view: self.view,
            exposure: fix(self.exposure, &Self::EXPOSURE_RANGE, defaults.exposure),
            gamma: fix(self.gamma, &Self::GAMMA_RANGE, defaults.gamma),
        }
    }

    /// Whether the chain is the identity (every stage a no-op).
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }
}

/// CPU reference: the full viewer chain on one linear rgb triple —
/// `exposure` (multiply by `2^ev`), then [`apply_display`] with `view`,
/// then display gamma (`v^(1/gamma)`, negatives clamped to 0 first so
/// the Raw view never produces NaN). Exposure 0 and gamma 1 are exact
/// fast paths: the identity settings return the input bit-for-bit.
/// A non-positive or non-finite gamma is treated as 1.
pub fn apply_display_chain(linear: [f32; 3], s: &DisplaySettings) -> [f32; 3] {
    let exposed = if s.exposure == 0.0 {
        linear
    } else {
        let scale = s.exposure.exp2();
        linear.map(|c| c * scale)
    };
    let viewed = apply_display(exposed, s.view);
    let gamma_usable = s.gamma.is_finite() && s.gamma > 0.0;
    if s.gamma == 1.0 || !gamma_usable {
        viewed
    } else {
        let inv = 1.0 / s.gamma;
        viewed.map(|c| c.max(0.0).powf(inv))
    }
}

/// Entries in the GPU display LUT: one per 8-bit linear input level.
pub const DISPLAY_LUT_ENTRIES: usize = 256;
/// The LUT's byte length: [`DISPLAY_LUT_ENTRIES`] RGBA8 texels.
pub const DISPLAY_LUT_BYTES: usize = DISPLAY_LUT_ENTRIES * 4;

/// The chain tabulated for the GPU display LUT: entry `i` is
/// [`apply_display_chain`] on the linear gray `i/255`, quantized to the
/// display byte (`clamp(v, 0, 1) · 255`, rounded — the Display panel's
/// swatch quantization; NaN saturates to 0), alpha 255. RGBA8, entry-
/// major (`[r0, g0, b0, a0, r1, …]`), ready for a 256×1 `Rgba8Unorm`
/// upload. The chain is per-channel identical, so r = g = b per entry;
/// the shader still reads each channel from its own entry's matching
/// component, which keeps a future per-channel (OCIO) LUT a data change.
///
/// The identity settings produce the identity table (`i → i`) byte-for-
/// byte: the chain returns `i/255` bit-exactly and `(i/255)·255` rounds
/// back to `i`.
pub fn build_display_lut(settings: &DisplaySettings) -> [u8; DISPLAY_LUT_BYTES] {
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut lut = [0u8; DISPLAY_LUT_BYTES];
    for (i, entry) in lut.chunks_exact_mut(4).enumerate() {
        let l = i as f32 / 255.0;
        let [r, g, b] = apply_display_chain([l, l, l], settings);
        entry.copy_from_slice(&[byte(r), byte(g), byte(b), 255]);
    }
    lut
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

/// ITU-R BT.709 OETF: linear → Rec.709 (f32 in 0..=1). The spec's
/// rounded constants (1.099 / 0.099 / 0.018 / 4.5 / 0.45) leave a tiny
/// step at the segment join (4.5·0.018 = 0.081 vs ≈0.081 25 from the
/// power branch); the inverse's 0.081 threshold sits inside that gap,
/// so each branch inverts its own segment exactly.
fn linear_to_rec709(l: f32) -> f32 {
    if l < 0.018 {
        l * 4.5
    } else {
        1.099 * l.powf(0.45) - 0.099
    }
}

/// BT.709 inverse OETF: Rec.709 → linear (f32 in 0..=1).
fn rec709_to_linear(v: f32) -> f32 {
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
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

    // ---- Rec.709 view -------------------------------------------------

    #[test]
    fn rec709_forward_known_points() {
        // Linear segment: exact 4.5·L (mirrored f32 product).
        assert_eq!(linear_to_rec709(0.0), 0.0);
        assert_eq!(linear_to_rec709(0.01), 0.01f32 * 4.5);
        // 1.099·1 − 0.099 = 1.0 up to f32 rounding of the constants.
        assert!((linear_to_rec709(1.0) - 1.0).abs() < 1e-6);
        // Mid-gray: 0.18^0.45 = e^(0.45·ln 0.18) = e^(0.45·−1.714 798)
        // = e^−0.771 659 = 0.462 245; ·1.099 = 0.508 007; −0.099 =
        // 0.409 007.
        assert!((linear_to_rec709(0.18) - 0.409_007).abs() < 1e-4);
        // The view goes through apply_display with the same values
        // (1e-6, not eq: the dev profile's opt-level 1 may const-fold one
        // powf call site and not the other).
        let out = apply_display([0.0, 0.01, 0.18], DisplayTransform::Rec709);
        let want = [0.0, linear_to_rec709(0.01), linear_to_rec709(0.18)];
        for (o, w) in out.iter().zip(want.iter()) {
            assert!((o - w).abs() < 1e-6, "{out:?} vs {want:?}");
        }
    }

    #[test]
    fn rec709_roundtrip_recovers_linear() {
        // Both sides of the 0.018 segment join, plus the endpoints.
        for &l in &[0.0f32, 0.005, 0.017, 0.018, 0.019, 0.18, 0.5, 0.9, 1.0] {
            let displayed = apply_display([l, l, l], DisplayTransform::Rec709);
            let back = invert_display(displayed, DisplayTransform::Rec709);
            for c in back {
                assert!(
                    (c - l).abs() < 1e-5,
                    "rec709 roundtrip failed for {l}: got {back:?}"
                );
            }
        }
    }

    #[test]
    fn rec709_differs_from_srgb() {
        // Distinct curves: mid-gray 0.409 (709) vs 0.461 (sRGB).
        let a = apply_display([0.18; 3], DisplayTransform::Rec709)[0];
        let b = apply_display([0.18; 3], DisplayTransform::Srgb)[0];
        assert!(b - a > 0.04, "709 {a} vs sRGB {b}");
    }

    #[test]
    fn labels_and_order_are_pinned() {
        let labels: Vec<_> = DisplayTransform::ALL.iter().map(|t| t.label()).collect();
        assert_eq!(labels, ["Raw", "sRGB", "Rec.709"]);
    }

    // ---- the viewer chain ---------------------------------------------

    #[test]
    fn identity_chain_returns_input_exactly() {
        let s = DisplaySettings::default();
        assert_eq!(s.view, DisplayTransform::None);
        assert_eq!(s.exposure, 0.0);
        assert_eq!(s.gamma, 1.0);
        assert!(s.is_identity());
        // Bit-exact, including out-of-range and negative values (Raw
        // never clamps; both fast paths skip the arithmetic).
        for v in [[0.0, 0.5, 1.0], [0.18, 2.5, -0.25], [1e-8, 0.3, 7.0]] {
            let out = apply_display_chain(v, &s);
            for (o, i) in out.iter().zip(v.iter()) {
                assert_eq!(o.to_bits(), i.to_bits(), "identity: {v:?} -> {out:?}");
            }
        }
    }

    #[test]
    fn exposure_plus_one_doubles_linear() {
        let s = DisplaySettings {
            exposure: 1.0,
            ..DisplaySettings::default()
        };
        // 2^1 is exactly 2.0, and doubling an f32 is exact (exponent
        // bump) — so the Raw view's output is exactly 2·input.
        assert_eq!(1.0f32.exp2(), 2.0);
        let v = [0.18, 0.5, 3.0];
        assert_eq!(apply_display_chain(v, &s), [0.36, 1.0, 6.0]);
        // And against the mirrored expression (libm-independent).
        assert_eq!(apply_display_chain(v, &s), v.map(|c| c * 1.0f32.exp2()));
    }

    #[test]
    fn gamma_on_srgb_view_matches_the_derived_chain() {
        let s = DisplaySettings {
            view: DisplayTransform::Srgb,
            exposure: 0.0,
            gamma: 2.2,
        };
        let out = apply_display_chain([0.18, 0.0, 1.0], &s);
        // Mirrored chain, same f32 ops in the same order (1e-6, not eq:
        // opt-level 1 may const-fold one powf call site, not the other).
        let mirror = |l: f32| linear_to_srgb(l).powf(1.0 / 2.2);
        let want = [mirror(0.18), mirror(0.0), mirror(1.0)];
        for (o, w) in out.iter().zip(want.iter()) {
            assert!((o - w).abs() < 1e-6, "{out:?} vs {want:?}");
        }
        // Derivation for mid-gray: sRGB(0.18) = 1.055·0.18^(1/2.4) −
        // 0.055 = 1.055·0.489 437 − 0.055 = 0.461 356; then
        // 0.461 356^(1/2.2) = e^(ln 0.461 356 / 2.2) = e^(−0.773 585/2.2)
        // = e^−0.351 630 = 0.703 541 → ·255 = 179.40 → byte 179.
        assert!((out[0] - 0.703_541).abs() < 1e-4, "got {}", out[0]);
        assert_eq!((out[0] * 255.0).round() as u8, 179);
        assert_eq!(out[1], 0.0);
        assert!((out[2] - 1.0).abs() < 1e-6);
        // Gamma brightens (the viewer convention): above the plain view.
        assert!(out[0] > linear_to_srgb(0.18));
    }

    #[test]
    fn chain_order_is_exposure_view_gamma() {
        // sRGB clamps after exposure: +1 EV on 0.6 saturates (1.2 → 1),
        // which only happens if exposure runs BEFORE the view.
        let s = DisplaySettings {
            view: DisplayTransform::Srgb,
            exposure: 1.0,
            gamma: 1.0,
        };
        let out = apply_display_chain([0.6, 0.6, 0.6], &s);
        assert!((out[0] - 1.0).abs() < 1e-6, "got {out:?}");
    }

    #[test]
    fn raw_view_gamma_never_produces_nan() {
        let s = DisplaySettings {
            view: DisplayTransform::None,
            exposure: 0.0,
            gamma: 2.2,
        };
        let out = apply_display_chain([-0.5, 0.25, 4.0], &s);
        assert_eq!(out[0], 0.0);
        assert!(out.iter().all(|c| c.is_finite()), "{out:?}");
    }

    #[test]
    fn degenerate_gamma_is_treated_as_one() {
        let v = [0.25, 0.5, 0.75];
        for gamma in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let s = DisplaySettings {
                gamma,
                ..DisplaySettings::default()
            };
            assert_eq!(apply_display_chain(v, &s), v, "gamma {gamma}");
        }
    }

    #[test]
    fn sanitized_clamps_and_repairs() {
        let s = DisplaySettings {
            view: DisplayTransform::Rec709,
            exposure: 9.0,
            gamma: f32::NAN,
        }
        .sanitized();
        assert_eq!(s.view, DisplayTransform::Rec709);
        assert_eq!(s.exposure, 5.0);
        assert_eq!(s.gamma, 1.0);
        let s = DisplaySettings {
            view: DisplayTransform::None,
            exposure: -7.5,
            gamma: 0.1,
        }
        .sanitized();
        assert_eq!((s.exposure, s.gamma), (-5.0, 0.5));
        // In-range values pass through untouched.
        let ok = DisplaySettings {
            view: DisplayTransform::Srgb,
            exposure: 1.25,
            gamma: 2.2,
        };
        assert_eq!(ok.sanitized(), ok);
    }

    // ---- the GPU display LUT ------------------------------------------

    #[test]
    fn identity_lut_is_byte_exact() {
        // The design's test 1: Raw / 0 EV / gamma 1 → entry i = (i, i,
        // i, 255) for every i, no exceptions (the golden render's
        // no-regression contract rests on this table).
        let lut = build_display_lut(&DisplaySettings::default());
        assert_eq!(lut.len(), 1024);
        for (i, entry) in lut.chunks_exact(4).enumerate() {
            let b = i as u8;
            assert_eq!(entry, [b, b, b, 255], "entry {i}");
        }
    }

    #[test]
    fn known_chain_gives_known_lut_bytes() {
        // The design's test 2: the LUT is the chain's math, tabulated.
        // Raw +1 EV: entry i = 2i (exact — doubling is an exponent bump,
        // and (2i/255)·255 rounds to 2i), saturating at 255 from i = 128.
        let plus_one = build_display_lut(&DisplaySettings {
            exposure: 1.0,
            ..DisplaySettings::default()
        });
        for (i, entry) in plus_one.chunks_exact(4).enumerate() {
            let want = (2 * i).min(255) as u8;
            assert_eq!(entry, [want, want, want, 255], "+1 EV entry {i}");
        }

        // sRGB view, hand-derived anchors (margins ≥ 0.15 from a
        // rounding tie): entry 1 = linear 1/255 → 0.049 840 (the
        // srgb_anchors test) · 255 = 12.71 → 13; entry 128 = linear
        // 0.501 961 → 1.055·0.501 961^(1/2.4) − 0.055 = 0.736 647 · 255
        // = 187.85 → 188; the ends pin 0 and 255.
        let srgb = DisplaySettings {
            view: DisplayTransform::Srgb,
            ..DisplaySettings::default()
        };
        let lut = build_display_lut(&srgb);
        let at = |i: usize| lut[i * 4];
        assert_eq!((at(0), at(1), at(128), at(255)), (0, 13, 188, 255));

        // sRGB + gamma 2.2 (the chain test's settings): every entry is
        // the chain at the grid point, quantized — and the entry nearest
        // linear 0.18 (46/255 = 0.180 392) lands on the chain test's
        // byte 179: sRGB 0.461 825, ^(1/2.2) = 0.703 865 · 255 = 179.486
        // → 179 (0.014 below the tie — f32 error is ~1e-4 here).
        let s = DisplaySettings {
            view: DisplayTransform::Srgb,
            exposure: 0.0,
            gamma: 2.2,
        };
        let lut = build_display_lut(&s);
        for (i, entry) in lut.chunks_exact(4).enumerate() {
            let l = i as f32 / 255.0;
            let v = apply_display_chain([l, l, l], &s)[0];
            let want = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert_eq!(entry, [want, want, want, 255], "entry {i}");
        }
        assert_eq!(lut[46 * 4], 179);
        // Monotone non-decreasing: the chain never reorders levels.
        for pair in lut.chunks_exact(4).collect::<Vec<_>>().windows(2) {
            assert!(pair[1][0] >= pair[0][0], "{pair:?}");
        }
    }

    #[test]
    fn settings_serde_round_trip_and_defaults() {
        // Default round-trips byte-identical, and "Raw" names None.
        let d = DisplaySettings::default();
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, r#"{"view":"Raw","exposure":0.0,"gamma":1.0}"#);
        let back: DisplaySettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
        // Modified settings: exact f32 round-trip.
        let m = DisplaySettings {
            view: DisplayTransform::Rec709,
            exposure: -1.37,
            gamma: 2.2,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: DisplaySettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.exposure.to_bits(), m.exposure.to_bits());
        // Missing keys fill from the identity defaults.
        let partial: DisplaySettings = serde_json::from_str(r#"{"view":"Srgb"}"#).unwrap();
        assert_eq!(
            partial,
            DisplaySettings {
                view: DisplayTransform::Srgb,
                ..DisplaySettings::default()
            }
        );
        let empty: DisplaySettings = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, DisplaySettings::default());
    }
}

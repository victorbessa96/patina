//! Named brush presets and their `.umberbrush` JSON file format.
//!
//! Wave-4 slices 1–2 (docs/specs/brush-presets-design.md): the data model.
//! A [`BrushPreset`] binds every parameter the engine already exposes —
//! [`BrushParams`](crate::BrushParams) (dab shape + response),
//! [`OneEuroParams`](crate::OneEuroParams) (stabilizer),
//! [`LazyMouseConfig`] + [`SpacingConfig`] (plain-data extracts of the
//! [`LazyMouse`](crate::LazyMouse) and
//! [`SpacingAccumulator`](crate::SpacingAccumulator) tunings) — plus two
//! 4-point pressure→dab [`ControlCurve`] mappings (pressure→alpha,
//! pressure→radius).
//!
//! The file envelope is `{ "format": "umber-brush-preset", "version": 1,
//! ...preset fields }`, written with `serde_json::to_string_pretty` so
//! write→read→write is byte-identical. Forward compatibility: every
//! [`BrushPreset`] field carries `#[serde(default)]`, so JSON missing a key
//! loads with that field's default instead of failing.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::{BrushParams, OneEuroParams};

/// File-format discriminator written into every `.umberbrush` file.
pub const PRESET_FORMAT: &str = "umber-brush-preset";
/// The only `.umberbrush` version this code reads and writes.
pub const PRESET_VERSION: u32 = 1;

fn default_format() -> String {
    PRESET_FORMAT.to_owned()
}

fn default_version() -> u32 {
    PRESET_VERSION
}

/// Strict-construction and file I/O errors for presets.
///
/// Follows the crate's `thiserror` style (cf. `OneEuroError`,
/// `SpacingError`). The `serde_json::Error` is never exposed with `#[from]`:
/// [`BrushPreset::from_json`] wraps it manually so the carrying variant
/// records the source line/column where parsing stopped.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PresetError {
    /// Malformed JSON: 1-based source line/column plus serde's message.
    #[error("invalid preset JSON at line {source_line}, column {source_column}: {message}")]
    Json {
        source_line: usize,
        source_column: usize,
        message: String,
    },
    /// The `format` envelope field is not [`PRESET_FORMAT`].
    #[error("wrong preset format: expected \"umber-brush-preset\", found \"{found}\"")]
    WrongFormat { found: String },
    /// The `version` envelope field is not [`PRESET_VERSION`].
    #[error("unsupported preset version: expected 1, found {found}")]
    UnsupportedVersion { found: u32 },
    /// Control-curve x coordinates are not strictly increasing
    /// (equal x counts: the bracketing segment would divide by zero).
    #[error("control-curve x values must be strictly increasing")]
    NonMonotonicX,
    /// A control-curve coordinate or parameter is NaN/infinite.
    #[error("control-curve points must be finite")]
    NonFinite,
    /// The preset name is empty.
    #[error("preset name must not be empty")]
    EmptyName,
    /// A parameter is outside its engine domain: `alpha`/`hardness` outside
    /// 0..=1, `pressure_gamma` non-finite or <= 0, one-euro fields outside
    /// their `try_new` domain, spacing density outside its `new` domain, or
    /// a control curve built with the wrong point count.
    #[error("preset parameter outside its valid domain")]
    InvalidParams,
}

/// A 4-point input→dab control curve (pressure→alpha, pressure→radius).
///
/// Exactly four `(x, y)` points are stored in [`Self::points`] (array, not
/// `Vec`, so the count is fixed by the type once constructed). Invariants,
/// enforced by [`Self::try_new`] and by `Deserialize`:
/// - every coordinate is finite ([`PresetError::NonFinite`] otherwise);
/// - x is strictly increasing, `x[0] < x[1] < x[2] < x[3]`
///   ([`PresetError::NonMonotonicX`] otherwise — equal x is a rejection,
///   since the bracketing segment would divide by zero).
///
/// JSON shape is the bare array `[[x,y] x 4]` (custom `Serialize`/
/// `Deserialize` impls — a derived struct would nest under a `"points"`
/// key, which the `.umberbrush` envelope does not use).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlCurve {
    /// The four control points, strictly increasing in x.
    pub points: [(f32, f32); 4],
}

impl ControlCurve {
    /// Build a curve, enforcing the finite + strictly-increasing-x
    /// invariants. Takes a slice (rather than the fixed array) so a wrong
    /// point count is expressible: anything but 4 points is rejected with
    /// [`PresetError::InvalidParams`].
    ///
    /// # Errors
    ///
    /// Returns [`PresetError::InvalidParams`] for a length other than 4,
    /// [`PresetError::NonFinite`] for NaN/infinite coordinates, and
    /// [`PresetError::NonMonotonicX`] when x is not strictly increasing.
    pub fn try_new(points: &[(f32, f32)]) -> Result<Self, PresetError> {
        if points.len() != 4 {
            return Err(PresetError::InvalidParams);
        }
        if points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(PresetError::NonFinite);
        }
        if !(points[0].0 < points[1].0 && points[1].0 < points[2].0 && points[2].0 < points[3].0) {
            return Err(PresetError::NonMonotonicX);
        }
        Ok(Self {
            points: [points[0], points[1], points[2], points[3]],
        })
    }

    /// The identity mapping: `[(0,0), (1/3,1/3), (2/3,2/3), (1,1)]` exactly
    /// (f32 `1.0/3.0` etc.). `pressure_gamma` on [`BrushParams`] remains the
    /// shortcut for this case.
    pub fn identity() -> Self {
        Self {
            points: [
                (0.0, 0.0),
                (1.0 / 3.0, 1.0 / 3.0),
                (2.0 / 3.0, 2.0 / 3.0),
                (1.0, 1.0),
            ],
        }
    }

    /// Clamped piecewise-linear interpolation: below `x[0]` yields `y[0]`,
    /// above `x[3]` yields `y[3]`, otherwise the lerp between the two points
    /// bracketing `x`.
    pub fn evaluate(&self, x: f32) -> f32 {
        let p = self.points;
        if x <= p[0].0 {
            return p[0].1;
        }
        if x >= p[3].0 {
            return p[3].1;
        }
        for i in 0..3 {
            if x <= p[i + 1].0 {
                let (x0, y0) = p[i];
                let (x1, y1) = p[i + 1];
                let t = (x - x0) / (x1 - x0);
                return y0 + (y1 - y0) * t;
            }
        }
        p[3].1
    }
}

impl Default for ControlCurve {
    fn default() -> Self {
        Self::identity()
    }
}

impl Serialize for ControlCurve {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.points.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ControlCurve {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let points = Vec::<(f32, f32)>::deserialize(deserializer)?;
        if points.len() != 4 {
            return Err(serde::de::Error::invalid_length(
                points.len(),
                &"exactly 4 control points",
            ));
        }
        Self::try_new(&points).map_err(serde::de::Error::custom)
    }
}

/// Plain-data tuning for the pulled-string smoother.
///
/// NOTE: the engine's [`LazyMouse`](crate::LazyMouse) currently takes only
/// `radius_px` at construction (`LazyMouse::new(radius_px)`) and holds no
/// strength state. This config struct is plain data; wiring a preset into
/// the engine happens at the call site (`LazyMouse::new(cfg.radius_px)`),
/// where `strength` is carried for the future pull-strength tuning and
/// ignored today. That wiring is a later slice, not this module.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LazyMouseConfig {
    /// Leash length in logical pixels.
    pub radius_px: f32,
    /// Pull strength (reserved for the call-site wiring; see above).
    pub strength: f32,
}

impl Default for LazyMouseConfig {
    fn default() -> Self {
        Self {
            radius_px: 12.0,
            strength: 0.75,
        }
    }
}

/// Plain-data tuning for the dab-spacing engine.
///
/// Domain mirrors [`SpacingAccumulator::new`](crate::SpacingAccumulator::new):
/// the density must be finite and strictly positive. Default is 12.0 dabs
/// per radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpacingConfig {
    /// Dabs emitted per one brush-radius of travel.
    pub dabs_per_radius: f32,
}

impl Default for SpacingConfig {
    fn default() -> Self {
        Self {
            dabs_per_radius: 12.0,
        }
    }
}

/// A named, saveable brush: the full engine parameter set plus curves.
///
/// Every field carries `#[serde(default)]` (the forward-compat rule): JSON
/// missing a key loads with that field's default instead of failing, so
/// older files and hand-written partial files keep loading as new fields
/// are added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrushPreset {
    /// Display name (must be non-empty per [`Self::validate`]).
    #[serde(default)]
    pub name: String,
    /// Dab color/shape/response.
    #[serde(default)]
    pub params: BrushParams,
    /// Stroke stabilizer tuning.
    #[serde(default)]
    pub one_euro: OneEuroParams,
    /// Pulled-string smoother tuning (plain data; see [`LazyMouseConfig`]).
    #[serde(default)]
    pub lazy_mouse: LazyMouseConfig,
    /// Dab-density tuning.
    #[serde(default)]
    pub spacing: SpacingConfig,
    /// Pressure→alpha mapping.
    #[serde(default)]
    pub alpha_curve: ControlCurve,
    /// Pressure→radius mapping.
    #[serde(default)]
    pub radius_curve: ControlCurve,
}

impl BrushPreset {
    /// Serialize to the `.umberbrush` envelope
    /// (`format`/`version` header + preset fields, pretty-printed; key order
    /// follows struct field order, so equal structs produce byte-identical
    /// output).
    ///
    /// # Errors
    ///
    /// Returns [`PresetError::Json`] if serialization fails (line/column
    /// are 0: serialization errors carry no source position).
    pub fn to_json(&self) -> Result<String, PresetError> {
        let file = PresetFile {
            format: PRESET_FORMAT.to_owned(),
            version: PRESET_VERSION,
            preset: self.clone(),
        };
        serde_json::to_string_pretty(&file).map_err(|e| PresetError::Json {
            source_line: e.line(),
            source_column: e.column(),
            message: e.to_string(),
        })
    }

    /// Parse a `.umberbrush` document, verifying the envelope before
    /// returning the preset.
    ///
    /// # Errors
    ///
    /// Returns [`PresetError::Json`] for malformed JSON (with the source
    /// line/column where parsing stopped), [`PresetError::WrongFormat`]
    /// when `format` is not [`PRESET_FORMAT`], and
    /// [`PresetError::UnsupportedVersion`] when `version` is not
    /// [`PRESET_VERSION`].
    pub fn from_json(s: &str) -> Result<Self, PresetError> {
        let file: PresetFile = serde_json::from_str(s).map_err(|e| PresetError::Json {
            source_line: e.line(),
            source_column: e.column(),
            message: e.to_string(),
        })?;
        if file.format != PRESET_FORMAT {
            return Err(PresetError::WrongFormat { found: file.format });
        }
        if file.version != PRESET_VERSION {
            return Err(PresetError::UnsupportedVersion {
                found: file.version,
            });
        }
        Ok(file.preset)
    }

    /// Check the preset against the engine's real parameter domains.
    ///
    /// One-euro constraints mirror
    /// [`OneEuroFilter::try_new`](crate::OneEuroFilter::try_new)
    /// (`one_euro.rs` `validate`): `min_cutoff` finite and > 0, `beta`
    /// finite and >= 0, `d_cutoff` finite and > 0. Spacing mirrors
    /// [`SpacingAccumulator::new`](crate::SpacingAccumulator::new):
    /// `dabs_per_radius` finite and > 0. Curves hold by construction
    /// ([`ControlCurve::try_new`]) and are re-checked defensively.
    ///
    /// # Errors
    ///
    /// Returns [`PresetError::EmptyName`] for an empty name,
    /// [`PresetError::InvalidParams`] for any out-of-domain parameter, and
    /// [`PresetError::NonMonotonicX`]/[`PresetError::NonFinite`] if a curve
    /// somehow violates its invariants.
    pub fn validate(&self) -> Result<(), PresetError> {
        if self.name.is_empty() {
            return Err(PresetError::EmptyName);
        }
        if !(0.0..=1.0).contains(&self.params.alpha)
            || !(0.0..=1.0).contains(&self.params.hardness)
            || !self.params.pressure_gamma.is_finite()
            || self.params.pressure_gamma <= 0.0
        {
            return Err(PresetError::InvalidParams);
        }
        if !self.one_euro.min_cutoff.is_finite()
            || self.one_euro.min_cutoff <= 0.0
            || !self.one_euro.beta.is_finite()
            || self.one_euro.beta < 0.0
            || !self.one_euro.d_cutoff.is_finite()
            || self.one_euro.d_cutoff <= 0.0
        {
            return Err(PresetError::InvalidParams);
        }
        if !self.spacing.dabs_per_radius.is_finite() || self.spacing.dabs_per_radius <= 0.0 {
            return Err(PresetError::InvalidParams);
        }
        for curve in [&self.alpha_curve, &self.radius_curve] {
            let p = curve.points;
            if p.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
                return Err(PresetError::NonFinite);
            }
            if !(p[0].0 < p[1].0 && p[1].0 < p[2].0 && p[2].0 < p[3].0) {
                return Err(PresetError::NonMonotonicX);
            }
        }
        Ok(())
    }
}

/// The on-disk `.umberbrush` envelope: `format`/`version` header plus the
/// preset fields flattened alongside (so the file has a single flat
/// namespace: `format`, `version`, `name`, `params`, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetFile {
    /// Must be [`PRESET_FORMAT`]; defaults to it when the key is missing.
    #[serde(default = "default_format")]
    pub format: String,
    /// Must be [`PRESET_VERSION`]; defaults to it when the key is missing.
    #[serde(default = "default_version")]
    pub version: u32,
    /// The preset body.
    #[serde(flatten)]
    pub preset: BrushPreset,
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn non_default_preset() -> BrushPreset {
        BrushPreset {
            name: "Test Chisel".to_owned(),
            params: BrushParams {
                color: [0.1, 0.2, 0.3, 0.9],
                alpha: 0.33,
                hardness: 0.77,
                pressure_gamma: 2.2,
            },
            one_euro: OneEuroParams {
                min_cutoff: 2.5,
                beta: 0.05,
                d_cutoff: 0.8,
            },
            lazy_mouse: LazyMouseConfig {
                radius_px: 20.0,
                strength: 0.5,
            },
            spacing: SpacingConfig {
                dabs_per_radius: 8.0,
            },
            alpha_curve: ControlCurve::try_new(&[(0.0, 0.0), (0.25, 0.1), (0.75, 0.9), (1.0, 1.0)])
                .expect("valid test curve"),
            radius_curve: ControlCurve::try_new(&[(0.0, 0.5), (0.3, 0.6), (0.7, 0.9), (1.0, 1.0)])
                .expect("valid test curve"),
        }
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let preset = non_default_preset();
        let first = preset.to_json().expect("serialize");
        let back = BrushPreset::from_json(&first).expect("parse");
        assert_eq!(back, preset, "round trip must preserve every field");
        let second = back.to_json().expect("re-serialize");
        assert_eq!(
            second, first,
            "serde_json pretty output is deterministic: same struct, same bytes"
        );
    }

    #[test]
    fn identity_curve_evaluates_exactly() {
        let curve = ControlCurve::identity();
        assert_eq!(
            curve.points,
            [
                (0.0, 0.0),
                (1.0 / 3.0, 1.0 / 3.0),
                (2.0 / 3.0, 2.0 / 3.0),
                (1.0, 1.0)
            ]
        );
        assert!((curve.evaluate(0.0) - 0.0).abs() <= EPS);
        // Midpoint: x=0.5 lands in segment [1/3, 2/3] at t=0.5, so
        // y = 1/3 + (2/3-1/3)*0.5 = 0.5 (within one f32 rounding step).
        assert!((curve.evaluate(0.5) - 0.5).abs() <= EPS);
        assert!((curve.evaluate(1.0) - 1.0).abs() <= EPS);
        assert!((curve.evaluate(-0.5) - 0.0).abs() <= EPS);
        assert!((curve.evaluate(1.5) - 1.0).abs() <= EPS);
    }

    #[test]
    fn steep_curve_lerps_within_bracketing_segment() {
        let curve = ControlCurve::try_new(&[(0.0, 0.0), (0.33, 0.1), (0.66, 0.9), (1.0, 1.0)])
            .expect("valid test curve");
        // Derivation: x=0.165 falls in segment 0 (0 <= 0.165 < 0.33), so
        // t = (0.165-0)/(0.33-0) = 0.5 and y = 0 + (0.1-0)*0.5 = 0.05.
        let got = curve.evaluate(0.165);
        assert!(
            (got - 0.05).abs() <= 1e-6,
            "bracketing-segment lerp must give ~0.05, got {got}"
        );
    }

    #[test]
    fn try_new_rejects_bad_curves() {
        // Equal x is a rejection (the segment would divide by zero).
        assert_eq!(
            ControlCurve::try_new(&[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]),
            Err(PresetError::NonMonotonicX)
        );
        // Decreasing x is a rejection.
        assert_eq!(
            ControlCurve::try_new(&[(0.0, 0.0), (0.7, 0.5), (0.6, 0.8), (1.0, 1.0)]),
            Err(PresetError::NonMonotonicX)
        );
        // NaN y is a rejection.
        assert_eq!(
            ControlCurve::try_new(&[(0.0, 0.0), (0.3, f32::NAN), (0.6, 0.8), (1.0, 1.0)]),
            Err(PresetError::NonFinite)
        );
        // Infinite x is a rejection.
        assert_eq!(
            ControlCurve::try_new(&[(0.0, 0.0), (0.3, 0.5), (f32::INFINITY, 0.8), (1.0, 1.0)]),
            Err(PresetError::NonFinite)
        );
        // Three points is a rejection.
        assert!(ControlCurve::try_new(&[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]).is_err());
    }

    #[test]
    fn from_json_gates_format_version_and_syntax() {
        let valid = non_default_preset().to_json().expect("serialize");

        let wrong_format = valid.replace(PRESET_FORMAT, "umber-brush-preset-v2");
        assert_eq!(
            BrushPreset::from_json(&wrong_format),
            Err(PresetError::WrongFormat {
                found: "umber-brush-preset-v2".to_owned()
            })
        );

        let wrong_version = valid.replace("\"version\": 1", "\"version\": 2");
        assert_eq!(
            BrushPreset::from_json(&wrong_version),
            Err(PresetError::UnsupportedVersion { found: 2 })
        );

        let truncated = &valid[..valid.len() / 2];
        match BrushPreset::from_json(truncated) {
            Err(PresetError::Json {
                source_line,
                source_column,
                message,
            }) => {
                assert!(source_line >= 1, "line info must be present");
                assert!(source_column >= 1, "column info must be present");
                assert!(!message.is_empty());
            }
            other => panic!("truncated JSON must fail with Json, got {other:?}"),
        }
    }

    #[test]
    fn from_json_missing_key_falls_back_to_default() {
        let valid = non_default_preset().to_json().expect("serialize");
        let mut value: serde_json::Value = serde_json::from_str(&valid).expect("valid JSON");
        value
            .as_object_mut()
            .expect("top-level object")
            .remove("spacing");
        let without_spacing = serde_json::to_string_pretty(&value).expect("re-serialize");
        let back = BrushPreset::from_json(&without_spacing).expect("missing key must default");
        assert_eq!(back.spacing, SpacingConfig::default());
        assert_eq!(back.name, "Test Chisel");
    }

    #[test]
    fn validate_enforces_engine_domains() {
        assert!(non_default_preset().validate().is_ok());

        let mut empty_name = non_default_preset();
        empty_name.name.clear();
        assert_eq!(empty_name.validate(), Err(PresetError::EmptyName));

        let mut bad_alpha = non_default_preset();
        bad_alpha.params.alpha = 1.5;
        assert_eq!(bad_alpha.validate(), Err(PresetError::InvalidParams));

        let mut bad_gamma = non_default_preset();
        bad_gamma.params.pressure_gamma = 0.0;
        assert_eq!(bad_gamma.validate(), Err(PresetError::InvalidParams));

        let mut bad_euro = non_default_preset();
        bad_euro.one_euro.beta = f32::NAN;
        assert_eq!(bad_euro.validate(), Err(PresetError::InvalidParams));

        let mut bad_spacing = non_default_preset();
        bad_spacing.spacing.dabs_per_radius = -3.0;
        assert_eq!(bad_spacing.validate(), Err(PresetError::InvalidParams));
    }

    #[test]
    fn shipped_library_loads_and_validates() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/brushes");
        let mut count = 0;
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .expect("assets/brushes must exist")
            .map(|entry| entry.expect("readable dir entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "umberbrush"))
            .collect();
        names.sort();
        for path in names {
            let text = std::fs::read_to_string(&path).expect("readable preset file");
            let preset = BrushPreset::from_json(&text)
                .unwrap_or_else(|e| panic!("{} must parse: {e}", path.display()));
            preset
                .validate()
                .unwrap_or_else(|e| panic!("{} must validate: {e}", path.display()));
            count += 1;
        }
        assert_eq!(count, 12, "the starter library ships twelve presets");
    }
}

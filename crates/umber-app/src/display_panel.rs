//! The Display panel: edits the viewer chain (requirements.md §9 —
//! display transform) — view transform, exposure, display gamma — as
//! [`umber_color::DisplaySettings`] on `AppState::display`, persisted in
//! the `.umber` project (`ProjectSettings::display`).
//!
//! The chain reaches the pixels through the GPU display LUT
//! (`LANDING_NOTES_DISPLAY_PANEL.md`): [`umber_color::build_display_lut`]
//! tabulates it into 256 RGBA8 entries, the app uploads them into its
//! `umber_gpu::DisplayLut`, and both GPU views sample that LUT as their
//! last step — the 3D viewport's mesh pass and the UV view's paint-
//! target display (`umber_gpu::texture_display`). What the panel does:
//!
//! - the settings rows write the data (it saves/loads with the project)
//!   and, on any change, mark the LUT dirty
//!   ([`DisplayLutState::mark_display_lut_dirty`]); the frame loop
//!   rebuilds and uploads it ([`DisplayLutState::take_rebuild`]);
//! - the preview strip runs the pure [`apply_display_chain`] over a
//!   synthetic linear ramp ([`PREVIEW_RAMP`]) and paints the results as
//!   egui swatches — the transform's effect on a known ramp;
//! - the status line names the active chain and the live consumers.
//!
//! v1 limits: the ground grid and wireframe overlays are not
//! transformed (only the two named consumers), and the LUT's input
//! clamps to 0..1. The panel also carries the UI-language row (the
//! locale picker of `docs/specs/i18n-design.md`) — the app's one
//! settings surface today. Real OCIO configs stay gated behind
//! `umber-color`'s `ocio` feature.

use crate::i18n;
use egui::{Color32, Ui};
use fluent::FluentValue;
use umber_color::{
    apply_display_chain, build_display_lut, DisplaySettings, DisplayTransform, DISPLAY_LUT_BYTES,
};

/// The GPU display LUT's CPU-side sync state: whether the table on the
/// device matches the current settings. The panel's write path marks it
/// dirty; the frame loop calls [`Self::take_rebuild`] once per frame and
/// uploads what it returns. Starts dirty, so the first frame builds the
/// LUT from whatever settings the app holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayLutState {
    dirty: bool,
    /// The settings the device LUT was last built from.
    built: Option<DisplaySettings>,
}

impl Default for DisplayLutState {
    fn default() -> Self {
        Self {
            dirty: true,
            built: None,
        }
    }
}

impl DisplayLutState {
    /// The settings changed: the next [`Self::take_rebuild`] rebuilds.
    pub fn mark_display_lut_dirty(&mut self) {
        self.dirty = true;
    }

    /// The table to upload, if the device LUT is stale: dirty, or built
    /// from other settings than `settings` (a safety net — a write path
    /// that forgot to mark still converges in one frame). Clears the
    /// dirty flag and records `settings` as built.
    pub fn take_rebuild(&mut self, settings: &DisplaySettings) -> Option<[u8; DISPLAY_LUT_BYTES]> {
        if !self.dirty && self.built.as_ref() == Some(settings) {
            return None;
        }
        self.dirty = false;
        self.built = Some(*settings);
        Some(build_display_lut(settings))
    }

    /// Whether the device LUT holds `settings` (what the status line
    /// reports as live).
    pub fn is_live(&self, settings: &DisplaySettings) -> bool {
        !self.dirty && self.built.as_ref() == Some(settings)
    }
}

/// Number of preview swatches.
pub const PREVIEW_STOPS: usize = 8;

/// The preview's synthetic linear ramp: one swatch per EV stop,
/// `0.2 · 2^(k−7)` for k = 0..8 — seven stops of shadow up to 0.2
/// (just above 18% gray). Topping out at 0.2 keeps even +2 EV (×4 =
/// 0.8) below the sRGB/Rec.709 clamp at every swatch, so the strip
/// stays readable while the exposure slider pushes it.
pub const PREVIEW_RAMP: [f32; PREVIEW_STOPS] = [
    0.2 / 128.0,
    0.2 / 64.0,
    0.2 / 32.0,
    0.2 / 16.0,
    0.2 / 8.0,
    0.2 / 4.0,
    0.2 / 2.0,
    0.2,
];

/// Swatch strip height (points).
const SWATCH_HEIGHT: f32 = 28.0;

/// The chain applied to each gray ramp stop (display-encoded 0..=1,
/// or raw values under the Raw view).
fn preview_values(s: &DisplaySettings) -> [[f32; 3]; PREVIEW_STOPS] {
    PREVIEW_RAMP.map(|l| apply_display_chain([l, l, l], s))
}

/// Display-encoded 0..=1 → egui color. `Color32` is already sRGB-
/// encoded display bytes, so the chain output is quantized directly —
/// never through `egui::Rgba` (linear), which would encode twice.
fn to_color32(v: [f32; 3]) -> Color32 {
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(byte(v[0]), byte(v[1]), byte(v[2]))
}

/// The preview strip's swatch colors for `s`.
fn swatches(s: &DisplaySettings) -> [Color32; PREVIEW_STOPS] {
    preview_values(s).map(to_color32)
}

/// The status line: the active chain, and where it is live — the two
/// LUT consumers once the device table matches (`live`), else that the
/// LUT is being rebuilt (the frame loop uploads it this frame).
fn chain_label(s: &DisplaySettings, live: bool) -> String {
    let chain = if s.is_identity() {
        i18n::tr("display.chain-identity")
    } else {
        let args = [
            ("exposure", FluentValue::from(format!("{:+.2}", s.exposure))),
            ("view", FluentValue::from(s.view.label())),
            ("gamma", FluentValue::from(format!("{:.2}", s.gamma))),
        ];
        i18n::tr_args("display.chain", args)
    };
    let status = if live {
        i18n::tr("display.live")
    } else {
        i18n::tr("display.rebuilding")
    };
    let args = [
        ("chain", FluentValue::from(chain)),
        ("status", FluentValue::from(status)),
    ];
    i18n::tr_args("display.active-chain", args)
}

/// Draws the panel and edits `settings` in place; any change marks `lut`
/// dirty (the frame loop rebuilds it). Returns whether any value changed
/// this frame.
pub fn show(ui: &mut Ui, settings: &mut DisplaySettings, lut: &mut DisplayLutState) -> bool {
    let mut changed = false;

    egui::ComboBox::from_label(i18n::tr("display.view"))
        .selected_text(settings.view.label())
        .show_ui(ui, |ui| {
            for candidate in DisplayTransform::ALL {
                changed |= ui
                    .selectable_value(&mut settings.view, candidate, candidate.label())
                    .changed();
            }
        });
    changed |= ui
        .add(
            egui::Slider::new(&mut settings.exposure, DisplaySettings::EXPOSURE_RANGE)
                .text(i18n::tr("display.exposure")),
        )
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut settings.gamma, DisplaySettings::GAMMA_RANGE)
                .text(i18n::tr("display.gamma")),
        )
        .changed();
    if ui
        .add_enabled(
            !settings.is_identity(),
            egui::Button::new(i18n::tr("button.reset")),
        )
        .clicked()
    {
        *settings = DisplaySettings::default();
        changed = true;
    }
    if changed {
        lut.mark_display_lut_dirty();
    }

    ui.add_space(4.0);
    ui.label(i18n::tr("display.preview"));
    let width = ui.available_width().max(PREVIEW_STOPS as f32);
    let size = egui::vec2(width, SWATCH_HEIGHT);
    let (strip, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(strip);
    let step = strip.width() / PREVIEW_STOPS as f32;
    for (i, color) in swatches(settings).into_iter().enumerate() {
        let cell = egui::Rect::from_min_size(
            egui::pos2(strip.left() + i as f32 * step, strip.top()),
            egui::vec2(step, strip.height()),
        );
        painter.rect_filled(cell, 0.0, color);
    }

    ui.separator();
    ui.label(chain_label(settings, lut.is_live(settings)));

    ui.separator();
    language_row(ui);
    changed
}

/// The picker's label for a locale tag: the language's own name for the
/// built-in catalogs, the bare tag for discovered ones.
fn locale_label(tag: &str) -> String {
    match tag {
        "en-US" => "English (US)".to_owned(),
        "pt-BR" => "Português (Brasil)".to_owned(),
        other => other.to_owned(),
    }
}

/// The UI-language settings row (an app preference, not a project
/// setting: [`crate::i18n::set_locale_persisted`] writes the per-user
/// locale file). Takes effect from the next frame.
fn language_row(ui: &mut Ui) {
    let (current, locales) = i18n::with(|i| (i.locale().to_owned(), i.locales()));
    let mut picked = current.clone();
    egui::ComboBox::from_label(i18n::tr("settings.language"))
        .selected_text(locale_label(&current))
        .show_ui(ui, |ui| {
            for tag in locales {
                let label = locale_label(&tag);
                ui.selectable_value(&mut picked, tag, label);
            }
        });
    if picked != current {
        i18n::set_locale_persisted(&picked);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_is_one_stop_per_swatch_below_the_clamp() {
        for pair in PREVIEW_RAMP.windows(2) {
            assert_eq!(pair[1], pair[0] * 2.0, "one EV per swatch");
        }
        // Nonzero, and +2 EV (×4) stays unclamped at every swatch.
        for &l in &PREVIEW_RAMP {
            assert!(l > 0.0 && l * 4.0 < 1.0, "stop {l}");
        }
    }

    #[test]
    fn exposure_is_strictly_monotonic_at_every_stop() {
        // The can-fail ordering: +2 EV > +1 EV > 0 EV at every swatch,
        // for every view and with or without display gamma.
        for view in DisplayTransform::ALL {
            for gamma in [1.0, 2.2, 0.5] {
                let at = |exposure: f32| {
                    preview_values(&DisplaySettings {
                        view,
                        exposure,
                        gamma,
                    })
                };
                let (e0, e1, e2) = (at(0.0), at(1.0), at(2.0));
                for k in 0..PREVIEW_STOPS {
                    for c in 0..3 {
                        assert!(
                            e2[k][c] > e1[k][c] && e1[k][c] > e0[k][c],
                            "{view:?} gamma {gamma} stop {k}: \
                             +2 {} / +1 {} / 0 {}",
                            e2[k][c],
                            e1[k][c],
                            e0[k][c]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn identity_swatches_are_the_raw_ramp_bytes() {
        // Raw / 0 / 1: the chain is the identity, so each swatch is the
        // linear value quantized directly (0.2·255 = 51 at the top).
        let colors = swatches(&DisplaySettings::default());
        for (color, l) in colors.iter().zip(PREVIEW_RAMP) {
            let b = (l * 255.0).round() as u8;
            assert_eq!(*color, Color32::from_rgb(b, b, b));
        }
        assert_eq!(colors[PREVIEW_STOPS - 1], Color32::from_rgb(51, 51, 51));
    }

    #[test]
    fn srgb_view_brightens_the_preview() {
        let raw = swatches(&DisplaySettings::default());
        let srgb = swatches(&DisplaySettings {
            view: DisplayTransform::Srgb,
            ..DisplaySettings::default()
        });
        for (r, s) in raw.iter().zip(srgb.iter()) {
            assert!(s.r() > r.r(), "sRGB {s:?} vs raw {r:?}");
        }
    }

    #[test]
    fn to_color32_clamps_and_survives_nan() {
        assert_eq!(
            to_color32([-1.0, 2.0, f32::NAN]),
            Color32::from_rgb(0, 255, 0)
        );
    }

    #[test]
    fn locale_labels_name_the_builtin_languages() {
        // Every built-in catalog gets a native name; discovered ones show
        // their tag.
        for tag in i18n::I18n::builtin().locales() {
            assert_ne!(locale_label(&tag), tag, "{tag} has no native name");
        }
        assert_eq!(locale_label("de-DE"), "de-DE");
    }

    #[test]
    fn chain_label_names_the_chain_and_the_live_consumers() {
        let identity = chain_label(&DisplaySettings::default(), true);
        assert!(identity.contains("identity"), "{identity}");
        assert!(
            identity.contains("Live in the 3D viewport (mesh pass) and the UV view"),
            "{identity}"
        );
        assert!(!identity.contains("Preview only"), "{identity}");
        let s = DisplaySettings {
            view: DisplayTransform::Rec709,
            exposure: 1.5,
            gamma: 2.2,
        };
        let label = chain_label(&s, true);
        assert!(
            label.contains("exposure +1.50 EV → Rec.709 → gamma 2.20"),
            "{label}"
        );
        let pending = chain_label(&s, false);
        assert!(
            pending.contains("Rebuilding the GPU display LUT"),
            "{pending}"
        );
        assert!(!pending.contains("Live in"), "{pending}");
    }

    #[test]
    fn lut_dirty_rebuild_cycle_drives_the_status() {
        // The design's test 4: settings change → rebuild → the status
        // reflects it, on the data path the frame loop runs.
        let mut settings = DisplaySettings::default();
        let mut lut = DisplayLutState::default();

        // Startup is dirty: the first frame builds the identity table —
        // the exact bytes the GPU side's identity fallback holds.
        assert!(!lut.is_live(&settings));
        let first = lut.take_rebuild(&settings).expect("startup build");
        assert_eq!(first, umber_gpu::identity_lut_bytes());
        assert!(lut.is_live(&settings));
        assert!(chain_label(&settings, lut.is_live(&settings)).contains("Live in"));
        // Clean + unchanged: no upload this frame.
        assert_eq!(lut.take_rebuild(&settings), None);

        // The panel's write path: +1 EV, marked dirty → not live yet.
        settings.exposure = 1.0;
        lut.mark_display_lut_dirty();
        assert!(!lut.is_live(&settings));
        assert!(chain_label(&settings, lut.is_live(&settings)).contains("Rebuilding"));
        // The frame loop rebuilds: the chain's table (entry 100 → 200
        // under Raw +1 EV), and the status flips back to live.
        let rebuilt = lut.take_rebuild(&settings).expect("dirty → rebuild");
        assert_eq!(rebuilt, build_display_lut(&settings));
        assert_eq!(&rebuilt[400..404], &[200, 200, 200, 255]);
        assert!(lut.is_live(&settings));
        assert_eq!(lut.take_rebuild(&settings), None);

        // A write path that forgot to mark (e.g. a project load) is
        // still caught: the built settings no longer match.
        settings.view = DisplayTransform::Srgb;
        assert!(!lut.is_live(&settings));
        assert_eq!(
            lut.take_rebuild(&settings),
            Some(build_display_lut(&settings))
        );
        assert!(lut.is_live(&settings));
    }

    #[test]
    fn color_and_gpu_agree_on_the_lut_shape() {
        // The crate boundary: umber-color builds the bytes, umber-gpu
        // uploads them — same length, same identity table.
        assert_eq!(DISPLAY_LUT_BYTES, umber_gpu::DISPLAY_LUT_BYTES);
        assert_eq!(
            build_display_lut(&DisplaySettings::default()),
            umber_gpu::identity_lut_bytes()
        );
    }
}

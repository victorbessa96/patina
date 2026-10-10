//! The Display panel: edits the viewer chain (requirements.md §9 —
//! display transform) — view transform, exposure, display gamma — as
//! [`umber_color::DisplaySettings`] on `AppState::display`, persisted in
//! the `.umber` project (`ProjectSettings::display`).
//!
//! The v1 boundary, stated plainly: the chain is CPU-only. The 3D
//! viewport's mesh pass and the UV view's paint-target display
//! (`umber_gpu::texture_display`) both render on the GPU, and a CPU
//! function cannot touch those pixels — so neither view is affected by
//! these settings yet. What the panel does today:
//!
//! - the settings rows write the data (and it saves/loads with the
//!   project), so the GPU consumer has its input ready;
//! - the preview strip runs the pure [`apply_display_chain`] over a
//!   synthetic linear ramp ([`PREVIEW_RAMP`]) and paints the results as
//!   egui swatches — the transform's effect, shown without any GPU;
//! - the status line names the active chain and says it is preview-only.
//!
//! The panel also carries the UI-language row (the locale picker of
//! `docs/specs/i18n-design.md`) — the app's one settings surface today.
//!
//! The GPU display LUT (the chain baked into a LUT the viewport and UV
//! view shaders sample) is the named follow-up; see
//! `LANDING_NOTES_DISPLAY_PANEL.md`. Real OCIO configs stay gated behind
//! `umber-color`'s `ocio` feature.

use crate::i18n;
use egui::{Color32, Ui};
use umber_color::{apply_display_chain, DisplaySettings, DisplayTransform};

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

/// The status line: the active chain, and the preview-only boundary.
fn chain_label(s: &DisplaySettings) -> String {
    let chain = if s.is_identity() {
        "identity (Raw, 0 EV, gamma 1)".to_owned()
    } else {
        format!(
            "exposure {:+.2} EV → {} → gamma {:.2}",
            s.exposure,
            s.view.label(),
            s.gamma
        )
    };
    format!(
        "Active chain: {chain}. Preview only — the viewport and UV view \
         are unaffected until the GPU display LUT lands."
    )
}

/// Draws the panel and edits `settings` in place. Returns whether any
/// value changed this frame.
pub fn show(ui: &mut Ui, settings: &mut DisplaySettings) -> bool {
    let mut changed = false;

    egui::ComboBox::from_label("View")
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
                .text("Exposure (EV)"),
        )
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut settings.gamma, DisplaySettings::GAMMA_RANGE).text("Gamma"))
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

    ui.add_space(4.0);
    ui.label("Preview (linear ramp, 1 EV per swatch):");
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
    ui.label(chain_label(settings));

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
    fn chain_label_names_the_chain_and_the_boundary() {
        let identity = chain_label(&DisplaySettings::default());
        assert!(identity.contains("identity"), "{identity}");
        assert!(identity.contains("Preview only"), "{identity}");
        let s = DisplaySettings {
            view: DisplayTransform::Rec709,
            exposure: 1.5,
            gamma: 2.2,
        };
        let label = chain_label(&s);
        assert!(
            label.contains("exposure +1.50 EV → Rec.709 → gamma 2.20"),
            "{label}"
        );
    }
}

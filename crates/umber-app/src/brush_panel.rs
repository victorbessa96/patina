//! The brush properties panel: live editing of the active [`BrushPreset`].
//!
//! Wave-4 item 3 (docs/specs/brush-presets-design.md, "The properties
//! panel data model"): the panel edits a [`BrushPreset`] live — engine
//! params bind directly to sliders, the two pressure curves bind to a
//! 4-point drag-the-control-points editor (straight polyline segments;
//! the doc's piecewise-linear v1, no spline rendering). Changes write
//! back to the preset file on explicit save only.
//!
//! [`BrushPanel::new`] assumes the working directory is the repo root in
//! dev builds (that is how the app runs today): it loads
//! `assets/brushes` as an extra search dir alongside the packaged path.
//! The packaged path override happens in a later slice.

use std::path::PathBuf;

use umber_brush::preset::{BrushPreset, ControlCurve};
use umber_brush::preset_library::{user_preset_dir, Library};

/// Size of the curve-editor plot (logical pixels).
const CURVE_PLOT_SIZE: egui::Vec2 = egui::Vec2::new(150.0, 100.0);
/// Radius of the draggable control-point handles (logical pixels).
const CURVE_HANDLE_RADIUS: f32 = 5.0;
/// Minimum x gap between adjacent control points: the strict-monotonicity
/// invariant of [`ControlCurve`] in curve-editor units.
const CURVE_MIN_GAP: f32 = 0.01;

/// The properties panel state: the loaded shelf plus the edit session.
pub struct BrushPanel {
    /// The loaded preset shelf (sorted by name; the loader guarantees it).
    pub library: Library,
    /// Index into [`Library::entries`] of the preset being edited.
    pub active: Option<usize>,
    /// True once the active preset was edited since load/save.
    pub dirty: bool,
    /// Directories the library was loaded from (kept so Save-As can
    /// reload the same search path and the new entry appears).
    dirs: Vec<PathBuf>,
    /// Last save/save-as/reset failure, shown inline (never a panic).
    save_error: Option<String>,
    /// Buffer for the Save-As name field.
    save_as_name: String,
}

impl BrushPanel {
    /// Load the dev search path: the user dir (when discoverable) plus
    /// the repo's `assets/brushes`. Assumes the working directory is the
    /// repo root (true for how the app runs today); nonexistent dirs skip
    /// silently per [`Library::load_from_dirs`].
    pub fn new() -> Self {
        let mut dirs = Vec::new();
        if let Some(user) = user_preset_dir() {
            dirs.push(user);
        }
        // The packaged install dir (`current_exe` parent + "brushes") is
        // covered by `Library::load_default_dirs`; in dev the exe lives
        // under target/, so name the repo dir explicitly instead.
        dirs.push(PathBuf::from("assets/brushes"));
        Self::with_dirs(dirs)
    }

    /// Load the shelf from an explicit priority-ordered search path
    /// (earlier dirs win). No preset is selected initially.
    pub fn with_dirs(dirs: Vec<PathBuf>) -> Self {
        let library = Library::load_from_dirs(&dirs);
        Self {
            library,
            active: None,
            dirty: false,
            dirs,
            save_error: None,
            save_as_name: String::new(),
        }
    }

    /// The panel body: preset selector + live editor + save row.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        self.preset_selector(ui);
        ui.add_space(8.0);
        ui.separator();
        let Some(idx) = self.active else {
            ui.centered_and_justified(|ui| {
                ui.label("No preset selected");
            });
            return;
        };
        if idx >= self.library.entries.len() {
            // The shelf reloaded out from under the selection (Save-As
            // restores `active` by name, but stay defensive anyway).
            self.active = None;
            self.dirty = false;
            return;
        }
        self.preset_editor(ui, idx);
        ui.add_space(8.0);
        ui.separator();
        self.save_row(ui, idx);
        if let Some(err) = self.save_error.clone() {
            ui.label(egui::RichText::new(err).color(ui.visuals().error_fg_color));
        }
    }

    /// Preset combo + broken-file badge + overridden count.
    fn preset_selector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("Preset");
            let current = self
                .active
                .and_then(|i| self.library.entries.get(i))
                .map_or_else(|| "None".to_owned(), |e| e.preset.name.clone());
            let mut picked = self.active;
            egui::ComboBox::from_id_salt("brush_preset_combo")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (i, entry) in self.library.entries.iter().enumerate() {
                        ui.selectable_value(&mut picked, Some(i), &entry.preset.name);
                    }
                });
            if picked != self.active {
                self.active = picked;
                self.dirty = false;
                self.save_error = None;
                // Prefill Save-As with the selected name for "duplicate
                // then rename" flows.
                self.save_as_name = self
                    .active
                    .and_then(|i| self.library.entries.get(i))
                    .map_or_else(String::new, |e| e.preset.name.clone());
            }
        });
        ui.horizontal_wrapped(|ui| {
            if !self.library.errors.is_empty() {
                let n = self.library.errors.len();
                ui.collapsing(format!("{n} broken"), |ui| {
                    for (path, err) in &self.library.errors {
                        ui.label(format!("{}: {err}", path.display()));
                    }
                });
            }
            if self.library.overridden > 0 {
                ui.weak(format!("{} overridden", self.library.overridden));
            }
        });
    }

    /// Live editor for every field of the active preset. Every change
    /// writes through to the preset struct and sets `dirty`.
    fn preset_editor(&mut self, ui: &mut egui::Ui, idx: usize) {
        let preset = &mut self.library.entries[idx].preset;
        let mut changed = false;

        ui.horizontal_wrapped(|ui| {
            ui.label("Name");
            changed |= ui
                .add(egui::TextEdit::singleline(&mut preset.name))
                .changed();
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Color");
            let mut rgb = [
                preset.params.color[0],
                preset.params.color[1],
                preset.params.color[2],
            ];
            if ui.color_edit_button_rgb(&mut rgb).changed() {
                preset.params.color[0] = rgb[0];
                preset.params.color[1] = rgb[1];
                preset.params.color[2] = rgb[2];
                changed = true;
            }
        });

        changed |= ui
            .add(egui::Slider::new(&mut preset.params.alpha, 0.0..=1.0).text("Alpha"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut preset.params.hardness, 0.0..=1.0).text("Hardness"))
            .changed();
        // Gamma > 1 favors light pressure; the 0.05 floor avoids
        // div-by-zero downstream.
        changed |= ui
            .add(
                egui::Slider::new(&mut preset.params.pressure_gamma, 0.05..=4.0)
                    .text("Pressure gamma"),
            )
            .changed();

        ui.add_space(4.0);
        ui.strong("Stabilizer (one-euro)");
        changed |= ui
            .add(egui::Slider::new(&mut preset.one_euro.min_cutoff, 0.1..=10.0).text("Min cutoff"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut preset.one_euro.beta, 0.0..=0.1).text("Beta"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut preset.one_euro.d_cutoff, 0.1..=10.0).text("d cutoff"))
            .changed();

        ui.add_space(4.0);
        ui.strong("Lazy mouse");
        changed |= ui
            .add(egui::Slider::new(&mut preset.lazy_mouse.radius_px, 0.0..=64.0).text("Radius px"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut preset.lazy_mouse.strength, 0.0..=1.0).text("Strength"))
            .changed();

        ui.add_space(4.0);
        ui.strong("Spacing");
        changed |= ui
            .add(
                egui::Slider::new(&mut preset.spacing.dabs_per_radius, 1.0..=64.0)
                    .text("Dabs per radius"),
            )
            .changed();

        ui.add_space(4.0);
        ui.strong("Pressure → alpha");
        changed |= curve_editor(ui, &mut preset.alpha_curve);
        ui.add_space(4.0);
        ui.strong("Pressure → radius");
        changed |= curve_editor(ui, &mut preset.radius_curve);

        if changed {
            self.dirty = true;
        }
    }

    /// Save (to the entry's source file) / Save-As (to the user dir) /
    /// Reset (reload from the source file).
    fn save_row(&mut self, ui: &mut egui::Ui, _idx: usize) {
        ui.horizontal_wrapped(|ui| {
            let save = ui.add_enabled(self.dirty, egui::Button::new("Save"));
            if save.clicked() {
                self.save_active();
            }
            let reset = ui.add_enabled(self.dirty, egui::Button::new("Reset"));
            if reset.clicked() {
                self.reset_active();
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Save as");
            ui.add(
                egui::TextEdit::singleline(&mut self.save_as_name)
                    .hint_text("New preset name")
                    .desired_width(140.0),
            );
            if ui.button("Save As…").clicked() {
                self.save_as();
            }
        });
    }

    /// Serialize the active preset back to its source file. IO/serialize
    /// failures surface inline via `save_error`; never a panic. Clears
    /// `dirty` on success.
    fn save_active(&mut self) {
        let Some(idx) = self.active else {
            return;
        };
        if idx >= self.library.entries.len() {
            return;
        }
        let entry = &self.library.entries[idx];
        let json = match entry.preset.to_json() {
            Ok(json) => json,
            Err(err) => {
                self.save_error = Some(format!("Save failed: {err}"));
                return;
            }
        };
        match std::fs::write(&entry.source_path, json) {
            Ok(()) => {
                self.dirty = false;
                self.save_error = None;
            }
            Err(err) => {
                self.save_error = Some(format!("Save failed: {err}"));
            }
        }
    }

    /// Write the active preset under a new name into the user library
    /// dir (created when missing), then reload the shelf so the new
    /// entry appears and select it by name lookup.
    fn save_as(&mut self) {
        let name = self.save_as_name.trim().to_owned();
        if name.is_empty() {
            self.save_error = Some("Save As needs a name".to_owned());
            return;
        }
        let Some(idx) = self.active else {
            self.save_error = Some("Save As needs an active preset".to_owned());
            return;
        };
        if idx >= self.library.entries.len() {
            return;
        }
        let Some(user_dir) = user_preset_dir() else {
            self.save_error = Some("Save As failed: no user preset dir".to_owned());
            return;
        };
        if let Err(err) = std::fs::create_dir_all(&user_dir) {
            self.save_error = Some(format!("Save As failed: {err}"));
            return;
        }
        let mut preset = self.library.entries[idx].preset.clone();
        preset.name = name.clone();
        let json = match preset.to_json() {
            Ok(json) => json,
            Err(err) => {
                self.save_error = Some(format!("Save As failed: {err}"));
                return;
            }
        };
        let path = user_dir.join(format!("{}.umberbrush", sanitize_file_stem(&name)));
        if let Err(err) = std::fs::write(&path, json) {
            self.save_error = Some(format!("Save As failed: {err}"));
            return;
        }
        // Reload the same search path (plus the user dir, in case this
        // panel was built from dirs that omit it) so the new entry
        // appears, then select it by name.
        if !self.dirs.contains(&user_dir) {
            let mut dirs = Vec::with_capacity(self.dirs.len() + 1);
            dirs.push(user_dir);
            dirs.extend(self.dirs.iter().cloned());
            self.dirs = dirs;
        }
        self.library = Library::load_from_dirs(&self.dirs);
        self.active = self
            .library
            .entries
            .iter()
            .position(|e| e.preset.name == name);
        self.dirty = false;
        self.save_error = None;
    }

    /// Discard edits by reloading the active preset from its source file.
    fn reset_active(&mut self) {
        let Some(idx) = self.active else {
            return;
        };
        if idx >= self.library.entries.len() {
            return;
        }
        let path = self.library.entries[idx].source_path.clone();
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                self.save_error = Some(format!("Reset failed: {err}"));
                return;
            }
        };
        match BrushPreset::from_json(&text).and_then(|preset| {
            preset.validate()?;
            Ok(preset)
        }) {
            Ok(preset) => {
                self.library.entries[idx].preset = preset;
                self.dirty = false;
                self.save_error = None;
            }
            Err(err) => {
                self.save_error = Some(format!("Reset failed: {err}"));
            }
        }
    }
}

impl Default for BrushPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Filename stem for Save-As: lowercase, spaces to dashes, everything
/// else outside `[a-z0-9-_.]` dropped. Falls back to `"preset"` so the
/// join always yields a usable file name.
fn sanitize_file_stem(name: &str) -> String {
    let stem: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c == ' ' { '-' } else { c })
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
        .collect();
    if stem.is_empty() {
        "preset".to_owned()
    } else {
        stem
    }
}

/// Clamp a dragged control point into its legal window: x stays strictly
/// between the neighbors ([`CURVE_MIN_GAP`] each side, the
/// [`ControlCurve`] strict-monotonicity invariant), endpoints additionally
/// stay in `[0, 1]`; y clamps to `0..=1`.
///
/// This is the exact function the curve-editor painter calls on every
/// drag, factored out so the clamp math is unit-testable headless.
/// Degenerate windows (neighbors closer than twice the gap — only
/// possible for hand-loaded files, since every edit goes through this
/// clamp) pin the point in place rather than panicking in `f32::clamp`.
pub(crate) fn clamp_control_point(
    points: &[(f32, f32); 4],
    idx: usize,
    new_x: f32,
    new_y: f32,
) -> (f32, f32) {
    let y = new_y.clamp(0.0, 1.0);
    let (lo, hi) = match idx {
        0 => (0.0, points[1].0 - CURVE_MIN_GAP),
        3 => (points[2].0 + CURVE_MIN_GAP, 1.0),
        _ => (
            points[idx - 1].0 + CURVE_MIN_GAP,
            points[idx + 1].0 - CURVE_MIN_GAP,
        ),
    };
    let x = if lo <= hi && new_x.is_finite() {
        new_x.clamp(lo, hi)
    } else {
        points[idx].0
    };
    (x, y)
}

/// Minimal 4-point curve editor: a 150x100 plot with the polyline
/// through the control points (straight segments — the doc's
/// piecewise-linear v1, no spline rendering) and 5px drag handles.
/// Dragging a handle moves x (clamped between neighbors via
/// [`clamp_control_point`]) and y (`0..=1`). Returns true when the curve
/// changed this frame.
fn curve_editor(ui: &mut egui::Ui, curve: &mut ControlCurve) -> bool {
    let mut changed = false;
    let (rect, _) = ui.allocate_exact_size(CURVE_PLOT_SIZE, egui::Sense::hover());
    let to_screen = |p: (f32, f32)| {
        egui::pos2(
            rect.left() + p.0 * rect.width(),
            rect.bottom() - p.1 * rect.height(),
        )
    };
    let painter = ui.painter().clone();
    painter.rect_stroke(
        rect,
        0.0,
        ui.visuals().widgets.noninteractive.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let screen: Vec<egui::Pos2> = curve.points.iter().map(|&p| to_screen(p)).collect();
    let stroke = egui::Stroke::new(1.5, ui.visuals().text_color());
    for pair in screen.windows(2) {
        painter.line_segment([pair[0], pair[1]], stroke);
    }
    for i in 0..4 {
        let center = to_screen(curve.points[i]);
        painter.circle_filled(
            center,
            CURVE_HANDLE_RADIUS,
            ui.visuals().widgets.active.bg_fill,
        );
        let grab = egui::Rect::from_center_size(center, egui::Vec2::splat(14.0));
        let response = ui.allocate_rect(grab, egui::Sense::drag());
        if response.dragged() {
            let delta = response.drag_delta();
            let candidate = (
                curve.points[i].0 + delta.x / rect.width(),
                curve.points[i].1 - delta.y / rect.height(),
            );
            let (x, y) = clamp_control_point(&curve.points, i, candidate.0, candidate.1);
            if (x, y) != curve.points[i] {
                curve.points[i] = (x, y);
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use umber_brush::preset::{LazyMouseConfig, SpacingConfig};
    use umber_brush::preset_library::user_preset_dir_from;
    use umber_brush::{BrushParams, OneEuroParams};

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fresh_tmpdir(tag: &str) -> PathBuf {
        let id = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "umber-app-panel-test-{}-{}-{}",
            std::process::id(),
            id,
            tag
        ));
        std::fs::create_dir_all(&dir).expect("test tmpdir must be creatable");
        dir
    }

    fn test_preset(name: &str, alpha: f32) -> BrushPreset {
        BrushPreset {
            name: name.to_owned(),
            params: BrushParams {
                color: [0.8, 0.2, 0.1, 1.0],
                alpha,
                hardness: 0.25,
                pressure_gamma: 1.0,
            },
            one_euro: OneEuroParams {
                min_cutoff: 1.2,
                beta: 0.02,
                d_cutoff: 1.0,
            },
            lazy_mouse: LazyMouseConfig::default(),
            spacing: SpacingConfig::default(),
            alpha_curve: ControlCurve::identity(),
            radius_curve: ControlCurve::identity(),
        }
    }

    fn write_preset(dir: &std::path::Path, file_name: &str, preset: &BrushPreset) -> PathBuf {
        let path = dir.join(file_name);
        std::fs::write(&path, preset.to_json().expect("test preset must serialize"))
            .expect("test preset must be writable");
        path
    }

    #[test]
    fn with_dirs_loads_fixtures_with_no_selection() {
        let dir = fresh_tmpdir("with-dirs");
        write_preset(&dir, "b_soft.umberbrush", &test_preset("Soft Round", 0.7));
        write_preset(&dir, "a_hard.umberbrush", &test_preset("Hard Round", 0.95));

        let panel = BrushPanel::with_dirs(vec![dir.clone()]);

        assert_eq!(panel.library.entries.len(), 2);
        assert!(panel.library.errors.is_empty());
        assert_eq!(panel.active, None);
        assert!(!panel.dirty);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clamp_pins_interior_points_between_neighbors() {
        let points = [(0.0, 0.0), (0.33, 0.4), (0.66, 0.7), (1.0, 1.0)];
        // NOTE: bounds are written as the same f32 arithmetic the clamp
        // performs (`0.66 - 0.01`, not the decimal `0.65`: the two differ
        // by one ulp), so these assert_eq!s are bitwise-exact.
        // Dragging point 1 left past point 0 pins to neighbor.x - gap...
        // (here the left neighbor is x=0.0, so the lower bound 0.01 wins).
        assert_eq!(
            clamp_control_point(&points, 1, -1.0, 0.5),
            (0.0 + 0.01, 0.5)
        );
        // ... and dragging right past point 2 pins to 0.66 - 0.01 exactly.
        assert_eq!(
            clamp_control_point(&points, 1, 10.0, 0.5),
            (0.66 - 0.01, 0.5)
        );
        // Over-dragging past the right neighbor pins to neighbor.x - 0.01
        // exactly: 1.05 is 0.05 past point 3 (x=1.0), window hi is 0.99.
        assert_eq!(
            clamp_control_point(&points, 2, 1.0 + 0.05, 0.5),
            (1.0 - 0.01, 0.5)
        );
        assert_eq!(
            clamp_control_point(&points, 2, 0.33 + 0.05, 0.5),
            (0.33 + 0.05, 0.5),
            "0.38 lies inside the (0.34, 0.99) window, so it passes through"
        );
        assert_eq!(
            clamp_control_point(&points, 2, 0.33 - 0.05, 0.5),
            (0.33 + 0.01, 0.5),
            "0.28 is below the window, so it pins to 0.33 + 0.01"
        );
    }

    #[test]
    fn clamp_pins_endpoints_to_unit_x_and_all_y() {
        let points = [(0.0, 0.0), (0.33, 0.4), (0.66, 0.7), (1.0, 1.0)];
        assert_eq!(clamp_control_point(&points, 0, -0.5, 0.5), (0.0, 0.5));
        assert_eq!(
            clamp_control_point(&points, 0, 0.9, 0.5),
            (0.33 - 0.01, 0.5)
        );
        assert_eq!(clamp_control_point(&points, 3, 1.5, 0.5), (1.0, 0.5));
        assert_eq!(
            clamp_control_point(&points, 3, 0.1, 0.5),
            (0.66 + 0.01, 0.5)
        );
        assert_eq!(clamp_control_point(&points, 1, 0.5, -2.0), (0.5, 0.0));
        assert_eq!(clamp_control_point(&points, 1, 0.5, 2.0), (0.5, 1.0));
    }

    #[test]
    fn user_dir_computation_honors_xdg() {
        assert_eq!(
            user_preset_dir_from("/home/ava", Some("/run/user/1000/data")),
            PathBuf::from("/run/user/1000/data/umber/brushes"),
        );
        assert_eq!(
            user_preset_dir_from("/home/ava", None),
            PathBuf::from("/home/ava/.local/share/umber/brushes"),
        );
        assert_eq!(
            user_preset_dir_from("/home/ava", Some("")),
            PathBuf::from("/home/ava/.local/share/umber/brushes"),
            "empty XDG falls back to the home default",
        );
    }

    #[test]
    fn save_round_trips_through_the_real_model() {
        let dir = fresh_tmpdir("save-roundtrip");
        write_preset(&dir, "round.umberbrush", &test_preset("Round", 0.5));

        let mut panel = BrushPanel::with_dirs(vec![dir.clone()]);
        assert_eq!(panel.library.entries.len(), 1);
        panel.active = Some(0);
        panel.library.entries[0].preset.params.alpha = 0.25;
        panel.dirty = true;
        panel.save_active();

        assert!(!panel.dirty, "successful save clears dirty");
        assert!(panel.save_error.is_none());

        let fresh = Library::load_from_dirs(std::slice::from_ref(&dir));
        assert_eq!(fresh.entries.len(), 1);
        let back = BrushPreset::from_json(
            &std::fs::read_to_string(&fresh.entries[0].source_path).expect("readable"),
        )
        .expect("parseable");
        assert!(
            (back.params.alpha - 0.25).abs() < f32::EPSILON,
            "on-disk alpha must reflect the edit, got {}",
            back.params.alpha
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

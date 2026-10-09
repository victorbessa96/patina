//! The Export panel: drives `umber-export`'s preset engine from the
//! egui shell.
//!
//! Wave 3 scope: a synchronous Export button that bakes today's
//! available source maps (AO via [`crate::bake_sources::bake_ao`] plus
//! a flat-normal placeholder — no painted maps feed export yet), filters
//! the chosen engine preset down to the outputs those maps can satisfy
//! (mirroring `umber-cli export`'s "export what we can, name what's
//! missing" behavior), and runs [`umber_export::run_preset`] to write
//! the files. See `LANDING_NOTES_EXPORT_DIALOG.md` for the
//! baked-maps-today vs. painted-maps-later story and why every built-in
//! preset collapses to its normal output alone right now.
//!
//! Synchronous for the same reason as the Bakes panel (`bakes_panel.rs`
//! module docs): no async job system exists yet, and one AO bake + a
//! handful of PNG writes at 512² is sub-second.
//!
//! GPU access follows the `viewport`/`paint_state`/`bakes_panel` pattern:
//! this module never names a `wgpu` type.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;

use crate::bake_sources;
use crate::bakes_panel;

/// The four built-in engine presets the dialog can drive
/// ([`umber_export::ExportPreset`]'s constructors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetChoice {
    /// `ExportPreset::gltf_metal_rough`.
    GltfMetalRough,
    /// `ExportPreset::unreal_orm`.
    UnrealOrm,
    /// `ExportPreset::unity_hdrp_urp`.
    UnityHdrpUrp,
    /// `ExportPreset::blender_principled`.
    BlenderPrincipled,
}

impl PresetChoice {
    /// Every selectable preset, in combo-box order.
    pub const ALL: [Self; 4] = [
        Self::GltfMetalRough,
        Self::UnrealOrm,
        Self::UnityHdrpUrp,
        Self::BlenderPrincipled,
    ];

    /// Builds the `umber_export` preset this choice names.
    pub fn build(self) -> umber_export::ExportPreset {
        match self {
            Self::GltfMetalRough => umber_export::ExportPreset::gltf_metal_rough(),
            Self::UnrealOrm => umber_export::ExportPreset::unreal_orm(),
            Self::UnityHdrpUrp => umber_export::ExportPreset::unity_hdrp_urp(),
            Self::BlenderPrincipled => umber_export::ExportPreset::blender_principled(),
        }
    }

    /// The combo-box / status-line label — matches the built preset's
    /// own `name` field (pinned by a test so the two never drift).
    pub fn label(self) -> &'static str {
        match self {
            Self::GltfMetalRough => "glTF metal-rough",
            Self::UnrealOrm => "Unreal ORM",
            Self::UnityHdrpUrp => "Unity HDRP/URP",
            Self::BlenderPrincipled => "Blender Principled",
        }
    }
}

/// One preset output the dialog could not write from today's available
/// maps, and which map kinds it was missing (by
/// [`umber_export::MapKind::token`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedOutput {
    /// The output's filename template (e.g. `$textureSet_baseColor.png`).
    pub filename: String,
    /// The map tokens this output needed that weren't available.
    pub missing: Vec<&'static str>,
}

/// Splits `preset`'s outputs into those satisfiable from `available`
/// map kinds and those that are not (with the reason) — pure logic, no
/// GPU/IO, so it is directly testable against a synthetic map list.
///
/// Mirrors `umber-cli export`'s filtering (never errors on a partial
/// map set: exports what it can, reports the rest), but names the
/// specific missing map tokens instead of a generic "needs maps not
/// baked" message.
pub fn filter_satisfiable(
    preset: &umber_export::ExportPreset,
    available: &[umber_export::MapKind],
) -> (umber_export::ExportPreset, Vec<SkippedOutput>) {
    let mut kept = Vec::new();
    let mut skipped = Vec::new();
    for output in &preset.outputs {
        let missing: Vec<&'static str> = output
            .maps
            .iter()
            .filter(|(kind, _)| !available.contains(kind))
            .map(|(kind, _)| kind.token())
            .collect();
        if missing.is_empty() {
            kept.push(output.clone());
        } else {
            skipped.push(SkippedOutput {
                filename: output.filename.clone(),
                missing,
            });
        }
    }
    (
        umber_export::ExportPreset {
            name: preset.name.clone(),
            outputs: kept,
        },
        skipped,
    )
}

/// What [`ExportDialog::show`] needs from the app each frame — the same
/// shape as `bakes_panel::BakesContext`.
pub struct ExportContext<'a> {
    /// The app's GPU context (carries device + queue).
    pub gpu: Option<&'a umber_gpu::GpuContext>,
    /// The loaded mesh to bake source maps against.
    pub mesh: Option<&'a umber_mesh::MeshData>,
    /// The mesh's source path (for the texture-set name); `None` falls
    /// back to [`bakes_panel::FALLBACK_TEXTURE_SET`].
    pub mesh_path: Option<&'a Path>,
}

/// One finished export's outcome: what was written, what was skipped,
/// and phase timings for the status line.
struct ExportOutcome {
    written: Vec<PathBuf>,
    skipped: Vec<SkippedOutput>,
    texture_set: String,
    bake_ms: u128,
    write_ms: u128,
}

/// The Export panel state: preset choice, output directory, and the
/// last export's status.
///
/// The egui drawing itself (`show`) is untested by construction; every
/// other method is pure logic or a `Result`-typed driver with no UI
/// dependency.
pub struct ExportDialog {
    preset: PresetChoice,
    output_dir: PathBuf,
    status: String,
    last_written: Vec<PathBuf>,
    last_skipped: Vec<SkippedOutput>,
}

impl ExportDialog {
    /// Creates the dialog writing to `output_dir` with the glTF
    /// metal-rough preset selected.
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            preset: PresetChoice::GltfMetalRough,
            output_dir,
            status: String::from("No export yet."),
            last_written: Vec::new(),
            last_skipped: Vec::new(),
        }
    }

    /// The selected preset.
    pub fn preset(&self) -> PresetChoice {
        self.preset
    }

    /// Changes the selected preset.
    pub fn set_preset(&mut self, preset: PresetChoice) {
        self.preset = preset;
    }

    /// The output directory exported files are written to.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Retargets the output directory.
    pub fn set_output_dir(&mut self, dir: PathBuf) {
        self.output_dir = dir;
    }

    /// The last export's one-line summary.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// The last export's written file paths.
    pub fn last_written(&self) -> &[PathBuf] {
        &self.last_written
    }

    /// The last export's skipped outputs (unsatisfiable from today's
    /// baked maps).
    pub fn last_skipped(&self) -> &[SkippedOutput] {
        &self.last_skipped
    }

    /// Whether the Export button is enabled: mesh + GPU present.
    pub fn can_export(&self, mesh_loaded: bool, gpu_ready: bool) -> bool {
        mesh_loaded && gpu_ready
    }

    /// Draws the panel: preset combo, output-dir row, the synchronous
    /// Export button, and the status line.
    ///
    /// Gated: with no mesh (or no GPU) a reason line replaces the export
    /// controls' effect — the Export button is disabled either way.
    pub fn show(&mut self, ui: &mut egui::Ui, ctx: ExportContext<'_>) {
        let mesh_loaded = ctx.mesh.is_some();
        let gpu_ready = ctx.gpu.is_some();
        if !mesh_loaded {
            ui.label("No mesh loaded — open a mesh to enable export.");
        } else if !gpu_ready {
            ui.label("No GPU device — export needs the wgpu device.");
        }

        egui::ComboBox::from_label("Preset")
            .selected_text(self.preset.label())
            .show_ui(ui, |ui| {
                for candidate in PresetChoice::ALL {
                    ui.selectable_value(&mut self.preset, candidate, candidate.label());
                }
            });

        ui.horizontal(|ui| {
            ui.label(format!("Out: {}", self.output_dir.display()));
            if ui.button("Choose…").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.output_dir = dir;
                }
            }
        });

        let enabled = self.can_export(mesh_loaded, gpu_ready);
        if ui
            .add_enabled(enabled, egui::Button::new("Export"))
            .clicked()
        {
            if let (Some(gpu), Some(mesh)) = (ctx.gpu, ctx.mesh) {
                self.export_now(&gpu.device, &gpu.queue, mesh, ctx.mesh_path);
            }
        }

        ui.separator();
        ui.label(self.status.clone());
    }

    /// Runs the export synchronously and records the outcome in
    /// `status`/`last_written`/`last_skipped` (never propagates: the
    /// panel reports failures as a status line, not a crash).
    fn export_now(
        &mut self,
        device: &umber_gpu::WgpuDevice,
        queue: &umber_gpu::WgpuQueue,
        mesh: &umber_mesh::MeshData,
        mesh_path: Option<&Path>,
    ) {
        match self.run(device, queue, mesh, mesh_path) {
            Ok(outcome) => {
                let names: Vec<String> = outcome
                    .written
                    .iter()
                    .map(|p| {
                        p.file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| p.display().to_string())
                    })
                    .collect();
                let mut status = format!(
                    "Exported {} outputs (bake {} ms, write {} ms): {}",
                    outcome.written.len(),
                    outcome.bake_ms,
                    outcome.write_ms,
                    names.join(", ")
                );
                if !outcome.skipped.is_empty() {
                    let parts: Vec<String> = outcome
                        .skipped
                        .iter()
                        .map(|s| {
                            let filename = umber_export::expand_template(
                                &s.filename,
                                &[("textureSet", &outcome.texture_set)],
                            );
                            format!("{filename} (missing {})", s.missing.join(", "))
                        })
                        .collect();
                    status.push_str(&format!(" — skipped: {}", parts.join("; ")));
                }
                self.status = status;
                self.last_written = outcome.written;
                self.last_skipped = outcome.skipped;
            }
            Err(err) => {
                log::error!("export failed: {err:#}");
                self.status = format!("Export failed: {err:#}");
            }
        }
    }

    /// The synchronous export driver: bakes the source `MapSet`, filters
    /// the selected preset to its satisfiable outputs, and runs it.
    /// `Result`-typed so failures carry context instead of unwrapping.
    fn run(
        &self,
        device: &umber_gpu::WgpuDevice,
        queue: &umber_gpu::WgpuQueue,
        mesh: &umber_mesh::MeshData,
        mesh_path: Option<&Path>,
    ) -> anyhow::Result<ExportOutcome> {
        let fallback = PathBuf::from(bakes_panel::FALLBACK_TEXTURE_SET);
        let texture_set = umber_mesh::texture_set_name(mesh_path.unwrap_or(&fallback), mesh);
        let size = bakes_panel::DEFAULT_RESOLUTION;

        let bake_started = Instant::now();
        let map_set =
            bake_sources::bake_export_map_set(device, queue, mesh, size, bakes_panel::DEFAULT_RAYS)
                .context("building export map set")?;
        let bake_ms = bake_started.elapsed().as_millis();

        let available: Vec<umber_export::MapKind> = map_set.maps_iter().collect();
        let full_preset = self.preset.build();
        let (filtered, skipped) = filter_satisfiable(&full_preset, &available);
        if filtered.outputs.is_empty() {
            anyhow::bail!(
                "preset '{}' has no outputs satisfiable from today's baked maps (AO + flat normal)",
                full_preset.name
            );
        }

        let write_started = Instant::now();
        let written = umber_export::run_preset(&filtered, &map_set, &texture_set, &self.output_dir)
            .context("run_preset")?;
        let write_ms = write_started.elapsed().as_millis();

        Ok(ExportOutcome {
            written,
            skipped,
            texture_set,
            bake_ms,
            write_ms,
        })
    }
}

impl Default for ExportDialog {
    fn default() -> Self {
        Self::new(PathBuf::from("export"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialog() -> ExportDialog {
        ExportDialog::new(PathBuf::from("/tmp/umber-export-test"))
    }

    #[test]
    fn defaults_match_documented_values() {
        let d = dialog();
        assert_eq!(d.preset(), PresetChoice::GltfMetalRough);
        assert_eq!(d.output_dir(), Path::new("/tmp/umber-export-test"));
        assert_eq!(d.status(), "No export yet.");
        assert!(d.last_written().is_empty());
        assert!(d.last_skipped().is_empty());
    }

    #[test]
    fn preset_and_output_dir_setters_roundtrip() {
        let mut d = dialog();
        d.set_preset(PresetChoice::UnrealOrm);
        assert_eq!(d.preset(), PresetChoice::UnrealOrm);
        d.set_output_dir(PathBuf::from("/tmp/elsewhere"));
        assert_eq!(d.output_dir(), Path::new("/tmp/elsewhere"));
    }

    #[test]
    fn can_export_gates_on_mesh_and_gpu() {
        let d = dialog();
        assert!(d.can_export(true, true));
        assert!(!d.can_export(false, true));
        assert!(!d.can_export(true, false));
        assert!(!d.can_export(false, false));
    }

    #[test]
    fn preset_label_matches_built_preset_name() {
        for choice in PresetChoice::ALL {
            assert_eq!(choice.build().name, choice.label(), "{choice:?}");
        }
    }

    #[test]
    fn filter_satisfiable_keeps_everything_when_every_map_is_available() {
        let available = [
            umber_export::MapKind::BaseColor,
            umber_export::MapKind::Roughness,
            umber_export::MapKind::Metallic,
            umber_export::MapKind::AmbientOcclusion,
            umber_export::MapKind::Normal,
            umber_export::MapKind::Height,
            umber_export::MapKind::Opacity,
            umber_export::MapKind::Emissive,
        ];
        for choice in PresetChoice::ALL {
            let preset = choice.build();
            let (kept, skipped) = filter_satisfiable(&preset, &available);
            assert_eq!(kept.outputs.len(), preset.outputs.len(), "{choice:?}");
            assert!(skipped.is_empty(), "{choice:?}");
        }
    }

    #[test]
    fn filter_satisfiable_keeps_only_the_normal_output_from_ao_and_normal() {
        // Today's export-ready map set: AO + flat normal. Every built-in
        // preset collapses to its normal passthrough alone — the
        // headline fact behind LANDING_NOTES_EXPORT_DIALOG.md.
        let available = [
            umber_export::MapKind::AmbientOcclusion,
            umber_export::MapKind::Normal,
        ];
        for choice in PresetChoice::ALL {
            let preset = choice.build();
            let (kept, skipped) = filter_satisfiable(&preset, &available);
            assert_eq!(
                kept.outputs.len(),
                1,
                "{choice:?}: only normal should survive"
            );
            assert_eq!(
                kept.outputs[0].maps,
                vec![(umber_export::MapKind::Normal, vec![])],
                "{choice:?}: kept output isn't the normal passthrough"
            );
            assert_eq!(skipped.len(), preset.outputs.len() - 1, "{choice:?}");
            for s in &skipped {
                assert!(
                    !s.missing.contains(&"ambient_occlusion"),
                    "{choice:?}: AO is available, must never be reported missing"
                );
            }
        }
    }

    #[test]
    fn unreal_orm_skip_names_roughness_and_metallic_not_ao() {
        let available = [
            umber_export::MapKind::AmbientOcclusion,
            umber_export::MapKind::Normal,
        ];
        let preset = PresetChoice::UnrealOrm.build();
        let (_, skipped) = filter_satisfiable(&preset, &available);
        let orm = skipped
            .iter()
            .find(|s| s.filename.contains("ORM"))
            .expect("ORM output should be skipped");
        assert_eq!(orm.missing, vec!["roughness", "metallic"]);
    }

    #[test]
    fn filtered_preset_satisfies_the_driver_validation() {
        // Proves filter_satisfiable's output never trips run_preset's
        // MissingMap/SizeMismatch checks — the one way the filter could
        // be wrong that actually matters.
        let size = 2;
        let ao = vec![200u8; (size * size * 4) as usize];
        let map_set = bake_sources::map_set_from(ao, size);
        let available: Vec<umber_export::MapKind> = map_set.maps_iter().collect();

        for choice in PresetChoice::ALL {
            let full_preset = choice.build();
            let (filtered, _skipped) = filter_satisfiable(&full_preset, &available);
            let out = std::env::temp_dir().join(format!(
                "umber-export-dialog-test-{choice:?}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&out);
            let written = umber_export::run_preset(&filtered, &map_set, "Probe", &out)
                .unwrap_or_else(|e| panic!("{choice:?}: filtered preset should validate: {e:#}"));
            assert_eq!(written.len(), filtered.outputs.len(), "{choice:?}");
            std::fs::remove_dir_all(&out).ok();
        }
    }
}

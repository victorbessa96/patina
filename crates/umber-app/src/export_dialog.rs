//! The Export panel: drives `umber-export`'s preset engine from the
//! egui shell.
//!
//! Wave 5 scope (painted-maps bridge, audit remainder #1): a
//! synchronous Export button that bakes today's available source maps
//! (AO via [`crate::bake_sources::bake_ao`], a flat-normal placeholder,
//! and — when a paint session is live — the LIVE paint target as Base
//! Color via [`crate::bake_sources::painted_base_color`]), filters the
//! chosen engine preset down to the outputs those maps can satisfy
//! (mirroring `umber-cli export`'s "export what we can, name what's
//! missing" behavior), and runs [`umber_export::run_preset`] with the
//! full driver-side token sources (`$mesh`/`$textureSet`/`$udim`/
//! `$colorSpace`/`$srcMap`/`$layerName`) to write the files. See
//! `LANDING_NOTES_EXPORT_DIALOG.md` for the baked-maps-today vs.
//! painted-maps story.
//!
//! UDIM (wave-5 slice 5): with a multi-tile paint session the export
//! runs once per present tile ([`PaintState::tiles_present`]) through
//! [`umber_export::run_preset_tiled`] — each tile's AO baked over its own
//! UV window, its Base Color read from THAT tile's target
//! ([`crate::bake_sources::painted_base_color_tile`]), and `$udim`
//! expanding to the tile's number. No session (or one tile) is the
//! single-tile `[1001]` path, byte-identical to before.
//!
//! Slice 6 makes the tile badge interactive: per-tile checkboxes
//! ([`TileSelection`], default every exportable tile) filter the planned
//! tiles before anything bakes ([`selected_export_tiles`]); the badge
//! counts the selected tiles. Single-tile exports show no checkboxes and
//! always export `[1001]`. Known v1 wart: narrowing a multi-tile session
//! to tile 1001 alone takes the lone-1001 path (whole-mesh AO, unsuffixed
//! names — the driver's own single-tile rule in `umber-export`).
//!
//! V1 painted scope is Base Color ONLY (paint-per-channel compositing
//! is a later slice): normal/AO keep their baked sources, and the
//! dialog always shows which source feeds Base Color (the
//! painted-source badge) so a flat-placeholder export never masquerades
//! as painted.
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

use crate::bake_sources::{self, BaseColorSource};
use crate::bakes_panel;
use crate::document::Document;
use crate::graph_panel::GraphPanel;
use crate::paint_state::PaintState;
use crate::tile_selection::{MeshTilesCache, TileSelection};

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
    /// The live paint session (`None` = no paint target: Base Color
    /// exports the flat placeholder — see the dialog's source badge).
    pub paint: Option<&'a PaintState>,
    /// The open document (for the `$layerName` token; `None` behaves
    /// like an empty document).
    pub doc: Option<&'a Document>,
    /// The graph panel (for the procedural Base Color bridge; `None`
    /// or no evaluated output behaves like no graph — see the badge).
    pub graph: Option<&'a GraphPanel>,
}

/// What the badge shows before an export: the shared pick-fn priority
/// over liveness probes (paint session present? graph output at the
/// export size?), so the badge and the driver can never disagree. Pure
/// logic — directly testable without a GPU.
pub fn prospective_base_source(
    painted_live: bool,
    graph: Option<&GraphPanel>,
    size: u32,
) -> BaseColorSource {
    let probe = graph.and_then(|g| {
        g.output_node().map(|node| {
            let len_ok = g
                .export_output_dims()
                .is_some_and(|dims| dims == (size, size));
            (node, len_ok)
        })
    });
    bake_sources::pick_base_color_source(painted_live, probe)
}

/// Which UDIM tiles an export writes, and which it skips: the paint
/// session's present tiles (`None` — no session — means the single-tile
/// `[1001]` path, unchanged). A single tile is always exported (the
/// pre-UDIM whole-mesh bake, byte-identical); in a multi-tile export a
/// tile owning no mesh triangles (`mesh_tiles` = the mesh's
/// `tile_of_triangle`) is skipped — its AO bake has nothing to rasterize
/// — and reported, never aborting the other tiles. Pure logic.
pub fn plan_export_tiles(
    paint_tiles: Option<Vec<u16>>,
    mesh_tiles: &[u16],
) -> (Vec<u16>, Vec<u16>) {
    let tiles = paint_tiles.unwrap_or_else(|| vec![umber_mesh::FIRST_TILE]);
    if tiles.len() <= 1 {
        return (tiles, Vec::new());
    }
    tiles
        .into_iter()
        .partition(|tile| mesh_tiles.contains(tile))
}

/// The tiles an Export click hands the tiled driver, and the present
/// tiles skipped for owning no mesh geometry: [`plan_export_tiles`]
/// narrowed by the dialog's tile checkboxes (`selection`, resolved
/// against the planned list so an unseen tile defaults to selected).
/// A single planned tile is exported whatever the selection holds. Pure
/// logic — the selection -> driver `tiles` mapping, testable headless.
pub fn selected_export_tiles(
    paint_tiles: Option<Vec<u16>>,
    mesh_tiles: &[u16],
    selection: &TileSelection,
) -> (Vec<u16>, Vec<u16>) {
    let (planned, skipped) = plan_export_tiles(paint_tiles, mesh_tiles);
    (selection.resolve(&planned), skipped)
}

/// The badge's tile note beside the Base Color source: `Some("2 tiles")`
/// for a multi-tile session, `None` for the single-tile case (the badge
/// reads exactly as before).
pub fn tile_badge(tile_count: usize) -> Option<String> {
    (tile_count > 1).then(|| format!("{tile_count} tiles"))
}

/// The status line's Base Color note over every exported tile: one
/// source named once when all tiles agree (plus the tile count when
/// several), else each tile's source listed. Pure logic.
pub fn base_note(sources: &[(u16, BaseColorSource)]) -> String {
    fn name(source: BaseColorSource) -> String {
        match source {
            BaseColorSource::Painted => "painted".to_string(),
            BaseColorSource::Graph(node) => format!("graph node {node}"),
            BaseColorSource::FlatPlaceholder => "flat placeholder".to_string(),
        }
    }
    let Some((_, first)) = sources.first() else {
        return "Base Color: none".to_string();
    };
    if sources.iter().all(|(_, s)| s == first) {
        match tile_badge(sources.len()) {
            Some(tiles) => format!("Base Color: {} ({tiles})", name(*first)),
            None => format!("Base Color: {}", name(*first)),
        }
    } else {
        let parts: Vec<String> = sources
            .iter()
            .map(|(tile, s)| format!("{tile} {}", name(*s)))
            .collect();
        format!("Base Color: {}", parts.join(", "))
    }
}

/// V1 `$layerName` source: the topmost layer's display name (stack order
/// is bottom-to-top, so the last layer is what the user sees on top).
/// `""` when there is no document or it holds no layers — the token
/// then stays literal per the template engine's empty-value contract.
pub fn layer_token(doc: Option<&Document>) -> &str {
    doc.and_then(|d| d.layers().last())
        .map(|layer| layer.name.as_str())
        .unwrap_or("")
}

/// One finished export's outcome: what was written, what was skipped,
/// and phase timings for the status line.
struct ExportOutcome {
    written: Vec<PathBuf>,
    skipped: Vec<SkippedOutput>,
    texture_set: String,
    /// Each exported tile's Base Color source, in export order.
    base_sources: Vec<(u16, BaseColorSource)>,
    /// Present tiles skipped for owning no mesh geometry.
    skipped_tiles: Vec<u16>,
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
    /// Which exportable tiles an Export covers (default: all).
    tiles: TileSelection,
    /// The loaded mesh's present tiles, cached across frames.
    mesh_tiles: MeshTilesCache,
}

impl ExportDialog {
    /// Creates the dialog writing to `output_dir` with the glTF
    /// metal-rough preset selected (and every exportable tile).
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            preset: PresetChoice::GltfMetalRough,
            output_dir,
            status: String::from("No export yet."),
            last_written: Vec::new(),
            last_skipped: Vec::new(),
            tiles: TileSelection::new(),
            mesh_tiles: MeshTilesCache::default(),
        }
    }

    /// The tile checkboxes' state (test hook: the UI drives the same
    /// `sync`/`set_selected` calls).
    #[cfg(test)]
    pub(crate) fn tiles_mut(&mut self) -> &mut TileSelection {
        &mut self.tiles
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

        // Source badge: the user must SEE which source feeds the Base
        // Color output — honesty in UI (three-way: painted, graph,
        // flat; normal/AO stay baked; paint-per-channel is a later
        // slice). The badge mirrors the driver's priority exactly via
        // the shared pick fn (painted live session wins over graph).
        // The tile note rides beside the source (UDIM: one file set per
        // selected tile; single-tile sessions read exactly as before).
        // The checkboxes list the tiles the driver would export (painted
        // tiles with mesh geometry) — only when there is a choice.
        if let Some(mesh) = ctx.mesh {
            let (exportable, _) = plan_export_tiles(
                ctx.paint.map(PaintState::tiles_present),
                self.mesh_tiles.get(mesh),
            );
            self.tiles.sync(&exportable);
        }
        if self.tiles.known_len() > 1 {
            self.tiles.show(ui);
            if self.tiles.nothing_selected() {
                ui.label("Select at least one tile to export.");
            }
        }
        let tiles_note = ctx
            .paint
            .and_then(|_| tile_badge(self.tiles.tiles().len()))
            .map(|tiles| format!(" [{tiles}]"))
            .unwrap_or_default();
        match prospective_base_source(
            ctx.paint.is_some(),
            ctx.graph,
            bakes_panel::DEFAULT_RESOLUTION,
        ) {
            BaseColorSource::Painted => {
                ui.label(format!(
                    "Base Color source: painted{tiles_note} — what you painted is what exports."
                ));
            }
            BaseColorSource::Graph(node) => {
                ui.label(format!(
                    "Base Color source: graph node {node}{tiles_note} — the panel's evaluated output."
                ));
            }
            BaseColorSource::FlatPlaceholder => {
                ui.label(
                    "Base Color source: flat placeholder — no paint session or graph output live.",
                );
            }
        }

        let enabled = self.can_export(mesh_loaded, gpu_ready) && !self.tiles.nothing_selected();
        if ui
            .add_enabled(enabled, egui::Button::new("Export"))
            .clicked()
        {
            self.export_now(&ctx);
        }

        ui.separator();
        ui.label(self.status.clone());
    }

    /// Runs the export synchronously and records the outcome in
    /// `status`/`last_written`/`last_skipped` (never propagates: the
    /// panel reports failures as a status line, not a crash).
    fn export_now(&mut self, ctx: &ExportContext<'_>) {
        match self.run(ctx) {
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
                let base_line = base_note(&outcome.base_sources);
                let mut status = format!(
                    "Exported {} outputs ({base_line}; bake {} ms, write {} ms): {}",
                    outcome.written.len(),
                    outcome.bake_ms,
                    outcome.write_ms,
                    names.join(", ")
                );
                if !outcome.skipped.is_empty() {
                    let sources = skipped_display_sources(&outcome.texture_set);
                    let parts: Vec<String> = outcome
                        .skipped
                        .iter()
                        .map(|s| {
                            let filename = sources.expand_static(&s.filename);
                            format!("{filename} (missing {})", s.missing.join(", "))
                        })
                        .collect();
                    status.push_str(&format!(" — skipped: {}", parts.join("; ")));
                }
                if !outcome.skipped_tiles.is_empty() {
                    let tiles: Vec<String> =
                        outcome.skipped_tiles.iter().map(u16::to_string).collect();
                    status.push_str(&format!(
                        " — tiles skipped (no mesh geometry): {}",
                        tiles.join(", ")
                    ));
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

    /// The synchronous export driver: bakes one source `MapSet` per
    /// exported tile (AO + flat normal + painted-or-graph-or-flat Base
    /// Color), filters the selected preset to its satisfiable outputs,
    /// and runs it per tile with the full driver-side token sources.
    /// `Result`-typed so failures carry context instead of unwrapping.
    fn run(&self, ctx: &ExportContext<'_>) -> anyhow::Result<ExportOutcome> {
        let (Some(gpu), Some(mesh)) = (ctx.gpu, ctx.mesh) else {
            anyhow::bail!("export needs a loaded mesh and a GPU device");
        };
        let (device, queue) = (&gpu.device, &gpu.queue);
        let mesh_path = ctx.mesh_path;
        let paint = ctx.paint;
        let doc = ctx.doc;
        let graph = ctx.graph;
        let fallback = PathBuf::from(bakes_panel::FALLBACK_TEXTURE_SET);
        let texture_set = umber_mesh::texture_set_name(mesh_path.unwrap_or(&fallback), mesh);
        let size = bakes_panel::DEFAULT_RESOLUTION;

        // UDIM (slice 5): one map set per present paint tile. No session,
        // or a single-tile one, is the pre-UDIM `[1001]` path unchanged.
        // Slice 6: narrowed to the checked tiles before anything bakes.
        let (tiles, skipped_tiles) = selected_export_tiles(
            paint.map(PaintState::tiles_present),
            &umber_mesh::present_tiles(mesh),
            &self.tiles,
        );
        // Only the lone-1001 export keeps the whole-mesh bake (whose
        // rasterized window IS tile 1001); any other tile bakes its own.
        let per_tile_bake = tiles != [umber_mesh::FIRST_TILE];

        let bake_started = Instant::now();
        // Per tile: the baked half (whole-mesh AO for the lone 1001 —
        // byte-identical to before; the tile's own UV window otherwise),
        // then the painted bridge (THAT tile's target), then the graph
        // bridge (tile 1001 only), else the flat placeholder (never
        // silent — each tile's source rides the outcome into the status).
        let graphed = graph.and_then(|g| g.output_node().zip(g.export_base_color()));
        let tile_sets = bake_sources::assemble_tile_map_sets(
            &tiles,
            size,
            |tile| {
                let baked = if per_tile_bake {
                    bake_sources::bake_export_map_set_tile(
                        device,
                        queue,
                        mesh,
                        size,
                        bakes_panel::DEFAULT_RAYS,
                        tile,
                    )
                } else {
                    bake_sources::bake_export_map_set(
                        device,
                        queue,
                        mesh,
                        size,
                        bakes_panel::DEFAULT_RAYS,
                    )
                };
                baked.context("building export map set")
            },
            |tile| bake_sources::painted_base_color_tile(paint, tile, device, queue),
            graphed,
        )?;
        let bake_ms = bake_started.elapsed().as_millis();
        let Some(first) = tile_sets.first() else {
            anyhow::bail!("no tile has mesh geometry to export");
        };

        // Every tile's set carries the same kinds (AO + normal + Base
        // Color), so one filter serves all tiles.
        let available: Vec<umber_export::MapKind> = first.set.maps_iter().collect();
        let full_preset = self.preset.build();
        let (filtered, skipped) = filter_satisfiable(&full_preset, &available);
        if filtered.outputs.is_empty() {
            anyhow::bail!(
                "preset '{}' has no outputs satisfiable from today's maps (AO + flat normal + Base Color)",
                full_preset.name
            );
        }

        // Full driver-side token sources: $mesh from the loaded path,
        // $layerName v1 from the document's top layer; $udim is each
        // tile's own number (set per tile by the tiled driver);
        // $srcMap/$colorSpace derived per output by the driver.
        let mesh_stem = mesh_path
            .map(umber_export::mesh_stem)
            .unwrap_or("")
            .to_string();
        let layer_name = layer_token(doc).to_string();
        let sources = umber_export::TokenSources {
            texture_set: &texture_set,
            mesh: &mesh_stem,
            layer_name: &layer_name,
            udim: umber_export::SINGLE_TILE_UDIM,
        };

        let write_started = Instant::now();
        let pairs: Vec<(u16, &umber_export::MapSet)> =
            tile_sets.iter().map(|t| (t.tile, &t.set)).collect();
        let written = umber_export::run_preset_tiled(&filtered, &pairs, &sources, &self.output_dir)
            .context("run_preset_tiled")?;
        let write_ms = write_started.elapsed().as_millis();

        Ok(ExportOutcome {
            written,
            skipped,
            texture_set,
            base_sources: tile_sets.iter().map(|t| (t.tile, t.base_source)).collect(),
            skipped_tiles,
            bake_ms,
            write_ms,
        })
    }
}

/// Minimal sources for the skipped-output display line (texture set
/// only — the mesh/layer context is not retained on the outcome).
fn skipped_display_sources(texture_set: &str) -> umber_export::TokenSources<'_> {
    umber_export::TokenSources::new(texture_set)
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
            let written = umber_export::run_preset(
                &filtered,
                &map_set,
                &umber_export::TokenSources::new("Probe"),
                &out,
            )
            .unwrap_or_else(|e| panic!("{choice:?}: filtered preset should validate: {e:#}"));
            assert_eq!(written.len(), filtered.outputs.len(), "{choice:?}");
            std::fs::remove_dir_all(&out).ok();
        }
    }

    #[test]
    fn painted_base_color_keeps_the_base_color_output() {
        // Post-bridge map set: AO + flat normal + painted Base Color.
        // Every built-in preset must now keep its baseColor output ALONGSIDE
        // normal (pre-bridge: normal alone).
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let mut set = bake_sources::map_set_from(ao, size);
        let source = bake_sources::apply_base_color(
            &mut set,
            size,
            Some(vec![7u8; (size * size * 4) as usize]),
        );
        assert_eq!(source, bake_sources::BaseColorSource::Painted);
        let available: Vec<umber_export::MapKind> = set.maps_iter().collect();
        assert!(available.contains(&umber_export::MapKind::BaseColor));
        for choice in PresetChoice::ALL {
            let preset = choice.build();
            let (kept, _) = filter_satisfiable(&preset, &available);
            assert!(
                kept.outputs.iter().any(|o| o
                    .maps
                    .iter()
                    .any(|(k, _)| *k == umber_export::MapKind::BaseColor)),
                "{choice:?}: painted Base Color must keep the baseColor output"
            );
            assert!(
                kept.outputs.iter().any(|o| o
                    .maps
                    .iter()
                    .any(|(k, _)| *k == umber_export::MapKind::Normal)),
                "{choice:?}: normal output must survive too"
            );
        }
    }

    #[test]
    fn prospective_source_picks_graph_when_set() {
        use crate::graph_panel::GraphPanel;
        // No graph at all: flat.
        assert_eq!(
            prospective_base_source(false, None, 512),
            BaseColorSource::FlatPlaceholder
        );
        // Evaluated uniform graph, no paint: Graph (naming the node).
        let mut panel = GraphPanel::new();
        let id = panel.add_node("uniform");
        panel.set_output_node(id);
        panel.evaluate();
        assert_eq!(
            prospective_base_source(false, Some(&panel), 512),
            BaseColorSource::Graph(id)
        );
        // A live paint session still wins over a ready graph.
        assert_eq!(
            prospective_base_source(true, Some(&panel), 512),
            BaseColorSource::Painted
        );
        // Graph evaluated at another size: flat (honest size gate).
        assert_eq!(
            prospective_base_source(false, Some(&panel), 256),
            BaseColorSource::FlatPlaceholder
        );
        // Unevaluated graph: flat.
        let fresh = GraphPanel::new();
        assert_eq!(
            prospective_base_source(false, Some(&fresh), 512),
            BaseColorSource::FlatPlaceholder
        );
    }

    #[test]
    fn layer_token_is_empty_without_layers_and_top_with_them() {
        use umber_core::layers::{LayerCommand, LayerKind};
        assert_eq!(layer_token(None), "");
        let mut doc = Document::default();
        assert_eq!(layer_token(Some(&doc)), "");
        doc.run(LayerCommand::add("Paint 1", LayerKind::Paint));
        assert_eq!(layer_token(Some(&doc)), "Paint 1");
        // Stack order is bottom-to-top: the LAST layer is the top.
        doc.run(LayerCommand::add("Paint 2", LayerKind::Paint));
        assert_eq!(layer_token(Some(&doc)), "Paint 2");
    }

    #[test]
    fn plan_export_tiles_single_path_and_geometry_skip() {
        // No session: the single-tile path, whatever the mesh spans.
        assert_eq!(plan_export_tiles(None, &[1002, 1002]), (vec![1001], vec![]));
        // One present tile: always exported (the whole-mesh bake).
        assert_eq!(
            plan_export_tiles(Some(vec![1001]), &[1002]),
            (vec![1001], vec![])
        );
        // Two present tiles over a two-tile mesh: both exported, sorted.
        assert_eq!(
            plan_export_tiles(Some(vec![1001, 1002]), &[1001, 1001, 1002, 1002]),
            (vec![1001, 1002], vec![])
        );
        // A mirror-created tile with no geometry is skipped, not fatal.
        assert_eq!(
            plan_export_tiles(Some(vec![1001, 1002, 1100]), &[1001, 1002]),
            (vec![1001, 1002], vec![1100])
        );
    }

    #[test]
    fn checked_tiles_are_the_drivers_tile_list() {
        let mesh_tiles = [1001, 1002, 1011];
        let paint = Some(vec![1001, 1002, 1011]);
        let mut d = dialog();
        // `show` syncs the checkboxes to the plan's exportable tiles.
        let (exportable, _) = plan_export_tiles(paint.clone(), &mesh_tiles);
        d.tiles_mut().sync(&exportable);
        // Default: every exportable tile checked -> all exported.
        assert_eq!(
            selected_export_tiles(paint.clone(), &mesh_tiles, d.tiles_mut()),
            (vec![1001, 1002, 1011], vec![])
        );
        assert_eq!(
            tile_badge(d.tiles_mut().tiles().len()).as_deref(),
            Some("3 tiles")
        );
        // Uncheck 1002: the driver receives exactly [1001, 1011].
        d.tiles_mut().set_selected(1002, false);
        assert_eq!(
            selected_export_tiles(paint.clone(), &mesh_tiles, d.tiles_mut()),
            (vec![1001, 1011], vec![])
        );
        assert_eq!(
            tile_badge(d.tiles_mut().tiles().len()).as_deref(),
            Some("2 tiles")
        );
        // A geometry-less painted tile stays reported as skipped and
        // never appears as a checkbox (it was never exportable).
        let (tiles, skipped) =
            selected_export_tiles(Some(vec![1001, 1002, 1100]), &[1001, 1002], d.tiles_mut());
        assert_eq!((tiles, skipped), (vec![1001], vec![1100]));
    }

    #[test]
    fn a_tile_painted_after_the_last_sync_exports_by_default() {
        let mut d = dialog();
        d.tiles_mut().sync(&[1001, 1002]);
        d.tiles_mut().set_selected(1001, false);
        // 1003 appeared this frame (painted into) before the UI synced:
        // it defaults to checked; the user's uncheck of 1001 holds.
        assert_eq!(
            selected_export_tiles(
                Some(vec![1001, 1002, 1003]),
                &[1001, 1002, 1003],
                d.tiles_mut()
            ),
            (vec![1002, 1003], vec![])
        );
    }

    #[test]
    fn single_tile_export_list_is_1001_regardless_of_selection() {
        // Regression: a single-tile mesh (or no paint session) exports
        // [1001] with no checkboxes and no badge — exactly as before.
        let single_mesh_tiles = [1001];
        let mut d = dialog();
        let (exportable, _) = plan_export_tiles(Some(vec![1001]), &single_mesh_tiles);
        d.tiles_mut().sync(&exportable);
        assert_eq!(d.tiles_mut().known_len(), 1);
        // The lone tile can't be unchecked away.
        d.tiles_mut().set_selected(1001, false);
        assert!(!d.tiles_mut().nothing_selected());
        assert_eq!(
            selected_export_tiles(Some(vec![1001]), &single_mesh_tiles, d.tiles_mut()),
            (vec![1001], vec![])
        );
        assert_eq!(
            selected_export_tiles(None, &single_mesh_tiles, d.tiles_mut()),
            (vec![1001], vec![])
        );
        assert_eq!(tile_badge(d.tiles_mut().tiles().len()), None);
        // A fresh dialog (no sync yet) is already the single-tile list.
        assert_eq!(
            selected_export_tiles(None, &[1001, 1002], dialog().tiles_mut()),
            (vec![1001], vec![])
        );
    }

    #[test]
    fn tile_badge_names_the_count_only_when_multi_tile() {
        assert_eq!(tile_badge(0), None);
        assert_eq!(tile_badge(1), None);
        assert_eq!(tile_badge(2).as_deref(), Some("2 tiles"));
    }

    #[test]
    fn base_note_reports_every_tiles_source_honestly() {
        // Single tile: the pre-UDIM wording exactly.
        assert_eq!(
            base_note(&[(1001, BaseColorSource::Painted)]),
            "Base Color: painted"
        );
        assert_eq!(
            base_note(&[(1001, BaseColorSource::Graph(4))]),
            "Base Color: graph node 4"
        );
        // Agreeing tiles: one source plus the count.
        assert_eq!(
            base_note(&[
                (1001, BaseColorSource::Painted),
                (1002, BaseColorSource::Painted)
            ]),
            "Base Color: painted (2 tiles)"
        );
        // Disagreeing tiles: each named (a flat tile never hides behind
        // a painted one).
        assert_eq!(
            base_note(&[
                (1001, BaseColorSource::Painted),
                (1002, BaseColorSource::FlatPlaceholder)
            ]),
            "Base Color: 1001 painted, 1002 flat placeholder"
        );
    }

    #[test]
    fn skipped_display_expands_the_texture_set_token() {
        let sources = skipped_display_sources("Blade");
        assert_eq!(
            sources.expand_static("$textureSet_baseColor.png"),
            "Blade_baseColor.png"
        );
    }
}

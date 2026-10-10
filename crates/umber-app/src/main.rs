//! umber-app — the desktop application shell.
//!
//! Wave 1: egui/eframe boot + dockable panel shell (egui_dock) + app state
//! wiring. The GPU pass lands the viewport widget: `UmberApp::new` captures
//! eframe's `wgpu_render_state` (available because `NativeOptions.renderer`
//! is forced to `Renderer::Wgpu` below) and builds a `GpuContext` from it —
//! no `wgpu::Instance`/`Adapter` is ever constructed in this crate.
//!
//! API note: egui 0.36 replaced menu::bar with MenuBar/MenuButton
//! containers, TopBottomPanel with the unified Panel API, and moved
//! NativeOptions to eframe; this file was written against the vendored
//! source, not remembered APIs. Threading: the UI thread owns everything
//! except Bakes-panel bakes, which run on a `std::thread` worker
//! (`bakes_panel::BakeJobHandle`, polled each frame below); the
//! paint-thread + ring-buffer architecture (docs/specs/architecture.md)
//! lands with the paint engine.

mod bake_sources;
mod bakes_panel;
mod brush_panel;
mod display_panel;
mod document;
mod env;
mod export_dialog;
mod graph_canvas;
mod graph_panel;
mod i18n;
#[cfg(test)]
mod input_photon;
mod paint_state;
#[cfg(test)]
mod perf_soak;
mod plugins;
mod size_presets;
mod tile_selection;
mod uv_view;
mod viewport;

use bakes_panel::{BakesContext, BakesPanel};
use brush_panel::BrushPanel;
use egui::containers::menu::{MenuBar, MenuButton};
use egui::{CentralPanel, Id, Ui, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use export_dialog::{ExportContext, ExportDialog};
use graph_panel::GraphPanel;
use i18n::tr;
use std::path::{Path, PathBuf};
use umber_gpu::GpuContext;
use uv_view::UvView;
use viewport::Viewport;

/// Every panel the shell can dock.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Panel {
    Viewport,
    UvView,
    LayerStack,
    Properties,
    Assets,
    History,
    TextureSets,
    Bakes,
    Export,
    Graph,
    Display,
}

/// Implements the egui_dock tab interface. Built fresh each frame, borrowing
/// the pieces of `UmberApp` it needs — `egui_dock::TabViewer` has no access
/// to the app otherwise, so this is how the viewport panel reaches the GPU
/// context and the camera/mesh state that outlive a single frame.
struct PanelViewer<'a> {
    viewport: &'a mut Viewport,
    uv_view: &'a mut UvView,
    bakes: &'a mut BakesPanel,
    export: &'a mut ExportDialog,
    graph: &'a mut GraphPanel,
    mesh: Option<&'a umber_mesh::MeshData>,
    mesh_path: Option<&'a std::path::Path>,
    gpu: &'a GpuContext,
    paint: Option<&'a mut paint_state::PaintState>,
    doc: &'a mut document::Document,
    brush: &'a mut BrushPanel,
    display: &'a mut umber_color::DisplaySettings,
}

impl TabViewer for PanelViewer<'_> {
    type Tab = Panel;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        Id::new(tab)
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        match tab {
            Panel::Viewport => tr("panel.viewport").into(),
            Panel::UvView => tr("panel.uv-view").into(),
            Panel::LayerStack => tr("panel.layers").into(),
            Panel::Properties => tr("panel.properties").into(),
            Panel::Assets => tr("panel.assets").into(),
            Panel::History => tr("panel.history").into(),
            Panel::TextureSets => tr("panel.texture-sets").into(),
            Panel::Bakes => tr("panel.bakes").into(),
            Panel::Export => tr("panel.export").into(),
            Panel::Graph => tr("panel.graph").into(),
            Panel::Display => tr("panel.display").into(),
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        match tab {
            Panel::Viewport => self.viewport.ui(ui, self.gpu, self.paint.as_deref_mut()),
            Panel::UvView => self
                .uv_view
                .ui(ui, self.mesh, self.gpu, self.paint.as_deref_mut()),
            Panel::LayerStack => document::layers_ui(ui, self.doc),
            Panel::Properties => {
                self.brush.show(ui);
            }
            Panel::Assets => {
                ui.label("Assets / shelf (Wave 4+)");
            }
            Panel::History => document::history_ui(ui, self.doc),
            Panel::TextureSets => {
                ui.label("Texture sets (Wave 2)");
            }
            Panel::Bakes => {
                let ctx = BakesContext {
                    gpu: Some(self.gpu),
                    mesh: self.mesh,
                    mesh_path: self.mesh_path,
                };
                self.bakes.show(ui, ctx);
            }
            Panel::Export => {
                let ctx = ExportContext {
                    gpu: Some(self.gpu),
                    mesh: self.mesh,
                    mesh_path: self.mesh_path,
                    // Shared reborrow (not `take()`): the Export panel
                    // reads the live target for the painted bridge while
                    // the center views keep staging strokes into it.
                    paint: self.paint.as_deref(),
                    doc: Some(&*self.doc),
                    graph: Some(&*self.graph),
                };
                self.export.show(ui, ctx);
            }
            Panel::Graph => {
                self.graph.show(ui);
            }
            Panel::Display => {
                display_panel::show(ui, self.display);
            }
        }
    }
}

/// Shared application state (grows per wave).
#[derive(Default)]
pub struct AppState {
    pub mesh: Option<umber_mesh::MeshData>,
    pub mesh_path: Option<PathBuf>,
    pub viewport: Viewport,
    pub uv_view: UvView,
    /// The paint session; `None` when the device lacks the storage-texture
    /// feature (constructed in `UmberApp::new`, falls back to view-only).
    pub paint: Option<paint_state::PaintState>,
    /// The open document: layers + undo journal.
    pub doc: document::Document,
    /// The bakes panel: mesh-map bake settings + last-bake status.
    pub bakes: BakesPanel,
    /// The export dialog: preset choice, output dir + last-export status.
    pub export: ExportDialog,
    /// The node-graph panel: procedural graph + cached eval (wave-5).
    pub graph: GraphPanel,
    /// The viewer chain (Display panel; saved in the project). CPU
    /// preview only until the GPU display LUT consumes it.
    pub display: umber_color::DisplaySettings,
}

/// The eframe app.
pub struct UmberApp {
    pub state: AppState,
    dock: DockState<Panel>,
    gpu: GpuContext,
    brush_panel: BrushPanel,
    /// Perf HUD toggle (View menu). `perf` feature only.
    #[cfg(feature = "perf")]
    show_perf_hud: bool,
    /// Previous frame's `egui::InputState::time`, for the HUD's frame-ms
    /// line (`perf` feature only).
    #[cfg(feature = "perf")]
    last_frame_time: Option<f64>,
    /// Cumulative `dabs_composited` at the previous frame, for the HUD's
    /// dab-throughput delta (`perf` feature only).
    #[cfg(feature = "perf")]
    last_dabs_composited: u64,
    /// Wireframe overlay toggle (View menu, W). Synced into the
    /// viewport each frame; off by default (off renders byte-identical
    /// to the mesh-only frame).
    show_wireframe: bool,
    /// Ground-grid toggle (View menu, G). Same sync + off contract.
    show_grid: bool,
}

impl UmberApp {
    /// Captures eframe's `wgpu_render_state` (see module docs) and builds
    /// the one `GpuContext` the app uses for its lifetime.
    pub fn new(cc: &eframe::CreationContext<'_>) -> anyhow::Result<Self> {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("umber requires eframe's wgpu backend"))?;
        let gpu = GpuContext::new(
            render_state.adapter.clone(),
            render_state.device.clone(),
            render_state.queue.clone(),
            render_state.target_format,
            // Depth: eframe threads NativeOptions::depth_buffer into the egui
            // renderer's depth_stencil_format (wgpu_integration.rs:232 ->
            // egui_wgpu::depth_format_from_bits), which attaches depth to the
            // shared render pass our callback draws into. Value comes from
            // umber-gpu so this crate never names a wgpu type directly
            // (review #1 + architecture rule).
            umber_gpu::renderer::depth_format(),
        );

        // Layout: [left column | center (3D + 2D UV tabs)] with history docked right.
        let mut dock = DockState::new(vec![Panel::Viewport, Panel::UvView]);
        {
            let tree = dock.main_surface_mut();
            let [left, _] = tree.split_left(
                NodeIndex::root(),
                0.25,
                vec![
                    Panel::LayerStack,
                    Panel::TextureSets,
                    Panel::Bakes,
                    Panel::Export,
                    Panel::Graph,
                    Panel::Display,
                ],
            );
            let [_top, _bottom] =
                tree.split_below(left, 0.4, vec![Panel::Properties, Panel::Assets]);
            tree.split_right(NodeIndex::root(), 0.2, vec![Panel::History]);
        }

        // Paint session on the eframe-owned device/queue (cloned for the
        // thread). Feature-missing devices degrade to view-only with a
        // logged warning, never a hard startup failure.
        let paint = match paint_state::PaintState::new(
            render_state.device.clone(),
            render_state.queue.clone(),
        ) {
            Ok(session) => Some(session),
            Err(err) => {
                log::warn!("paint disabled: {err:#}");
                None
            }
        };

        let mut graph = GraphPanel::new();
        graph.load_plugins(&plugins::plugin_paths());

        Ok(Self {
            state: AppState {
                paint,
                graph,
                ..AppState::default()
            },
            dock,
            gpu,
            brush_panel: BrushPanel::new(),
            #[cfg(feature = "perf")]
            show_perf_hud: false,
            #[cfg(feature = "perf")]
            last_frame_time: None,
            #[cfg(feature = "perf")]
            last_dabs_composited: 0,
            show_wireframe: false,
            show_grid: false,
        })
    }
}

/// App-level actions (kept out of the App impl to keep that block small).
impl UmberApp {
    /// Loads the committed studio default if present (procedural
    /// otherwise) — called once at startup, after construction, so a
    /// missing asset degrades to the fallback instead of failing.
    fn load_default_env(&mut self) {
        if let Some(env) = env::load_default_environment(&self.gpu) {
            self.state.viewport.set_environment(&self.gpu, Some(env));
        }
    }

    /// File > Load Environment…: file dialog → decode → convolve →
    /// viewport swap. Failures log; the current environment is kept.
    fn load_environment(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Environment", &["png", "exr"])
            .pick_file()
        else {
            return;
        };
        match env::load_environment_file(&self.gpu, &path) {
            Ok(ibl) => {
                self.state.viewport.set_environment(&self.gpu, Some(ibl));
                log::info!("loaded environment: {}", path.display());
            }
            Err(err) => log::error!("environment load failed: {err:#}"),
        }
    }
    /// Exports the current paint target as an sRGB PNG via a save dialog.
    ///
    /// Readback is blocking (one full-target copy); acceptable for a
    /// user-triggered export at 512².
    fn export_paint_png(&mut self) {
        let Some(paint) = self.state.paint.as_ref() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("painted_map.png")
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };
        let target = paint.paint_target();
        match target.read_back_rgba8(&self.gpu.device, &self.gpu.queue) {
            Ok(bytes) => {
                let (w, h) = target.dimensions();
                match umber_export::png::write_png(
                    &path,
                    w,
                    h,
                    &bytes,
                    umber_export::png::Transfer::Srgb,
                ) {
                    Ok(()) => log::info!("exported painted map: {}", path.display()),
                    Err(e) => log::error!("png write failed: {e:#}"),
                }
            }
            Err(e) => log::error!("readback failed: {e:#}"),
        }
    }

    /// The texture-set name for project save: the loaded mesh's set
    /// name, else the fallback default.
    fn project_texture_set_name(&self) -> String {
        umber_mesh::texture_set_name(
            self.state
                .mesh_path
                .as_deref()
                .unwrap_or(Path::new("TextureSet")),
            self.state
                .mesh
                .as_ref()
                .unwrap_or(&umber_mesh::MeshData::default()),
        )
    }

    /// Saves the document as a `.umber` project directory via a save
    /// dialog: one texture set (named from the loaded mesh), the
    /// layer stack + project settings through `umber_core::project`,
    /// plus the graph panel's graph as a `.mtlx` string (the additive
    /// wave-5 carry — omitted when the panel's graph is empty, so
    /// graph-less projects save exactly the old shape).
    fn save_project(&mut self) {
        let set_name = self.project_texture_set_name();
        let Some(dir) = rfd::FileDialog::new()
            .set_file_name(format!("{set_name}.umber"))
            .save_file()
        else {
            return;
        };
        let graphs_mtlx = if self.state.graph.graph().nodes.is_empty() {
            Vec::new()
        } else {
            vec![self.state.graph.to_mtlx()]
        };
        let model = umber_core::project::ProjectModel::new(
            vec![umber_core::TextureSet::new_default(&set_name)],
            vec![umber_core::project::TextureSetLayers {
                texture_set: set_name.clone(),
                stack: self.state.doc.stack.clone(),
            }],
            umber_core::project::ProjectSettings {
                active_texture_set: Some(set_name.clone()),
                display: self.state.display,
            },
        )
        .with_graphs_mtlx(graphs_mtlx);
        match umber_core::project::save_to_dir(&model, &dir) {
            Ok(()) => log::info!("project saved: {}", dir.display()),
            Err(e) => log::error!("project save failed: {e}"),
        }
    }

    /// Loads a `.umber` project directory: restores the active texture
    /// set's layer stack into the document (undo history starts fresh;
    /// the journal is session-scoped by design) and the first carried
    /// `.mtlx` graph into the graph panel (a parse failure keeps the
    /// current panel graph and logs — the panel's status line says so).
    fn open_project(&mut self) {
        let Some(dir) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        match umber_core::project::load_from_dir(&dir) {
            Ok(model) => {
                let active = model
                    .settings
                    .active_texture_set
                    .or_else(|| model.texture_sets.first().map(|ts| ts.name.clone()));
                let Some(name) = active else {
                    log::error!("project has no texture sets: {}", dir.display());
                    return;
                };
                let Some(entry) = model.layers.iter().find(|l| l.texture_set == name) else {
                    log::error!("texture set {name:?} has no layers entry");
                    return;
                };
                self.state.doc.load_stack(entry.stack.clone());
                // Hand-edited files can carry out-of-range values; the
                // sliders' ranges are the contract.
                self.state.display = model.settings.display.sanitized();
                if let Some(doc) = model.graphs_mtlx.first() {
                    if self.state.graph.load_mtlx(doc) {
                        log::info!("project graph restored into the Graph panel");
                    } else {
                        log::error!(
                            "project graph failed to parse; panel graph unchanged: {}",
                            self.state.graph.status()
                        );
                    }
                }
                log::info!(
                    "project loaded: {} ({} layers in {name:?})",
                    dir.display(),
                    self.state.doc.stack.layers.len()
                );
            }
            Err(e) => log::error!("project load failed: {e}"),
        }
    }
}

impl eframe::App for UmberApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        profiling::scope!("frame");
        // Async bake (wave 5): drain a finished bake job here, not only in
        // the Bakes tab's `show` (egui_dock skips hidden tabs), and keep
        // frames coming while one runs — egui otherwise repaints only on
        // input, so the result would sit until the mouse moved. (The click
        // frame itself starts the job after this poll; the tab's spinner
        // requests the next frame, and this keeps it going from there.)
        if self.state.bakes.poll_job() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
        // Overlay keybinds (Wave-4 item 7): W toggles the wireframe, G
        // the ground grid. Skipped while a text edit has focus (brush
        // preset names, save-as dialog) so typing never flips the
        // viewport.
        if !ui.ctx().text_edit_focused() {
            let (wire_key, grid_key) =
                ui.input(|i| (i.key_pressed(egui::Key::W), i.key_pressed(egui::Key::G)));
            if wire_key {
                self.show_wireframe = !self.show_wireframe;
            }
            if grid_key {
                self.show_grid = !self.show_grid;
            }
        }
        self.state.viewport.set_show_wireframe(self.show_wireframe);
        self.state.viewport.set_show_grid(self.show_grid);
        // Top bar: MenuBar container (egui 0.36 API).
        MenuBar::new().ui(ui, |ui| {
            MenuButton::new(tr("menu.file")).ui(ui, |ui| {
                if ui.button(tr("menu.open-mesh")).clicked() {
                    if let Some(path) = rfd_pick_mesh() {
                        match umber_mesh::load(&path) {
                            Ok(mesh) => {
                                let n = mesh.vertex_count();
                                match self.state.viewport.load_mesh(&self.gpu, &mesh) {
                                    Ok(()) => {
                                        self.state.mesh = Some(mesh);
                                        self.state.mesh_path = Some(path);
                                        log::info!("loaded mesh: {n} vertices");
                                    }
                                    Err(e) => log::error!("mesh upload failed: {e:#}"),
                                }
                            }
                            Err(e) => log::error!("mesh load failed: {e}"),
                        }
                    }
                }
                if ui.button(tr("menu.open-project")).clicked() {
                    self.open_project();
                }
                if ui.button(tr("menu.save-project")).clicked() {
                    self.save_project();
                }
                let export_enabled = self.state.paint.is_some();
                if ui
                    .add_enabled(export_enabled, egui::Button::new(tr("menu.export-png")))
                    .clicked()
                {
                    self.export_paint_png();
                }
                if ui.button(tr("menu.load-environment")).clicked() {
                    self.load_environment();
                }
            });
            MenuButton::new(tr("menu.view")).ui(ui, |ui| {
                #[cfg(feature = "perf")]
                ui.checkbox(&mut self.show_perf_hud, tr("menu.perf-hud"));
                #[cfg(not(feature = "perf"))]
                ui.label(tr("menu.perf-hud-unavailable"));
                ui.checkbox(&mut self.show_wireframe, tr("menu.wireframe"));
                ui.checkbox(&mut self.show_grid, tr("menu.grid"));
            });
            MenuButton::new(tr("menu.help")).ui(ui, |ui| {
                if ui.button(tr("menu.about")).clicked() {
                    ui.label(tr("menu.about-text"));
                }
            });
        });

        CentralPanel::default().show(ui, |ui| {
            let mut viewer = PanelViewer {
                viewport: &mut self.state.viewport,
                uv_view: &mut self.state.uv_view,
                bakes: &mut self.state.bakes,
                export: &mut self.state.export,
                graph: &mut self.state.graph,
                mesh: self.state.mesh.as_ref(),
                mesh_path: self.state.mesh_path.as_deref(),
                gpu: &self.gpu,
                paint: self.state.paint.as_mut(),
                doc: &mut self.state.doc,
                brush: &mut self.brush_panel,
                display: &mut self.state.display,
            };
            DockArea::new(&mut self.dock)
                .style(Style::from_egui(ui.style()))
                .show_inside(ui, &mut viewer);
        });

        // Drain the paint channel once per frame after all panels ran.
        if let Some(paint) = self.state.paint.as_mut() {
            if let Err(err) = paint.process_pending() {
                log::warn!("paint processing failed: {err:#}");
            }
        }

        // Perf HUD (read-only overlay, `perf` feature only): frame ms from
        // egui's input-time delta, dab throughput from FrameStats deltas,
        // paint-target bytes, undo depth. No budget enforcement — display
        // only, per the instrumentation design.
        #[cfg(feature = "perf")]
        {
            let now = ui.input(|i| i.time);
            let frame_ms = self
                .last_frame_time
                .map(|t| (now - t) * 1000.0)
                .unwrap_or(0.0);
            self.last_frame_time = Some(now);
            if self.show_perf_hud {
                let dabs_total = self
                    .state
                    .paint
                    .as_ref()
                    .map(|p| p.last_stats().dabs_composited)
                    .unwrap_or(0);
                let dabs_delta = dabs_total.saturating_sub(self.last_dabs_composited);
                self.last_dabs_composited = dabs_total;
                let target_bytes = self
                    .state
                    .paint
                    .as_ref()
                    .map(|p| p.paint_target().byte_len())
                    .unwrap_or(0);
                let undo_depth = self.state.doc.history.len();
                egui::Window::new("Perf HUD").show(ui.ctx(), |ui| {
                    ui.label(format!("frame: {frame_ms:.2} ms"));
                    ui.label(format!(
                        "dabs this frame: {dabs_delta} (total {dabs_total})"
                    ));
                    ui.label(format!("paint target: {target_bytes} bytes"));
                    ui.label(format!("undo depth: {undo_depth}"));
                });
            }
        }
    }
}

/// Minimal file picker (rfd; wrapped so the call site stays clean if we
/// swap pickers later).
fn rfd_pick_mesh() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Meshes", &["obj", "gltf", "glb", "fbx", "usda", "usd"])
        .pick_file()
}

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("umber — starting (wave-2: paint core)");
    // UI strings: catalogs + locale on this (the UI) thread, before any
    // panel renders (docs/specs/i18n-design.md).
    i18n::init();

    // Paint-compositor requirement (umber-gpu paint claw reviewer checklist
    // item #1): read_write storage textures need the adapter-specific format
    // features requested on the device, or the compute splat pass panics at
    // first real stroke. eframe owns the device, so the requirement must be
    // injected here — through the device descriptor closure.
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(ref mut create_new) = wgpu_options.wgpu_setup {
        create_new.device_descriptor =
            std::sync::Arc::new(|_adapter| eframe::egui_wgpu::wgpu::DeviceDescriptor {
                required_features:
                    eframe::egui_wgpu::wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            });
    }

    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Umber"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        // Gives the egui renderer's render pass a depth attachment matching
        // umber-gpu's pipeline (review #1). Value sourced from umber-gpu so
        // this crate stays free of direct wgpu types.
        depth_buffer: umber_gpu::DEPTH_FORMAT_BITS,
        ..Default::default()
    };
    eframe::run_native(
        "Umber",
        native,
        Box::new(|cc| match UmberApp::new(cc) {
            Ok(mut app) => {
                app.load_default_env();
                Ok(Box::new(app) as Box<dyn eframe::App>)
            }
            Err(err) => Err(format!("{err:#}").into()),
        }),
    )
}

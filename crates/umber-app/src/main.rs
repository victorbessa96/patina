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
//! source, not remembered APIs. Threading: single-threaded skeleton — the
//! paint-thread + ring-buffer architecture (docs/specs/architecture.md)
//! lands with the paint engine.

mod uv_view;
mod viewport;

use egui::containers::menu::{MenuBar, MenuButton};
use egui::{CentralPanel, Id, Ui, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use std::path::PathBuf;
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
}

/// Implements the egui_dock tab interface. Built fresh each frame, borrowing
/// the pieces of `UmberApp` it needs — `egui_dock::TabViewer` has no access
/// to the app otherwise, so this is how the viewport panel reaches the GPU
/// context and the camera/mesh state that outlive a single frame.
struct PanelViewer<'a> {
    viewport: &'a mut Viewport,
    uv_view: &'a mut UvView,
    mesh: Option<&'a umber_mesh::MeshData>,
    gpu: &'a GpuContext,
}

impl TabViewer for PanelViewer<'_> {
    type Tab = Panel;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        Id::new(tab)
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        match tab {
            Panel::Viewport => "Viewport".into(),
            Panel::UvView => "2D UV".into(),
            Panel::LayerStack => "Layers".into(),
            Panel::Properties => "Properties".into(),
            Panel::Assets => "Assets".into(),
            Panel::History => "History".into(),
            Panel::TextureSets => "Texture Sets".into(),
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        match tab {
            Panel::Viewport => self.viewport.ui(ui, self.gpu),
            Panel::UvView => self.uv_view.ui(ui, self.mesh),
            Panel::LayerStack => {
                ui.label("Layer stack (Wave 2)");
            }
            Panel::Properties => {
                ui.label("Properties (Wave 2)");
            }
            Panel::Assets => {
                ui.label("Assets / shelf (Wave 4+)");
            }
            Panel::History => {
                ui.label("History (Wave 2)");
            }
            Panel::TextureSets => {
                ui.label("Texture sets (Wave 2)");
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
}

/// The eframe app.
pub struct UmberApp {
    pub state: AppState,
    dock: DockState<Panel>,
    gpu: GpuContext,
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
                vec![Panel::LayerStack, Panel::TextureSets],
            );
            let [_top, _bottom] =
                tree.split_below(left, 0.4, vec![Panel::Properties, Panel::Assets]);
            tree.split_right(NodeIndex::root(), 0.2, vec![Panel::History]);
        }

        Ok(Self {
            state: AppState::default(),
            dock,
            gpu,
        })
    }
}

impl eframe::App for UmberApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        // Top bar: MenuBar container (egui 0.36 API).
        MenuBar::new().ui(ui, |ui| {
            MenuButton::new("File").ui(ui, |ui| {
                if ui.button("Open Mesh…").clicked() {
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
            });
            MenuButton::new("Help").ui(ui, |ui| {
                if ui.button("About Umber").clicked() {
                    ui.label("Umber v0.1.0 — Wave 2 in progress");
                }
            });
        });

        CentralPanel::default().show(ui, |ui| {
            let mut viewer = PanelViewer {
                viewport: &mut self.state.viewport,
                uv_view: &mut self.state.uv_view,
                mesh: self.state.mesh.as_ref(),
                gpu: &self.gpu,
            };
            DockArea::new(&mut self.dock)
                .style(Style::from_egui(ui.style()))
                .show_inside(ui, &mut viewer);
        });
    }
}

/// Minimal file picker (rfd; wrapped so the call site stays clean if we
/// swap pickers later).
fn rfd_pick_mesh() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Meshes", &["obj", "gltf", "glb", "fbx"])
        .pick_file()
}

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("umber — starting (wave-2: paint core)");

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
            Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
            Err(err) => Err(format!("{err:#}").into()),
        }),
    )
}

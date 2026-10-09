//! umber-app — the desktop application shell.
//!
//! Wave 1: egui/eframe boot + dockable panel shell (egui_dock) + app state
//! wiring. The viewport widget (wgpu surface inside the dock tree) is the
//! claw-built piece of this wave; the shell here compiles and runs now.
//!
//! API note: egui 0.36 replaced menu::bar with MenuBar/MenuButton
//! containers, TopBottomPanel with the unified Panel API, and moved
//! NativeOptions to eframe; this file was written against the vendored
//! source, not remembered APIs. Threading: single-threaded skeleton — the
//! paint-thread + ring-buffer architecture (docs/specs/architecture.md)
//! lands with the GPU pass.

use egui::containers::menu::{MenuBar, MenuButton};
use egui::{CentralPanel, Id, Ui, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use std::path::PathBuf;

/// Every panel the shell can dock.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Panel {
    Viewport,
    LayerStack,
    Properties,
    Assets,
    History,
    TextureSets,
}

/// Implements the egui_dock tab interface. Stateless in the skeleton;
/// panel state moves into the shared app state with the GPU pass.
pub struct PanelViewer;

impl TabViewer for PanelViewer {
    type Tab = Panel;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        Id::new(tab)
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        match tab {
            Panel::Viewport => "Viewport".into(),
            Panel::LayerStack => "Layers".into(),
            Panel::Properties => "Properties".into(),
            Panel::Assets => "Assets".into(),
            Panel::History => "History".into(),
            Panel::TextureSets => "Texture Sets".into(),
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        match tab {
            Panel::Viewport => {
                // Claw pass replaces this placeholder with the wgpu
                // surface widget. Kept honest: it says what it is.
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0);
                    ui.weak("viewport (wgpu surface) lands with the GPU pass");
                });
            }
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
}

/// The eframe app.
pub struct UmberApp {
    pub state: AppState,
    dock: DockState<Panel>,
    viewer: PanelViewer,
}

impl Default for UmberApp {
    fn default() -> Self {
        // Layout: [left column | viewport] with history docked right.
        let mut dock = DockState::new(vec![Panel::Viewport]);
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

        Self {
            state: AppState::default(),
            dock,
            viewer: PanelViewer,
        }
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
                                self.state.mesh = Some(mesh);
                                self.state.mesh_path = Some(path);
                                log::info!("loaded mesh: {n} vertices");
                            }
                            Err(e) => log::error!("mesh load failed: {e}"),
                        }
                    }
                }
            });
            MenuButton::new("Help").ui(ui, |ui| {
                if ui.button("About Umber").clicked() {
                    ui.label("Umber v0.1.0 — Wave 1 skeleton");
                }
            });
        });

        CentralPanel::default().show(ui, |ui| {
            DockArea::new(&mut self.dock)
                .style(Style::from_egui(ui.style()))
                .show_inside(ui, &mut self.viewer);
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
    log::info!("umber — starting (wave-1 skeleton)");

    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Umber"),
        ..Default::default()
    };
    eframe::run_native(
        "Umber",
        native,
        Box::new(|_cc| Ok(Box::new(UmberApp::default()))),
    )
}

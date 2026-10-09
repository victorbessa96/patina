//! The 3D viewport: owns the orbit camera and the GPU-resident mesh, and
//! paints into the dock's `Panel::Viewport` tab each frame.
//!
//! Never touches a `wgpu` type by name — every value it holds or passes to
//! `umber_gpu` is opaque to this crate (received from or handed to
//! `umber_gpu`'s API), per the architecture rule that GPU objects stay
//! behind `umber-gpu`.

use egui::{Color32, PointerButton, Response, Sense, Ui};
use umber_gpu::{GpuContext, MeshBuffers, OrbitCamera};

/// Background shown when there is no mesh loaded (and behind the mesh
/// otherwise, since the shared egui render pass has no clear op of its
/// own — see `GpuContext::new` docs on why there's no depth/clear here).
const EMPTY_VIEWPORT_COLOR: Color32 = Color32::from_rgb(28, 28, 30);

/// Degrees-per-pixel for drag-orbit; chosen so a full-width drag on a
/// typical viewport is roughly one full turn.
const ORBIT_SENSITIVITY: f32 = 0.01;
/// Fraction of the viewport's visible size a full-width pan drag covers.
const PAN_SENSITIVITY: f32 = 0.0015;
/// Zoom step per wheel notch (multiplicative, see `OrbitCamera::zoom`).
const ZOOM_SENSITIVITY: f32 = 0.08;

/// Owns the camera and the currently-loaded mesh's GPU buffers.
#[derive(Default)]
pub struct Viewport {
    camera: OrbitCamera,
    mesh: Option<MeshBuffers>,
}

impl Viewport {
    /// Uploads `mesh` to the GPU and re-frames the camera on its bounds,
    /// keeping the current yaw/pitch so loading a new mesh doesn't spin
    /// the view out from under the user.
    pub fn load_mesh(
        &mut self,
        gpu: &GpuContext,
        mesh: &umber_mesh::MeshData,
    ) -> anyhow::Result<()> {
        let buffers = MeshBuffers::upload(gpu, mesh)?;
        if let Some((min, max)) = mesh.bounds() {
            self.camera = OrbitCamera::framing(min, max, self.camera.yaw, self.camera.pitch);
        }
        self.mesh = Some(buffers);
        Ok(())
    }

    /// Draws the viewport and handles orbit/pan/zoom input for this frame.
    pub fn ui(&mut self, ui: &mut Ui, gpu: &GpuContext) {
        let rect = ui.available_rect_before_wrap();
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return;
        }
        let response = ui.allocate_rect(rect, Sense::click_and_drag());

        self.handle_input(ui, &response);

        ui.painter().rect_filled(rect, 0.0, EMPTY_VIEWPORT_COLOR);

        if let Some(mesh) = &self.mesh {
            let aspect = rect.width() / rect.height();
            let view_proj = self.camera.view_proj(aspect);
            let light_dir = glam::Vec3::new(-0.4, -1.0, -0.3);
            let uniform = umber_gpu::CameraUniform::new(view_proj, light_dir);
            let callback = mesh.paint_callback(gpu, uniform);
            let shape = umber_gpu::mesh_paint_shape(rect, callback);
            ui.painter().add(shape);
        }
    }

    fn handle_input(&mut self, ui: &Ui, response: &Response) {
        if response.dragged_by(PointerButton::Primary) {
            let delta = response.drag_delta();
            if ui.input(|i| i.modifiers.shift) {
                self.camera
                    .pan(-delta.x * PAN_SENSITIVITY, delta.y * PAN_SENSITIVITY);
            } else {
                self.camera
                    .orbit(delta.x * ORBIT_SENSITIVITY, -delta.y * ORBIT_SENSITIVITY);
            }
        } else if response.dragged_by(PointerButton::Middle) {
            let delta = response.drag_delta();
            self.camera
                .pan(-delta.x * PAN_SENSITIVITY, delta.y * PAN_SENSITIVITY);
        }

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll.signum() * ZOOM_SENSITIVITY);
            }
        }
    }
}

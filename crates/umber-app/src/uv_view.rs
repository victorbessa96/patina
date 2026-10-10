//! The 2D UV view: draws the loaded mesh's UV triangles as a wireframe
//! on the 0..1 UV square (requirements.md §5: "2D UV view alongside 3D
//! view (synchronized)" — Wave 2 scope: the wireframe + fit; painting INTO
//! this view arrives with the paint engine).

use egui::{Color32, Sense, Stroke, Ui};
use umber_mesh::MeshData;

/// Wireframe color for UV island edges.
const UV_EDGE_COLOR: Color32 = Color32::from_rgb(140, 190, 255);
/// Background inside the 0..1 UV square.
const UV_BG_COLOR: Color32 = Color32::from_rgb(38, 38, 42);
/// Background outside the UV square (padding area).
const PAD_COLOR: Color32 = Color32::from_rgb(28, 28, 30);

/// The 2D UV view panel state. Wave 2: stateless drawing; zoom/pan for
/// this view lands with the paint-engine sync work.
#[derive(Default)]
pub struct UvView;

impl UvView {
    /// Draws the UV wireframe for `mesh` (or a hint when none is loaded)
    /// and feeds pointer strokes into the paint session. The paint
    /// canvas presents through `display_lut` (the Display panel's chain;
    /// `None` = the identity table).
    pub fn ui(
        &mut self,
        ui: &mut Ui,
        mesh: Option<&MeshData>,
        gpu: &umber_gpu::GpuContext,
        mut paint: Option<&mut crate::paint_state::PaintState>,
        display_lut: Option<&umber_gpu::DisplayLut>,
    ) {
        let rect = ui.available_rect_before_wrap();
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return;
        }
        let response = ui.allocate_rect(rect, Sense::click_and_drag());

        // The UV square occupies the largest centered square that fits.
        let side = rect.width().min(rect.height());
        let origin = egui::Pos2::new(
            rect.left() + (rect.width() - side) / 2.0,
            rect.top() + (rect.height() - side) / 2.0,
        );
        let square = egui::Rect::from_min_size(origin, egui::vec2(side, side));

        // Stroke input: primary-button drag inside the UV square maps to
        // UV space and feeds the paint conditioner. Left-drag orbits in
        // the 3D view; here left-drag paints.
        if let Some(paint) = paint.as_deref_mut() {
            let interact = ui.interact(
                square,
                ui.id().with("uv-paint"),
                egui::Sense::click_and_drag(),
            );
            let pointer_down =
                interact.dragged_by(egui::PointerButton::Primary) || interact.clicked();
            if pointer_down {
                if let Some(pos) = interact.interact_pointer_pos() {
                    if let Some(uv) = crate::paint_state::uv_from_pointer(square, pos) {
                        if !paint.is_stroking() {
                            paint.begin_stroke(uv);
                        } else {
                            paint.extend_stroke(uv);
                        }
                    }
                }
            } else if paint.is_stroking() {
                paint.end_stroke();
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, PAD_COLOR);
        painter.rect_filled(square, 0.0, UV_BG_COLOR);

        // Live paint canvas: the paint target presented through the
        // texture-display callback, clipped to the UV square, through the
        // display LUT. Drawn beneath the wireframe so strokes show
        // through (the egui-drawn UV wireframe is not LUT-transformed).
        if let Some(paint) = paint {
            let callback = gpu.texture_display().callback(
                &gpu.device,
                paint.paint_target(),
                square,
                display_lut,
            );
            let shape = umber_gpu::texture_display::texture_display_shape(square, callback);
            painter.add(shape);
        }

        // UV-space -> screen-space: v is flipped (UV origin is bottom-left,
        // screen origin is top-left).
        let to_screen = |uv: [f32; 2]| {
            egui::Pos2::new(
                square.left() + uv[0].clamp(0.0, 1.0) * side,
                square.bottom() - uv[1].clamp(0.0, 1.0) * side,
            )
        };

        match mesh {
            Some(mesh) => {
                // Draw each triangle as a 3-segment line loop. Per-triangle
                // iteration is fine for Wave 2 viewport counts; batching
                // into a single mesh arrives with paint-engine sync.
                let stroke = Stroke::new(1.0, UV_EDGE_COLOR);
                for tri in mesh.indices.chunks_exact(3) {
                    let pts: Option<Vec<egui::Pos2>> = (0..3)
                        .map(|i| mesh.uvs.get(tri[i] as usize).map(|uv| to_screen(*uv)))
                        .collect();
                    if let Some([a, b, c]) = pts
                        .as_deref()
                        .and_then(|p: &[egui::Pos2]| <&[egui::Pos2; 3]>::try_from(p).ok())
                    {
                        let (a, b, c) = (*a, *b, *c);
                        painter.line_segment([a, b], stroke);
                        painter.line_segment([b, c], stroke);
                        painter.line_segment([c, a], stroke);
                    }
                }
            }
            None => {
                painter.text(
                    square.center(),
                    egui::Align2::CENTER_CENTER,
                    "no mesh loaded",
                    egui::FontId::proportional(13.0),
                    egui::Color32::from_gray(120),
                );
            }
        }
        let _ = response;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_view_is_stateless_default() {
        // The panel holds no state in Wave 2; the default exists purely so
        // AppState can derive Default. This test pins that decision: if UV
        // view gains state (zoom/pan), update this test deliberately.
        let _view = UvView;
    }
}

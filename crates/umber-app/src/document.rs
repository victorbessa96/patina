//! App-side document state: the layer stack plus its undo journal, and
//! the egui panels that view/drive them (Layers + History).

use umber_core::layers::{Layer, LayerCommand, LayerKind, LayerStack};
use umber_core::undo::UndoStack;

/// The open document: layer stack + undo history.
///
/// Wave 2 scope: one texture-set layer stack; multi-set documents and
/// the paint-channel binding to GPU targets arrive with the tile pool.
pub struct Document {
    pub stack: LayerStack,
    pub history: UndoStack<LayerCommand>,
}

/// Undo journal capacity (entries). Sized for long sessions; the memory
/// budget system (bytes) arrives with GPU-tile snapshots.
const HISTORY_MAX_ENTRIES: usize = 256;

impl Default for Document {
    fn default() -> Self {
        Self {
            stack: LayerStack::default(),
            history: UndoStack::new(HISTORY_MAX_ENTRIES),
        }
    }
}

impl Document {
    /// Runs `command` through the journal (applies + records for undo).
    pub fn run(&mut self, command: LayerCommand) {
        self.history.push(command, &mut self.stack);
    }

    /// One-level undo; returns whether anything was undone.
    pub fn undo(&mut self) -> bool {
        self.history.undo(&mut self.stack)
    }

    /// One-level redo; returns whether anything was redone.
    pub fn redo(&mut self) -> bool {
        self.history.redo(&mut self.stack)
    }

    /// Layers bottom-to-top (stack order == draw order).
    pub fn layers(&self) -> &[Layer] {
        &self.stack.layers
    }
}

/// Draws the Layers panel: per-layer visibility toggle, name, kind badge.
pub fn layers_ui(ui: &mut egui::Ui, doc: &mut Document) {
    if ui.button("＋ Add paint layer").clicked() {
        let n = doc.layers().len() + 1;
        doc.run(LayerCommand::add(format!("Paint {n}"), LayerKind::Paint));
    }
    let mut remove_id = None;
    egui::ScrollArea::vertical().show(ui, |ui| {
        // Snapshot the display rows first so command runs below can't
        // borrow-conflict with the iteration (E0502).
        #[derive(Clone)]
        struct Row {
            id: u64,
            name: String,
            kind: &'static str,
            visible: bool,
        }
        let rows: Vec<Row> = doc
            .layers()
            .iter()
            .rev()
            .map(|layer| Row {
                id: layer.id,
                name: layer.name.clone(),
                kind: match layer.kind {
                    LayerKind::Paint => "paint",
                    LayerKind::Fill => "fill",
                    LayerKind::Folder { .. } => "folder",
                },
                visible: layer.visible,
            })
            .collect();
        for layer in &rows {
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(layer.visible, if layer.visible { "👁" } else { "·" })
                    .clicked()
                {
                    doc.run(LayerCommand::set_visible(layer.id, !layer.visible));
                }
                ui.label(&layer.name);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(layer.kind)
                            .small()
                            .color(egui::Color32::from_gray(140)),
                    );
                    if ui.small_button("✕").clicked() {
                        remove_id = Some(layer.id);
                    }
                });
            });
        }
    });
    if let Some(id) = remove_id {
        doc.run(LayerCommand::remove(id));
    }
}

/// Draws the History panel: journal size + undo/redo buttons.
pub fn history_ui(ui: &mut egui::Ui, doc: &mut Document) {
    ui.horizontal(|ui| {
        let can_undo = doc.history.can_undo();
        let can_redo = doc.history.can_redo();
        if ui
            .add_enabled(can_undo, egui::Button::new("↶ Undo"))
            .clicked()
        {
            doc.undo();
        }
        if ui
            .add_enabled(can_redo, egui::Button::new("↷ Redo"))
            .clicked()
        {
            doc.redo();
        }
    });
    ui.separator();
    // The journal doesn't expose entry labels yet (Wave 2 core keeps
    // commands opaque); show the live counts instead of a fake list.
    let n = doc.history.len();
    let entry_word = if n == 1 { "entry" } else { "entries" };
    ui.label(format!("{n} {entry_word} in journal"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_undo_roundtrip() {
        let mut doc = Document::default();
        doc.run(LayerCommand::add("base", LayerKind::Fill));
        assert_eq!(doc.layers().len(), 1);
        assert!(doc.undo());
        assert_eq!(doc.layers().len(), 0);
        assert!(doc.redo());
        assert_eq!(doc.layers().len(), 1);
    }

    #[test]
    fn fresh_document_is_empty() {
        let doc = Document::default();
        assert!(doc.layers().is_empty());
        assert!(!doc.history.can_undo());
        assert!(!doc.history.can_redo());
    }
}

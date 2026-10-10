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

    /// Replaces the document's layer stack wholesale — the
    /// Open-Project path. The journal holds ids and indices captured
    /// against the *old* stack, so it is cleared with the swap:
    /// undo history is session-scoped by design and does not follow
    /// the file across an open.
    pub fn load_stack(&mut self, stack: LayerStack) {
        self.history.clear();
        self.stack = stack;
    }

    /// Layers bottom-to-top (stack order == draw order).
    pub fn layers(&self) -> &[Layer] {
        &self.stack.layers
    }

    /// Approximate in-memory size of the document in bytes (layer metadata
    /// only — see [`umber_core::layers::LayerStack::memory_bytes`]).
    ///
    /// An honest approximation, documented, not the §12 budget: the undo
    /// soak test samples this after each push to assert sub-linear growth
    /// plus an empirical ceiling (2x the observed size, recorded in the
    /// test output — the §12 ~2GB @ 4K target needs tile accounting first).
    pub fn in_memory_bytes(&self) -> usize {
        self.stack.memory_bytes()
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

    /// The Open-Project contract: the journal does not follow the
    /// file. Undo entries recorded against the previous document
    /// must never replay against the loaded stack — both stacks
    /// allocate layer ids from 0, so a stale `Add` revert would
    /// delete the loaded project's layer 0 by coincidental id.
    #[test]
    fn load_stack_clears_journal_no_stale_undo() {
        let mut doc = Document::default();
        // Arm the journal in document A: one Add, one Remove.
        doc.run(LayerCommand::add("A-base", LayerKind::Fill));
        doc.run(LayerCommand::add("A-detail", LayerKind::Paint));
        assert!(doc.history.can_undo());

        // Load document B's stack — the Open Project path.
        let mut b = LayerStack::new();
        let b0 = b.add_layer("B-base", LayerKind::Fill);
        let b1 = b.add_layer("B-top", LayerKind::Paint);
        doc.load_stack(b);

        // Journal is fresh: undo is a no-op, never replays A's Add.
        assert!(!doc.history.can_undo());
        assert!(!doc.history.can_redo());
        assert!(!doc.undo());
        let ids: Vec<u64> = doc.layers().iter().map(|l| l.id).collect();
        assert_eq!(
            ids,
            [b0, b1],
            "undo after open project must not touch the loaded stack"
        );
        // And the loaded stack still behaves: a new Add works and is
        // itself undoable.
        doc.run(LayerCommand::add("B-new", LayerKind::Paint));
        assert_eq!(doc.layers().len(), 3);
        assert!(doc.undo());
        assert_eq!(doc.layers().len(), 2);
    }

    /// Red-state proof of the exact corruption `load_stack` closes:
    /// swapping the stack raw, journal untouched (the pre-fix
    /// `open_project` path), replays document A's `Add{id: 0}` revert
    /// as a deletion of the loaded stack's layer 0.
    #[test]
    fn raw_stack_swap_keeps_stale_journal_and_corrupts() {
        let mut doc = Document::default();
        // Document A: one Add — the journal holds `Add{id: Some(0)}`.
        doc.run(LayerCommand::add("A-only", LayerKind::Fill));

        // Document B loaded via the RAW swap the fix replaces.
        let mut b = LayerStack::new();
        let b0 = b.add_layer("B-keep-me", LayerKind::Fill);
        let b1 = b.add_layer("B-top", LayerKind::Paint);
        doc.stack = b;

        // One undo click on the stale journal: Add.revert deletes by
        // id 0 — which is now document B's innocent bottom layer.
        assert!(doc.undo());
        assert_eq!(
            doc.layers().iter().map(|l| l.id).collect::<Vec<_>>(),
            [b1],
            "pre-fix path: stale Add (id 0) revert silently deleted the loaded layer {b0}"
        );
    }
}

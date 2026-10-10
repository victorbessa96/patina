//! The layer stack: paint/fill/folder layers with blend modes, opacity, and
//! masks (requirements.md §2).
//!
//! [`LayerStack`] is a flat, ordered `Vec<Layer>`. Folder *membership* (which
//! layers nest inside a given folder) is not modeled yet — see
//! LANDING_NOTES.md for why that is an open question left to the Wave 2
//! compositor rather than guessed at here.

use serde::{Deserialize, Serialize};

use std::mem::size_of;

use crate::undo::Command;

/// What a layer contains and how its children (if any) relate to the stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerKind {
    /// A layer painted into directly with brush strokes.
    Paint,
    /// A layer filled procedurally via a projection (UV/tri-planar/etc,
    /// Wave 3).
    Fill,
    /// A group of layers.
    ///
    /// `passthrough: true` means child layers blend directly with the
    /// layers below the folder, as if the folder were not there. `false`
    /// means the folder's contents are flattened to a single result first,
    /// and that result blends with layers below using the folder layer's
    /// own `blend_mode`/`opacity`. See [`BlendMode::Passthrough`] for how
    /// this flag relates to the blend-mode enum.
    Folder {
        /// See the variant's own doc comment above.
        passthrough: bool,
    },
}

/// The core 12 blend modes (requirements.md §2: "P0 core 12, full set 4").
///
/// Blending is computed in linear space, per channel (architecture.md). A
/// layer's `blend_mode` is meaningful for `Paint`/`Fill` layers and for a
/// `Folder` layer's own composite step; `Passthrough` only makes sense on a
/// `Folder` layer and is the blend-mode-enum spelling of what
/// [`LayerKind::Folder`]'s `passthrough` flag already controls. The two are
/// kept as separate fields (flag vs. enum variant) because the task's
/// authoritative source for "does this folder pass through" is the flag;
/// the compositor (not built in this task) is expected to treat a folder's
/// `blend_mode: Passthrough` as redundant with `passthrough: true` and
/// should not need to special-case a mismatch, but resolving that precisely
/// is left open for Wave 2's compositor work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BlendMode {
    /// Top layer fully replaces what's below it, modulated by opacity.
    #[default]
    Normal,
    /// Folder-only: children blend directly with layers below the folder,
    /// as if the folder were not there. See [`LayerKind::Folder`].
    Passthrough,
    /// Result × base — darkens; black stays black, white is a no-op.
    Multiply,
    /// Inverse-multiply — lightens; white stays white, black is a no-op.
    Screen,
    /// Multiply below 0.5 result luminance, Screen above — contrast boost.
    Overlay,
    /// Per-channel minimum of result and base.
    Darken,
    /// Per-channel maximum of result and base.
    Lighten,
    /// Linear-light additive blend; clamps at white.
    Add,
    /// Linear-light subtractive blend; clamps at black.
    Subtract,
    /// Absolute difference between result and base.
    Difference,
    /// Low-contrast Overlay variant (Photoshop/Substance-style soft light).
    SoftLight,
    /// Higher-contrast Overlay variant (roles of result/base swapped).
    HardLight,
}

/// A paint mask attached to a layer.
///
/// Pixel data lives in the tile pool (painting core, Wave 2); this is only
/// the metadata half — name and whether the mask currently affects
/// compositing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerMask {
    /// Display name, shown in the layer-stack panel next to the layer's own.
    pub name: String,
    /// Whether the mask currently affects compositing.
    pub enabled: bool,
}

/// One entry in a [`LayerStack`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Monotonically-allocated within the owning [`LayerStack`]; stable
    /// across save/load (see project.rs).
    pub id: u64,
    /// Display name, shown in the layer-stack panel.
    pub name: String,
    /// What this layer contains (paint/fill/folder).
    pub kind: LayerKind,
    /// Clamped to `0.0..=1.0` by every mutator on [`LayerStack`].
    pub opacity: f32,
    /// Whether this layer contributes to compositing.
    pub visible: bool,
    /// How this layer's result combines with the layers below it.
    pub blend_mode: BlendMode,
    /// Attached paint mask, if any.
    pub mask: Option<LayerMask>,
}

impl Layer {
    /// Approximate RAM held by this layer, in bytes: the inline struct
    /// plus heap name bytes. See [`LayerStack::memory_bytes`] for the
    /// honesty contract (metadata only, no GPU tiles yet).
    pub fn memory_bytes(&self) -> usize {
        size_of::<Layer>() + self.name.len() + self.mask.as_ref().map(|m| m.name.len()).unwrap_or(0)
    }
}

/// An ordered stack of layers for one texture set.
///
/// Index 0 is the bottom of the stack; the last element is the top.
/// Mutating methods return the *previous* value — an "op handle" callers
/// (notably [`LayerCommand`]) use to build undo without re-querying the
/// stack afterwards.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStack {
    /// Stack order, bottom (index 0) to top.
    pub layers: Vec<Layer>,
    next_id: u64,
}

impl Default for LayerStack {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerStack {
    /// An empty stack with id allocation starting at 0.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            next_id: 0,
        }
    }

    /// Rebuilds a stack from layers loaded from disk plus the id counter
    /// that was in effect when they were saved.
    ///
    /// Takes the saved counter rather than deriving `max(id) + 1` so that
    /// ids stay monotonic across a save/load cycle even if the top layer
    /// was deleted before saving (see project.rs LayerSetOrder).
    pub fn from_parts(layers: Vec<Layer>, saved_next_id: u64) -> Self {
        let min_safe = layers.iter().map(|l| l.id + 1).max().unwrap_or(0);
        Self {
            layers,
            next_id: saved_next_id.max(min_safe),
        }
    }

    /// The id the next [`add_layer`](Self::add_layer) call will allocate.
    pub fn next_layer_id(&self) -> u64 {
        self.next_id
    }

    /// The current stack position of the layer with `id`, if it exists.
    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// Looks up a layer by id.
    pub fn layer(&self, id: u64) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Looks up a layer by id, mutably.
    pub fn layer_mut(&mut self, id: u64) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// Pushes a new layer onto the top of the stack, returning its
    /// newly-allocated id.
    pub fn add_layer(&mut self, name: impl Into<String>, kind: LayerKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.layers.push(Layer {
            id,
            name: name.into(),
            kind,
            opacity: 1.0,
            visible: true,
            blend_mode: BlendMode::Normal,
            mask: None,
        });
        id
    }

    /// Re-inserts a layer at the top of the stack under a *specific* id,
    /// without allocating a fresh one.
    ///
    /// Only used by [`LayerCommand::Add`]'s redo path, so that redoing an
    /// add restores the exact same id a later command in the journal (e.g.
    /// a `SetOpacity` on that id) may still refer to. Keeps `next_id`
    /// monotonic even though `id` was not issued by this call.
    fn add_layer_with_id(&mut self, id: u64, name: String, kind: LayerKind) {
        self.layers.push(Layer {
            id,
            name,
            kind,
            opacity: 1.0,
            visible: true,
            blend_mode: BlendMode::Normal,
            mask: None,
        });
        self.next_id = self.next_id.max(id + 1);
    }

    /// Removes the layer with `id`, returning its former `(index, Layer)` so
    /// it can be reinserted verbatim (see [`insert_layer`](Self::insert_layer)).
    pub fn remove_layer(&mut self, id: u64) -> Option<(usize, Layer)> {
        let index = self.index_of(id)?;
        Some((index, self.layers.remove(index)))
    }

    /// Re-inserts a previously-removed layer at `index` (clamped to the
    /// current length).
    pub fn insert_layer(&mut self, index: usize, layer: Layer) {
        self.next_id = self.next_id.max(layer.id + 1);
        let index = index.min(self.layers.len());
        self.layers.insert(index, layer);
    }

    /// Moves the layer at `from` to `to` (clamped to the valid range),
    /// returning the index it actually landed at, or `None` if `from` was
    /// out of range.
    pub fn reorder(&mut self, from: usize, to: usize) -> Option<usize> {
        if from >= self.layers.len() {
            return None;
        }
        let layer = self.layers.remove(from);
        let to = to.min(self.layers.len());
        self.layers.insert(to, layer);
        Some(to)
    }

    /// Sets opacity, clamped to `0.0..=1.0`. Non-finite input is ignored
    /// (the layer's opacity is left unchanged). Returns the previous value.
    pub fn set_opacity(&mut self, id: u64, value: f32) -> Option<f32> {
        let layer = self.layer_mut(id)?;
        let old = layer.opacity;
        if value.is_finite() {
            layer.opacity = value.clamp(0.0, 1.0);
        }
        Some(old)
    }

    /// Returns the previous visibility.
    pub fn set_visible(&mut self, id: u64, value: bool) -> Option<bool> {
        let layer = self.layer_mut(id)?;
        let old = layer.visible;
        layer.visible = value;
        Some(old)
    }

    /// Returns the previous blend mode.
    pub fn set_blend_mode(&mut self, id: u64, mode: BlendMode) -> Option<BlendMode> {
        let layer = self.layer_mut(id)?;
        let old = layer.blend_mode;
        layer.blend_mode = mode;
        Some(old)
    }

    /// Approximate RAM held by the whole stack, in bytes: per-layer
    /// [`Layer::memory_bytes`] plus the `Vec` buffer itself.
    ///
    /// An honest approximation, not a budget: layer metadata only (names,
    /// struct fields). GPU-tile bytes are not modeled here — they arrive
    /// with the tile-pool slice, at which point this sum grows a tile term.
    /// The undo soak test samples this after each push to assert the growth
    /// curve's shape (sub-linear) plus an empirical byte ceiling.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(Layer::memory_bytes).sum::<usize>()
            + self.layers.capacity() * size_of::<Layer>()
    }
}

/// A reversible [`LayerStack`] edit, for use with `UndoStack<LayerCommand>`.
///
/// One enum rather than a family of small command structs, because
/// [`crate::undo::UndoStack`] is monomorphic over a single command type —
/// callers need one type that covers every op to run a single journal over
/// a layer stack.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerCommand {
    /// Adds a new layer to the top of the stack.
    Add {
        /// Name of the layer to create.
        name: String,
        /// Kind of the layer to create.
        kind: LayerKind,
        /// `None` until the first `apply`; afterwards holds the allocated
        /// id so redo can restore the exact same one.
        id: Option<u64>,
    },
    /// Removes an existing layer by id.
    Remove {
        /// Id of the layer to remove.
        id: u64,
        /// Captured `(index, Layer)` after `apply`, for `revert`.
        removed: Option<(usize, Layer)>,
    },
    /// Moves a layer from one stack position to another.
    Reorder {
        /// Source index.
        from: usize,
        /// Requested destination index (clamped on apply).
        to: usize,
        /// The index the layer actually landed at after `apply`.
        applied_to: Option<usize>,
    },
    /// Sets a layer's opacity.
    SetOpacity {
        /// Id of the layer to update.
        id: u64,
        /// New opacity value.
        new: f32,
        /// Previous opacity, captured by `apply` for `revert`.
        old: Option<f32>,
    },
    /// Sets a layer's visibility.
    SetVisible {
        /// Id of the layer to update.
        id: u64,
        /// New visibility value.
        new: bool,
        /// Previous visibility, captured by `apply` for `revert`.
        old: Option<bool>,
    },
    /// Sets a layer's blend mode.
    SetBlendMode {
        /// Id of the layer to update.
        id: u64,
        /// New blend mode.
        new: BlendMode,
        /// Previous blend mode, captured by `apply` for `revert`.
        old: Option<BlendMode>,
    },
}

impl LayerCommand {
    /// Builds an [`Add`](LayerCommand::Add) command for a new layer.
    pub fn add(name: impl Into<String>, kind: LayerKind) -> Self {
        LayerCommand::Add {
            name: name.into(),
            kind,
            id: None,
        }
    }

    /// Builds a [`Remove`](LayerCommand::Remove) command for an existing layer.
    pub fn remove(id: u64) -> Self {
        LayerCommand::Remove { id, removed: None }
    }

    /// Builds a [`Reorder`](LayerCommand::Reorder) command.
    pub fn reorder(from: usize, to: usize) -> Self {
        LayerCommand::Reorder {
            from,
            to,
            applied_to: None,
        }
    }

    /// Builds a [`SetOpacity`](LayerCommand::SetOpacity) command.
    pub fn set_opacity(id: u64, value: f32) -> Self {
        LayerCommand::SetOpacity {
            id,
            new: value,
            old: None,
        }
    }

    /// Builds a [`SetVisible`](LayerCommand::SetVisible) command.
    pub fn set_visible(id: u64, value: bool) -> Self {
        LayerCommand::SetVisible {
            id,
            new: value,
            old: None,
        }
    }

    /// Builds a [`SetBlendMode`](LayerCommand::SetBlendMode) command.
    pub fn set_blend_mode(id: u64, mode: BlendMode) -> Self {
        LayerCommand::SetBlendMode {
            id,
            new: mode,
            old: None,
        }
    }
}

impl Command for LayerCommand {
    type Doc = LayerStack;

    fn apply(&mut self, doc: &mut LayerStack) {
        match self {
            LayerCommand::Add { name, kind, id } => match id {
                None => *id = Some(doc.add_layer(name.clone(), kind.clone())),
                Some(existing) => doc.add_layer_with_id(*existing, name.clone(), kind.clone()),
            },
            LayerCommand::Remove { id, removed } => *removed = doc.remove_layer(*id),
            LayerCommand::Reorder {
                from,
                to,
                applied_to,
            } => {
                *applied_to = doc.reorder(*from, *to);
            }
            LayerCommand::SetOpacity { id, new, old } => *old = doc.set_opacity(*id, *new),
            LayerCommand::SetVisible { id, new, old } => *old = doc.set_visible(*id, *new),
            LayerCommand::SetBlendMode { id, new, old } => *old = doc.set_blend_mode(*id, *new),
        }
    }

    fn revert(&mut self, doc: &mut LayerStack) {
        match self {
            // Deliberately does not take() the id: redo re-applies Add via
            // the Some(existing) arm above, which needs it to restore the
            // same id rather than allocating a new one.
            LayerCommand::Add { id, .. } => {
                if let Some(id) = id {
                    doc.remove_layer(*id);
                }
            }
            LayerCommand::Remove { removed, .. } => {
                if let Some((index, layer)) = removed.take() {
                    doc.insert_layer(index, layer);
                }
            }
            LayerCommand::Reorder {
                from, applied_to, ..
            } => {
                if let Some(to) = applied_to.take() {
                    doc.reorder(to, *from);
                }
            }
            LayerCommand::SetOpacity { id, old, .. } => {
                if let Some(value) = old.take() {
                    doc.set_opacity(*id, value);
                }
            }
            LayerCommand::SetVisible { id, old, .. } => {
                if let Some(value) = old.take() {
                    doc.set_visible(*id, value);
                }
            }
            LayerCommand::SetBlendMode { id, old, .. } => {
                if let Some(value) = old.take() {
                    doc.set_blend_mode(*id, value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::undo::UndoStack;

    fn ids(stack: &LayerStack) -> Vec<u64> {
        stack.layers.iter().map(|l| l.id).collect()
    }

    #[test]
    fn add_and_remove_round_trip_index_and_id() {
        let mut stack = LayerStack::new();
        let a = stack.add_layer("Base", LayerKind::Paint);
        let b = stack.add_layer("Detail", LayerKind::Paint);
        assert_eq!(stack.layers.len(), 2);
        assert_eq!(stack.layers[0].id, a);
        assert_eq!(stack.layers[1].id, b);

        let (index, removed) = stack.remove_layer(a).expect("layer a exists");
        assert_eq!(index, 0);
        assert_eq!(removed.id, a);
        assert_eq!(stack.layers.len(), 1);

        stack.insert_layer(index, removed);
        assert_eq!(stack.layers[0].id, a);
    }

    #[test]
    fn reorder_moves_layer_and_clamps_target() {
        let mut stack = LayerStack::new();
        let a = stack.add_layer("A", LayerKind::Paint);
        let b = stack.add_layer("B", LayerKind::Paint);
        let c = stack.add_layer("C", LayerKind::Paint);
        assert_eq!(ids(&stack), vec![a, b, c]);

        let landed = stack.reorder(0, 2).unwrap();
        assert_eq!(landed, 2);
        assert_eq!(ids(&stack), vec![b, c, a]);

        let landed = stack.reorder(0, 999).unwrap();
        assert_eq!(landed, 2); // clamped to the end
        assert_eq!(ids(&stack), vec![c, a, b]);

        assert_eq!(stack.reorder(99, 0), None); // out-of-range source
    }

    #[test]
    fn set_opacity_clamps_and_ignores_non_finite() {
        let mut stack = LayerStack::new();
        let a = stack.add_layer("A", LayerKind::Paint);

        assert_eq!(stack.set_opacity(a, 0.5), Some(1.0));
        assert_eq!(stack.layer(a).unwrap().opacity, 0.5);

        stack.set_opacity(a, 5.0);
        assert_eq!(stack.layer(a).unwrap().opacity, 1.0);

        stack.set_opacity(a, -5.0);
        assert_eq!(stack.layer(a).unwrap().opacity, 0.0);

        stack.set_opacity(a, 0.75);
        stack.set_opacity(a, f32::NAN);
        assert_eq!(stack.layer(a).unwrap().opacity, 0.75); // NaN ignored, not stored
    }

    #[test]
    fn layer_command_interleaved_undo_redo_matches_direct_mutation() {
        let mut stack = LayerStack::new();
        let mut undo: UndoStack<LayerCommand> = UndoStack::new(100);

        let snapshot_empty = stack.layers.clone();

        undo.push(LayerCommand::add("Base", LayerKind::Paint), &mut stack);
        let id = stack.layers[0].id;
        let snapshot_after_add = stack.layers.clone();

        undo.push(LayerCommand::set_opacity(id, 0.5), &mut stack);
        let snapshot_after_opacity = stack.layers.clone();
        assert_eq!(stack.layer(id).unwrap().opacity, 0.5);

        undo.push(LayerCommand::set_visible(id, false), &mut stack);
        let snapshot_after_visible = stack.layers.clone();
        assert!(!stack.layer(id).unwrap().visible);

        // Undo back through each intermediate state.
        assert!(undo.undo(&mut stack));
        assert_eq!(stack.layers, snapshot_after_opacity);
        assert!(undo.undo(&mut stack));
        assert_eq!(stack.layers, snapshot_after_add);
        assert!(undo.undo(&mut stack));
        assert_eq!(stack.layers, snapshot_empty);
        assert!(!undo.undo(&mut stack));

        // Redo forward through each intermediate state, re-using the same id.
        assert!(undo.redo(&mut stack));
        assert_eq!(stack.layers, snapshot_after_add);
        assert_eq!(stack.layers[0].id, id);

        assert!(undo.redo(&mut stack));
        assert_eq!(stack.layers, snapshot_after_opacity);

        assert!(undo.redo(&mut stack));
        assert_eq!(stack.layers, snapshot_after_visible);
        assert!(!undo.can_redo());
    }

    #[test]
    fn remove_command_undo_restores_layer_at_original_index() {
        let mut stack = LayerStack::new();
        let a = stack.add_layer("A", LayerKind::Paint);
        let b = stack.add_layer("B", LayerKind::Paint);
        let mut undo: UndoStack<LayerCommand> = UndoStack::new(100);

        let before = stack.layers.clone();
        undo.push(LayerCommand::remove(a), &mut stack);
        assert_eq!(ids(&stack), vec![b]);

        assert!(undo.undo(&mut stack));
        assert_eq!(stack.layers, before);
    }
}

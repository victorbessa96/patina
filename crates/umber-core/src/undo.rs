//! The command-pattern undo journal.
//!
//! [`UndoStack`] is generic over any [`Command`] implementation, so the same
//! journal type works for layer-stack edits ([`crate::layers::LayerCommand`])
//! and any future document-mutating command (stroke commits, procedural-graph
//! edits, ...) without this module needing to know about them.

use std::collections::VecDeque;

/// A reversible edit to a document of type [`Command::Doc`].
///
/// `apply` and `revert` must be exact inverses: applying a command and then
/// reverting it must leave the document indistinguishable from before
/// `apply` ran.
pub trait Command {
    /// The document type this command mutates.
    type Doc;

    /// Performs the edit.
    fn apply(&mut self, doc: &mut Self::Doc);

    /// Undoes the edit performed by the most recent `apply`.
    fn revert(&mut self, doc: &mut Self::Doc);

    /// Approximate RAM held by this command's undo state (captured "before"
    /// data, snapshots, ...), in bytes.
    ///
    /// Defaults to 0 for commands that only hold small scalars. Commands
    /// wrapping future GPU-tile snapshots should override this so
    /// [`UndoStack`]'s memory budget reflects what is actually pinned in RAM.
    ///
    /// Must return the same value for the lifetime of the command: `UndoStack`
    /// reads it once on `push` and assumes it doesn't change afterwards (in
    /// particular, don't let `revert` free data that `memory_bytes` still
    /// counts — the command may be re-applied by a later `redo`).
    fn memory_bytes(&self) -> usize {
        0
    }
}

/// A bounded undo/redo journal over commands of type `C`.
///
/// Bounded by entry count (`max_entries`) and, optionally, by an approximate
/// memory budget (`max_bytes`, backed by [`Command::memory_bytes`]). When a
/// bound is exceeded, the oldest entry on the undo stack is evicted and
/// handed to the `on_evict` callback — the hook future GPU-tile snapshot
/// cleanup links into. Entries dropped from the redo stack by a new [`push`]
/// are evicted the same way, since they hold the same kind of RAM.
///
/// [`push`]: UndoStack::push
pub struct UndoStack<C: Command> {
    undo: VecDeque<C>,
    redo: Vec<C>,
    max_entries: usize,
    max_bytes: Option<usize>,
    bytes_held: usize,
    on_evict: Option<Box<dyn FnMut(C)>>,
}

impl<C: Command> UndoStack<C> {
    /// A journal bounded only by entry count.
    pub fn new(max_entries: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            max_entries,
            max_bytes: None,
            bytes_held: 0,
            on_evict: None,
        }
    }

    /// Also caps total [`Command::memory_bytes`] held across both stacks.
    #[must_use]
    pub fn with_memory_budget(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    /// Registers a callback invoked with each evicted command, oldest first.
    #[must_use]
    pub fn with_on_evict(mut self, on_evict: impl FnMut(C) + 'static) -> Self {
        self.on_evict = Some(Box::new(on_evict));
        self
    }

    /// Applies `command` to `doc`, pushes it onto the undo stack, and clears
    /// the redo stack — a fresh edit invalidates any previously-undone
    /// future.
    pub fn push(&mut self, mut command: C, doc: &mut C::Doc) {
        profiling::scope!("undo_push");
        command.apply(doc);
        self.bytes_held += command.memory_bytes();

        for stale in self.redo.drain(..) {
            Self::evict(&mut self.bytes_held, &mut self.on_evict, stale);
        }

        self.undo.push_back(command);
        self.enforce_budget();
    }

    fn evict(bytes_held: &mut usize, on_evict: &mut Option<Box<dyn FnMut(C)>>, command: C) {
        *bytes_held = bytes_held.saturating_sub(command.memory_bytes());
        if let Some(cb) = on_evict.as_mut() {
            cb(command);
        }
    }

    fn enforce_budget(&mut self) {
        while self.undo.len() > self.max_entries
            || self.max_bytes.is_some_and(|max| self.bytes_held > max)
        {
            let Some(oldest) = self.undo.pop_front() else {
                break;
            };
            Self::evict(&mut self.bytes_held, &mut self.on_evict, oldest);
        }
    }

    /// Reverts the most recently applied command, moving it to the redo
    /// stack. Returns `false` if there was nothing to undo.
    pub fn undo(&mut self, doc: &mut C::Doc) -> bool {
        let Some(mut command) = self.undo.pop_back() else {
            return false;
        };
        command.revert(doc);
        self.redo.push(command);
        true
    }

    /// Re-applies the most recently undone command. Returns `false` if there
    /// was nothing to redo.
    pub fn redo(&mut self, doc: &mut C::Doc) -> bool {
        let Some(mut command) = self.redo.pop() else {
            return false;
        };
        command.apply(doc);
        self.undo.push_back(command);
        true
    }

    /// Whether [`undo`](Self::undo) would do anything right now.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Whether [`redo`](Self::redo) would do anything right now.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Drops every journal entry, undo and redo both — the journal
    /// starts fresh, as if just constructed. Evicted entries fire
    /// [`with_on_evict`](Self::with_on_evict) and release their
    /// [`Command::memory_bytes`] against the budget, exactly as
    /// capacity eviction does, so cleanup hooks (the future
    /// GPU-tile-snapshot pool) stay coherent.
    ///
    /// The load-a-different-document path: command journals capture
    /// ids and indices from the stack they were recorded against, so
    /// replaying them across a stack swap silently corrupts the
    /// loaded document (an `Add` recorded as id 0 in the previous
    /// document reverts by deleting the new document's layer 0).
    /// Replace documents through this, never by swapping the stack
    /// underneath a live journal.
    pub fn clear(&mut self) {
        for stale in self.undo.drain(..) {
            Self::evict(&mut self.bytes_held, &mut self.on_evict, stale);
        }
        for stale in self.redo.drain(..) {
            Self::evict(&mut self.bytes_held, &mut self.on_evict, stale);
        }
        self.bytes_held = 0;
    }

    /// Number of entries currently held in the undo journal.
    pub fn len(&self) -> usize {
        self.undo.len()
    }

    /// Whether the journal holds no entries.
    pub fn is_empty(&self) -> bool {
        self.undo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    struct Counter(i64);

    #[derive(Debug, Clone)]
    struct AddCommand {
        delta: i64,
    }

    impl Command for AddCommand {
        type Doc = Counter;

        fn apply(&mut self, doc: &mut Counter) {
            doc.0 += self.delta;
        }

        fn revert(&mut self, doc: &mut Counter) {
            doc.0 -= self.delta;
        }
    }

    #[test]
    fn interleaved_undo_redo_returns_each_intermediate_state() {
        let mut doc = Counter::default();
        let mut stack: UndoStack<AddCommand> = UndoStack::new(100);

        stack.push(AddCommand { delta: 1 }, &mut doc);
        assert_eq!(doc, Counter(1));
        stack.push(AddCommand { delta: 10 }, &mut doc);
        assert_eq!(doc, Counter(11));
        stack.push(AddCommand { delta: 100 }, &mut doc);
        assert_eq!(doc, Counter(111));

        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(11));
        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(1));

        assert!(stack.redo(&mut doc));
        assert_eq!(doc, Counter(11));

        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(1));
        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(0));
        assert!(!stack.undo(&mut doc));
        assert_eq!(doc, Counter(0));

        assert!(stack.redo(&mut doc));
        assert_eq!(doc, Counter(1));
        assert!(stack.redo(&mut doc));
        assert_eq!(doc, Counter(11));
        assert!(stack.redo(&mut doc)); // the 100 from the very first undo, still pending
        assert_eq!(doc, Counter(111));
        assert!(!stack.can_redo());
    }

    #[test]
    fn push_after_undo_clears_redo_through_the_evict_callback() {
        let evicted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let evicted_clone = evicted.clone();
        let mut doc = Counter::default();
        let mut stack: UndoStack<AddCommand> =
            UndoStack::new(100).with_on_evict(move |c: AddCommand| {
                evicted_clone.borrow_mut().push(c.delta);
            });

        stack.push(AddCommand { delta: 1 }, &mut doc);
        stack.push(AddCommand { delta: 2 }, &mut doc);
        stack.undo(&mut doc);
        assert!(stack.can_redo());

        // A fresh push clears the redo stack; the discarded entry must flow
        // through on_evict, not just vanish.
        stack.push(AddCommand { delta: 3 }, &mut doc);
        assert_eq!(*evicted.borrow(), vec![2]);
        assert!(!stack.can_redo());
        assert_eq!(doc, Counter(4));
    }

    #[test]
    fn eviction_drops_oldest_entry_at_capacity() {
        let evicted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let evicted_clone = evicted.clone();
        let mut doc = Counter::default();
        let mut stack: UndoStack<AddCommand> =
            UndoStack::new(2).with_on_evict(move |c: AddCommand| {
                evicted_clone.borrow_mut().push(c.delta);
            });

        stack.push(AddCommand { delta: 1 }, &mut doc);
        stack.push(AddCommand { delta: 2 }, &mut doc);
        stack.push(AddCommand { delta: 3 }, &mut doc); // over capacity -> evicts delta 1

        assert_eq!(*evicted.borrow(), vec![1]);
        assert_eq!(doc, Counter(6));

        // Deltas 2 and 3 are still revertible; delta 1's effect is now
        // permanently baked into `doc` since its command was evicted.
        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(3));
        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(1));
        assert!(!stack.undo(&mut doc)); // delta 1 was evicted, nothing further back
        assert_eq!(doc, Counter(1));
    }

    #[test]
    fn memory_budget_evicts_oldest_when_over_budget() {
        #[derive(Debug, Clone)]
        struct HeavyCommand {
            delta: i64,
            bytes: usize,
        }

        impl Command for HeavyCommand {
            type Doc = Counter;

            fn apply(&mut self, doc: &mut Counter) {
                doc.0 += self.delta;
            }

            fn revert(&mut self, doc: &mut Counter) {
                doc.0 -= self.delta;
            }

            fn memory_bytes(&self) -> usize {
                self.bytes
            }
        }

        let mut doc = Counter::default();
        let mut stack: UndoStack<HeavyCommand> = UndoStack::new(100).with_memory_budget(150);

        stack.push(
            HeavyCommand {
                delta: 1,
                bytes: 100,
            },
            &mut doc,
        );
        stack.push(
            HeavyCommand {
                delta: 2,
                bytes: 100,
            },
            &mut doc,
        ); // 200 > 150 budget

        assert_eq!(doc, Counter(3));
        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(1)); // only delta 2 reverted; delta 1 was evicted
        assert!(!stack.undo(&mut doc));
        assert_eq!(doc, Counter(1));
    }

    #[test]
    fn clear_empties_both_stacks_and_fires_evict_for_every_entry() {
        let evicted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let evicted_clone = evicted.clone();
        let mut doc = Counter::default();
        let mut stack: UndoStack<AddCommand> =
            UndoStack::new(100).with_on_evict(move |c: AddCommand| {
                evicted_clone.borrow_mut().push(c.delta);
            });

        stack.push(AddCommand { delta: 1 }, &mut doc);
        stack.push(AddCommand { delta: 2 }, &mut doc);
        assert!(stack.undo(&mut doc)); // delta 2 now sits on the redo stack
        assert!(stack.can_redo());

        stack.clear();

        // Every held entry — undo and redo both — flowed through
        // on_evict; nothing silently vanished. Drain order is
        // chronological: undo entries oldest-first, then redo.
        assert_eq!(*evicted.borrow(), vec![1, 2]);
        assert!(stack.is_empty());
        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
        // The document itself is untouched: clear is a journal
        // operation, not an undo.
        assert_eq!(doc, Counter(1));
    }

    #[test]
    fn clear_releases_budget_and_accepts_new_entries() {
        let mut doc = Counter::default();
        let mut stack: UndoStack<AddCommand> = UndoStack::new(100).with_memory_budget(10);

        stack.push(AddCommand { delta: 5 }, &mut doc);
        stack.clear();
        // Budget accounting must have released the cleared entry:
        // a same-cost push after clear must NOT trigger eviction.
        stack.push(AddCommand { delta: 5 }, &mut doc);
        assert!(stack.can_undo());
        assert!(stack.undo(&mut doc));
        assert_eq!(doc, Counter(5));
        assert!(!stack.undo(&mut doc));
    }
}

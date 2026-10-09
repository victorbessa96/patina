# umber-core — Wave 2 document model landing notes

Scope: `layers.rs` (layer stack), `undo.rs` (command-pattern journal),
`project.rs` (`.umber` project format). `lib.rs`'s `TextureSet`/`Channel`
were extended (serde + `PartialEq` derives only — see "The one deliberate
touch to lib.rs" below) but not reshaped. Pure CPU Rust throughout: no wgpu,
no egui, no umber-gpu; `serde` + `serde_json` + `thiserror` only.

16 unit tests, `cargo fmt` clean, `cargo clippy -p umber-core --all-targets
-- -D warnings` clean, `cargo check --workspace` clean.

## What's built

### `layers.rs`

- `LayerKind`: `Paint`, `Fill`, `Folder { passthrough: bool }`.
- `BlendMode`: the core 12 (Normal, Passthrough, Multiply, Screen, Overlay,
  Darken, Lighten, Add, Subtract, Difference, SoftLight, HardLight). See
  "Blend-mode rationale" below for the `Passthrough` vs. `Folder.passthrough`
  relationship — it's deliberately left as an open question for the Wave 2
  compositor, not resolved here.
- `LayerMask { name, enabled }` — metadata only; pixel data is Wave 2
  painting-core/tile-pool territory.
- `Layer { id: u64, name, kind, opacity: f32, visible, blend_mode, mask }`.
- `LayerStack { pub layers: Vec<Layer>, next_id: u64 }` — flat, ordered,
  index 0 = bottom of stack. Mutators (`add_layer`, `remove_layer`,
  `insert_layer`, `reorder`, `set_opacity`, `set_visible`, `set_blend_mode`)
  all take `&mut self` and return the *previous* value (an "op handle") so
  callers can build undo without a second query. `set_opacity` clamps to
  `0.0..=1.0` and **ignores non-finite input** (NaN/±inf) rather than
  storing it — otherwise a NaN opacity serializes as JSON `null` and fails
  to deserialize on load. `next_id` is monotonic and private; the only way
  to get a specific id onto the stack from outside the module is through
  `LayerStack::from_parts` (project load) or by having previously received
  it from `add_layer`.
- `LayerCommand`: one enum (`Add`, `Remove`, `Reorder`, `SetOpacity`,
  `SetVisible`, `SetBlendMode`) implementing `undo::Command` with
  `Doc = LayerStack`. It's one enum rather than six command structs because
  `UndoStack<C>` is monomorphic over a single `C` — a real undo journal over
  a layer stack needs one type that covers every op.

  The one subtlety worth flagging for review: **`LayerCommand::Add` must
  restore the same id on redo.** `apply` allocates a fresh id only the
  *first* time (`id: None`); once `id` is `Some(n)`, `revert` removes layer
  `n` but does not clear the field, and a later `apply` call (redo)
  re-inserts under that same id via a private `LayerStack::add_layer_with_id`
  rather than calling `add_layer` again. Without this, redoing an `Add`
  would allocate id `n+1`, and any later command in the same journal that
  captured id `n` (e.g. a `SetOpacity` on the newly-added layer) would
  silently no-op after an undo/redo cycle. `layer_command_interleaved_undo_redo_matches_direct_mutation`
  exercises exactly this: add → set opacity → set visible, undo three times,
  redo three times, asserting the id is unchanged and the stack matches a
  byte-for-byte snapshot taken at each intermediate step.

### `undo.rs`

- `Command` trait: associated `type Doc`, `fn apply(&mut self, doc: &mut
  Doc)`, `fn revert(&mut self, doc: &mut Doc)`, and a `fn memory_bytes(&self)
  -> usize { 0 }` default — commands that wrap future GPU-tile snapshots
  override this so the budget below means something.
- `UndoStack<C: Command>`: `undo: VecDeque<C>` (the real history) + `redo:
  Vec<C>`. `push` applies, clears `redo` (evicting each cleared entry
  through `on_evict` — they hold the same kind of RAM a forgotten undo entry
  does, so the eviction hook is the right place for them too), then enforces
  the budget. `undo`/`redo` move a command between the two stacks without
  touching the budget (both stacks count as "held"). Two independent caps:
  `max_entries` (required, via `new`) and an optional `max_bytes` (via
  `.with_memory_budget()`, backed by `Command::memory_bytes`) — either one
  tripping evicts the oldest *undo* entry and calls `on_evict`
  (`.with_on_evict()`).
- **Eviction does not undo the evicted command's effect on the document** —
  it only forgets the *ability* to revert it. This matches how real
  "100 steps of undo" budgets work (requirements.md §12): old history
  becomes permanently baked into the document once it falls off the back of
  the journal. `eviction_drops_oldest_entry_at_capacity` checks this
  explicitly (push 1, 2, 3 at capacity 2 evicts delta-1's command, but the
  document still reflects `1+2+3`; undoing twice reverts only 3 then 2,
  landing on 1, not 0).
- Not `Debug`-derived: `Box<dyn FnMut(C)>` isn't `Debug`, and the type
  carries no other state worth inspecting that way.

### `project.rs`

- `ProjectModel { version, texture_sets: Vec<TextureSet>, layers:
  Vec<TextureSetLayers>, settings: ProjectSettings }` — the real in-memory
  document, full `Layer` bodies included.
- `TextureSetLayers { texture_set: String, stack: LayerStack }` — one per
  texture set; `texture_set` must match a `TextureSet.name`.
- `ProjectSettings { active_texture_set: Option<String> }` — deliberately
  minimal; it's a real, usable field (which set the UI was last looking at),
  not a placeholder. Extend when Wave 2's OCIO/display-transform settings
  need a home.
- `save_to_dir` / `load_from_dir`: the on-disk schema (`ProjectFile` +
  `LayerSetOrder`, both private to the module) is **not** the same shape as
  `ProjectModel`. `project.json` holds `version`, `texture_sets`, settings,
  and per-set *ordering* (`layer_ids: Vec<u64>` plus a persisted
  `next_layer_id`) — never layer bodies. Each layer body is its own
  `layers/<set>/<id>.json`. That split is what makes the format diffable
  (two artists editing different layers touch different files) and is why
  `ProjectModel` and `ProjectFile` are deliberately two different Rust
  types rather than one struct wearing two hats.
- `next_layer_id` is persisted per set rather than re-derived as
  `max(layer_ids) + 1` on load: if the highest-id layer was deleted before
  the save, re-deriving would let a future `add_layer` reissue its old id,
  breaking the "ids are stable identity, not just array position" guarantee
  undo and any future layer-instancing/anchors feature (requirements.md §2)
  will depend on.
- Save fully replaces each `layers/<set>/` directory (`remove_dir_all` then
  rewrite) rather than diffing — so a save into an existing project never
  leaves a stale `<id>.json` behind for a layer deleted since the last save.
  `save_overwrite_prunes_deleted_layer_files` checks this directly.
- Validation on save: texture-set names are rejected if empty, `.`/`..`, or
  containing `< > : " | ? * / \` (they become directory names — this list
  is the union of what's unsafe on Windows and POSIX, and Windows CI is
  live for this repo); duplicate set names are rejected too.
- Validation on load: the version is probed from a tiny `{version}`-only
  struct *before* the full schema is parsed, so a future-version project
  fails with `ProjectError::UnsupportedVersion` instead of an opaque JSON
  error; every `layer_sets` entry must name a known texture set; layer ids
  within a set must be unique; each loaded layer file's internal `id` must
  match the order entry that pointed at it (catches a renamed/corrupted
  file immediately rather than silently reassigning identity).
- JSON is written pretty-printed with a trailing newline (git-diff-friendly,
  matches the "diffable, mergeable" requirement verbatim).

## Round-trip determinism test design (the SPEC §8 acceptance criterion)

Two tests, intentionally not merged into one:

1. `project_file_json_bytes_are_deterministic_across_serializations` —
   serializes the same `ProjectFile` value twice and compares bytes. This is
   the literal "serialize twice, compare" the task text calls out, but on
   its own it's a close-to-trivial check (`serde_json` is deterministic for
   a fixed value by construction; the only way to break it is a `HashMap` in
   the serialized shape, which this format avoids throughout — everything is
   `Vec`/struct fields in declaration order).

2. `save_load_save_round_trip_is_byte_identical` — the test that actually
   matters, covering requirements.md §8's real wording ("save→load→export
   identical bytes"): save a fixture to directory A, load it back, assert
   the loaded `ProjectModel` equals the original (`PartialEq`), save the
   loaded model to directory B, then walk both directories and assert the
   file listing and every file's bytes match exactly. The fixture
   deliberately exercises `Folder { passthrough: true }`, a populated
   `LayerMask`, opacities that stress float formatting (`0.1`, `1.0/3.0`),
   two texture sets, several blend modes, a layer that was added then
   removed (to prove `next_layer_id` persistence survives a gap), and a
   non-default `ProjectSettings`. Temp directories are namespaced by PID +
   an atomic counter so parallel test runs never collide.

## Blend-mode set rationale

Requirements.md §2 calls for "P0 core 12" now and a full ~32-mode set later
(Wave 4). The 12 implemented are exactly the task's list — the standard
separable blend-mode family (Normal/Multiply/Screen/Overlay/Darken/
Lighten/Difference/SoftLight/HardLight) plus the three a layer-stack
engine needs structurally from day one (Passthrough for folders, Add and
Subtract as the simplest linear-light modes, already common in brush
eraser/additive workflows). The full set (HSV modes, normal-map
combine/detail) is Wave 4 scope per the requirements table and intentionally
not started here — adding unused enum variants now would just be dead code
with no compositor to exercise it.

## Known open questions / explicitly not built (read before extending)

- **`BlendMode::Passthrough` vs. `LayerKind::Folder { passthrough }`:** the
  task asked for both, and as implemented they can say the same thing twice
  for a folder layer (`kind: Folder { passthrough: true }` and
  `blend_mode: Passthrough` independently settable, nothing enforces they
  agree). I did not invent a cross-validation rule because there's no
  compositor yet to tell me which one should win if they disagree. Folder's
  own flag is documented as authoritative; `BlendMode::Passthrough` exists
  so the blend-mode enum matches the full named list from requirements.md
  §2. Whoever builds the compositor should either (a) make `blend_mode` the
  single source of truth and drop the redundant flag, or (b) keep the flag
  authoritative and stop treating `Passthrough` as a selectable value for
  non-folder layers (currently nothing prevents setting it on a `Paint`
  layer, where it's meaningless).
- **No folder membership model.** `LayerStack` is a flat `Vec<Layer>` — a
  `Folder` layer is just an entry with no record of which other entries are
  "inside" it. Real nesting (and the indentation/traversal that implies for
  compositing and for the eventual UI tree view) is not modeled. I did not
  guess at a parent-pointer or nested-`Vec` scheme because the task
  description's own wording ("folders with passthrough flag") only asked
  for the flag, not hierarchy, and guessing wrong here would be expensive to
  unwind later.
- **Mask pixel data** is categorically absent (`LayerMask` is name+enabled
  only) — correct per the task, since pixel/tile data belongs to the
  Wave 2 tile-pool work, not this crate's document model.
- **`ProjectSettings` has exactly one field.** It's real (active-set UI
  state), not a stub, but it is not where OCIO config or other Wave 2/3
  settings should necessarily land — that's a call for whoever builds those
  features, not pre-empted here.

## The one deliberate touch to lib.rs

The task said "`TextureSet`/`Channel` stay untouched," but `ProjectModel`
also needed `texture_sets: Vec<TextureSet>` to actually serialize — the only
way to satisfy both is to add `Serialize, Deserialize, PartialEq` derives to
`Channel`, `ChannelKind`, and `TextureSet` in `lib.rs`. No field, variant, or
method on any of the three changed; this is strictly additive. `PartialEq`
specifically is required for `save_load_save_round_trip_is_byte_identical`'s
structural-equality assertion (`assert_eq!(loaded, model)`), not just for
serde. Flag this explicitly in review since it's the one line of this task
that touches code outside `layers.rs`/`undo.rs`/`project.rs`.

## Reviewer checklist

- [ ] `LayerCommand::Add`'s id-preservation-across-redo logic
      (`layers.rs`, `apply`/`revert` match arms) — this is the subtlest
      correctness point in the whole landing; walk through add → mutate →
      undo × N → redo × N by hand if the test isn't convincing enough.
- [ ] `set_opacity`'s NaN handling — confirm silently ignoring non-finite
      input (vs. an error or a saturating clamp) is the right call for the
      brush-engine callers that will drive this in Wave 2's painting core.
- [ ] `save_to_dir`'s full-directory-replace strategy for `layers/<set>/` —
      confirm this is acceptable given autosave (requirements.md §8, P1,
      Wave 5) will call this far more often than a manual save; a future
      incremental-write optimization may be worth it before autosave lands,
      but isn't needed yet.
- [ ] The `BlendMode::Passthrough` / `Folder.passthrough` redundancy above —
      needs a decision before the compositor is built, not before this
      lands.
- [ ] `ProjectError` variants — confirm the granularity (nine variants, two
      of them `#[from]`) is what callers in umber-app's save/load UI will
      actually want to match on, versus wanting fewer, broader categories.
- [ ] Derives added to `lib.rs`'s `TextureSet`/`Channel`/`ChannelKind` — the
      one touch outside this task's three new files; confirm it doesn't
      collide with whatever the parallel `umber-brush` agent is doing this
      wave.

# Export dialog (Wave 3) — landing notes

Headless-driven Export dialog for the egui shell: `crates/umber-app/src/export_dialog.rs`
(new) + `crates/umber-app/src/bake_sources.rs` (new, shared with the Bakes panel) +
dock wiring in `crates/umber-app/src/main.rs`. `umber-cli`, `umber-export`, and
`umber-bake` internals were read-only sources — no lines touched there.

## What was built

**`crates/umber-app/src/bake_sources.rs`** (new)
- `bake_ao(device, queue, mesh, size, rays) -> Result<Vec<u8>>` — the AO arm lifted
  verbatim out of `BakesPanel::run_all` (same `10.0`/`0.01`/dummy-plane params, same
  `bake_ao_mesh` call). Returns the **raw, undilated** bake.
- `flat_normal_rgba8(size) -> Vec<u8>` — `[128, 128, 255, 255]` at every texel, the
  same placeholder `umber-cli export` writes.
- `map_set_from(ao, size) -> MapSet` — pure composition (no GPU), directly testable.
- `bake_export_map_set(device, queue, mesh, size, rays) -> Result<MapSet>` — the
  GPU-touching entry point: bakes AO, composes it with the flat normal.
- `bakes_panel.rs`'s own Ao arm now calls `bake_sources::bake_ao` and dilates the
  result itself — **the one edit this slice makes outside the two new files**,
  required by the brief ("ONE shared helper both panels use"). Behavior is
  unchanged: same calls, same order, just no longer duplicated.

**`crates/umber-app/src/export_dialog.rs`** (new)
- `PresetChoice` — the four `umber_export::ExportPreset` built-ins
  (`GltfMetalRough`/`UnrealOrm`/`UnityHdrpUrp`/`BlenderPrincipled`), each with
  `.build()` and `.label()` (pinned equal to the built preset's own `name` by a test).
- `filter_satisfiable(preset, available) -> (ExportPreset, Vec<SkippedOutput>)` — pure
  logic: splits a preset's outputs into what `available` map kinds can satisfy and
  what can't, naming the specific missing tokens (`MapKind::token()`) per skipped
  output instead of the CLI's generic "needs maps not baked headless". Mirrors the
  CLI's never-error-on-partial-set behavior.
- `ExportDialog` — preset choice + output dir + last-export status. `show(&mut self,
  ui, ExportContext)` draws: preset combo, output-dir row with an rfd folder picker,
  an Export button, and a status line (`Exported N outputs (bake T ms, write T ms):
  <files> — skipped: <file> (missing tok1, tok2); …`).
- `ExportContext<'a> { gpu: Option<&GpuContext>, mesh: Option<&MeshData>, mesh_path:
  Option<&Path> }` — identical shape to `BakesContext`; same gating story ("No mesh
  loaded" / "No GPU device" disables Export via `can_export`).
- Export driver (`run`): `bake_sources::bake_export_map_set` (timed) → `MapSet` with
  `{AmbientOcclusion, Normal}` → `filter_satisfiable` against the chosen preset →
  `umber_export::run_preset` (timed) → `ExportOutcome`. Texture-set name via
  `umber_mesh::texture_set_name`, same fallback (`bakes_panel::FALLBACK_TEXTURE_SET`)
  the Bakes panel uses. Fixed resolution/rays (`bakes_panel::DEFAULT_RESOLUTION`/
  `DEFAULT_RAYS`) — no resolution control in this dialog; the brief's UI list names
  four controls only (preset, output dir, status, button), and the shared defaults
  already match what the Bakes panel would produce at the same settings.
- No `unwrap` outside tests; failures become `status = "Export failed: …"` +
  `log::error`, never a crash.

**Wiring** (`main.rs`)
- `Panel::Export` ("Export" tab) docked in the left column beside Layers / Texture
  Sets / Bakes; `PanelViewer` carries `export: &mut ExportDialog`; `AppState::export`
  (manual construction via `ExportDialog::default()` → `./export`, `AppState` keeps
  its derive).
- No `Cargo.toml` changes — `umber-export`/`umber-bake`/`rfd` were already
  `umber-app` dependencies (the Bakes panel and `export_paint_png` use them).

## The baked-maps-today vs. painted-maps-later story (read before filing "export is useless")

**With the two maps this dialog can build today — AO and a flat normal — every
built-in preset collapses to exactly one surviving output: the normal passthrough.**
This is not a bug in `filter_satisfiable`; it is the honest consequence of what the
app can bake headless right now:

- glTF / Unity / Blender all want `BaseColor` (+ Roughness/Metallic for
  glTF/Unity) — none of those are baked or painted yet, so their baseColor/
  metallic-rough/smoothness outputs are skipped.
- Unreal ORM is the one preset that *does* want AO — but its ORM output also needs
  Roughness and Metallic, so AO alone isn't enough either; it's skipped too
  (`missing: ["roughness", "metallic"]`, confirmed by test — AO itself is never
  reported missing).
- The normal output in every preset wires only `MapKind::Normal`, which the flat
  placeholder always satisfies — so it's the one file every preset writes. For the
  three OpenGL-convention presets it comes out as the flat `[128,128,255]` texel
  unchanged; Unreal ORM's DirectX convention flips the green channel at export time
  (`run_preset`'s existing behavior), so Unreal's normal file is `[128,127,255]`.

This click is a genuine, if narrow, win today (a plugged-in Unreal/Unity/Blender
pipeline gets *a* normal map file at the right name, right now), and it's a
deliberately honest preview of the real payoff: **once a source exists for
BaseColor/Roughness/Metallic — painted layers composited to a texture (the paint
engine's eventual export hook) or additional bakers — `filter_satisfiable` needs no
changes.** It already keeps whatever the `MapSet` carries; feed it more map kinds and
more outputs survive automatically. The dialog's only job on that day is adding more
`map_set.set(...)` calls in `bake_sources` (or a painted-map readback) — the preset
filtering, the driver call, and the status-line reporting are already general.

### Why AO is undilated here (unlike the Bakes panel)

The Bakes panel dilates every map before writing (seam-filled PNGs for inspection/
reimport). The Export dialog's `bake_sources::bake_ao` returns the **raw** bake,
matching `umber-cli export`'s own behavior — both intentionally skip dilation for
the export-sourced AO. Today this is moot (no built-in preset's surviving output
touches AO, per above), but if a future preset or map set starts consuming AO
through this path, revisit whether export-quality AO should dilate like the panel's
copy does — flag it then, don't assume the current choice is permanent.

## Wiring map (where to look if something's wrong)

| Symptom | Look here |
|---|---|
| Export button always disabled | `ExportDialog::can_export` / `ExportContext` construction in `main.rs`'s `Panel::Export` arm |
| Wrong/no preset selected | `PresetChoice::label`/`build` — test `preset_label_matches_built_preset_name` pins the pairing |
| An output is skipped that shouldn't be | `filter_satisfiable` — check `available` (from `map_set.maps_iter()`) against the output's `maps` list |
| Status line times look wrong | `ExportDialog::run`'s two `Instant` windows (bake vs. write) — see "phase timings", not per-file |
| AO/normal bytes look wrong | `bake_sources::bake_ao` / `flat_normal_rgba8` — shared with `bakes_panel.rs`'s Ao arm |
| `run_preset` errors (`MissingMap`/`SizeMismatch`) despite filtering | `filtered_preset_satisfies_the_driver_validation` test should catch this; if it's green but prod still errors, the `MapSet`'s `size` doesn't match `bakes_panel::DEFAULT_RESOLUTION` |

## Tests

8 pure-logic unit tests in `export_dialog.rs` (no GPU, no `egui::Ui` construction):
defaults pin, preset/output-dir setter round-trip, `can_export` gating (all four
mesh/gpu combinations), preset label ↔ built-preset-name pairing for all four
presets, full-satisfiability (every map available → nothing skipped), the headline
AO+normal-only-keeps-normal fact across all four presets (plus "AO is never reported
missing"), Unreal ORM's specific missing-token list, and an integration-style test
that runs the *filtered* preset through the real `umber_export::run_preset` into a
pid-suffixed temp dir to prove the filter's output never trips the driver's own
`MissingMap`/`SizeMismatch` validation. 3 more in `bake_sources.rs` (flat-normal
shape, zero-size edge case, `MapSet` composition). `cargo test -p umber-app`: 14 → 25.

## Reviewer checklist

- [ ] **Every built-in preset reduces to its normal output today** (see story above)
  — do not file "export only writes one file" as a bug; do file it if the *skip
  reasons* in the status line are wrong (missing tokens should never include
  `ambient_occlusion` when AO is in the `MapSet`).
- [ ] **`bakes_panel.rs`'s Ao arm was edited** (now calls `bake_sources::bake_ao` then
  dilates) — this is the one intentional change outside the two new files; confirm
  `git diff` shows no behavior change, only the extraction.
- [ ] **Export-sourced AO is undilated, Bakes-panel AO is dilated** — same source
  function, different post-processing, both deliberate (see "Why AO is undilated").
- [ ] **No resolution/rays control in this dialog** — fixed to
  `bakes_panel::DEFAULT_RESOLUTION`/`DEFAULT_RAYS` by design (brief's UI list has four
  controls, not six); flag if export should expose its own resolution independent of
  the Bakes panel's.
- [ ] **Synchronous Export blocks the frame** — same tradeoff as the Bakes panel
  (`LANDING_NOTES_BAKES_PANEL.md`), same justification (sub-second at today's fixed
  512², no async job system yet, no progress bar exists to stay responsive *for*).
- [ ] **`umber-bake`/`umber-export`/`umber-cli` untouched** — `git status` should show
  only `umber-app` files (plus this note and `LANDING_NOTES_BAKES_PANEL.md`
  unchanged).
- [ ] **Green**: `cargo fmt --check`, `cargo clippy -p umber-app --all-targets -- -D
  warnings`, `cargo test -p umber-app` (25), `cargo test -p umber-export` unchanged
  (39 — this slice adds no export-crate tests and touches no export-crate code).

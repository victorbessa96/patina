# Bakes panel (Wave 3) — landing notes

Headless-driven Bakes panel for the egui shell: `crates/umber-app/src/bakes_panel.rs`
(new) + dock wiring in `crates/umber-app/src/main.rs`. `umber-cli`, `umber-export`,
and `umber-bake` internals were read-only sources — no lines touched there.

## What was built

**`crates/umber-app/src/bakes_panel.rs`** (new)
- `BakesPanel` — settings + last-bake status. `show(&mut self, ui, BakesContext)`
  draws: resolution combo (128/256/512/1024/2048), per-map checkboxes
  (AO/curvature/thickness/position), rays slider (4..=64, default 16),
  dilation slider (0..=64, default 16), output-dir row with an rfd folder
  picker, a Bake button, and a status line (`Baked N maps in T ms:
  <file> (t ms), …`).
- `BakesContext<'a> { gpu: Option<&GpuContext>, mesh: Option<&MeshData>,
  mesh_path: Option<&Path> }` — `Option` so the panel gates itself: no mesh
  → "No mesh loaded", no GPU → "No GPU device", either (or zero maps
  checked) disables Bake via `can_bake`. The panel never names a `wgpu`
  type: GPU handles travel as `&GpuContext` / `umber_gpu::WgpuDevice` /
  `WgpuQueue` aliases, per the `viewport`/`paint_state` rule.
- Bake driver (`run_all`): per checked map, the mesh-fed `umber-bake` entry
  point (`ao::bake_ao_mesh`, `curvature::bake_curvature_mesh`,
  `thickness::bake_thickness_mesh`, `position::bake_position_map`), then
  `dilation::dilate_map` with the slider's iterations, then
  `umber_export::png::write_png(Transfer::Srgb)` to
  `umber_mesh::format_mesh_map(<dir>, <set>, <kind>, "png")`. Texture-set
  name via `umber_mesh::texture_set_name` (falls back to `"TextureSet"`).
  Same composition as `umber-cli bake-all`, so panel and CLI outputs match.
- Two judgment calls, both documented in code: position f32 → RGBA8 uses the
  CLI's own bounds-normalization (`encode_position_rgba8`, ported verbatim
  in behavior); PNG transfer is `Srgb` for every map to match `bake-all`
  (data-map linearity is an open question for the export pass, not this panel).
- No `unwrap` outside tests; failures become `status = "Bake failed: …"` +
  `log::error`, never a crash. `thiserror` not needed — fallible paths use
  the app's existing `anyhow` with `.context(...)`.

**Wiring** (`main.rs` + `Cargo.toml`)
- `Panel::Bakes` ("Bakes" tab) docked in the left column beside Layers /
  Texture Sets; `PanelViewer` carries `bakes: &mut BakesPanel` + `mesh_path`;
  `AppState::bakes` (manual `Default` → `./bakes`, so `AppState` keeps its derive).
- `umber-app/Cargo.toml`: `umber-bake = { workspace = true }` (default
  features — the bake entry points need no `gpu` feature; `pollster` stays
  behind it).

## The synchronous-bake tradeoff (read before asking for threads)

The Bake click runs all selected bakes inline and blocks the frame. This is
deliberate and honest, not a missing feature:

- A 512² headless bake is sub-second per map; even 4 maps at 512² is a few
  seconds of one user-triggered click — the same blocking-readback tradeoff
  `export_paint_png` already makes at 512².
- There is **no async runtime in the app yet** (single-threaded shell per
  `main.rs` docs). A hand-rolled thread + channel just for bakes would
  duplicate what Wave 5 must build properly anyway (below), while adding
  lifetime hazards around `GpuContext`/egui repaints for zero visual gain
  (no progress bar exists to stay responsive *for*).
- Cost ceiling is bounded by UI: resolutions top out at 2048 and rays at 64,
  so the worst click is large but finite and user-initiated.

## Wave-5 async job plan (not built here)

When the job system lands, this panel is its first client:
1. `run_all` already returns owned `Vec<BakeRecord>` and takes only shared
   refs — it moves onto a worker thread almost as-is (device/queue are
   `Clone`; mesh + settings cross by value).
2. Per-map `Instant` timings become progress events (`started {map}`,
   `finished {record}`) over the job channel; the status line renders the
   in-flight job instead of only the last one.
3. Cancellation = drop the in-flight submission between maps (bakes are
   per-map sequential, so the abort points already exist).
4. Auto-re-bake-on-parameter-change (requirements §3 P1) reuses the same
   path with a debounce + dirty flag; pipeline caching (fresh pipeline per
   call today, per every baker's docs) should be revisited at the same time.

## Tests

9 pure-logic unit tests in `bakes_panel.rs` (no GPU, no `egui::Ui`
construction): defaults pin, resolution accept/reject (incl. keeping the old
value on reject), selection toggling → `any_selected`, `can_bake` gating on
all four (mesh, gpu, selection) combinations, slider clamping, output paths
equal `format_mesh_map` **and** round-trip through `parse_mesh_map`
(set + kind recovered), label/suffix coverage, position-encode normalization
(bounds, degenerate-span mid-gray, uncovered zeros), all-uncovered/empty
encode. `cargo test -p umber-app`: 5 → 14.

## Reviewer checklist

- [ ] **Blocking Bake is intentional** (see tradeoff above) — do not file
  "UI freezes on Bake" as a bug before Wave 5; do file it if a bake *fails
  silently* (status line must always say what happened).
- [ ] **`umber-bake`/`umber-export`/`umber-cli` untouched** — `git status`
  should show only `umber-app` files (plus this note). The position-encode
  port duplicates CLI logic deliberately (CLI owns its copy); if the
  normalization changes, both must change — flag if you'd rather share it.
- [ ] **`Transfer::Srgb` on data maps** matches `bake-all` today but is
  arguably wrong for linear data (AO/curvature/thickness/position) — confirm
  this stays consistent with whatever the export pass decides.
- [ ] **Dilation applies to all four maps including position-viz** (nearest-
  donor smear invents positions in the margin — fine for a viz PNG, but say
  so if position should ship undilated).
- [ ] **AO `max_distance = 10 / bias = 0.01` + dummy plane** mirror the CLI
  (plane unused by the mesh-fed path) — if bake defaults ever centralize,
  this panel must read them instead of hard-coding.
- [ ] **2048² × 64-ray worst case is slow by design** — the sliders bound it,
  the status line times it; no hidden background work to leak.
- [ ] **Green**: `cargo fmt --check`, `cargo clippy -p umber-app --all-targets
  -- -D warnings`, `cargo test -p umber-app` (14), `cargo test -p umber-bake`
  unchanged (this slice adds no bake tests and touches no bake code).

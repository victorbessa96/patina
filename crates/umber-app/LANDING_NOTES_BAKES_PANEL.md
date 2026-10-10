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

## Async bake jobs (wave 5 — built)

The Bake click no longer blocks the frame (the wave-3 synchronous tradeoff
is retired). Shape:

- **What moves to the thread:** `bake_now` snapshots an owned `BakeRequest`
  on the main thread — `device.clone()` + `queue.clone()`, `mesh.clone()`
  (`MeshData: Clone`), resolution/rays/dilation, and every planned job with
  its output path already resolved (texture-set name + tiled/untiled
  naming). The old `run_all(&self, …)` became the free `run_bake(request)`
  with the same body; it runs on a named `umber-bake` `std::thread` and
  sends one `anyhow::Result<Vec<BakeRecord>>` back over `std::sync::mpsc`.
  No new deps.
- **wgpu threading shape:** in wgpu 30 on native, `Device`/`Queue` are
  Arc-backed `Clone + Send + Sync`, and `Queue::submit`/`Device::poll` are
  internally synchronized — the worker's bake submissions and blocking
  readback waits interleave safely with eframe's frame submissions on the
  same queue. Nothing in the bake needs main-thread-only state, so the
  whole bake (GPU submission included) is off-thread. The `Send + 'static`
  bound on `BakeJobHandle::spawn` pins this at compile time.
- **Precedent note:** `umber_gpu::PaintThread` is *not* an OS thread despite
  its name — it is a same-thread mpsc command queue drained by
  `process_pending` once per frame. This slice adds the codebase's first
  real worker thread; it borrows PaintThread's owned-clone handle shape
  (`PaintThread::new(device.clone(), queue.clone(), …)`), not a thread.
- **Polling:** `BakesPanel::poll_job` drains the channel (`try_recv`); it is
  called from `UmberApp::ui` every frame (egui_dock skips hidden tabs, so
  the result lands even if the Bakes tab is closed mid-bake) and from
  `show`. While a job runs, the app requests a repaint every 100 ms so
  completion shows up without mouse input.
- **Guards:** one bake at a time — `can_bake` is false while a job is in
  flight (the button disables, a spinner + `Baking N maps…` shows), and
  `bake_now`/`start_job` are no-ops if a job exists.
- **Worker death:** a worker that panics (e.g. wgpu's uncaptured-error
  handler fires on the submitting thread) drops its sender; `poll_job`
  treats `Disconnected` as `Bake failed: bake worker exited without a
  result…` and re-enables the button — never "Baking…" forever.

V1 limits (follow-ups):
1. No progress events — the job reports once, at the end. Per-map
   `started`/`finished` events over the same channel are the next step.
2. No cancellation (abort points exist between maps; not wired).
3. Quitting mid-bake abandons the detached worker; a PNG being written at
   exit can be left truncated.
4. **Async export is not in this slice.** The Export dialog still bakes AO
   synchronously through `bake_sources` inside its run; giving it the same
   `std::thread` + `mpsc` treatment is the follow-up.
5. Auto-re-bake-on-parameter-change (requirements §3 P1) and pipeline
   caching remain open.

## Tests

9 pure-logic unit tests in `bakes_panel.rs` (no GPU, no `egui::Ui`
construction): defaults pin, resolution accept/reject (incl. keeping the old
value on reject), selection toggling → `any_selected`, `can_bake` gating on
all four (mesh, gpu, selection) combinations, slider clamping, output paths
equal `format_mesh_map` **and** round-trip through `parse_mesh_map`
(set + kind recovered), label/suffix coverage, position-encode normalization
(bounds, degenerate-span mid-gray, uncovered zeros), all-uncovered/empty
encode. `cargo test -p umber-app`: 5 → 14.

Async jobs (wave 5) add headless job-plumbing tests driven through
`start_job` with gated closures (no GPU): the result lands only through
`poll_job` after the worker finishes (`Baking 1 map…` → `Baked …` with the
skipped note); an in-flight job disables `can_bake` and refuses a second
start; an `Err` reaches the status line with its context chain and keeps
the previous records; a panicking worker reports failure instead of
baking forever. Two GPU-gated (adapter-skip) tests drive the real
`bake_now`: a 128² AO bake of a quad lands on disk via the worker, and an
empty mesh's `ao bake` error crosses the channel to the status line.

## Reviewer checklist

- [ ] **Bake runs on a worker** (see async jobs above) — "UI freezes on
  Bake" is now a bug; so is a bake that *fails silently* or leaves the
  panel stuck on "Baking…" (status line must always say what happened).
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

# umber-gpu render pass — landing notes

Wave-1 GPU claw: orbit camera, WGSL mesh shader, device context, and the
egui paint-callback that draws a normal-shaded mesh into the viewport.
Delete `docs/claw-artifacts/umber-gpu/` (per its own README) once this is
cross-reviewed.

## What was built

**`crates/umber-gpu/src/`**
- `camera.rs` — `OrbitCamera`: yaw/pitch/distance around a `target: Vec3`,
  `view_matrix`/`projection_matrix`/`view_proj`, `orbit`/`pan`/`zoom`, and
  `framing(min, max, yaw, pitch)` to fit a mesh's AABB. Pure CPU math, no
  wgpu dependency, fully unit-tested.
- `shaders.rs` — `MESH_SHADER`: WGSL vertex (`view_proj * position`) +
  fragment (ambient + directional-diffuse from the normal) pass.
- `renderer.rs` — `GpuContext` (pipeline + bind group layout, built once
  from eframe-provided adapter/device/queue), `MeshBuffers` (per-mesh
  vertex/index/uniform buffers + bind group), `MeshPaintCallback`
  (implements `egui_wgpu::CallbackTrait`), `mesh_paint_shape` (wraps a
  callback into an `epaint::Shape` ready for `ui.painter().add`),
  `CameraUniform`, `Vertex`, `GpuError`, and `compute_vertex_normals`
  (smooth-normal fallback for meshes without usable normals).
- `lib.rs` — module wiring, re-exports, `#![warn(missing_docs)]` (fixed
  every resulting warning), and `Backend::from_wgpu` now maps the real
  `wgpu::Backend` enum instead of the old placeholder `u8`.

**`crates/umber-app/src/`**
- `viewport.rs` — `Viewport`: owns the camera + `Option<MeshBuffers>`,
  `load_mesh(gpu, mesh)` (upload + re-frame), `ui(ui, gpu)` (input handling
  + paint). Never names a `wgpu`/`egui_wgpu`/`epaint` type — everything it
  holds or passes is opaque, received from or handed back to `umber_gpu`.
- `main.rs` — `UmberApp::new(cc)` captures `cc.wgpu_render_state` and builds
  the one `GpuContext`; `PanelViewer<'a>` is now built fresh each frame
  (borrowing `&mut Viewport` + `&GpuContext`) since `egui_dock::TabViewer`
  has no other way to reach app state; `NativeOptions.renderer` is forced
  to `eframe::Renderer::Wgpu`.

**Cargo.toml**: added to workspace deps — `egui-wgpu = "0.36"`,
`epaint = "0.36"`, `bytemuck = { version = "1", features = ["derive"] }`,
`pollster = "1"`. `umber-gpu` depends on `umber-mesh`, `glam`, `wgpu`,
`egui-wgpu`, `epaint`, `bytemuck`, and `pollster` (optional, behind a new
`gpu` feature gating the GPU unit tests). **`umber-app`'s `Cargo.toml` is
unchanged** — see "egui-wgpu reachability" below.

## Live-verified vs needs-GPU-check

This sandbox unexpectedly has a working wgpu adapter: `cargo test -p
umber-gpu --features gpu` actually *runs* the GPU tests (not the
skip-on-no-adapter path) in ~8s, consistent with a software rasterizer
(llvmpipe/lavapipe). So more was empirically verified than a typical
headless CI box would allow:

**Live-verified** (ran in this sandbox):
- `GpuContext::new` builds the pipeline (shader module, bind group layout,
  pipeline layout, render pipeline) without validation errors, with
  `color_format: Rgba8Unorm`, `depth_format: None`.
- `MeshBuffers::upload` roundtrip: vertex/index/uniform buffer creation,
  bind group creation, `paint_callback` construction (resource cloning).
- `compute_vertex_normals` correctness (flat triangle → unit normals;
  degenerate triangle → finite fallback normals).
- `OrbitCamera` math: framing centers on bounds and preserves orientation,
  degenerate (zero-size) bounds don't panic, pitch/zoom clamping holds,
  `view_proj` is finite for a normal aspect ratio.
- Full workspace: `cargo build --workspace`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo clippy -p umber-gpu --all-targets
  --features gpu -- -D warnings`, `cargo fmt -- --check`, and
  `cargo test` all green at hand-off.

**Needs-GPU-check** (no display server was reachable from this sandbox —
and starting one would have needed Bash tools outside this task's
pre-approved `cargo build|check|test|fmt|clippy` scope, so this was not
attempted):
1. **The actual windowed frame loop.** `MeshPaintCallback::paint` has never
   executed inside a real eframe/egui render pass. The pipeline's
   `color_format` is threaded from `render_state.target_format` at
   construction time, which should make it match, but that path is
   untested end-to-end.
2. **Depth testing is off** (`depth_format: None` in `UmberApp::new`).
   This is a deliberate, safety-first choice — see below — but it means
   concave/self-occluding mesh geometry draws in index order with no
   z-buffer. Visually verify whether this matters before wiring depth up.
3. **Camera framing/sensitivity tuning** (`framing`'s distance/near/far
   formulas; `ORBIT_SENSITIVITY`/`PAN_SENSITIVITY`/`ZOOM_SENSITIVITY` in
   `viewport.rs`) — never exercised with real pointer input.
4. **Lighting direction** (`light_dir = (-0.4, -1.0, -0.3)` in
   `viewport.rs`) — arbitrary, never seen rendered.
5. Whether `eframe::Renderer::Wgpu` actually yields a non-`None`
   `wgpu_render_state` on the target platform/windowing backend — Cargo.lock
   confirms both `egui_glow` and `egui-wgpu` are compiled in today (so the
   feature is reachable), but the runtime window/surface creation path was
   never run.

## API decisions and their source

- `CreationContext.wgpu_render_state: Option<egui_wgpu::RenderState>` —
  `eframe_epi.rs:84`/`:803`. Read via `cc.wgpu_render_state.as_ref()` in
  `UmberApp::new`.
- `RenderState` fields `adapter`/`device`/`queue`/`target_format` —
  `egui_wgpu_lib.rs:107-137`.
- `egui_wgpu::Callback::new_paint_callback` + `CallbackTrait` — exact
  `prepare`/`finish_prepare`/`paint` signatures from
  `egui_wgpu_renderer.rs:31-121`; `MeshPaintCallback` implements only
  `prepare`/`paint` (the `finish_prepare` default is a no-op, which is
  correct here — nothing needs the post-all-prepare ordering).
- **Pipeline descriptor shapes copied, not written from memory**, from
  `egui_wgpu_renderer.rs:354-436` (`Renderer::new`), because wgpu 30
  changed several field shapes versus older wgpu: `depth_write_enabled`/
  `depth_compare` on `DepthStencilState` are now `Option<bool>`/
  `Option<CompareFunction>` (not bare values); `PipelineLayoutDescriptor`
  takes `immediate_size`; `VertexState`/`FragmentState.entry_point` is
  `Option<&str>` with an explicit `compilation_options`;
  `RenderPipelineDescriptor` has `multiview_mask`/`cache`.
- `ScreenDescriptor`/`CallbackResources` — defined in
  `egui_wgpu_renderer.rs` and re-exported by `egui_wgpu` via `pub use
  renderer::*` (confirmed in `egui_wgpu_lib.rs:26`).
- `PaintCallbackInfo` is **not** re-exported by `egui_wgpu` — the vendored
  renderer only `use epaint::{PaintCallbackInfo, ...}`s it internally, it
  isn't part of `renderer::*`. Added `epaint = "0.36"` as an explicit
  `umber-gpu` dependency (matching Cargo.lock's pinned `epaint 0.36.2`) to
  name it in the `CallbackTrait` impl.
- **egui-wgpu reachability (the task's explicit fork point):** the
  vendored README didn't establish whether `eframe` re-exports `egui_wgpu`
  publicly, and `eframe`'s own `lib.rs` isn't vendored. Rather than guess,
  `umber-gpu` depends on `egui-wgpu = "0.36"` directly (this crate owns the
  `CallbackTrait` impl regardless) and exposes `renderer::mesh_paint_shape`
  — a thin wrapper around `egui_wgpu::Callback::new_paint_callback` that
  returns a plain `epaint::Shape`. Result: **`umber-app`'s `Cargo.toml`
  needed zero new dependencies** — it never spells out a `wgpu`/
  `egui_wgpu`/`epaint` type by name anywhere, only holds/passes opaque
  values whose types are inferred. Verified empirically (build succeeds
  with the unmodified `umber-app` `Cargo.toml`). This also satisfies the
  architecture rule more strictly than the task's literal wording implied.
- `eframe`'s default features already compile in **both** `egui_glow` and
  `egui-wgpu` (confirmed in `Cargo.lock`: `eframe 0.36.2`'s dependency list
  includes both). So `wgpu_render_state` was already reachable with no
  `Cargo.toml` feature changes. `NativeOptions.renderer` is still
  explicitly set to `eframe::Renderer::Wgpu` in `main.rs` rather than
  relying on `Renderer::default()`'s "prefer wgpu when both are compiled
  in" fallback (`eframe_epi.rs:598-621`), so a future feature-flag change
  elsewhere in the dependency graph can't silently flip the app to glow
  (where `wgpu_render_state` would be `None` and `UmberApp::new` would
  error out).

## Signature deviations from the task sketch (and why)

- `GpuContext::new(adapter, device, queue, color_format, depth_format)` —
  added the last two beyond the task's literal `(adapter, device, queue)`.
  The pipeline's color-target format and depth-stencil format must match
  the render pass it's later used in, or wgpu raises a validation error at
  draw time; there's no safe way to hardcode a format.
- `MeshBuffers::upload(gpu: &GpuContext, mesh: &MeshData)` rather than the
  sketched `(device, &MeshData)` — needs `gpu.bind_group_layout` (private)
  to build the per-mesh bind group, and taking `gpu` alone avoids a
  redundant second parameter at every call site (`gpu.device` is already
  reachable).

## The depth-buffer decision (read this before enabling depth testing)

`GpuContext::new`'s `depth_format` parameter is `None` from
`UmberApp::new` today, and `NativeOptions.depth_buffer` is left at its
default (`0`) in `main.rs`. These two defaults are mutually consistent —
no depth attachment on the shared render pass, no depth state in the mesh
pipeline — so this is the empirically-safe configuration (confirmed:
pipeline creation with `depth_stencil: None` succeeds against this
sandbox's real adapter).

The alternative — setting `NativeOptions.depth_buffer = 32` and passing
`Some(wgpu::TextureFormat::Depth32Float)` into `GpuContext::new` — depends
on eframe threading `NativeOptions.depth_buffer` through to the wgpu
`Renderer`'s `RendererOptions.depth_stencil_format` for the *shared* render
pass this crate draws into. `egui_wgpu::depth_format_from_bits` (seen in
`egui_wgpu_lib.rs:441-451`) strongly implies this wiring exists somewhere
in eframe's native wgpu integration, but that module isn't vendored, so
it's inferred, not confirmed. Getting it wrong means a wgpu validation
panic on the very first painted frame, on real hardware, in a window —
exactly the kind of failure this sandbox cannot catch (no display server).
**This is the #1 item for a reviewer with a real windowed GPU run.**

## Reviewer checklist

- [ ] Run the app on a real display; confirm a window opens and the first
      paint doesn't panic.
- [ ] File > Open Mesh with an OBJ/glTF/FBX file; confirm the mesh appears
      centered, normal-shaded, and lit.
- [ ] Orbit (primary drag), pan (shift+primary drag or middle drag), zoom
      (wheel) — tune the `*_SENSITIVITY` constants in
      `crates/umber-app/src/viewport.rs` if the feel is off.
- [ ] Decide on the depth-buffer follow-up above; test with a concave mesh
      (e.g. a torus) to judge whether the lack of depth testing is
      visually acceptable before investing in wiring it up.
- [ ] `glam::camera::rh::{view, proj}` is new in glam 0.34 (it replaced
      `Mat4::look_at_rh`/`Mat4::perspective_rh`, discovered by reading the
      actual glam source — it isn't in the vendored artifacts). If glam is
      ever pinned to an older version transitively, `camera.rs` won't
      compile; re-check this module path on any glam bump.
- [ ] `umber-mesh`'s glTF/FBX loaders landed concurrently (another agent,
      same wave). `MeshBuffers::upload` was only tested against synthetic
      triangle data here — worth a real multi-material mesh end-to-end.
- [ ] Delete `docs/claw-artifacts/umber-gpu/` once this is cross-reviewed
      (per that directory's own README).

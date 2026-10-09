# Claw artifacts: umber-gpu render pass

Hand-off artifacts for the GPU-pass claw (headless claude cannot WebFetch or
run cargo doc without pre-approval; these are verbatim copies of the exact
dependency sources so the API can be verified without leaving the repo).
Delete this directory when the render pass lands and is cross-reviewed.

## Verified API facts (checked against the vendored sources 2026-10-09, wgpu pinned 30.0.1 in Cargo.lock)

- `eframe::CreationContext.wgpu_render_state: Option<egui_wgpu::RenderState>`
  (eframe 0.36.2 src/epi.rs:84; accessor `.wgpu_render_state()` at :803)
- `egui_wgpu::RenderState` fields (egui-wgpu 0.36.2 src/lib.rs):
  `adapter`, `available_adapters`, `instance`, `device`, `queue` (+ target,
  renderer in the full struct)
- Custom pass hook: `egui_wgpu::Callback` + `CallbackTrait` (prepare/finish/
  paint) — see `egui_wgpu_renderer.rs` (renderer.rs:26-48: "Implement
  CallbackTrait and call Callback::new_paint_callback")
- eframe feature for wgpu backend: default `wgpu` feature (feature chain:
  `wgpu = ["wgpu_no_default_features", "wgpu", ...]`)
- Files: `eframe_epi.rs` (CreationContext + RenderState accessors),
  `egui_wgpu_lib.rs` (RenderState struct), `egui_wgpu_renderer.rs`
  (CallbackTrait full source)
- Our app currently uses `eframe::run_native("Umber", native, Box::new(|_cc|
  Ok(Box::new(UmberApp::default()))))` — the `_cc` CreationContext is where
  RenderState is obtained. UmberApp currently takes no cc data; you will need
  to restructure `UmberApp::new(cc: &CreationContext)` to capture RenderState.

# The GPU Display LUT — Design (the display panel's named follow-up)

The display panel's landing notes. Written 2026-10-10 15:30
against `ae7e8f9`. The panel's own doc points here
(display_panel.rs:24 — the reference was dangling; this doc is
the landing it meant).

## The gap, plainly

The display chain (view transform + exposure + gamma) is CPU
and preview-only: the panel's swatch strip shows it, but the 3D
viewport's mesh pass and the UV view's paint-target display
render on the GPU and never see the settings. The §9 row wants
the chain ON THE PIXELS.

## The design: a 1D LUT texture, sampled in the display passes

1. **The LUT**: `DisplaySettings` → a 256×1 RGBA8 texture
   (the chain baked per-entry via `apply_display_chain` — the
   same pure fn the preview strip uses). Rebuilt when the
   settings change (the panel's write path calls
   `mark_display_lut_dirty`).
2. **The consumers**: `texture_display` (the UV view's paint
   target display — umber-gpu) and the viewport mesh pass's
   fragment shader each gain a LUT binding in their display
   path: `texture(lut, linear-rgb)` → the displayed color.
   The default (identity) LUT renders byte-identically (a
   golden test pins it).
3. **The ownership**: umber-app owns the LUT texture (created
   on the device at startup + on settings change); the GPU
   consumers take a bind-group entry. `umber-gpu` stays
   DisplaySettings-free (the LUT IS the interface — the
   settings never cross the crate boundary).
4. **The honest v1 scope**: two consumers wired (UV view +
   viewport mesh), the golden identity test, the panel's
   status line flipping from "preview-only" to naming the
   live consumers. The `ocio` feature (real OCIO config
   loading) stays gated — the built-in chain is the LUT's
   only source until the OCIO adoption probe's condition
   fires (docs/specs/ocio-adoption-probe-2026-10-10.md).

## Tests

1. The LUT build: the identity settings produce the identity
   entries (0..255 → 0..255, byte-exact).
2. A non-identity chain: known settings → the known output
   bytes (the same math as the CPU chain's existing tests —
   the LUT is that math, tabulated).
3. The golden: the UV view's default render is byte-identical
   with vs without the identity LUT bound (the no-regression
   proof).
4. The dirty/rebuild cycle: settings change → rebuild → the
   panel's status reflects it (the data path, CPU-assertable).

Tests 1/2/4 are CPU (umber-color + the app's state); test 3 is
GPU-gated (the golden harness's adapter rule).

## Build

One claw slice, code-only: the LUT builder (umber-app or
umber-color — the chain fn lives in umber-color, so the
builder sits beside it), the two shader bindings + bind-group
wiring (umber-gpu), the panel status flip, the four tests.
No new deps.

## Landed (code-only, un-built — awaiting review)

- **Format**: 256×1 `Rgba8Unorm` D2 (not `…Srgb`, not D1).
  The eframe surface is non-sRGB (`preferred_framebuffer_format`
  picks `Rgba8Unorm`/`Bgra8Unorm`), so entries are final display
  bytes — no double encode. D2 mirrors every other binding and
  avoids GL-backend 1D quirks.
- **Read**: `textureLoad` at `round(clamp(c,0,1)·255)`, no
  sampler, each channel from its own entry's component
  (`umber_gpu::display_lut::DISPLAY_LUT_WGSL`, pasted verbatim
  into both shaders; a test pins the copies).
- **Sampling point**: the last step before the target — the
  mesh pass after diffuse + specular; the texture display on
  the sampled texel's rgb (alpha still forced to 1).
- **Binding**: group 1 of both pipelines (additive — group 0 and
  the wireframe pass are untouched). Consumers take
  `Option<&DisplayLut>`; `None` binds the context's identity LUT.
- **Ownership**: `UmberApp::display_lut` (created at startup);
  `AppState::display_lut` (`DisplayLutState`) starts dirty, the
  panel marks it on edit, `open_project` on load; the frame loop
  `take_rebuild`s after the panels and uploads in place
  (`queue.write_texture` — no bind-group churn).
- **Limits**: input clamps to 0..1; 256 nearest levels band in the
  shadows on the mesh pass under sRGB/Rec.709; the grid,
  wireframe, and egui-drawn UV wireframe are not transformed.

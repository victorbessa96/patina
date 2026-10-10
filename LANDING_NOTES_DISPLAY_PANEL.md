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

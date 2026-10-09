# Umber — Component Architecture

> Wave-0 consolidation. Sources: docs/research/05 (brush engine), 03 (ecosystem), 04 (capability bar), 02 (ArmorPaint lessons — GPU-resident paint loop, stroke-as-pass, bus-factor-proof crate boundaries).

## Bird's-eye view

```
┌─────────────────────────────────────────────────────────────────┐
│                        umber-app (binary)                        │
│   egui + egui_dock shell · winit event loop · app state          │
└──────┬──────────────┬───────────────┬───────────────┬───────────┘
       │              │               │               │
  ┌────▼────┐   ┌────▼─────┐   ┌─────▼─────┐  ┌─────▼─────┐
  │ stylus  │   │ umber-    │   │ umber-    │  │ umber-    │
  │ (input) │   │ brush     │   │ gpu       │  │ mesh      │
  └────┬────┘   └────┬─────┘   └─────┬─────┘  └─────┬─────┘
       │              │               │               │
       └──────────────┴───────┬───────┴───────────────┘
                               │
                        ┌──────▼──────┐
                        │ umber-core  │  document model + undo
                        └──────┬──────┘
                   ┌───────────┼───────────┬─────────────┐
             ┌─────▼────┐ ┌────▼─────┐ ┌────▼────┐ ┌────▼─────┐
             │ umber-   │ │ umber-   │ │ umber-  │ │ umber-   │
             │ graph    │ │ bake     │ │ export  │ │ color    │
             └──────────┘ └──────────┘ └─────────┘ └──────────┘
```

**Hard rules:** GPU work is submitted only through `umber-gpu` (single device context, single render-graph abstraction). UI never touches wgpu objects directly. Core (document model) is pure Rust, no wgpu dependency — testable headless in CI without lavapipe.

## Threading model [05]

```
Thread 1: winit main loop          Thread 2: paint scheduler        Thread 3+: workers
┌────────────────────────┐        ┌────────────────────────┐      ┌──────────────────┐
│ drain ALL events      │        │ stroke job queue       │      │ rayon pool:      │
│   → route pen →       │ lock-  │  - dab batcher         │      │  - mesh raycast  │
│     stroke input      │ free   │  - tile job scheduler  │────▶ │  - encode dabs   │
│ acquire frame         │ ring   │  - undo snapshot coales│      │  - image encode  │
│ build+submit UI/      │        │  (Krita-shaped typing: │      │  - bake jobs     │
│   viewport passes     │◀───────│   paint vs present     │      │                  │
│ present               │ dirty- │   never block)         │      │                  │
└────────────────────────┘ tiles └────────────────────────┘      └──────────────────┘
```

- **Event flow:** winit drains events first, then acquires the frame (wgpu#2269 — avoids a frame of latency; never vsync-block with pending input).
- Pen events bypass egui's input model entirely (egui#2104 workaround): `stylus` crate → normalized `StrokeEvent` → ring buffer → brush thread. Mouse stays through egui.
- Paint submits never wait on present; present never blocks paint (separate command buffers; dirty-tile list bridges them).

## Paint data flow (stylus → texel) [05]

```
stylus event (pressure, tilt, position, time)
  → one-euro filter + lazy-mouse pull (stroke input conditioning)
  → umber-brush state machine (libmypoint-lineage: inputs→curves→dab params;
    residual partial-dab spacing; dabs-per-radius)
  → dab stream {uv, size, color, flow, hardness, alpha_mode}
  → 3D-space stroke evaluation:
        raycast brush cursor → face set under screen-space disc
        → UV-space triangle projection (emit tris xy=uv, w=1; falloff
          evaluated in screen space — seam splitting and UDIM clipping
          fall out of rasterization naturally)   [Psy-Fi design, 05 §6]
  → dab batch in storage buffer
  → one wgpu compute dispatch per dirty 256² tile (premultiplied RGBA)
  → per-stroke accumulation texture (wash mode = flow semantics for free)
  → end-of-stroke merge into layer texture; dirty-tile list → undo journal
  → viewport samples layer textures directly (no CPU copy)
```

**Why this shape:** GPU-resident paint loop is the ArmorPaint-proven pattern (16K on consumer GPUs) [02]; UV-rasterization stamping makes seams/UDIM correctness fall out of triangle rasterization rather than bespoke stitching [05]; per-stroke accumulation gives Substance/Krita flow semantics without order-dependent alpha-over per frame [05].

## Virtual texturing (software VT) [04 P0-3]

- Physical page atlas (e.g. 8192² of 256² pages) + page-table indirection texture + background upload thread.
- Paint targets own per-channel page pools; bake targets reuse the same pool allocator.
- Eviction policy under paint traffic + mip updates = the hard 20% (budgeted as core-engine work, Wave 2–3).
- No wgpu sparse residency exists; this design uses only stable primitives (storage-buffer arrays, per-draw rebinds) with the `as_hal` Vulkan hatch as insurance [03].

## Layer stack evaluation

- Stack = DAG of layers/masks/effects, composited per dirty tile in compute.
- Blending in linear space per channel; per-channel blend mode + opacity.
- Procedural nodes (umber-graph) evaluate as MaterialX-document graphs on the same tile scheduler; graphs either interpret per-tile (early) or codegen WGSL passes (later) [04 P1-14].
- Flatten = bake a group to bitmaps (Painter 12.0 parity); smart material = serialized folder graph with mesh-map inputs.

## Undo [05]

- Stroke = command with lazy first-write dirty-tile blits; no GPU readback.
- Cumulative merge + memory budget (~100 steps at 4K within ~2GB) + disk swap for older steps.
- History window maps over the journal (click-to-time-travel); journal is in-RAM, not persisted (matches Painter).

## Project format (.umber) [04 P1-15]

```
myproject.umber/            # a directory, not a file
  project.json             # scene, texture sets, channel configs, settings
  layers/<hash>.json       # one JSON per layer (stack topology, params)
  textures/<content-hash>.exr|.png   # tiles by content hash
  assets/                  # imported meshes referenced by hash
```
- Diffable, mergeable, git/LFS-friendly; two artists editing different layers never conflict.
- Round-trip determinism CI-tested (save→load→export byte-identical).

## Viewport

- OpenPBR 1.1 über-shader (WGSL, from ASWF reference); IBL + analytical lights.
- Render order: mesh pass (samples layer-composited textures via VT) → post → UI composite (egui on same wgpu surface).
- HDR surface output via wgpu 30 SurfaceColorSpace behind a display-capability check (P1, Wave 5).

## Bake engine

- Compute passes: BVH build (CPU, rayon) → per-pixel raycasts on GPU (or rayon fallback) → accumulation → dilation pass.
- Cage = distance-based offsets (+ custom map); skew map paintable + edge protection in Wave 5 (Marmoset-class polish).
- Bakers and the paint engine share the tile-pool allocator; bake output lands as mesh-map channels (AO/curvature/…) usable as generator inputs.

## Plugin boundary (Wave 6)

- wasmtime + WASI 0.3 components; host API = stamps/ops orchestration (exporters, generators, presets), never inner-loop execution.
- Node-graph extension API = the same component interface (a "node" is a plugin exposing params + a per-tile op).
- Rust-native hot-reload (hot-lib-reloader) is a dev-time feature only.

## Bus-factor defense [02]

ArmorPaint died on the vine for 2–3 years per release because one maintainer carried everything. Umber's countermeasures are structural: crate boundaries with documented contracts (this doc + tech-stack.md), headless-testable core (umber-core/brush/mesh have zero GPU dependency in their logic layers), golden-image CI from Wave 3, and DECISIONS.md as institutional memory.

# Umber — Tech Stack

> Wave-0 consolidation. Source: docs/research/03 (Rust ecosystem survey — 100+ crates verified against crates.io/docs.rs/GitHub, 2026-10-09) cross-checked against alignment decisions (DECISIONS.md). This is the crate-level contract for the workspace.

## Core stack (locked)

| Layer | Crate | Why | Maturity (1–5) | Risk |
|---|---|---|---|---|
| GPU API | **wgpu 30.x** | One API over DX12/Vulkan/Metal; compute-first painting/baking; `as_hal` escape hatch reserved for Vulkan-only needs (timeline semaphores, external memory) | 5 | Low core; med on experimental flags |
| Shader IR | **naga** (wgpu-locked) | WGSL validation + module IR for composition | 5 | Low |
| Shader composition | **naga_oil 0.23** | Module import/compose (brush dab shader, OpenPBR über-shader layers) | 4 | Version-locked to wgpu 30 — pin trio together |
| Windowing | **winit 0.30.13** (→0.31 when stable) | 0.31 beta reworked pointer events; pin 0.30 until final, wrap all events behind own enum | 5 / pen 2.5 | Med (pen input gaps) |
| Stylus input | **own `stylus` crate** (fork/absorb octotablet + wintab_lite) | The must-own layer: dual Windows path (Ink WM_POINTER + Wintab32), Wayland zwp_tablet_v2 + X11 XI2; no production-grade crate exists | 2 (built up to 5) | **High — core differentiator** |
| UI | **egui 0.36 + egui_dock 0.21** | Immediate-mode iteration speed; dockable panel trees; renders on our wgpu stack; pen events routed raw from winit, bypassing egui input | 4.5 | Med (pressure not in input model — bypass proven in egui#2104) |
| Math | **glam 0.34** | SIMD vec/mat/quat; the Rust standard | 5 | Low |
| Concurrency | **rayon, parking_lot, dashmap, crossbeam** | Paint-job parallelism, locks; all boringly reliable | 5 | Low |
| Async | **none in core** | DCC = thread-pool + channels; async never in the paint path | — | — |
| Profiling | **tracy-client + `profiling` facade** | Frame + job profiling from day one; puffin optional | 5 | Low |
| Errors | **thiserror (libs) + anyhow (app) + miette** (import diagnostics) | Layered error strategy | 5 | Low |

## Asset & format layer

| Layer | Crate | Why | Maturity | Risk |
|---|---|---|---|---|
| glTF/OBJ | **gltf 1.4, tobj 4.0, meshopt** | Solid, slow-moving, correct | 4 | Low |
| FBX | **ufbx 0.11.5** | MIT C lib with official Rust bindings; only credible FBX route (Autodesk SDK non-redistributable) | 4 | Low-med |
| USD | **openusd 0.7** (pure-Rust) | usda/usdc/usdz R/W, LIVRPS composition; scope: import + UsdShade/primvar export; golden-test vs pxr | 3 | Med-high (pre-1.0) |
| Images | **image 0.25, exr 1.74, half 2.7** | PNG/JPEG/TIFF + pure-Rust OpenEXR + f16/bf16 | 5 | Low |
| GPU texture containers | **ktx2, ddsfile, ctt** (BCn/ASTC) | Compressed cache/pipeline textures | 3–4 | Med |
| Color | **ocio-rs 0.2** (vendored OCIO 2.5) + **lcms2 6.2** | OCIO v2 configs (ACES 2.0) + GPU shader extraction; ICC embed | 3 | Med (single maintainer — golden tests + LUT fallback) |
| Materials | **own .mtlx subset parser** (quick-xml) + **OpenPBR WGSL port** | Zero usable MaterialX bindings exist; spec + reference C++ (Apache-2.0) are the source of truth; OpenPBR über-shader implemented once in WGSL. Serialization claim is honest: standard-equivalent nodes map to MaterialX standard nodes; painter-specific nodes ship as declared custom nodedefs — no automatic full-fidelity interchange | 1 (built to 4) | **High — own it** |
| Mesh topology | **in-house half-edge/DCEL** | Painter's topology needs (seam awareness, per-layer masks, UV islands) are bespoke; ecosystem crates are hobby-grade | — | Owned |
| Subdivision | in-house Catmull-Clark CPU first; opensubdiv-rs only if hard-verified | Preview-grade only at first | 1–2 | Deferred |

## Platform & distribution

| Layer | Crate | Why | Maturity | Risk |
|---|---|---|---|---|
| Windows installer | **cargo-dist** (MSI + scripts) | Full GH Actions release pipeline from a tag | 4 | Low |
| Linux | **Flatpak manifest + AppImage** (cargo-packager) | Flatpak = best Vulkan-driver sandbox story | 4 | Low |
| CI | **GH Actions**: windows-latest + ubuntu; **lavapipe** software-Vulkan for headless GPU tests | wgpu's own CI pattern | 4 | Low |
| Plugin runtime | **wasmtime 49 + WASI 0.3 + wit-bindgen** | Sandboxed plugins orchestrate via host API; never execute paint hot paths. WASI 0.3 ratified 2026-06. Native dynlib plugins (abi_stable — stale 3y) deferred | 5 | Low |
| Undo | **own command journal** (undo 0.52 as reference) | Stroke merging, memory-budgeted history, dirty-tile snapshots | 4 | Low |
| Hot reload (dev) | notify + naga pipeline rebuild; hot-lib-reloader for Rust code | Fast iteration loop | 4 | Low |

## The three budget-to-own engineering bets [03 §(c)]

1. **Stylus stack** — dual Windows backend + Linux tablet protocols under one `stylus` crate; event-rate-independent stamp spacing, pressure smoothing, proximity. All API paths proven to exist in Rust (wintab_lite, octotablet, efude-input); nobody has productionized them. This is Umber's tablet-feel moat.
2. **Painter-class texture streaming on wgpu** — software virtual texturing (page pool + indirection table) on stable wgpu primitives only; no sparse residency exists, so the page atlas + async upload thread is mandatory for 4K–16K. Main performance bet.
3. **Interchange/material layer solo** — .mtlx parser + OpenPBR WGSL + ocio-rs + openusd covers ~90% of real asset traffic; golden-file conformance against C++ reference tools on every commit is the mitigation.

## Version policy

- **wgpu + naga + naga_oil pin as a trio** in one workspace update (naga_oil lags wgpu majors).
- winit pinned 0.30.13 until 0.31 final; own event enum isolates the swap.
- ocio-rs vendored → if bindings rot, baked-LUT fallback path is pre-designed.
- No experimental wgpu features in the core paint path (RT, mesh shaders, bindless) — cargo-feature-gated only.
- **Dependency license audit gate (Wave 1):** THIRD_PARTY.md manifest listing every dependency's license before first merge to master; MIT relicensing of standalone crates requires upstream license compatibility verified per crate (octotablet absorption, ctt's vendored ISPC/astcenc components, ocio-sys, openusd crate — all checked at manifest time, not assumed). libmypaint relationship = documented-semantics reimplementation, never a code derivative (LGPL line drawn explicitly).

## Cargo workspace shape (Wave 1)

```
crates/
  umber-app        # binary: egui shell + winit loop + app state wiring
  umber-core       # document model: texture sets, layers, channels, undo journal
  umber-gpu        # wgpu device context, render graph, tile pool, shaders (naga_oil)
  umber-brush      # libmypaint-lineage state machine + dab batching + stroke jobs
  umber-mesh       # half-edge DCEL, import (gltf/tobj/ufbx), raycast picking
  umber-bake      # mesh-map bakers (compute passes)
  umber-graph      # procedural node engine (evaluates as MaterialX documents)
  umber-export     # template/preset engine + encoders
  umber-color      # OCIO/ICC integration + display transforms
  stylus           # owned input crate (dual Windows + Linux tablet protocols)
  umber-cli        # headless automation (bake/export/batch)
xtask/             # workspace tasks (dist, golden-image tests, bench)
```

Library crates get clean boundaries; the app is a thin composition. Standalone-usable crates (stylus, umber-graph, umber-color) keep MIT-option per license decision.

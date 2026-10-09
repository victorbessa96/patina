# Umber — Requirements

> Wave-0 consolidation of the research corpus (docs/research/01–05). This is the component-and-capability contract that drives the wave plan and the eventual per-subsystem specs. Provenance: every section cites its research source. Priority tags map to waves.

**Sources:** [01] Substance Painter inventory · [02] competitor landscape · [03] Rust ecosystem survey · [04] next-gen capability bar · [05] brush-engine architecture.

**Priorities:** P0 = professional floor (cannot ship v0.1 without) · P1 = differentiator (wins users, post-0.1 but on the roadmap) · P2 = moonshot (2+ years). Wave tags show where each lands.

---

## 1. Painting tools (brush engine) — [01][05]

| Requirement | Priority | Wave |
|---|---|---|
| Stamp-based brush engine: size, flow (per-stamp, pressure-bindable), stroke opacity (per-stroke, end-of-stroke), spacing (dabs-per-radius, residual partial-dab carry), angle, alignment (camera/tangent-wrap/tangent-planar/UV), jitter quartet (size/flow/angle/position) | P0 | 2 |
| Pressure dynamics mapped through input→curve model (MyPaint-style); log-radius; `opaque_linearize` flow semantics (per-stamp flow with per-stroke accumulation = Krita wash-mode; Build-up mode as alternative) | P0 | 2 |
| Lazy mouse (radius-based stroke offset smoothing) + one-euro filter smoothing (Casiez CHI'12) with Krita-style stabilizer panel (samples@min/max speed, delay dead-zone) | P0 | 2 |
| Hardness/stamp falloff (radial gradient dabs); alpha source: bitmap textures + procedural | P0 | 2 |
| Eraser (sets layer alpha to zero — non-destructive; per-channel enable) | P0 | 2 |
| Fill layer + fill projections: UV, Tri-planar, Planar, Spherical, Cylindrical | P0 | 3 |
| Smudge (sample+blur blend), Clone (source offset + follow-stroke) | P1 | 3 |
| Symmetry: mirror (axis+offset+manipulator) and radial (count, flip U/V) — brush + fill-layer projections | P1 | 3 |
| Straight-line constraint; angle snap | P1 | 3 |
| Polygon fill (triangle/polygon/mesh/UV-chunk pixel-mask fills) | P1 | 3 |
| Quick mask (temporary painted mask, invert) | P1 | 4 |
| Path tools (paint/erase/smudge along 3D curve; per-vertex size+opacity; filled path) | P1 | 5 |
| Brush presets (.abr import optional; native preset files) | P0 | 2 |
| Seam-aware stamping: strokes crossing UV seams land consistently in both islands; 3D-space-neighbor padding/dilation | P0 | 2 |
| Stroke evaluated in 3D space, not per-tile (UDIM continuity) | P0 | 2 |
| Paint-what-you-see projection modes beyond UV (Mari-style paint buffer) | P1 | 4 |

## 2. Layer + channel system — [01]

| Requirement | Priority | Wave |
|---|---|---|
| Texture sets (per material/mesh UV set); per-set channel stack: baseColor, roughness, metallic, normal, height, emissive, opacity, AO + user channels | P0 | 1 (model), 2 (full) |
| Layer types: paint layer, fill layer; folders with passthrough | P0 | 2 |
| Per-channel blending mode + opacity; blending computed in linear space per channel | P0 | 2 |
| Blending modes: full pro set (~32: normal, passthrough, multiply, screen, overlay family, HSV modes, normal-map combine/detail) | P0 core 12, full set 4 | 2/4 |
| Masks: grayscale paint mask, bitmap mask, black/white; mask stacks with effects | P0 | 2 |
| Layer instancing (source-edits propagate, cycle detection) | P1 | 4 |
| Anchors (reference any layer/channel/mask as generator input within same texture set) | P1 | 4 |
| Per-channel resolution; in-app 4K default, 8K export support | P0 | 2/3 |
| UDIM multi-tile texture sets; per-tile resolution; cross-tile painting | P0 | 4 (data model from 1) |
| Smart materials (folder presets applied with mesh-map-driven generators) | P1 | 4 |
| Dynamic material layering (shader-declared sub-stacks) | P2 | 6+ |
| Storage formats per channel: sRGB8/RGB16/RGB16F/RGB32F | P0 | 2 |

## 3. Baking engine — [01][04]

| Requirement | Priority | Wave |
|---|---|---|
| Bakers: AO, normal (mesh + high→low), curvature, position, thickness, ID (material/vertex/mesh), height, bent normals, opacity | P0 (AO/normal/curvature/position/ID first) | 3 |
| Common settings: output size, dilation width, antialiasing (supersampling), ignore backfaces, low-as-high toggle, match by mesh name | P0 | 3 |
| Cage: distance-based + custom offset; front/back projection distance | P0 | 3 |
| Compute-shader raycast/bvh bake path (no RT hardware dependency); wgpu ray-query acceleration when stable | P0 | 3 |
| Skew correction: paintable skew map + edge protection (Marmoset-class bake polish [02]) | P1 | 5 |
| Auto-rebake of affected regions on parameter change | P1 | 5 |
| Baking mode UI: per-set/per-map checklist, bake log with jump-to-setting, cage visualization | P1 | 3 (log) / 5 (viz) |

## 4. Procedural system (original — .sbsar is legally untouchable) — [01][04]

| Requirement | Priority | Wave |
|---|---|---|
| Original node-graph engine: DAG, topological eval, dirty-region propagation; graphs serialize as **MaterialX documents** (nodes = MaterialX standard nodes; interchange with Houdini/UE/UsdMtlx free) | P0 | 4 |
| Node set v1 (~40 core): noise family (perlin/value/worley), gradients, patterns, blur/sharpen, levels, curves, color ops, direction warp, flood fill from masks, edge detect, histogram ops | P0 | 4 |
| Generator nodes driven by baked mesh maps (AO/curvature/position/WS-normal inputs plumbed automatically) | P0 | 4 |
| Smart masks (effect-stack presets) | P1 | 4 |
| Example-based texture synthesis (EmbarkStudios/texture-synthesis style) as first-class generator primitive | P1 | 4 |
| Graph evaluated on GPU (WGSL compute passes per dirty tile) | P0 | 4 |
| Community extension API for nodes (see plugins §10) | P1 | 6 |

## 5. Viewport + shading — [01][03][04]

| Requirement | Priority | Wave |
|---|---|---|
| PBR viewport: **OpenPBR 1.1 über-shader in WGSL** (native channel model = OpenPBR parameter names; spec is ASWF/Apache-2.0 with reference C++ to transliterate) | P0 | 1 (basic Blinn-Phong-ish PBR), 2 (OpenPBR) |
| IBL environment lighting (HDR env maps, exposure, rotation, blur) | P0 | 1 |
| View modes: lit / unlit single-channel solo (±display transform) / mesh-map preview | P0 | 1 |
| 2D UV view alongside 3D view (synchronized) | P0 | 2 |
| Camera: orbit/pan/zoom, perspective + orthographic | P0 | 1 |
| Wireframe overlay; grid | P1 | 2 |
| Displacement/tessellation preview | P2 | 6+ |
| Post-effects stack (DoF, bloom, vignette, sharpen, tone-map) | P1 | 5 |
| HDR surface output (wgpu 30 SurfaceColorSpace; DX12/Metal, Vulkan driver-dependent) | P1 | 5 |
| Custom user shaders (WGSL plugins replacing GLSL import) | P2 | 6 |
| Performance floor: 60fps viewport on mid-range GPU with 1M-tri mesh + 4K texture set | P0 | 1 |

## 6. Export pipeline — [01][04]

| Requirement | Priority | Wave |
|---|---|---|
| Template-driven export: naming tokens ($mesh/$textureSet/$udim/$colorSpace/$srcMap/$layerName), folder mapping, saved-as-files presets | P0 | 3 |
| Channel packing (ORM etc.); per-output RGB/R/G/B/A/grayscale slot mapping | P0 | 3 |
| Engine presets: glTF metal-rough, Unreal (packed, DX normals), Unity (HDRP/URP, GL normals), Blender Principled + the long tail as community presets | P0 first 4 | 3 |
| Bit depths 8/16/32F; formats: PNG, EXR, TIFF, JPEG (via image/exr crates); dithering option | P0 | 3 |
| Normal-convention conversion (DirectX vs OpenGL Y-flip) at export | P0 | 3 |
| Padding: dilation (finite/infinite), transparent/default-color fill; 3D-neighbor aware | P0 | 3 |
| 8K export (from 4K in-app) | P1 | 5 |
| PSD export (layered) | P2 | 6+ |
| MaterialX document export (OpenPBR nodedef + image/UDIM tokens) — "beyond raster" interchange | P0 | 3 (basic), 4 (full) |
| USD export (textures + .usda + material binding via UsdShade→OpenPBR) | P1 | 5 |

## 7. Import — [01][03]

| Requirement | Priority | Wave |
|---|---|---|
| glTF/GLB (gltf crate), OBJ (tobj), PLY | P0 | 1 |
| FBX via ufbx (MIT C lib, official Rust bindings — the only credible route; Autodesk FBX SDK is non-redistributable) | P0 | 1 |
| USD import (openusd pure-Rust: usda first, usdc when stable) | P1 | 5 |
| Image formats: png/jpeg/tiff/exr/hdr (image + exr crates); SVG (later) | P0 (png/exr) | 2 |
| Mesh-map import by naming convention (TextureSetName_map) | P1 | 3 |
| **Never:** .spp (undocumented HDF5), .sbsar execution (Adobe Developer Additional Terms — closed, no FOSS license exists) | hard wall | — |

## 8. Project format (.umber — the .spp kill-shot) — [04]

| Requirement | Priority | Wave |
|---|---|---|
| Project = directory; `project.json` (scene, texture sets, settings) + per-layer JSON files (diffable, mergeable) | P0 | 2 |
| Textures as content-hash-named PNG/EXR tiles (git/LFS friendly; assets referenced by hash, never embedded blobs) | P0 | 2 |
| Save/load round-trip determinism (CI-tested: save→load→export identical bytes) | P0 | 2 |
| Autosave + crash recovery (journal) | P1 | 5 |
| Merge story documented (per-layer files = two artists editing different layers never conflict) | P1 | 2 (doc) |

## 9. Color management — [01][03][04]

| Requirement | Priority | Wave |
|---|---|---|
| Scene-linear working space, OCIO v2 configs (ocio-rs, vendored build) incl. ACES 2.0 CG/Studio built-ins | P0 | 2 (viewport), 3 (full configs) |
| Per-channel color-managed flags (baseColor yes; roughness/metallic/normal = data) | P0 | 2 |
| ICC profile embed on export (lcms2) | P1 | 3 |
| Display transform via GPU LUT (OCIO GPU shader extraction → 3D texture) | P0 | 2 |

## 10. Platform, input, plugins, distribution — [03]

| Requirement | Priority | Wave |
|---|---|---|
| Windows 10/11 + Linux (Vulkan 1.3; DX12 on Windows) | P0 | 1 |
| Stylus/tablet: **owned `stylus` crate** — Windows dual-path (Windows Ink WM_POINTER + Wintab32 polling), Linux (Wayland zwp_tablet_v2 + X11 XI2); pressure/tilt/proximity/hover; fork/absorb octotablet; upstream to winit | P0 | 1 (windows-ink basic) → 2 (dual) → 5 (matrix) |
| Pressure into paint pipeline bypassing egui input (winit raw events → canvas) | P0 | 2 |
| Plugin runtime: wasmtime + WASI 0.3 component model (sandboxed; host-API for stamps/ops — plugins orchestrate, never execute inner paint loops) | P1 | 6 |
| Scripting/automation: headless CLI (bake/export/render-batch), project format is the API | P1 | 6 |
| Distribution: cargo-dist (MSI + GH releases), Flatpak + AppImage on Linux; CI on windows-latest + ubuntu with lavapipe GPU tests | P0 (CI) | 1 (CI), 6 (installers) |
| Profiling: tracy-client + profiling facade from day one | P0 | 1 |

## 11. UI shell — [01]

| Requirement | Priority | Wave |
|---|---|---|
| Dockable panels (egui_dock): layer stack, properties, assets/shelf, texture-set list, history, display settings, export dialog | P0 | 1 (shell) → 2 (dense) |
| Brush properties panel: input→curve editor (MyPaint model), flow×opacity, stabilizer, jitter quartet [05] | P0 | 2 |
| History window (session-global, click-to-time-travel; not persisted) | P1 | 2 |
| Dark pro theme, 4K-DPI scaling | P0 | 1 |
| Node-canvas widget (custom, for graph editor) | P0 | 4 |
| Localization (EN first; architecture i18n-ready) | P1 | 5 |

## 12. Performance budgets — [02][05]

| Budget | Target |
|---|---|
| Input-to-photon (stylus event → composited pixel on screen) | < 20 ms at 60Hz; drain-events-then-acquire frame loop [05] |
| Stroke throughput | 4K texture set, no dropped dabs at 200Hz input streams |
| Viewport | 60 fps, 1M-tri mesh, 4K set, mid-range GPU |
| Undo RAM | 100 steps at 4K within ~2 GB (dirty-tile snapshots + cumulative merge) [05] |
| App footprint | release binary < 50 MB (no runtime deps) |

---

## Explicit non-goals (guardrails)

- Mesh modeling/editing, sculpting — out of scope for v0.1 horizon [02: "basic everything-adjacent features diluting the core"]
- .sbsar/.spp execution — permanent legal wall
- Cloud/accounts/telemetry — none, ever
- macOS — later, contributor-driven
- Real-time collaboration (CRDT) — P2 research only [04]

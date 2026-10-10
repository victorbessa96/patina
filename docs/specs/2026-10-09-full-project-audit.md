# Umber — Full Project Audit & Wave-3 Closeout (2026-10-09 21:00)

Complete evaluation against docs/specs/requirements.md. Every claim
verified against the tree at `05a11b9` (99 commits, 11 crates) this
evening: **workspace 272 tests passed / 0 failed, clippy
`--workspace --all-targets -D warnings` clean, fmt clean, CI green on
ubuntu-latest + windows-latest for every commit.** No cargo was run
after the audit verification (the machine-load directive holds).

## The honest scoreboard

| § | Area | State | The real gaps |
|---|---|---|---|
| 1 | Painting tools | **Partial (engine core done)** | StrokeEvent/OneEuro/lazy-mouse/spacing/dab-adapter exist; hardness+flow stamping live in umber-gpu paint. **Missing: brush presets (§1 P0 — none anywhere), seam-aware stamping (the P0 core differentiator — no seam handling in paint.rs/brush), fill layer projections, smudge/clone/symmetry/path tools (P1 wave-3 rows not started), eraser per-channel.** |
| 2 | Layers + channels | **Model done, engine partial** | LayerStack + 10 blend modes (contract wanted core 12: multiply/screen/overlay family present — count the HSV + passthrough family), masks, undo journal (Bessa-hardened), texture-set model, storage formats plumbed. **Missing: per-channel blending computed in linear space end-to-end, folder passthrough GPU compositing, UDIM (wave 4 as planned), instancing/anchors (wave 4).** |
| 3 | Bake engine | **Mostly done — the wave-3 story** | Six bakers (AO/position/world-normal/curvature/thickness/tangent-normal) + dilation (finite/infinite/fill) — 75 GPU tests on real adapters, deterministic, golden-verifiable. **Missing from the P0 row: ID bake (material/vertex/mesh), height bake, bent normals, high→low projection (cage/match-by-name/skew settings), antialiasing-supersampling, the bake-log UI. The current bakers are UV-space mesh-space bakes — the high→low transfer bakers are a distinct subsystem (wave-3 remainder or wave-4 re-scope).** |
| 4 | Procedural node graph | **Not started (wave 4 — as planned)** | umber-graph has DAG + topological sort only (the eval skeleton). Node set, MaterialX serialization, GPU graph eval all wave-4. |
| 5 | Viewport + shading | **Core done** | OpenPBR über-shader live with conformance tests, orbit camera, environment irradiance fn (procedural — NOT image-based IBL yet), 2D UV view, view modes. **Missing: HDR env-map IBL (P0 wave-1 row — the procedural stand-in shipped instead), wireframe/grid overlays, post-effects (P1/5).** |
| 6 | Export pipeline | **Done** (audit doc has the row detail) | Presets, driver, formats, dilation, dithering, flips, ICC. Remaining: driver token sources beyond $textureSet, 3D-neighbor padding, 8K (P1/5), PSD/MaterialX/USD (P2/later). |
| 7 | Import | **Done for P0s** | glTF/GLB/OBJ/FBX + png/exr read. Mesh-map import-by-convention: parser exists (fd58791), **not wired into the app UI**. USD (P1/5). |
| 8 | Project format | **Core done** | ProjectModel, per-layer JSON, content-hash asset store (blake3), round-trip determinism, Open/Save Project in the app. **Missing: textures actually routed through the asset store during paint sessions (the store exists; the paint pipeline writes tile-pool PNGs directly), autosave journal (P1/5).** |
| 9 | Color management | **Bridge done, display partial** | CM flags, per-output transfer rule, ocio stub-gated bridge, ICC embed. CPU sRGB display path is the reference. **Missing: real-OCIO build (manual dispatch), GPU LUT display (P0 wave-2 row — the fixed sRGB stands in), display-transform panel.** |
| 10 | Platform/input/CI | **CI done, stylus partial** | Dual-platform CI with lavapipe/WARP, headless CLI (bake/export — ahead of the wave-6 plan), profiling **NOT wired (tracy absent — a §10 P0 miss)**. Stylus: winink skeleton only (wave-1→2 progression incomplete — Wayland/XI2 not started, Wintab not started). |
| 11 | UI shell | **Shell done, density partial** | Dockable panels, layer stack, history window, bakes panel, export dialog, project I/O. **Missing: brush properties panel (§11 P0 — none found), assets/shelf, display settings, dark pro theme polish/4K pass.** |
| 12 | Performance budgets | **Unmeasured** | No docs/perf protocol, no timestamp instrumentation, no soak tests. The budget targets are written but nothing measures them. This is the audit's biggest honest gap: **the numbers exist only as targets.** |

## What this audit changes in the plan

The original wave tags assumed bake/export would consume wave-3 and
leave painting/UX for waves 1-2 already complete. The truth is more
honest: **the painting core (§1/§11 P0s) is the thinnest P0 surface in
the repo** — brush presets, seam-aware stamping, the brush properties
panel, and the high→low bake transfer are the real wave-4 entry
points, more urgent than the node graph that the original plan
slotted there.

## Renewed roadmap (supersedes the wave tags where they drifted)

**Wave 4 — Painting completeness (re-prioritized from node graph):**
1. Seam-aware stamping (§1 P0) — the UV-seam graph + 3D-neighbor
   dilation over seams; the paint.rs stamping path gains the seam
   lookup. This unblocks §6's 3D-neighbor padding too.
2. Brush presets (§1 P0) — native preset files + a starter library.
3. Brush properties panel (§11 P0) — input→curve editor, flow×opacity,
   stabilizer, jitter quartet.
4. High→low bake transfer (§3 P0 remainder) — cage, match-by-name,
   the height/normal-from-high-poly path.
5. ID bake + bent normals (the §3 P0 set completion).
6. Env-map IBL (the §5 P0 that shipped procedural-only).
7. Wireframe/grid overlays (§5 P1).
8. Perf protocol (§12) — docs/perf + timestamp instrumentation + the
   60fps/undo-RAM soak tests. Measure what we build.

**Wave 5 — Pro polish + procedural:** node graph engine (§4 — DAG
exists), UDIM, USD import, async bake/export jobs, display-transform
panel + real-OCIO build, post-effects, 8K/PSD.

**Wave 6 — Ecosystem:** plugins (WASM), installers, localization,
scripting API on the CLI, AI interface hook.

## Immediate next-session hunts (in order)

1. Seam-aware stamping design note → implementation (biggest P0 gap).
2. Brush presets + properties panel (paired UX work).
3. High→low bake brief for the claws.
4. Perf protocol skeleton (docs/perf + instrumentation plan).

Each has a natural claw-dispatch shape; the recipe (vendored
artifacts, excluded files, verify-from-artifacts) is in STATE.md.

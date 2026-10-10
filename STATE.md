# umber — State

> Updated every wave. This file is the bridge across context windows. Read this first on any new session before deciding what to do.

## Status line

Current wave: **3 of 6** (bake & export pipeline complete; painting core from Wave 2 landed through the brush/layer/undo/viewport surface)
State: WAVE-3-BAKE/EXPORT-COMPLETE
Last checkpoint: Wave 1 @ 905a389 (exit 0, clean — 2026-10-09 09:28 UTC)
Last green commit: 8b79e48 (README current; workspace 250+ tests green on both platforms)

**WAVE-1-ACTIVATION (2026-10-09 03:25):** Bessa's /loop instruction ("continue working autonomously... writing the code and expanding until 11am") explicitly authorizes implementation past the ratification gate. Wave 1 opened under loop authorization with SPEC v1.0 as the working contract — ratification still pending; any Bessa edits at ratification rescope per the spec-change rule. Full trail: DECISIONS.md [03:25].

**RENAME 2026-10-09 02:41:** Patina → **Umber** (name-collision sweep found live USPTO marks + in-category Patina® Mac paint app; claw recommended Umber — no software trademark, no in-category product, on-theme: raw earth pigment containing iron oxide). Repo: **github.com/victorbessa96/umber** (old URL 301-redirects). Executed under overnight grant; **Bessa can revert on wake** — full trail in DECISIONS.md [02:41].

**Overnight autonomy grant active** (Bessa asleep, granted 2026-10-09 02:15): take all decisions needed until wake. Green-light standing for everything except: (a) SPEC.md v1.0 remains DRAFT until Bessa ratifies, (b) .sbsar/.spp wall stays absolute, (c) license stays GPL-3.0-or-later.

## Active wave

**Name:** Wave 2 — Painting core (THE BIG ONE)
**Goal:** Full brush engine (libmypaint-documented-semantics dynamics, one-euro + lazy mouse); seam-aware UV-rasterization stamping + golden-image harness; 3D-space stroke evaluation; layer stack + masks + per-channel blending (core 12 blend modes); undo journal; OCIO viewport; OpenPBR über-shader + conformance suite; bundled brush presets; tile-pool memory architecture foundation; .umber project format with CPU-path deterministic round-trip; HDR image import; history window; Windows dual stylus path (Ink + Wintab); 2D UV view; wireframe/grid overlays.
**Entry state:** Wave 1 closed clean — see git log 342e8f1..905a389 and the checkpoint manifest.
**Exit criteria (from SPEC Wave 2):** every item above lands with CI-green gates + cross-review; golden-image harness running on lavapipe/WARP; project round-trip byte-identical in CI.

**Wave-1 progress (2026-10-09 03:25–05:57):**
- [x] 11-crate workspace scaffolded — builds clean, clippy 0 warnings, fmt clean (ba9fecc)
- [x] App binary verified live: window boots, wgpu enumerates Vulkan (Intel HD 530) + llvmpipe + GL adapters
- [x] CLI verified end-to-end: OBJ inspect (3 verts/1 tri/bounds correct)
- [x] CI workflow (ubuntu+windows matrix) + THIRD_PARTY.md license manifest (rfd + claw deps recorded)
- [x] glTF/GLB/FBX loaders landed (opencode claw + 14-finding cross-review, fa393d2); real binary-FBX fixture (ufbx upstream cube) un-ignored and green
- [x] wgpu viewport render pass landed (claude claw + 10-finding cross-review, 93e0140): OrbitCamera, WGSL normal-shaded pass, GpuContext/MeshBuffers/paint-callback, depth via NativeOptions::depth_buffer=32 (verified against vendored eframe source), 32B vertex pad, GPU tests 10/10 on real adapter
- [x] **CI GREEN BOTH PLATFORMS on 93e0140** (windows-latest ✅ + ubuntu-latest ✅) — toolchain pin 1.97.0 + platform-agnostic tests held
- [ ] IBL environment lighting (bundled env maps — viewport currently directional+ambient normal-shaded)
- [ ] Stylus crate Windows-Ink backend skeleton (event vocabulary done; WM_POINTER backend feature-gated = not started)
- [ ] Wave-1 checkpoint (dragon-checkpoint) once the two items above land

**Claw lessons learned this wave (both claws hit sandbox walls, both rescued the same way):** headless opencode cannot read the cargo registry (auto-rejects external_directory); headless claude cannot WebFetch/cargo-doc without pre-approved allowlists. Fix pattern that worked: vendor the exact dependency sources into docs/claw-artifacts/<crate>/ with a verified-API-facts README, and pass --allowedTools explicitly on claude dispatches. Both rescued dispatches delivered full working code + honest LANDING_NOTES.

## Handoff addendum (next loop iteration)

**Wave-3 bake & export surface is code-complete** (2026-10-09, ending 8b79e48): six mesh-map bakers (AO/position/world-normal/curvature/thickness/tangent-normal, 71 GPU tests), 8-neighbor UV dilation, four engine presets (glTF/Unreal/Unity/Blender, JSON-roundtripping), the two-pass export driver (no partial exports, per-output color/data transfer, DirectX normal flip), format writers (PNG 8/16, EXR 32F, TIFF, JPEG), the §9 ocio bridge (stub-gated: built-in ACES 2.0 config surface, 9/9 with feature / 6/6 default), ICC iCCP embed (bundled sRGB profile, no lcms2), the headless CLI (inspect / bake-ao / bake-all --dilate / export --preset), and the egui app (Bakes panel, Export dialog, .umber Open/Save Project). Workspace 250+ tests green, CI both platforms every commit. Cross-review catches this wave: Bessa's sRGB-on-unit-vectors dilate fix (b5a2e2d), the claw brief-error rejections (Chebyshev square, DirectX fixed-point), the driver's partial-export hole.

**Machine-load directive (2026-10-09 18:50, standing):** one cargo command at a time, no gate stacking, after stacked full-parallelism gates crashed Hermes. Claw dispatches must exclude files another session is editing.

**Wave-4 progress (2026-10-09 23:32, FINAL for the evening):** ALL EIGHT ITEMS DONE — the full audit-driven re-prioritization executed in one evening. (1) seam-aware stamping (fb984d1 + aca7f8d + 7106a79; the time_ns fix revived production painting), (2) brush presets (f18a0ce + 57ce764 + 0d49d83), (3) properties panel (d58bdb4), (4) high→low transfer (057f9cf + bb9ec90 + 2cb7b08 — §3 closed, 8 bakers), (5) ID bake (ab1dec5) + bent normals (60324a4), (6) env-map IBL (8065908 — §5's P0 closed), (7) wireframe + grid (09eb4e9), (8) perf instrumentation (e4925a1 — first baselines: 4K stroke soak zero drops, P95 15ms). Evening totals: 18 commits, 15 claw kills all cross-reviewed, 6 brief errors caught by claws, 1 CI lint caught+fixed (4dde836). Test counts: mesh 22→43, gpu 37→46, app 27→41, brush 25→39, bake 48→70 CPU, core 31. CI green both platforms through 8065908; 09eb4e9 queued. Next session: wave-5 entry (node graph on umber-graph's DAG, UDIM, USD, async jobs, display panel + real-OCIO, specular IBL tier, per-texel TBN) — all wave-5 items named in the audit; wave-4's remaining dependency-weighted notes (painted-maps bridge, token sources) fold in as listed in the delivery audit.

**WAVE-5 OPENED (2026-10-09 23:49):** the painted-maps export bridge LANDED (b1aa05a) — the delivery audit's remainder #1 + the token-sources row both closed: painted Base Color flows to export (decode-back can-fail test: the exported PNG MUST contain the painted pixels), the full TokenSources ($mesh/$layerName/$udim/$srcMap/$colorSpace) wired driver-side, the dialog shows its source honestly, the CLI path unchanged. Claw's corrections: PaintThread has NO background thread (same-thread readback is the honest pattern), the panel take()→as_deref_mut() fix so Export shares the live session. The app's core loop is WHOLE for the first time: paint → export → the PNG carries the paint. 16 claw kills tonight. CI on b1aa05a in flight at midnight — next session watches it green, then opens the node graph (wave-5's first designed hunt).

**Next items (RENEWED ROADMAP — full audit 2026-10-09 21:00, docs/specs/2026-10-09-full-project-audit.md):** the audit re-prioritized Wave 4 from the node graph to PAINTING COMPLETENESS — the thinnest P0 surface in the repo. Wave-4 order: (1) seam-aware stamping (§1 P0, unblocks §6 3D-neighbor padding), (2) brush presets (§1 P0), (3) brush properties panel (§11 P0), (4) high→low bake transfer (§3 P0 remainder: cage, match-by-name, height/normal-from-high), (5) ID bake + bent normals, (6) env-map IBL (§5 P0 shipped procedural-only), (7) wireframe/grid overlays, (8) the perf protocol (§12 — currently unmeasured targets; biggest honest gap). Wave 5: node graph (DAG exists in umber-graph), UDIM, USD, async jobs, display panel + real-OCIO. Wave 6: plugins/installers/i18n/CLI-scripting/AI hook. The prior wave-3 remainders (real-OCIO build, GPU display LUT, UV-derivative TBN, painted-maps bridge) fold into waves 4-5 as listed.

Claw dispatch recipe (proven repeatedly): vendor dependency sources into docs/claw-artifacts/<crate>/ + verified-API-facts README before dispatch; opencode gets the brief pointing at the artifacts; claude gets --allowedTools with cargo read/write allowlist. Cross-review: the OTHER claw's model reviews each landing. Never trust claw self-reports: verify from artifacts (fmt/clippy/test + behavioral probes) before committing.

## Blocked / waiting

None. Waiting only on background research claws (deleg_a2943965, 4 subagents).

## Recent decisions affecting now

See DECISIONS.md for all-time log. Last 3 inline:

- [01:41] UI = egui on wgpu + egui_dock, custom stylus input layer, custom node canvas
- [01:41] GPU = wgpu + WGSL, compute-first, naga_oil, hot-reload
- [01:40] Name = Umber; License = GPL-3.0-or-later (MIT for standalone crates); Wave 1 = vertical slice MVP

## Handoff note (what the next session must know)

**Wave 0 is functionally complete.** All 5 research reports in docs/research/ (01 Painter inventory, 02 competitors + name sweep, 03 Rust ecosystem, 04 next-gen P0/P1/P2, 05 brush-engine architecture). Consolidation docs written: docs/specs/requirements.md (12 sections, P0/P1/P2 + wave tags), tech-stack.md (crate table + 3 budget-to-own bets + workspace shape), architecture.md (threading model, paint data flow, VT, project format, bus-factor defense). SPEC.md is at **v1.0 DRAFT with a ratification block for Bessa** — DO NOT start Wave 1 coding until he ticks the block. Do not re-run research; it is done.

On Bessa's wake: (1) he reads SPEC.md ratification block, (2) any edits → apply → re-check, (3) ratified → dragon-checkpoint umber --wave 0, (4) Wave 1 opens (workspace skeleton via rust-workspace-greenfield skill; claws opencode/claude available for implementation with cross-review per standing rule). Rename Patina→Umber was executed overnight under autonomy grant — he may revert (DECISIONS.md [02:41] has the one-command path).

**Public repo: https://github.com/victorbessa96/umber** (old patina URL 301s). Community files, topics, discussions, wave-tracker issues #1–7 all live. Overnight autonomy grant (2026-10-09 02:15) remains active until Bessa wakes.

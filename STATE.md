# umber — State

> Updated every wave. This file is the bridge across context windows. Read this first on any new session before deciding what to do.

## Status line

Current wave: **2 of 6** (READY — Wave 1 closed at checkpoint 905a389, 2026-10-09 09:28 UTC)
State: WAVE-1-CLOSED / WAVE-2-READY
Last checkpoint: Wave 1 @ 905a389 (exit 0, clean — 2026-10-09 09:28 UTC)
Last green commit: 905a389 (both platforms green incl. winink tests on windows-latest)

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

Wave-1 vertical slice is code-complete through the GPU viewport: workspace + loaders + render pass all committed and CI-green on both platforms (93e0140). Remaining for Wave-1 exit: (1) IBL env-map lighting — the claw-artifacts pattern applies: vendor an HDR example or ship a procedural gradient env-light in shaders.rs first; (2) stylus Windows-Ink WM_POINTER backend skeleton behind the feature gate (event vocabulary already compiles). Then dragon-checkpoint umber --wave 1, and Wave 2 (painting core — the big one: brush engine, seam-aware stamping, layer stack, undo) opens per SPEC.

Claw dispatch recipe (proven twice this wave): vendor dependency sources into docs/claw-artifacts/<crate>/ + verified-API-facts README before dispatch; opencode gets the brief pointing at the artifacts; claude gets --allowedTools with cargo read/write allowlist. Cross-review: the OTHER claw's model reviews each landing (nemotron reviewed both this wave; findings were substantive both times — depth blocker, OBJ validation, GLB spec confirmation). Never trust claw self-reports: verify from artifacts (fmt/clippy/test + behavioral probes) before committing.

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

# umber — State

> Updated every wave. This file is the bridge across context windows. Read this first on any new session before deciding what to do.

## Status line

Current wave: **1 of 6** (IN-PROGRESS)
State: WAVE-1-ACTIVE — workspace skeleton + mesh import + PBR viewport
Last checkpoint: pending (Wave 0 checkpoint fires when Bessa ratifies SPEC v1.0)
Last green commit: fb29d7c (Wave 0 close-out)

**WAVE-1-ACTIVATION (2026-10-09 03:25):** Bessa's /loop instruction ("continue working autonomously... writing the code and expanding until 11am") explicitly authorizes implementation past the ratification gate. Wave 1 opened under loop authorization with SPEC v1.0 as the working contract — ratification still pending; any Bessa edits at ratification rescope per the spec-change rule. Full trail: DECISIONS.md [03:25].

**RENAME 2026-10-09 02:41:** Patina → **Umber** (name-collision sweep found live USPTO marks + in-category Patina® Mac paint app; claw recommended Umber — no software trademark, no in-category product, on-theme: raw earth pigment containing iron oxide). Repo: **github.com/victorbessa96/umber** (old URL 301-redirects). Executed under overnight grant; **Bessa can revert on wake** — full trail in DECISIONS.md [02:41].

**Overnight autonomy grant active** (Bessa asleep, granted 2026-10-09 02:15): take all decisions needed until wake. Green-light standing for everything except: (a) SPEC.md v1.0 remains DRAFT until Bessa ratifies, (b) .sbsar/.spp wall stays absolute, (c) license stays GPL-3.0-or-later.

## Active wave

**Name:** Wave 1 — Workspace skeleton + mesh import + PBR viewport
**Goal:** 11-crate workspace builds green on Linux+Windows CI; umber-app boots (eframe/wgpu + egui_dock shell); glTF/OBJ/FBX import in umber-mesh; basic PBR viewport with camera; THIRD_PARTY.md license manifest; stylus crate skeleton. Dev GPU note: Intel HD 530 + Vulkan 1.4 Mesa — correctness dev target only; 60fps reference-class validation happens on other hardware.
**Exit criteria (from SPEC Wave 1):**
- [ ] Workspace builds on Windows+Linux CI (fmt/build/test/clippy gates)
- [ ] App boots with dockable shell
- [ ] Loads glTF/OBJ/FBX
- [ ] IBL-lit viewport using bundled env maps, camera controls
- [ ] Stylus crate skeleton (Windows Ink path = feature-gated backend)
- [ ] Dependency license audit gate: THIRD_PARTY.md before first claw-authored merge
- [ ] Claws: opencode → umber-mesh importers; claude → umber-gpu + viewport integration; cross-review per standing rule

**Wave-1 progress (2026-10-09 03:25–04:30):**
- [x] 11-crate workspace scaffolded — builds clean, 12 tests green, clippy 0 warnings, fmt clean (commit ba9fecc)
- [x] App binary verified live: window boots, wgpu enumerates Vulkan (Intel HD 530) + llvmpipe + GL adapters
- [x] CLI verified end-to-end: OBJ inspect (3 verts/1 tri/bounds correct)
- [x] CI workflow (ubuntu+windows matrix) + THIRD_PARTY.md license manifest
- [ ] glTF/FBX loaders (opencode claw IN FLIGHT, proc_6f43c67f2557, result → /tmp/umber-opencode-result.txt)
- [ ] wgpu viewport render pass (claude claw IN FLIGHT, proc_372f90fdd7fc, result → /tmp/umber-claude-result.json)
- [ ] Claw integration + cross-review (next loop iteration)
- [ ] Push + CI green on GitHub

## Handoff addendum (next loop iteration)

Two claws in flight on the live tree (single-writer discipline: both were told do-not-commit; I integrate their diffs, run gates, cross-review, then commit). opencode owns crates/umber-mesh ONLY (glTF+FBX loaders + LANDING_NOTES.md); claude owns crates/umber-gpu + the Viewport panel in umber-app (wgpu render pass, orbit camera, LANDING_NOTES.md). On wake: (1) poll both processes; (2) read their LANDING_NOTES.md; (3) verify from artifacts — cargo fmt/clippy/test --workspace; (4) cross-review each claw's diff (opencode↔claude per standing rule); (5) commit + push; (6) check CI on GitHub (first real run — windows runner is unproven); (7) if CI green and time remains before 11am, next slice: stylus Windows-Ink backend skeleton or bake-crate compute-pass skeleton.

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

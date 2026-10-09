# umber — State

> Updated every wave. This file is the bridge across context windows. Read this first on any new session before deciding what to do.

## Status line

Current wave: **0 of 6** (IN-PROGRESS — consolidation complete, awaiting ratification)
State: WAVE-0-CONSOLIDATED — research corpus landed; SPEC v1.0 DRAFT packaged for Bessa
Last checkpoint: not yet (fires when Bessa ratifies SPEC v1.0)
Last green commit: see git log — docs/specs consolidation + report 05

**RENAME 2026-10-09 02:41:** Patina → **Umber** (name-collision sweep found live USPTO marks + in-category Patina® Mac paint app; claw recommended Umber — no software trademark, no in-category product, on-theme: raw earth pigment containing iron oxide). Repo: **github.com/victorbessa96/umber** (old URL 301-redirects). Executed under overnight grant; **Bessa can revert on wake** — full trail in DECISIONS.md [02:41].

**Overnight autonomy grant active** (Bessa asleep, granted 2026-10-09 02:15): take all decisions needed until wake. Green-light standing for everything except: (a) SPEC.md v1.0 remains DRAFT until Bessa ratifies, (b) .sbsar/.spp wall stays absolute, (c) license stays GPL-3.0-or-later.

## Active wave

**Name:** Wave 0 — Research + spec ratification
**Goal:** Five research claws (Substance Painter feature inventory, competitor autopsy incl. ArmorPaint + name-collision sweep, Rust graphics ecosystem stack, next-gen capability bar, brush-engine architecture) return; findings merge into docs/research/; adversarial review of the consolidation; SPEC.md goes DRAFT → v1.0 with Bessa review.
**Exit criteria:**
- [x] All five claw reports landed in docs/research/ (01–05)
- [x] Name-collision sweep executed — Patina→Umber rename done (DECISIONS.md [02:41])
- [x] Requirements + tech-stack + component-architecture documents written from claw findings
- [x] Adversarial review of SPEC + consolidation docs run; 26 findings reconciled, fixes applied (docs/research/06)
- [x] Roadmap validated against research
- [ ] Bessa reviews + approves SPEC.md v1.0

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

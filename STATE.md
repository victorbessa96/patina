# patina — State

> Updated every wave. This file is the bridge across context windows. Read this first on any new session before deciding what to do.

## Status line

Current wave: **0 of 6** (IN-PROGRESS)
State: WAVE-0-ACTIVE (research + spec v1.0)
Last checkpoint: never
Last green commit: pending first commit

## Active wave

**Name:** Wave 0 — Research + spec ratification
**Goal:** Four research claws (Substance Painter feature inventory, competitor autopsy incl. ArmorPaint + name-collision sweep, Rust graphics ecosystem stack, next-gen capability bar) return; findings merge into docs/research/; SPEC.md goes DRAFT → v1.0 with Bessa review.
**Exit criteria:**
- [ ] All four claw reports landed in docs/research/
- [ ] Name-collision sweep confirms Patina (or triggers rename decision)
- [ ] Requirements + tech-stack + component-architecture document written from claw findings
- [ ] Roadmap (waves 1-6) validated against research
- [ ] Bessa reviews + approves SPEC.md v1.0

## Blocked / waiting

None. Waiting only on background research claws (deleg_a2943965, 4 subagents).

## Recent decisions affecting now

See DECISIONS.md for all-time log. Last 3 inline:

- [01:41] UI = egui on wgpu + egui_dock, custom stylus input layer, custom node canvas
- [01:41] GPU = wgpu + WGSL, compute-first, naga_oil, hot-reload
- [01:40] Name = Patina; License = GPL-3.0-or-later (MIT for standalone crates); Wave 1 = vertical slice MVP

## Handoff note (what the next session must know)

Wave 0 in flight. Four research claws dispatched (deleg_a2943965): (1) Substance Painter exhaustive feature inventory, (2) competitor landscape incl. ArmorPaint deep-dive + Patina name-collision sweep, (3) Rust graphics/DCC crate stack survey, (4) next-gen capability bar (P0/P1/P2). When their results land: save each report to docs/research/<topic>.md, cross-check name collision, then write the consolidated REQUIREMENTS + TECH-STACK + COMPONENTS doc and bring SPEC.md DRAFT → v1.0 for Bessa review. Do NOT start Wave 1 coding — brainstorming skill hard-gate: no implementation before spec approval. Alignment decisions already locked in DECISIONS.md (name, license, wave-1 shape, GPU, UI).

**Public repo live: https://github.com/victorbessa96/patina** (origin, master, GPL-3.0 LICENSE + README + docs tree pushed 2026-10-09 ~02:15). Bessa authorized repo + full claw usage (opencode/claude at any moment, subagents at will) — logged in DECISIONS.md [02:10]. Standby for implementation once spec v1.0 ratified: claws will be used for Wave 1 (workspace skeleton via rust-workspace-greenfield skill), cross-reviewed per the standing rule.

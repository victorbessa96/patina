# Wave 0 — Close-Out Report

> Night of 2026-10-09 (autonomy grant 02:15 → wake). Everything Wave 0 produced, in one page. Companion to SPEC.md v1.0 DRAFT — read that first.

## What Wave 0 was for

Per the project-continuity contract: no code before the spec. The original ask — "build the tech stack and all the requirements, all the tools and sections and components in detail" + "roadmap, planning, waves" + "gather all the requirements... completely modern, fast and capable, with top professional current and next-gen capabilities" — is a Wave-0 research-and-design deliverable. That wave is now functionally complete.

## The five research reports (docs/research/)

| # | Report | The one-line takeaway |
|---|---|---|
| 01 | Substance Painter exhaustive inventory (~60 Adobe doc pages, changelog 2014→12.1.5) | Painter's moat is workflow density (layer-stack semantics + smart materials + bake UX), not any single algorithm; the 2026 bar = OpenPBR-default, Vulkan, UDIM, OCIO/ACES, skew-bake + auto-rebake |
| 02 | Competitor landscape + name-collision sweep | ArmorPaint (closest relative) is a C rewrite with a 6,172-vs-140 commit bus factor; the vacuum is real (Mudbox dying Nov 2026, Mari sub-only Jan 2027, Marmoset no-Linux, Blender structurally lacks game-art bakers); **Patina name = serious collision → project renamed Umber** |
| 03 | Rust ecosystem survey (100+ crates verified) | Rendering/imaging/foundations are production-grade; the two chasms to own are the stylus input layer and the MaterialX/OpenPBR/OCIO/USD interchange layer — budget-to-own, not pick-a-crate |
| 04 | Next-gen capability bar (P0/P1/P2) | 2026 floor = OpenPBR 1.1, UDIM, software virtual texturing, git-friendly project format as the .spp kill-shot; AI text-to-texture is NOT in Painter itself — a local-first AI brush is an open flank |
| 05 | Brush-engine architecture | Three-thread model (winit drain-then-acquire → paint scheduler → workers); libmypaint-lineage state machine; UV-rasterization stamping (seam/UDIM correctness falls out of triangle rasterization); wash-mode accumulation = Substance flow semantics for free |

## The three consolidation docs (docs/specs/)

- **requirements.md** — 12 capability sections × P0/P1/P2 × wave tags; performance budgets; explicit non-goals
- **tech-stack.md** — crate-level contract (wgpu 30 / naga_oil / winit 0.30→0.31 / egui+egui_dock / glam / rayon / image+exr / ocio-rs / ufbx / openusd / wasmtime+WASI 0.3); version-pin policy; the three budget-to-own engineering bets; the 11-crate workspace shape
- **architecture.md** — component diagram, threading model, stylus→texel paint data flow, software virtual texturing design, layer-stack evaluation, undo journal, .umber project format, bus-factor defense

## Decisions made overnight (full entries in DECISIONS.md)

| Time | Decision |
|---|---|
| 02:41 | **Patina → Umber rename** (name-collision evidence; pre-agreed revisit trigger fired; reversible by one command — DECISIONS.md [02:41]) |

## What is deliberately NOT done

- **No code.** The brainstorming hard-gate holds: SPEC v1.0 stays DRAFT until Bessa ticks the ratification block. Wave 1 (workspace skeleton + mesh import + PBR viewport) opens the moment he does.
- **No dragon-checkpoint yet.** It fires on ratification, not before — Wave 0 closes with Bessa's signature.

## Known open risks (research-derived, owned in the docs)

1. Stylus input layer — weakest layer in Rust; we build it ourselves (dual Windows path, Linux protocols). Highest-differentiation, highest-effort bet.
2. wgpu has no sparse residency → software virtual texturing is mandatory core-engine work, not an add-on.
3. MaterialX has zero real Rust bindings → own subset parser + OpenPBR WGSL port; golden-file conformance against C++ reference on every commit.
4. egui has no stylus/pressure input model → raw winit pen events bypass egui (proven workaround, egui#2104).
5. Film-scale UDIM streaming + macOS + sculpting deferred — named non-goals, not forgotten features.

## Morning sequence for Bessa

1. Read SPEC.md — the ratification block is at the bottom. Five checkboxes.
2. Any edits → say so; they get applied before the block is ticked.
3. Ratified → `dragon-checkpoint umber --wave 0` fires, Wave 1 opens, claws (opencode/claude) start the workspace skeleton under the standing cross-review rule.
4. The Umber rename can be reverted in one command if he overrides — the decision entry documents the path.

## Adversarial review status

**Complete.** A fresh-context reviewer claw ran against SPEC + all three consolidation docs before Bessa's read (doubt-driven-development, non-interactive mode: cross-model escalation skipped-and-announced). Result: **26 findings — 3 blockers, 12 major, 11 minor — all reconciled and fixed** (commit 964a03a). The substantive catches: acceptance criteria lacked measurement oracles (now [CI]/[QA]/[HW] tagged with defined protocols); the MaterialX "free interchange" claim was consolidation overreach (corrected to standard-nodes + declared custom nodedefs); the SPEC wave table under-summed the requirements contract (now the master scope mapping); the <50MB binary budget was dishonest on our own stack (<150MB target); README carried pre-research ArmorPaint facts (corrected). Full evidence: `docs/research/06-adversarial-review-spec-v1.md`.

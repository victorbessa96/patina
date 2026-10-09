# patina — Decisions

> Alignment-session + mid-execution decisions, append-only. Never edit history; supersede by adding a new entry that names the old one.

Format per entry:

```
## [YYYY-MM-DD HH:MM] <short tag>
Decision: <what we decided>
Context: <the question / ambiguity / trade-off>
Options rejected: <list, one line each>
Owner: <who made the call — usually "Bessa + Razul" or "Razul under delegation X">
Revisit when: <condition that would reopen this>
```

---

(Entry log below — newest at bottom)

## [2026-10-09 01:40] project-name
Decision: Project name is **Patina**.
Context: Brand must be short, professional, evoke the rust/pigment metaphor (iron oxide = the oldest pigment family in human art). "Patina" = the story a surface tells over time — exactly what this tool authors.
Options rejected: Hematite, Umber, Impasto (all viable; name-collision sweep running in parallel to confirm Patina is clean).
Owner: Bessa + Razul
Revisit when: research claw finds a serious name collision or trademark holder.

## [2026-10-09 01:40] license
Decision: **GPL-3.0-or-later** for the application (Blender/Krita/GIMP community-DCC model). Standalone utility crates extracted later may be MIT.
Context: Bessa said "Either A or MIT" (A = GPL-3.0-or-later). Every thriving community DCC is GPL: it grows contributors, blocks proprietary enclosure, and corporations still use and contribute freely. MIT/Apache remain available for any standalone library crates we spin out.
Options rejected: Apache-2.0 (permits enclosure, weakens copyleft ecosystem), MPL-2.0 (file-level copyleft too weak for an app), dual-licensing (complexity with no community-DCC precedent).
Owner: Bessa + Razul
Revisit when: a corporate partner demands a permissive exception for a standalone crate.

## [2026-10-09 01:40] wave1-shape
Decision: Wave 1 = **vertical-slice MVP**: import mesh → paint layer stacks in real 3D viewport → export PBR maps. Exercises every architectural pillar (GPU engine, UI shell, formats, undo) with smallest surface.
Context: Full Painter parity is multi-year; first usable milestone must prove the architecture end-to-end.
Options rejected: baking-first CLI (defers the hard UI/viewport bets), feature-parity crash program (generators+smart materials+baking all in v1 = unshippable scope).
Owner: Bessa + Razul
Revisit when: vertical slice proves architecture and Wave 2+ scope gets ratified.

## [2026-10-09 01:41] gpu-backend
Decision: **wgpu + WGSL** (Vulkan/DX12/Metal backends), compute-first for painting/baking, naga_oil for shader composition, hot-reload during dev.
Context: One API across target platforms, mature in 2026, avoids doubling maintenance of Vulkan-native (ash) for years of zero benefit.
Options rejected: wgpu + GL fallback tier (decision deferred, not rejected — old-hardware tier stays open), Vulkan-first via ash (doubles maintenance, buys nothing short-term).
Owner: Bessa (sovereignty: accepted Razul's lead recommendation)
Revisit when: wgpu blocks a hard requirement (e.g. hardware RT timeline) — then backend abstraction gets revisited.

## [2026-10-09 02:10] repo-and-claws
Decision: Public GitHub repo **victorbessa96/patina** (origin), created under Bessa's explicit in-message authorization (2026-10-09 02:08: "feel free to have a project github repository"); claw roster (opencode default + claude deliberate third) usable at any moment under the standing cross-review rule; subagents at will.
Context: Standing grants reconfirmed mid-Wave-0. Upstream-PR gate remains (present before posting to public repos); the project's OWN repo is explicitly authorized — the gate doesn't apply to patina's own repo.
Options rejected: private repo until v0.1 (public-by-default builds in the open from day one, matches OSS-first intent); solo execution (claws parallelize research + later cross-review implementation).
Owner: Bessa (authorization) + Razul (execution)
Revisit when: never — this is a standing-grant confirmation, not a scoped call.

## [2026-10-09 01:41] ui-framework
Decision: **egui on wgpu + egui_dock**, custom input layer for stylus/pressure feeding custom events, purpose-built node-canvas widget for the procedural graph editor.
Context: Bessa asked for a recommendation instead of picking. Immediate-mode gives the fastest tool-iteration loop in the Rust ecosystem; egui_dock provides Painter-style dockable panel trees; renders on our own wgpu surface (one GPU stack for UI + viewport). Known weaknesses (stylus pressure, dense node-graph editing) are solved with custom input layer + custom node canvas — work we'd own under any toolkit. Ecosystem claw is pressure-testing this integration story; dealbreaker findings will reopen this.
Options rejected: Iced (slower to build complex dock UIs), GPUI (young, opinionated, Zed-internal defaults), fully custom retained stack (years of cost before shipping).
Owner: Razul under delegation; Bessa ratified by accepting recommendation
Revisit when: ecosystem research surfaces a hard egui blocker (e.g. per-tablet-pixel pressure latency structurally unachievable) or node-canvas work proves unbounded.

# umber — Decisions

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
Revisit when: research claw finds a serious name collision or trademark holder. *(Fired 2026-10-09 02:41 — see [02:41] project-rename.)*

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

## [2026-10-09 01:41] ui-framework
Decision: **egui on wgpu + egui_dock**, custom input layer for stylus/pressure feeding custom events, purpose-built node-canvas widget for the procedural graph editor.
Context: Bessa asked for a recommendation instead of picking. Immediate-mode gives the fastest tool-iteration loop in the Rust ecosystem; egui_dock provides Painter-style dockable panel trees; renders on our own wgpu surface (one GPU stack for UI + viewport). Known weaknesses (stylus pressure, dense node-graph editing) are solved with custom input layer + custom node canvas — work we'd own under any toolkit. Ecosystem claw is pressure-testing this integration story; dealbreaker findings will reopen this.
Options rejected: Iced (slower to build complex dock UIs), GPUI (young, opinionated, Zed-internal defaults), fully custom retained stack (years of cost before shipping).
Owner: Razul under delegation; Bessa ratified by accepting recommendation
Revisit when: ecosystem research surfaces a hard egui blocker (e.g. per-tablet-pixel pressure latency structurally unachievable) or node-canvas work proves unbounded.

## [2026-10-09 02:10] repo-and-claws
Decision: Public GitHub repo **victorbessa96/patina** (origin), created under Bessa's explicit in-message authorization (2026-10-09 02:08: "feel free to have a project github repository"); claw roster (opencode default + claude deliberate third) usable at any moment under the standing cross-review rule; subagents at will.
Context: Standing grants reconfirmed mid-Wave-0. Upstream-PR gate remains (present before posting to public repos); the project's OWN repo is explicitly authorized — the gate doesn't apply to patina's own repo.
Options rejected: private repo until v0.1 (public-by-default builds in the open from day one, matches OSS-first intent); solo execution (claws parallelize research + later cross-review implementation).
Owner: Bessa (authorization) + Razul (execution)
Revisit when: never — this is a standing-grant confirmation, not a scoped call. *(Repo renamed with the project — see [02:41] project-rename.)*

## [2026-10-09 02:41] project-rename
Decision: Project renamed **Patina → Umber** (repo, directory ~/Projects/umber, docs, brand). Supersedes [01:40] project-name.
Context: The [01:40] decision pre-committed the revisit trigger ("research claw finds a serious name collision or trademark holder"), and the competitor claw's name-collision sweep fired it: Patina® is an established Mac drawing/paint app (patinaapp.com) in the same broad category, with live USPTO marks (Patina LLC Reg #7213631, The Patina Group Reg #2375467) and a crowded GitHub namespace. Umber was one of the four names on Bessa's original ballot, and the claw independently recommended it sight-unseen: no in-category product, no active software trademark, on-theme (raw earth pigment containing iron oxide — literally rust-colored paint), short, spellable. Executed under the 2026-10-09 02:15 overnight-autonomy grant ("take all the decisions needed").
Options rejected: Impasto (runner-up; generic-term trademark weakness, and a 310★ active Linux desktop shell shares the name), Hematite (1,920★ canonical Rust Minecraft clone + crates.io collision), keeping Patina (legal exposure + brand confusion in-category from day one).
Owner: Razul under overnight-autonomy delegation; **reversible by Bessa on wake — one `gh repo rename` + git mv if he overrides.**
Revisit when: Bessa overrides on wake (pre-agreed revert path), or a Umber trademark/collision surfaces before 0.1 branding investment begins.

## [2026-10-09 03:25] wave1-loop-activation
Decision: Wave 1 implementation opened under Bessa's /loop instruction (03:23: "continue working autonomously on the project and writing the code and expanding until 11am... use Claude code and opencode... plan and implement the next steps"), with **SPEC v1.0 as the working contract pending ratification**. Claw usage reconfirmed: opencode + claude dispatched with cross-review.
Context: The brainstorming hard-gate said "no implementation before spec approval," but Bessa's explicit loop instruction to write code until 11am supersedes the wait. Ratification remains the Wave-0 close gate; any edits he makes at ratification rescope Wave 1 work per the spec-change rule (name it, fix the spec, then the work, in that order).
Options rejected: continue waiting for ratification (contradicts explicit loop instruction); treating SPEC as fully ratified (it is not — DRAFT-as-working-contract is the honest middle).
Owner: Bessa (loop instruction) + Razul (execution)
Revisit when: Bessa wakes and either ratifies (Wave 0 checkpoint fires) or edits (rescope).

## [2026-10-09 19:20] icc-embed-via-png-crate
Decision: ICC profile embed (§9 P1) ships via the png crate's native iCCP chunk write + a bundled 2576-byte sRGB ICC constant. **lcms2 dropped from the dependency plan entirely.**
Context: requirements.md §9 named lcms2 for the embed, but the need is embedding, not a CMS — the png crate (already a dependency) reads and writes iCCP natively (Info::icc_profile), verified against its 0.17.16 source. lcms2 would add a C dependency for a byte-chunk write the existing stack already performs. The bundled profile was validated (acsp signature, header-size match) before checking in.
Options rejected: lcms2 (unnecessary C dep for a solved problem); icc-profile crate (a full pure-Rust CMS — heavier than the need); shipping profile-less PNGs (fails §9 P1).
Owner: Razul
Revisit when: a non-sRGB output color space needs embedding (then the profile becomes a parameter, still no lcms2).

## [2026-10-09 19:30] ocio-stub-gated-bridge
Decision: ocio-rs lands as an optional stub-gated feature (umber-color's `ocio` feature, default OFF); the vendored/bundled OCIO C++ build is a manual-dispatch job, never in the per-commit path.
Context: the §9 survey (docs/research/ocio-aces-integration.md) found ocio-rs 0.2.1 targets OCIO v2.5.2 — built-in ACES 2.0 CG/Studio configs, so no config files need vendoring. The crate's stub mode is CI-safe (compiles, safe defaults, clean errors) while the bundled mode is a minutes-long C++ build. The machine-load directive (one command at a time, no stacking) makes the heavy build a deliberate act. Stub-mode probe confirmed all three API surfaces (create_from_builtin_config, BuiltinConfigRegistry, GpuShaderDesc::create).
Options rejected: bundled-in-CI (multi-minute C++ compile per commit — violates the load directive); pre-installed system OCIO (packagers' path, unportable); no ocio at all (§9 P0 requires it).
Owner: Razul
Revisit when: the display-transform panel ships — then the bundled build gets its one-time manual run + golden tests against the CPU reference path.

# Umber — Spec

> Written ONCE at project start during the alignment session. Changed only by explicit Bessa+Razul respec, not by execution drift. Spec drift is a bug — name it, fix the spec, then the work, in that order.
>
> Status: DRAFT v0.1 (2026-10-09) — alignment locked, research in flight. v1.0 lands after research claws return + Bessa review.

## One-sentence goal

An open-source, GPL-3.0-or-later, Rust-native 3D texture-painting DCC (Substance Painter class) for Windows and Linux — modern, fast, community-driven, with professional current and next-gen capabilities.

## Non-goals

What this project will NOT do. (Guardrails against scope creep.)

- No mesh modeling/editing (this is a painter, not a modeler — Blender exists)
- No sculpting (v1 horizon; revisit only after painting core is rock-solid)
- No Adobe format execution: we do not load or execute .sbsar/.spp — an original procedural node graph engine ships instead
- No macOS support in the v1 horizon (Windows + Linux are the contract; macOS later if contributors arrive)
- No cloud/account requirement — local-first always
- No subscription, telemetry, or license servers

## Deliverables

What exists at the end that does not exist today.

1. A paint-capable desktop application (Windows + Linux): import mesh → paint layer stacks in a 3D viewport → export PBR texture maps.
2. A crate-structured Rust workspace: render engine, painting core, UI shell, format I/O, procedural graph engine, baking engine — each independently reusable.
3. An original procedural node graph engine (generators, filters, masks, smart materials) replacing the .sbsar ecosystem.
4. A bake engine: AO, normal-from-mesh, curvature, position, thickness, ID maps with cage controls.
5. A git-friendly open project format (.umber) — diffable, mergeable alternative to .spp.
6. Plugin system (Rust-native + sandboxed WASM/WASI runtime) + headless CLI for automation.
7. Reproducible CI on GitHub Actions building Windows + Linux releases.

## Acceptance criteria

Testable statements. Each one observable, not vibes.

- [ ] App opens a glTF/OBJ/FBX mesh, renders PBR-lit in real time at 60fps on a mid-range GPU
- [ ] Brush strokes with stylus pressure land < 16.7ms paint-to-photon on Vulkan/DX12
- [ ] Layer stack (paint + fill layers, masks, blending modes, per-channel control) paints and exports correctly
- [ ] Export produces engine-correct maps (Unity/Unreal/glTF presets: channel packing, normal conventions, bit depth)
- [ ] Bake engine produces AO/normal/curvature maps on a test mesh, verifiable against golden images (tolerance-checked)
- [ ] Project round-trips: save → close → open → identical rendered + exported output (deterministic serialization)
- round-trips + bake golden images run headless in CI on both OSes
- [ ] Undo/redo survives full painting sessions without history corruption
- [ ] OpenPBR-style PBR shading + OCIO color management in viewport and export
- [ ] Plugin sandbox loads a sample WASM plugin and executes it under resource limits

## Waves

Ordered execution plan. Wave N must pass its exit criteria before Wave N+1 starts its main work.

| Wave | Goal | Exit criteria |
|---|---|---|
| 0 | Research + spec v1.0 (in flight) | This SPEC.md approved by Bessa; feature inventory, competitor analysis, ecosystem stack, next-gen capability list merged into docs/ |
| 1 | Vertical-slice MVP: workspace skeleton + mesh import + PBR viewport | App boots (wgpu+egui), loads glTF/OBJ/FBX, PBR-lit viewport at 60fps, camera controls, layer-stack UI shell exists |
| 2 | Painting core: brush engine + layer stack + undo | Pressure-sensitive painting on mesh surface, stroke compositing into layers, full undo/redo, save/load project |
| 3 | Export + bake: map export presets, bake engine | Engine-correct export (Unity/Unreal/glTF), AO+normal+curvature bake with golden-image CI, project round-trip |
| 4 | Procedural graph engine: nodes, generators, filters, smart materials | Original node engine executing in viewport, smart materials apply non-destructively, graph saved/loaded |
| 5 | Pro polish: tablet pressure pipeline end-to-end, dockable UI maturity, theming | Painter-grade workflow density; external artist dogfood on real assets |
| 6 | Ecosystem: plugin runtime, headless CLI, docs site, 0.1 release | GitHub releases for Windows+Linux, sample plugins, contributor docs |

## Constraints

Hard limits the work must respect (cost caps, forbidden libraries, runtime boundaries).

- Rust-only core; FFI only where no viable Rust crate exists (OIDN, OCIO, ufbx) and license-compatible
- No Adobe SDKs, no .sbsar/.spp execution ever (legal wall)
- GPL-3.0-or-later for the application; standalone utility crates may be MIT
- Target hardware floor: Vulkan 1.3 / DX12 on Windows; Vulkan 1.3 on Linux (GL fallback tier deferred)
- No network-required features; local-first
- Dev machine: 8 cores / 16GB (this box) — CI must build with public runners
- Scope cap for v0.1: no sculpting, no macOS, no film-UDIM (single-tile UV sets first)

## Sources of truth

If docs/code conflict with this spec during execution, this list is what wins.

1. This SPEC.md (contract) — ratified only by Bessa+Razul respec
2. Adobe Substance 3D Painter official docs + release notes (feature-reference bar)
3. OpenPBR specification (material model)
4. glTF 2.0 spec + Unity/Unreal texture conventions (export correctness)
5. wgpu / egui upstream docs (engine + UI reality)

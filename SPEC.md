# Umber — Spec v1.0 (DRAFT — awaiting Bessa's ratification)

> The project contract. v0.1 draft written at project start; this v1.0 draft folds in the Wave-0 research corpus (docs/research/01–05) and the adversarial-review fixes (docs/research/06). **Ratified only when Bessa approves.** Until then, no implementation work beyond what this spec names as Wave-0.

## One-sentence goal

**Umber** — an open-source, GPL-3.0-or-later, Rust-native 3D texture-painting DCC (Substance Painter class) for Windows and Linux: modern, fast, local-first, community-driven, with professional current and next-gen capabilities.

## Non-goals

What this project will NOT do. (Guardrails against scope creep.)

- No mesh modeling/editing (this is a painter, not a modeler — Blender exists)
- No sculpting in the v0.1 horizon
- **No Adobe format execution, ever:** .sbsar runs only under Adobe's closed Materials SDK terms — bundling it in an OSS app is legally unresolved at best (docs/research/01 §4) and violates local-first principles regardless; .spp is undocumented HDF5. Umber ships an **original node-graph engine** whose graphs serialize as **MaterialX documents with declared custom nodedefs** for painter-specific nodes.
- No cloud/account requirement — local-first always; no telemetry
- No subscription — free, GPL, community-owned
- No macOS in the v0.1 horizon (Windows + Linux is the contract; macOS when contributors arrive)
- No film-scale UDIM streaming (100+ tiles) and no >4K in-app painting in v0.1 — the tile-pool architecture is built in v0.1; streaming-scale operation is post-0.1
- No in-app generative AI in v0.1 — a local-first AI brush is a P2 research track (docs/research/04 §13); the plugin/provider interface is the design hook, delivery is post-0.1

## Deliverables

What exists at the end (v0.1) that does not exist today.

1. A paint-capable desktop application (Windows 10/11 + Linux): import mesh → paint layered PBR materials on it in a real-time viewport → export engine-ready texture maps.
2. A crate-structured Rust workspace (see docs/specs/tech-stack.md §workspace shape) — each crate builds, tests, and clippy-gates independently in CI on both OSes.
3. An original procedural node-graph engine (generators, filters, masks, smart materials). Graphs serialize as **.mtlx documents**: nodes with MaterialX standard-node equivalents map to them; painter-specific nodes ship as declared custom nodedefs (consuming DCCs read the document and evaluate what they support — no automatic full-fidelity interchange is claimed).
4. A bake engine: AO, normal (mesh + high→low), curvature, position, thickness, ID maps with cage controls.
5. A git-friendly open project format (`.umber` directory) — diffable, mergeable, deterministic round-trip.
6. A headless CLI (bake/export/batch automation) + sandboxed plugin runtime (wasmtime + WASI 0.3 components; native Rust dynlib plugins are deferred — dev-time hot-reload only).
7. Reproducible CI on GitHub Actions building Windows + Linux releases, with headless GPU tests via lavapipe (Linux) and WARP (Windows) plus golden-image bake/paint verification.

## Acceptance criteria

Testable statements. Each names its oracle. **[CI]** = enforced in CI on both OSes; **[QA]** = manual verification with a defined protocol; **[HW]** = hardware-dependent, validated on the reference GPU class (one named mid-range GPU — e.g. RTX 3060 / RX 6600 class, current driver — recorded in docs/perf when the harness lands).

- [ ] [CI] Workspace: build + tests + clippy green on windows-latest and ubuntu
- [ ] [CI] Import: glTF/OBJ/FBX sample meshes load with correct vertex/UV counts vs known-good references
- [ ] [HW] Viewport: 60fps sustained (frame-time P95 < 16.7ms) on the reference GPU class with the standard benchmark scene (1M-tri mesh, 4K texture set, bundled HDRI) — measured with built-in tracy frame captures; protocol recorded in docs/perf
- [ ] [QA] Stylus: input-to-photon (stylus event → present, timestamp instrumentation) < 20ms at 60Hz on the reference rig; protocol in docs/perf. Budget note: at 60Hz, scanout alone consumes up to ~16.7ms — if measurement shows >20ms, the criterion is renegotiated with data, not vibes
- [ ] [CI] Stroke across a UV seam lands consistently on both islands — golden-image test vs committed reference images (tolerance comparison)
- [ ] [CI] Layer stack: paint + fill layers, masks, per-channel blending produce golden-image-correct composites on fixed test projects
- [ ] [CI] Export: 4 engine presets (glTF metal-rough, Unreal packed/DX, Unity GL, Blender Principled) produce byte-identical output given identical decoded tiles; channel packing, naming tokens, PNG 8/16 + EXR 16F/32F verified against committed fixtures
- [ ] [CI] Bake: AO/normal/curvature/position/ID on the test mesh match committed golden images within tolerance (lavapipe on Linux, WARP on Windows)
- [ ] [CI] Project round-trip: save → load → save is byte-identical for the serialization layer (CPU path, both OSes); export-after-reload is byte-identical given the CPU decode path. Cross-GPU parity is golden-image tolerance, not bit-exact — named so nobody promises bit-exact across vendors
- [ ] [CI] Undo: scripted 100-step soak test at 4K stays within the ~2GB budget and time-travels to the correct state at every step
- [ ] [QA] OpenPBR viewport: Umber's WGSL über-shader compared against the MaterialX reference renderer on a fixed material set, tolerance defined in the conformance suite doc — the suite is built as part of Wave 2; it is scope, and it is named
- [ ] [CI] OCIO: display transforms verified in golden-image viewport captures; per-channel CM flags respected in export metadata
- [ ] [CI] Plugins: sample WASM component loads and executes under resource limits (fuel + memory)
- [ ] [CI] CLI: headless bake+export on a fixed project produces outputs identical to the GUI path
- [ ] [CI] Release: tag → CI produces Windows MSI + Linux Flatpak manifest artifacts + AppImage on both runners

## Waves

Ordered scope milestones. Waves carry no calendar commitment — velocity is unknowable in advance (solo human + agents + future contributors); size honesty: **Waves 2 and 4 are the big ones**. Research 02's ArmorPaint evidence (one maintainer, 2–3 years per point release) is priced in by scope-gating, not by promising dates. Wave N must pass its exit criteria before Wave N+1 starts its main work. **This table is the master scope mapping** — every requirements.md item appears in exactly one wave; items not listed here are not in v0.1.

| Wave | Goal | Exit criteria (all [CI] items run in CI) |
|---|---|---|
| 0 | Research + spec ratification | 5 research reports landed ✓; consolidation docs ✓; adversarial review + fixes ✓; **Bessa ratifies SPEC v1.0** (pending) |
| 1 | Workspace skeleton + mesh import + PBR viewport | Workspace builds on Windows+Linux CI; app boots (wgpu+egui+winit), loads glTF/OBJ/FBX, IBL-lit viewport using bundled env maps (HDR file import lands W2), camera controls, dockable UI shell; **stylus risk callout: first backend of the owned `stylus` crate = raw Windows Ink WM_POINTER path** (winit 0.30 has no pen events — we bypass it; winit 0.31 pen events adopted when stable); **dependency license audit gate: THIRD_PARTY.md manifest before first merge** |
| 2 | Painting core | Brush engine (libmypaint-documented-semantics dynamics, one-euro + lazy mouse); seam-aware UV-rasterization stamping + golden-image harness; 3D-space stroke evaluation (tile-count-independent — works single-tile); layer stack + masks + per-channel blending (core 12 blend modes); undo journal; OCIO viewport; OpenPBR über-shader + conformance suite; bundled brush presets; **tile-pool memory architecture foundation**; .umber project format with CPU-path deterministic round-trip; HDR image import (env maps + bitmaps); history window; Windows dual stylus path (Ink + Wintab); 2D UV view; wireframe/grid overlays |
| 3 | Export + bake + fill layer | Template/preset export (4 engine presets, packing, naming tokens, PNG/EXR); bake engine (AO/normal/curvature/position/ID) with golden-image CI + cage controls + mesh-map import by naming convention; **fill layers with fill projections (UV/tri-planar/planar/spherical/cylindrical)**; smudge + clone; straight-line/angle snap; **tile-pool bake-target integration** |
| 4 | Procedural graph engine + UDIM | Node engine executing on the tile scheduler; **graphs save/load as .mtlx with declared custom nodedefs** (round-trip + re-evaluation identity in CI); ~40 core nodes; generators driven by baked mesh maps; smart materials + smart masks; **UDIM multi-tile texture sets + cross-tile painting + per-tile resolution**; layer instancing + anchors; full blending-mode set (~32); texture-synthesis generator primitive; custom node-canvas editor UI; symmetry suite (mirror + radial); polygon fill; quick mask |
| 5 | Pro polish | Linux X11 stylus + hardware test matrix (Wacom/Huion/XP-Pen); bake skew painting + auto-rebake; path tools; post-effects stack; HDR display output; 8K export (up-sampled from 4K in-app); autosave + crash recovery; theming; localization architecture; **dogfood gate: ≥2 external artists each complete a full asset start-to-finish, feedback logged as issues** |
| 6 | Ecosystem + v0.1 | wasmtime/WASI plugin runtime + sample plugins (exporter + generator); headless CLI (if not already pulled forward); docs site (user guide covers install → first export); cargo-dist release pipeline; **Windows MSI + Flatpak manifest + AppImage artifacts**; v0.1 tagged |

## Constraints

Hard limits the work must respect.

- Rust-only core; FFI only where no viable Rust crate exists (OCIO via ocio-rs vendored, ufbx for FBX) and license-compatible
- **Dependency license audit is a Wave-1 merge gate:** every dependency's license recorded in a THIRD_PARTY.md manifest before first merge to master; standalone-crate MIT relicensing requires the upstream license to permit it (verified per crate — e.g. octotablet absorption happens only after its license is confirmed compatible); the libmypaint relationship is a documented-semantics reimplementation, never a code derivative
- No Adobe SDKs, no .sbsar/.spp execution ever
- GPL-3.0-or-later for the application; standalone utility crates may be MIT
- Hardware floor: Vulkan 1.3 / DX12 (GL fallback tier not built in v0.1)
- No network-required features; local-first
- Dev machine 8 cores / 16GB; CI on public GitHub Actions runners
- No experimental wgpu features in the core paint path (RT/mesh-shaders/bindless are feature-gated only)
- wgpu+naga+naga_oil version trio pinned together; winit pinned 0.30 until 0.31 stable
- Installer budget: target < 150 MB (wasmtime + vendored OCIO + compression codecs make smaller numbers dishonest — size is tracked per release, not gated); no runtime deps beyond the installer contents

## Risks (the honest section)

The three budget-to-own engineering bets (docs/specs/tech-stack.md §(c)) and their failure modes:

1. **Stylus input layer** (highest differentiation, highest effort): no production-grade Rust crate exists; we build dual Windows paths + Linux protocols ourselves. *Failure mode:* latency/compat gaps on some hardware. *Fallback:* Windows Ink single path first, Wintab behind a runtime switch; hardware matrix grows in W5. *Kill criterion:* if two release cycles pass without the owned crate reaching parity with Krita's tablet stack on top-3 vendors, reconsider the FFI route (SDL3 pen API).
2. **Painter-class texture streaming on wgpu** (main performance bet): no sparse residency in wgpu → software page-pool. *Failure mode:* eviction/upload contention under paint traffic at scale. *Fallback:* `as_hal` Vulkan escape hatch for affected paths. *Scope honesty:* v0.1 ships the tile-pool foundation; >4K streaming operation is post-0.1.
3. **Interchange/material layer solo** (MaterialX subset parser + OpenPBR WGSL + ocio-rs + openusd): zero viable MaterialX bindings exist. *Failure mode:* conformance drift vs the C++ reference. *Mitigation:* golden-file conformance on every commit; LUT fallback if ocio-rs rots. *Kill criterion:* if MaterialX Rust bindings appear upstream, evaluate adoption instead of maintaining our parser.

Secondary named risks: egui pressure gap (bypass proven — egui#2104; revisit only if per-pixel latency fails), naga_oil lockstep (pin trio; vendor locally if it lags), ocio-rs single maintainer (vendored + LUT fallback), openusd pre-1.0 (scope to import + UsdShade export; pxr FFI escape hatch).

## Sustainability (research 02's survival criteria, answered honestly)

Prior OSS texture tools died of: full-DCC scope + single maintainer + no funding + "Blender already exists" gravity. Current Umber reality: **Bessa + Razul + claws** — a human bus factor of one. Defenses that exist: narrow scope (this SPEC's non-goals), engine-on-GPU from day one (W1 architecture), crate boundaries + DECISIONS.md as institutional memory, headless-testable core. What is NOT solved, named plainly: **funding and a second human maintainer.** Plan: recruit from the first dogfood cohort (W5), Open Collective / GitHub Sponsors post-0.1 (Material Maker's Patreon precedent funds adjacent OSS today), plugin API (W6) as contributor on-ramp. This is the program's biggest non-technical risk and it is written here so the ratifier sees it.

## Sources of truth

If docs/code conflict with this spec during execution, this list is what wins.

1. This SPEC.md (contract) — changes only by explicit Bessa decision, recorded in DECISIONS.md
2. docs/specs/requirements.md + tech-stack.md + architecture.md (Wave-0 consolidation, research-derived)
3. Adobe Substance 3D Painter official docs + release notes (feature-reference bar — docs/research/01)
4. OpenPBR 1.1 spec (ASWF) + MaterialX spec (material model + graph interchange)
5. docs/research/03 (crate maturity verdicts) for any dependency question
6. glTF 2.0 spec + Unity/Unreal texture conventions (export correctness)

---

*Ratification block (Bessa — morning):*

- [ ] One-sentence goal — approved / edits:
- [ ] Non-goals — approved / edits:
- [ ] Deliverables + acceptance criteria (with oracles) — approved / edits:
- [ ] Wave plan (master scope mapping) — approved / edits:
- [ ] Constraints + Risks + Sustainability — approved / edits:
- [ ] **SPEC v1.0 RATIFIED** (all five above)

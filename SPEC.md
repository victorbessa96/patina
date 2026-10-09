# Umber — Spec v1.0 (DRAFT — awaiting Bessa's ratification)

> The project contract. v0.1 draft was written at project start; this v1.0 draft folds in the full Wave-0 research corpus (docs/research/01–05) and the Patina→Umber rename. **Ratified only when Bessa approves.** Until then, no implementation work beyond what this spec names as Wave-0.

## One-sentence goal

**Umber** — an open-source, GPL-3.0-or-later, Rust-native 3D texture-painting DCC (Substance Painter class) for Windows and Linux: modern, fast, local-first, community-driven, with professional current and next-gen capabilities.

## Non-goals

What this project will NOT do. (Guardrails against scope creep.)

- No mesh modeling/editing (this is a painter, not a modeler — Blender exists)
- No sculpting in the v0.1 horizon
- **No Adobe format execution, ever:** we do not load or execute .sbsar or .spp — .sbsar runs only under Adobe's closed Materials SDK terms (no FOSS-compatible license exists — docs/research/01 §4); .spp is undocumented HDF5. Umber ships an **original node-graph engine** whose graphs serialize as **MaterialX documents** instead.
- No cloud/account requirement — local-first always; no telemetry
- No subscription — free, GPL, community-owned
- No macOS in the v0.1 horizon (Windows + Linux is the contract; macOS when contributors arrive)
- No film-scale UDIM streaming (100+ tiles) in v0.1 — UDIM data model and cross-tile painting are in; Mari-scale streaming is post-0.1

## Deliverables

What exists at the end (v0.1) that does not exist today.

1. A paint-capable desktop application (Windows 10/11 + Linux): import mesh → paint layered PBR materials on it in a real-time viewport → export engine-ready texture maps.
2. A crate-structured Rust workspace — render engine, painting core, UI shell, format I/O, procedural graph engine, bake engine, stylus input layer — each independently reusable.
3. An original procedural node-graph engine (generators, filters, masks, smart materials) whose graphs are MaterialX documents.
4. A bake engine: AO, normal (mesh + high→low), curvature, position, thickness, ID maps with cage controls.
5. A git-friendly open project format (`.umber` directory) — diffable, mergeable, deterministic round-trip.
6. Plugin runtime (wasmtime + WASI 0.3 components) + headless CLI for automation.
7. Reproducible CI on GitHub Actions building Windows + Linux releases (cargo-dist), with lavapipe headless GPU tests and golden-image bake/paint verification.

## Acceptance criteria

Testable statements. Each one observable, not vibes.

- [ ] App opens glTF/OBJ/FBX, renders PBR-lit at 60fps on a mid-range GPU with a 1M-tri mesh + 4K texture set
- [ ] Stylus input-to-photon < 20ms at 60Hz (drain-events-then-acquire loop); pressure curves drive brush dynamics end-to-end
- [ ] Layer stack (paint + fill layers, masks, per-channel blending, linear-space compositing) paints and exports correctly
- [ ] Stroke across a UV seam lands consistently on both islands (golden-image test)
- [ ] Export presets produce engine-correct output (glTF metal-rough, Unreal packed/DX normals, Unity GL normals; channel packing; naming tokens; PNG 8/16 + EXR 16F/32F)
- [ ] Bake engine produces AO/normal/curvature/position/ID maps verifiable against golden images in CI (lavapipe)
- [ ] Project save → close → load → export is byte-identical (deterministic serialization, CI-enforced)
- [ ] Undo: 100 steps at 4K within ~2GB; history window time-travel; no corruption across long sessions
- [ ] OpenPBR 1.1 viewport shader matches reference renders within tolerance (spec conformance suite)
- [ ] OCIO display transforms active in viewport; per-channel color-management flags respected
- [ ] Plugin sandbox loads a sample WASM component and executes it under resource limits
- [ ] Windows (MSI) and Linux (Flatpak + AppImage) installers built by CI from a tag

## Waves

Ordered execution plan. Wave N must pass its exit criteria before Wave N+1 starts its main work.

| Wave | Goal | Exit criteria |
|---|---|---|
| 0 | Research + spec ratification | 5 research reports landed ✓; requirements/tech-stack/architecture docs written ✓; name-collision sweep executed — Patina→Umber rename done ✓; **Bessa ratifies SPEC v1.0** (pending) |
| 1 | Workspace skeleton + mesh import + PBR viewport | Cargo workspace builds clean on Windows+Linux CI; app boots (wgpu+egui+winit), loads glTF/OBJ/FBX, IBL-lit viewport 60fps, camera controls, dockable UI shell; stylus basic (Windows Ink) |
| 2 | Painting core | Full brush engine (MyPaint-lineage dynamics, one-euro + lazy mouse); seam-aware UV-rasterization stamping; layer stack + masks + per-channel blending; undo journal; OCIO viewport; .umber project format with deterministic round-trip; Windows dual stylus path |
| 3 | Export + bake | Template/preset export (4 engine presets, packing, naming tokens, PNG/EXR); bake engine (AO/normal/curvature/position/ID) with golden-image CI; material-map import; cage controls |
| 4 | Procedural graph engine | Node engine executing on tile scheduler; graphs save/load as MaterialX documents; ~40 core nodes; generators driven by baked mesh maps; smart materials; UDIM cross-tile painting; layer instancing + anchors |
| 5 | Pro polish | Linux X11 stylus + hardware matrix; bake skew painting + auto-rebake; path tools; symmetry suite; post-effects; HDR output; 8K export; autosave/crash recovery; theming; external-artist dogfood |
| 6 | Ecosystem | wasmtime/WASI plugin runtime + sample plugins; headless CLI; docs site; cargo-dist release pipeline; v0.1 tagged |

## Constraints

Hard limits the work must respect.

- Rust-only core; FFI only where no viable Rust crate exists (OCIO via ocio-rs vendored, ufbx for FBX) and license-compatible
- No Adobe SDKs, no .sbsar/.spp execution ever (legal wall — docs/research/01 §4)
- GPL-3.0-or-later for the application; standalone utility crates may be MIT
- Hardware floor: Vulkan 1.3 / DX12 (GL fallback tier not built in v0.1)
- No network-required features; local-first
- Dev machine 8 cores / 16GB; CI on public GitHub Actions runners
- No experimental wgpu features in the core paint path (RT/mesh-shaders/bindless are feature-gated only)
- wgpu+naga+naga_oil version trio pinned together; winit pinned 0.30 until 0.31 stable

## Sources of truth

If docs/code conflict with this spec during execution, this list is what wins.

1. This SPEC.md (contract) — ratified only by Bessa+Razul respec
2. docs/specs/requirements.md + tech-stack.md + architecture.md (Wave-0 consolidation, research-derived)
3. Adobe Substance 3D Painter official docs + release notes (feature-reference bar — docs/research/01)
4. OpenPBR 1.1 spec (ASWF) + MaterialX spec (material model + graph interchange)
5. docs/research/03 (crate maturity verdicts) for any dependency question
6. glTF 2.0 spec + Unity/Unreal texture conventions (export correctness)

---

*Ratification block (Bessa — morning):*

- [ ] One-sentence goal — approved / edits:
- [ ] Non-goals — approved / edits:
- [ ] Deliverables + acceptance criteria — approved / edits:
- [ ] Wave plan — approved / edits:
- [ ] Constraints — approved / edits:
- [ ] **SPEC v1.0 RATIFIED** (all five above)

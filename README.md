# Umber

**Open-source, Rust-native 3D texture painting — the community's answer to Adobe Substance 3D Painter.**

Umber is a ground-up reimagining of the 3D texture-painting DCC: paint PBR materials directly on your meshes, build non-destructive layer stacks, bake mesh maps, and export engine-ready texture sets — fast, local-first, and free forever. Built completely in Rust for **Windows and Linux**.

> Status: **Wave 3 — bake & export pipeline landed.** Mesh-map bakers (AO, position, world normal, curvature, thickness, tangent normal), UV-seam dilation, engine presets (glTF / Unreal / Unity / Blender), the export driver, a headless CLI (`inspect` / `bake-ao` / `bake-all` / `export`), and the egui app with Bakes panel, Export dialog, and `.umber` project save/load are all in. 250+ workspace tests green, CI on both platforms every commit. Painting, the node graph, and a first release are ahead — watch the repo.

## Why

Substance Painter is the industry standard for game/film texturing — and it is subscription-only, closed, and cloud-pressured. The open-source alternatives each miss something: ArmorPaint is close but single-maintainer with a thin material ecosystem; Blender's texture paint is a module of a larger tool, not a focused painter; everything else is either dead, film-priced (Mari ~$2k/yr), or discontinued (Quixel Mixer). The world deserves a Rust-native, GPL, community-owned painter with a professional workflow.

## Principles

- **Free forever, GPL-3.0-or-later.** No subscriptions, no license servers, no telemetry. Community-owned, community-extended.
- **Local-first.** Your machine, your files. No account. No cloud requirement — ever.
- **Fast by construction.** Rust + wgpu (Vulkan/DX12). Compute-first painting and baking. 60fps viewport floor.
- **Original procedural engine.** We do not touch Adobe's .sbsar/.spp formats — Umber ships its own node-graph engine for generators, filters, and smart materials.
- **Git-friendly project format.** Diffable, mergeable projects — the anti-.spp.
- **Windows + Linux first-class.** One codebase, no second-class platform.

## Scope (v0.1 horizon)

| Capability | Wave |
|---|---|
| glTF / OBJ / FBX mesh import, PBR viewport | 1 |
| Pressure-sensitive painting, layer stack, undo/redo, project save/load | 2 |
| Engine-correct export (Unity / Unreal / glTF), AO/normal/curvature bake | 3 |
| Procedural node graph: generators, filters, smart materials | 4 |
| Tablet-pressure pipeline end-to-end, dockable UI maturity, theming | 5 |
| Sandboxed plugins (WASM/WASI), headless CLI, v0.1 release | 6 |


## Try it

```bash
# headless: bake every P0 mesh map from a mesh, seam-padded
cargo run -p umber-cli -- bake-all sword.obj out/ --size 1024 --rays 16 --dilate 16
# -> sword_ambient_occlusion.png, sword_curvature.png, sword_position.png,
#    sword_world_space_normal.png, sword_normal_base.png, sword_thickness.png

# headless: pack through an engine preset (DirectX normals for Unreal)
cargo run -p umber-cli -- export sword.obj out/ --preset unreal

# the app: viewport, layer stack, Bakes panel, Export dialog
cargo run -p umber-app
```

## License

GPL-3.0-or-later for the application. Standalone library crates extracted from Umber may be released under MIT where it helps the ecosystem.

## Repository layout

```
umber/
├── SPEC.md            # the contract — what Umber is and is not
├── DECISIONS.md       # append-only architecture decision log
├── STATE.md           # wave state (read this first on any new session)
├── docs/
│   ├── research/      # claw research reports (Painter inventory, competitors, ecosystem)
│   └── specs/         # design specs per subsystem
└── crates/            # Rust workspace — 11 crates, 250+ tests, CI green
```

## Contributing

Early stage — the design docs are the product right now. Read `SPEC.md`, open issues for anything missing from the requirements, and watch `docs/research/` as the feature inventory lands. Code contributions open when Wave 1 breaks ground (see `STATE.md`).

## Credit

Built by [Bessa](https://github.com/victorbessa96) and Razul, with research claws.

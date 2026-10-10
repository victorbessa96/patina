# MaterialX Document Export — Frontier Design (the §6 P0 row)

The texture-set's MaterialX document — "beyond raster"
interchange. Written 2026-10-10 12:12 against `0f75881`.

## What exists already

- **The graph-side writer** (`umber-graph/src/mtlx.rs:271`
  `to_mtlx`) — graph documents with declared custom nodedefs
  (the wave-5 kill, byte-identical round-trip).
- **The OpenPBR nodedef ground truth** vendored at
  `docs/claw-artifacts/openpbr/open_pbr_surface.mtlx` — the
  parameter names/order/types the document must cite.
- **The token engine** (`$srcMap`, `$colorSpace`, `$udim`) —
  the export driver's filename expansion, reusable for the
  image node's `file` strings.
- **The GPU material struct** (`umber-gpu/src/material.rs`) —
  the app-side OpenPBR parameter set, already spec-verified.

## The design: texture_set -> document

A new graph-side or export-side pure function:

`to_mtlx_document(params: &OpenPbrParams, maps: &[(MapKind,
file_name)], udim: Option<u32>) -> String`

- **The document**: `<materialx>` root, the standard
  OpenPBR nodedef IMPORTED (a `<nodegraph>` instantiation
  reference or the full nodedef per the consuming DCC's
  expectation — v1 cites the vendored spec's own document
  shape), one `<surfacematerial>` + one OpenPBR `surface`
  node instance, `<image>` nodes per map wired to the
  matching OpenPBR inputs.
- **The image nodes**: `file` = the map's exported filename
  (possibly `$udim`-tokened per the texture-set's tiling),
  `colorspace` = sRGB for baseColor/emissive, `acescg`/
  linear for data maps — mirroring the export driver's
  §9 per-channel transfer rule.
- **Params**: the non-textured parameters (roughness value
  when no roughness map, etc.) as node input values from
  the app's OpenPbrParams.
- **The honest scope**: v1 = the document generator + the
  export-dialog toggle ("MaterialX (.mtlx)" alongside the
  format outputs) + the wiring so a preset run also emits
  `<textureSet>.mtlx` next to the raster files. Full
  OpenPBR fidelity is NOT claimed — the doc references the
  exported maps; consuming DCCs evaluate what they support
  (the requirements' own honest line).

## Tests

1. The document parses with `from_mtlx` (the graph-side
   reader — the round-trip through the existing engine).
2. The OpenPBR parameter cites match the vendored nodedef's
   names/types (a test reading the vendored file).
3. The image files' colorspace wiring: color maps sRGB, data
   maps linear (per-map assert).
4. The UDIM token: tiled sets emit `$udim` in the file
   strings; single-tile sets emit the concrete name.
5. The preset integration: an export run with the mtlx
   toggle produces the .mtlx next to the PNGs, referencing
   the exact filenames written.

## As built (2026-10-10)

- **Generator**: `umber-graph/src/mtlx_doc.rs` `to_mtlx_document(name,
  &SurfaceParams, &[MtlxTexture])`. Graph-local types, NOT
  `OpenPbrParams`/`MapKind`: `MapKind` lives in umber-export (which now
  depends on umber-graph — a cycle otherwise) and `OpenPbrParams` in
  umber-gpu (wgpu; umber-graph also feeds umber-wasm).
  `SurfaceParams::from_gpu_slots` unpacks the GPU struct's six slots.
  `TextureMap` mirrors `MapKind`, and the driver's exhaustive match pins that.
- **Inputs** (vendored nodedef lines): base_color L10, base_metalness
  L14, specular_roughness L20, emission_color L76, geometry_opacity L78,
  geometry_normal L82 (through a stdlib `<normalmap>`, which isn't in the
  vendored file). AO and height have no OpenPBR input, so they're not wired.
- **UDIM**: tiled runs emit MaterialX's own `<UDIM>` filename token,
  NOT `$udim`. `$udim` is this app's export token, and no MaterialX
  consumer resolves it. Lone-1001 runs emit the concrete names.
  UNVERIFIED: `<UDIM>` and the `srgb_texture`/`lin_rec709` names come
  from recall of the MaterialX spec ("Filename Substitutions",
  "Color Spaces"). Check them against
  `documents/Specification/MaterialX.Specification.md` upstream.
- **Colorspace**: `srgb_texture` / `lin_rec709`, taken from the bytes
  actually written (`driver::output_transfer`: PNG follows §9, TIFF is
  always sRGB, EXR/JPEG are as-is). Packed scalar channels read through `<extract>`.
  DirectX normals aren't wired, because `normalmap` expects +Y.
- **Reader boundary**: `from_mtlx` parses the document as an EMPTY
  graph (the doc has no painter `<node>`s). Structure is asserted by
  walking the same reader's element tree.
- **App**: the Export dialog has a "MaterialX (.mtlx)" checkbox, off by
  default. Preset JSON takes `"materialx": true`, and the CLI honours it.
  The viewport renders `OpenPbrParams::default()`, which a pin test
  shows unpacks to `SurfaceParams::default()`, so the default-carrying
  `run_preset*` entry points ARE the app's material today.
  `run_preset_with_surface` is for when the material becomes editable.

## Build

One claw slice: the generator (graph-side pure fn — no new
deps; umber-export already depends on umber-graph? CHECK —
if not, the generator lives IN umber-graph and umber-export
calls it), the toggle, the five tests. claude's limit resets
12:40 — dispatch at reset or write it dragon-own.

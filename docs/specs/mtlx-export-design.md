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

## Build

One claw slice: the generator (graph-side pure fn — no new
deps; umber-export already depends on umber-graph? CHECK —
if not, the generator lives IN umber-graph and umber-export
calls it), the toggle, the five tests. claude's limit resets
12:40 — dispatch at reset or write it dragon-own.

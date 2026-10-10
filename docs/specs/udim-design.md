# UDIM Multi-Tile Support — Wave-5 Design (§2/§6)

Wave-5's UDIM slice. Written 2026-10-10 01:36 against the tree at
`05a5f1c`. The requirements name it twice: §2's data model row
("per-tile resolution; cross-tile painting" — the cross-tile part
is the P1 remainder) and §6's token row ($udim already emits its
'1001' constant, driver.rs:128). This note contracts the data-model
slice; the stroke-continuity-across-tiles is a named follow-up.

## The model (§2's row)

**Tile addressing.** A texture set's channels become per-UDIM-tile:
`TileId(u16)` — the 1001..1100 range, stored as `u16` (1001+); the
canonical address is `(channel, tile)`. The v1 API:

```rust
// umber-core: the texture-set model grows tiles
pub struct TextureSet {
    pub name: String,
    /// Channel -> tile -> image. Tile 1001 always present.
    pub tiles: BTreeMap<u16, TileChannels>,
}
pub struct TileChannels { /* base_color/normal/... as today, per-tile */ }
```

The mesh side: `MeshData` UVs may exceed [0,1] — the UDIM convention
maps integer offsets to tiles: tile = 1001 + floor(u) + 10*floor(v)
(WITHIN the 10x10 grid; the u/v floor computed per-triangle at
load). **The loader tags each triangle with its tile** (a
`tile_of_triangle: Vec<u16>` built at import from the UVs' floor)
— this is the piece the rest hangs on: the bakers bake per-tile
(each tile's UV window [floor..floor+1] becomes the PlaneDesc
region — the existing bakes parameterize over the UV window
already? CHECK: ao/position bake the 0..1 square via PlaneDesc's
extent — the tile window is the same mechanism with offset bounds),
the paint target becomes per-tile, the export writes per-tile files
with $udim = the real tile number.

**Single-tile compatibility**: a mesh whose UVs are all in [0,1] is
tile 1001 only — the existing single-tile paths are the degenerate
case, byte-identical behavior (the regression contract: every
existing test passes unchanged; the tile map has exactly one entry).

## Where each piece lands (the lean slices)

| Piece | Crate | Note |
|---|---|---|
| `tile_of_triangle` from UV floors | umber-mesh (import path) | the foundation — pure CPU, exact tests per the floor math |
| TextureSet tiles map + per-tile resolution | umber-core | the data model; per-tile resolution = each tile's own w/h |
| Per-tile bake window (PlaneDesc offset) | umber-bake | the bakes loop tiles; the 1001-only case unchanged |
| Per-tile paint targets | umber-app PaintState | one PaintTarget per tile; the stroke router picks the tile from the UV (the cross-tile STROKE continuity is the named follow-up — v1 strokes land in the tile under the pointer) |
| Export per-tile + real $udim | umber-export | the driver iterates tiles; the token emits each tile's number; SINGLE_TILE_UDIM retires |
| Bakes/Export panels tile lists | umber-app UI | the checklist per tile |

## Tests (the can-fail cores)

1. **Tile floor math**: a mesh with UVs in [0,2]x[0,1] → triangles
   span tiles 1001/1002 exactly as their u-floors dictate (assert
   the full tile_of_triangle vec, derived).
2. **The 10x10 grid**: UV (2.5, 0.3) → tile 1003; (0.5, 1.2) → 1011
   (the v-offset adds tens — the classic UDIM formula; assert both
   with derivations).
3. **Single-tile regression**: the all-[0,1] mesh → one tile, the
   existing exports byte-identical (the strongest regression).
4. **Per-tile bake**: the two-tile mesh bakes AO per tile — each
   tile's output covers only its UV window's geometry (the
   boundary triangle lands in exactly one tile per its floor).
5. **Export tokens**: the two-tile export writes two files with
   ..._1001 and ..._1002 names (assert both paths); $udim emits
   per-tile (the driver test extends the token suite).
6. **Paint routing**: a pointer event at UV (1.2, 0.5) lands in
   tile 1002's PaintTarget (the staged batch's target assert).

## The honest v1 boundary

Cross-tile stroke continuity (a stroke dragging across a tile seam
paints both) is §2's P1 row and this design's named follow-up — v1
routes per-event to the event's tile. The seam-graph work (wave-4's
SeamGraph) extends naturally: tile seams are seam edges whose
correspondence crosses tiles — the mirror math already handles
island correspondence; the tile case reuses it with the tile-offset
UV shift. Not built here; the design holds the slot.

## Build order

1. umber-mesh tile_of_triangle (the foundation, pure CPU) — lean
   claw slice.
2. umber-core TextureSet tiles (the model).
3. umber-bake per-tile window (the PlaneDesc offset — small).
4. umber-app paint routing + per-tile targets.
5. umber-export per-tile + token.
6. UI tile lists (rides 4-5).

Slices 1-3 are one lean evening; 4-5 the next; 6 rides.

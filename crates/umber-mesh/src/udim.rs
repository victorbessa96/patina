//! UDIM tile addressing (Wave-5 slice 1: the foundation).
//!
//! Implements the tile formula from `docs/specs/udim-design.md` ("The
//! model"): `tile = 1001 + floor(u) + 10 * floor(v)` within the 10x10
//! UDIM grid (tiles 1001..=1100).
//!
//! # Assignment policy
//!
//! [`tile_of_triangle`] assigns a triangle to the tile of its FIRST
//! vertex's UV. A triangle spanning a tile boundary therefore lands in
//! exactly one tile — its first vertex's. The per-vertex split across
//! tiles is the later cross-tile work named in the design doc, not v1.
//!
//! # Clamping (v1)
//!
//! Out-of-range UVs are importer bugs; we do not propagate garbage.
//! Each floored axis is clamped to the grid (`0..=9`) before the tile
//! is composed, so the result always lies in `1001..=1100`:
//! negative floors (e.g. `floor(-0.3) = -1`, which would naively give
//! tile 1000) clamp up to the grid edge (tile 1001), and huge floors
//! clamp down to the far edge (column/row 9).
//!
//! # No panics
//!
//! NaN UV components map to [`FIRST_TILE`], and triangles whose first
//! index (or UV lookup) is out of bounds map to [`FIRST_TILE`]. All
//! lookups go through [`slice::get`]; nothing here panics.
//!
//! # Single-tile compatibility
//!
//! A mesh whose UVs are all in `[0, 1]` maps every triangle to tile
//! 1001 — the degenerate case; existing single-tile paths behave
//! byte-identically (per the design doc's regression contract).

use crate::MeshData;

/// The first (and default) UDIM tile: UV `[0, 0]` lives here.
pub const FIRST_TILE: u16 = 1001;

/// Last tile of the v1 10x10 UDIM grid (`1001 + 9 + 10 * 9`).
const LAST_TILE: u16 = 1100;

/// Tile for a single UV point: `1001 + floor(u) + 10 * floor(v)`,
/// with each floored axis clamped to `0..=9` so the result stays in
/// `1001..=1100` (see the module docs for the v1 clamp policy).
///
/// NaN in either component yields [`FIRST_TILE`].
pub fn tile_of_uv(uv: [f32; 2]) -> u16 {
    if uv[0].is_nan() || uv[1].is_nan() {
        return FIRST_TILE;
    }
    let cu = (uv[0].floor() as i64).clamp(0, 9);
    let cv = (uv[1].floor() as i64).clamp(0, 9);
    (FIRST_TILE as i64 + cu + 10 * cv) as u16
}

/// One UDIM tile per triangle: the tile of the triangle's FIRST
/// vertex's UV (see the module docs for why the first vertex wins).
///
/// Malformed triangles (first index out of bounds, or no UV at that
/// index) default to [`FIRST_TILE`] — never panics, per the crate rule.
pub fn tile_of_triangle(mesh: &MeshData) -> Vec<u16> {
    let tri_count = mesh.indices.len() / 3;
    let mut tiles = Vec::with_capacity(tri_count);
    for k in 0..tri_count {
        let tile = mesh
            .indices
            .get(3 * k)
            .and_then(|i| mesh.uvs.get(*i as usize))
            .map_or(FIRST_TILE, |uv| tile_of_uv(*uv));
        tiles.push(tile);
    }
    debug_assert!(tiles.iter().all(|t| (FIRST_TILE..=LAST_TILE).contains(t)));
    tiles
}

/// The triangle indices assigned to `tile`: the positions `k` in
/// [`tile_of_triangle`]'s output whose tile equals `tile`, in ascending
/// order. An empty vec means the tile has no geometry — the per-tile
/// bakers treat that as "nothing to rasterize" (their filtered mesh is
/// empty and the bake is rejected as [`EmptyMesh`]-equivalent, never
/// silently blank).
pub fn triangles_for_tile(mesh: &MeshData, tile: u16) -> Vec<u32> {
    tile_of_triangle(mesh)
        .into_iter()
        .enumerate()
        .filter_map(|(k, t)| (t == tile).then_some(k as u32))
        .collect()
}

/// The tiles `mesh` has geometry in: [`tile_of_triangle`]'s output,
/// sorted ascending and deduplicated. An all-`[0, 1]` mesh yields
/// `[1001]`; a mesh with no triangles yields `[]` (callers that need a
/// tile anyway — the single-tile UI path — fall back to [`FIRST_TILE`]
/// themselves). The Bakes/Export panels' tile lists are built from this.
pub fn present_tiles(mesh: &MeshData) -> Vec<u16> {
    let mut tiles = tile_of_triangle(mesh);
    tiles.sort_unstable();
    tiles.dedup();
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh(uvs: Vec<[f32; 2]>) -> MeshData {
        let n = uvs.len() as u32;
        MeshData {
            positions: vec![[0.0; 3]; n as usize],
            normals: vec![[0.0; 3]; n as usize],
            uvs,
            indices: vec![0, 1, 2, 0, 2, 3],
            material_names: vec!["m".into()],
        }
    }

    #[test]
    fn all_unit_uvs_map_to_first_tile() {
        // The degenerate single-tile case: every UV in [0, 1].
        let mesh = quad_mesh(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert_eq!(tile_of_triangle(&mesh), vec![1001, 1001]);
    }

    #[test]
    fn grid_math_tens_come_from_v() {
        // 1001 + floor(u) + 10 * floor(v): the v term adds TENS —
        // the classic UDIM trap (1011, not 1002, for the v-offset case).
        assert_eq!(tile_of_uv([2.5, 0.3]), 1003); // 1001 + 2 + 10*0
        assert_eq!(tile_of_uv([0.5, 1.2]), 1011); // 1001 + 0 + 10*1
        assert_eq!(tile_of_uv([0.0, 0.0]), 1001);
        // 1001 + 9 + 10*9 = 1100: the far corner of the 10x10 grid.
        assert_eq!(tile_of_uv([9.9, 9.9]), 1100);
    }

    #[test]
    fn two_triangles_span_two_tiles() {
        // Triangle 0's first vertex UV (0.5, 0.5) -> 1001 + 0 + 0;
        // triangle 1's first vertex UV (1.5, 0.5) -> 1001 + 1 + 0.
        let mesh = MeshData {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0; 3]; 4],
            uvs: vec![[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [1.5, 0.5]],
            indices: vec![0, 1, 2, 3, 1, 2],
            material_names: vec!["m".into()],
        };
        assert_eq!(tile_of_triangle(&mesh), vec![1001, 1002]);
    }

    #[test]
    fn edges_clamp_to_the_grid() {
        // NaN UVs carry no tile information -> the default tile.
        assert_eq!(tile_of_uv([f32::NAN, 0.5]), 1001);
        assert_eq!(tile_of_uv([0.5, f32::NAN]), 1001);
        // floor(-0.3) = -1 would naively give tile 1000 (below the
        // grid); the v1 clamp pins it to the grid edge: 1001, not 1000.
        assert_eq!(tile_of_uv([-0.3, 0.5]), 1001);
        // floor(50.5) = 50 clamps to column 9: 1001 + 9 + 10*0 = 1010,
        // the grid's right edge on the bottom row.
        assert_eq!(tile_of_uv([50.5, 0.3]), 1010);
    }

    #[test]
    fn oob_index_defaults_without_panicking() {
        let mesh = MeshData {
            positions: vec![[0.0; 3]],
            normals: vec![[0.0; 3]],
            uvs: vec![[2.5, 0.3]],
            indices: vec![7, 8, 9],
            material_names: vec!["m".into()],
        };
        assert_eq!(tile_of_triangle(&mesh), vec![1001]);
    }

    #[test]
    fn triangles_for_tile_selects_each_tile_exactly() {
        // Same two-triangle two-tile shape as `two_triangles_span_two_tiles`.
        let mesh = MeshData {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0; 3]; 4],
            uvs: vec![[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [1.5, 0.5]],
            indices: vec![0, 1, 2, 3, 1, 2],
            material_names: vec!["m".into()],
        };
        assert_eq!(triangles_for_tile(&mesh, 1001), vec![0]);
        assert_eq!(triangles_for_tile(&mesh, 1002), vec![1]);
        // No triangle lives in 1003: empty vec = no geometry (documented).
        assert_eq!(triangles_for_tile(&mesh, 1003), Vec::<u32>::new());
    }

    #[test]
    fn present_tiles_is_sorted_unique() {
        // Two-tile fixture: triangle 0 -> 1001, triangle 1 -> 1002.
        let two = MeshData {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0; 3]; 4],
            uvs: vec![[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [1.5, 0.5]],
            indices: vec![0, 1, 2, 3, 1, 2],
            material_names: vec!["m".into()],
        };
        assert_eq!(present_tiles(&two), vec![1001, 1002]);

        // Out-of-order, repeated tiles: triangles map to 1011, 1001,
        // 1011 (first-vertex UVs (0.5, 1.5), (0.5, 0.5), (0.5, 1.5)) —
        // sorted + deduplicated to [1001, 1011].
        let shuffled = MeshData {
            positions: vec![[0.0; 3]; 2],
            normals: vec![[0.0; 3]; 2],
            uvs: vec![[0.5, 1.5], [0.5, 0.5]],
            indices: vec![0, 1, 1, 1, 0, 0, 0, 1, 1],
            material_names: vec!["m".into()],
        };
        assert_eq!(tile_of_triangle(&shuffled), vec![1011, 1001, 1011]);
        assert_eq!(present_tiles(&shuffled), vec![1001, 1011]);

        // Single-tile: every UV in [0, 1].
        let single = quad_mesh(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert_eq!(present_tiles(&single), vec![1001]);

        // No triangles: no tiles (not a defaulted 1001).
        let empty = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            material_names: vec![],
        };
        assert_eq!(present_tiles(&empty), Vec::<u16>::new());
    }
}

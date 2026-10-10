//! UV-seam topology for seam-aware stamping (Wave-4 slice 1).
//!
//! Implements the `SeamGraph` data structure from
//! `docs/specs/seam-aware-stamping-design.md`: built once per mesh load,
//! it records every 3D edge whose UV endpoints differ between the two
//! adjacent triangles (a UV "cut"), plus a per-triangle seam flag and a
//! midpoint-proximity query used at stroke start.
//!
//! # Adjacency key
//!
//! Edges are keyed on the bit-exact 3D endpoint positions, NOT on vertex
//! indices. UV splits duplicate vertices along the cut (distinct indices,
//! identical positions, distinct UVs — this is how glTF/OBJ/FBX carry
//! seams), so index adjacency would only ever find island-internal edges:
//! two triangles sharing vertex indices trivially share per-vertex UVs.
//! Position keying subsumes the index-shared case (identical indices imply
//! identical positions; their UVs then match exactly and produce no entry).
//!
//! # Non-manifold policy
//!
//! A 3D edge shared by more than two triangles pairs the FIRST two
//! triangles (in triangle-index order) and ignores the rest: at most one
//! [`SeamEdge`] per distinct 3D edge. A self-pair (one degenerate triangle
//! contributing the same 3D edge twice) is skipped in favor of the first
//! pair of distinct triangles; an edge used only by a single (possibly
//! degenerate) triangle yields no entry.
//!
//! # No panics
//!
//! All attribute lookups go through [`slice::get`]; triangles referencing
//! missing positions/UVs, short index lists, and zero-length position edges
//! are skipped rather than panicked on.

use crate::MeshData;
use glam::Vec3;
use std::collections::HashMap;

/// One UV seam: two 3D-adjacent triangles sharing a 3D edge whose UV
/// endpoints differ between the triangles' islands.
///
/// `uv_a[0]`/`uv_b[0]` are the two triangles' UVs at the SAME 3D endpoint
/// (likewise `[1]`), so `uv_a <-> uv_b` is the edge correspondence the
/// mirror-dab slice maps across.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeamEdge {
    /// First triangle sharing the 3D edge (triangle index, lower index of
    /// the recorded pair).
    pub tri_a: u32,
    /// Second triangle sharing the 3D edge (triangle index).
    pub tri_b: u32,
    /// The shared edge's UV endpoints in `tri_a`'s island.
    pub uv_a: [[f32; 2]; 2],
    /// The shared edge's UV endpoints in `tri_b`'s island.
    pub uv_b: [[f32; 2]; 2],
    /// Midpoint of the shared 3D edge, for proximity queries.
    pub mid_3d: Vec3,
}

/// UV-seam topology of a mesh, built once per mesh load by
/// [`build_seam_graph`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeamGraph {
    /// Seam edges, sorted by `mid_3d` (x, then y, then z) for deterministic
    /// order. Linearly scanned by [`seam_edges_near`]; a spatial index
    /// arrives only if profiling demands it (per the design doc).
    pub edges: Vec<SeamEdge>,
    /// Per-triangle flag: `tri_touches_seam[t]` is true iff triangle `t`
    /// participates in at least one seam edge. Length equals the mesh
    /// triangle count at build time.
    pub tri_touches_seam: Vec<bool>,
}

/// Bit-exact position key: seam-split duplicates carry bitwise-identical
/// coordinates, so exact matching finds them with no tolerance tuning.
type PositionKey = [u32; 3];
/// Canonical 3D-edge key: the two endpoint [`PositionKey`]s, lesser first.
type EdgeKey = (PositionKey, PositionKey);
/// One triangle's use of a 3D edge: (triangle, UV at key.0, UV at key.1).
type EdgeUse = (u32, [f32; 2], [f32; 2]);

fn position_key(p: &[f32; 3]) -> PositionKey {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

/// Inverse of [`position_key`], for recovering edge endpoints at record time.
fn key_position(key: &[u32; 3]) -> Vec3 {
    Vec3::new(
        f32::from_bits(key[0]),
        f32::from_bits(key[1]),
        f32::from_bits(key[2]),
    )
}

/// Builds the seam graph for `mesh`.
///
/// For each pair of triangles sharing a 3D edge (position-keyed adjacency,
/// see the module docs), compares the edge's UV endpoints across the two
/// triangles with exact `f32` equality — UV cuts are exact, never
/// near-equal. Edges whose UVs match exactly are island-internal and
/// produce no entry. Infallible: malformed triangles are skipped (see the
/// module docs); an empty or seam-free mesh yields an empty [`SeamGraph`].
pub fn build_seam_graph(mesh: &MeshData) -> SeamGraph {
    let tri_count = mesh.indices.len() / 3;
    // Canonical 3D-edge key -> per-triangle uses (tri, uv at key.0, uv at key.1).
    let mut uses: HashMap<EdgeKey, Vec<EdgeUse>> = HashMap::new();
    for (tri_idx, tri) in mesh.indices.chunks_exact(3).enumerate() {
        let tri_u32 = tri_idx as u32;
        let corners = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
            let (Some(&pa), Some(&pb)) = (
                mesh.positions.get(corners[a]),
                mesh.positions.get(corners[b]),
            ) else {
                continue;
            };
            let (Some(&uv_a), Some(&uv_b)) = (mesh.uvs.get(corners[a]), mesh.uvs.get(corners[b]))
            else {
                continue;
            };
            let (ka, kb) = (position_key(&pa), position_key(&pb));
            if ka == kb {
                continue; // Zero-length position edge: not a seam.
            }
            let (key, uv_lo, uv_hi) = if ka < kb {
                ((ka, kb), uv_a, uv_b)
            } else {
                ((kb, ka), uv_b, uv_a)
            };
            uses.entry(key).or_default().push((tri_u32, uv_lo, uv_hi));
        }
    }

    let mut graph = SeamGraph {
        edges: Vec::new(),
        tri_touches_seam: vec![false; tri_count],
    };
    for ((ka, kb), list) in &uses {
        // First pair of DISTINCT triangles (non-manifold: first two win;
        // degenerate self-uses are skipped, never self-paired).
        let mut pair: Option<(usize, usize)> = None;
        'outer: for (i, _) in list.iter().enumerate() {
            for (j, _) in list.iter().enumerate().skip(i + 1) {
                if list[i].0 != list[j].0 {
                    pair = Some((i, j));
                    break 'outer;
                }
            }
        }
        let Some((i, j)) = pair else { continue };
        let (tri_a, uv_lo_a, uv_hi_a) = list[i];
        let (tri_b, uv_lo_b, uv_hi_b) = list[j];
        if uv_lo_a == uv_lo_b && uv_hi_a == uv_hi_b {
            continue; // Island-internal edge: continuous UVs.
        }
        let mid_3d = (key_position(ka) + key_position(kb)) * 0.5;
        let (lo, hi) = (tri_a as usize, tri_b as usize);
        graph.tri_touches_seam[lo] = true;
        graph.tri_touches_seam[hi] = true;
        graph.edges.push(SeamEdge {
            tri_a,
            tri_b,
            uv_a: [uv_lo_a, uv_hi_a],
            uv_b: [uv_lo_b, uv_hi_b],
            mid_3d,
        });
    }
    // Deterministic order: sort by midpoint (total float order), then pair.
    graph.edges.sort_by(|a, b| {
        a.mid_3d
            .x
            .total_cmp(&b.mid_3d.x)
            .then(a.mid_3d.y.total_cmp(&b.mid_3d.y))
            .then(a.mid_3d.z.total_cmp(&b.mid_3d.z))
            .then(a.tri_a.cmp(&b.tri_a))
            .then(a.tri_b.cmp(&b.tri_b))
    });
    graph
}

/// Returns the seam edges whose 3D midpoint lies within radius `r` of `p`.
///
/// Straight-line distance from the edge midpoint (v1 approximation per the
/// design doc). Linear scan over [`SeamGraph::edges`]; call once per stroke
/// start, not per dab. A negative `r` matches nothing.
pub fn seam_edges_near(graph: &SeamGraph, p: Vec3, r: f32) -> Vec<&SeamEdge> {
    graph
        .edges
        .iter()
        .filter(|e| e.mid_3d.distance(p) <= r)
        .collect()
}

/// Returns true iff triangle `tri` touches any seam edge.
///
/// Out-of-range indices (e.g. against a graph built from a different mesh)
/// return false rather than panicking.
pub fn tri_touches_seam(graph: &SeamGraph, tri: usize) -> bool {
    graph.tri_touches_seam.get(tri).copied().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quad corners: A=(0,0,0) B=(1,0,0) C=(1,1,0) D=(0,1,0), diagonal A-C.
    fn single_island_quad() -> MeshData {
        MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_names: vec![],
        }
    }

    /// Same quad, but the diagonal A-C is a UV cut: tri 1 duplicates the
    /// shared vertices (indices 3,4) with island-offset UVs.
    fn two_island_quad() -> MeshData {
        MeshData {
            positions: vec![
                [0.0, 0.0, 0.0], // 0: A (tri 0)
                [1.0, 0.0, 0.0], // 1: B
                [1.0, 1.0, 0.0], // 2: C (tri 0)
                [0.0, 0.0, 0.0], // 3: A (tri 1, duplicated)
                [1.0, 1.0, 0.0], // 4: C (tri 1, duplicated)
                [0.0, 1.0, 0.0], // 5: D
            ],
            normals: vec![[0.0, 0.0, 1.0]; 6],
            uvs: vec![
                [0.0, 0.0],   // A in tri 0
                [1.0, 0.0],   // B
                [1.0, 1.0],   // C in tri 0
                [10.0, 10.0], // A in tri 1 (other island)
                [11.0, 11.0], // C in tri 1 (other island)
                [11.0, 10.0], // D
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
            material_names: vec![],
        }
    }

    #[test]
    fn single_island_quad_has_no_seams() {
        let graph = build_seam_graph(&single_island_quad());
        assert!(
            graph.edges.is_empty(),
            "continuous UVs must yield zero seam edges, got {:?}",
            graph.edges
        );
        assert_eq!(graph.tri_touches_seam, vec![false, false]);
    }

    #[test]
    fn two_island_quad_has_exactly_one_seam_with_exact_endpoints() {
        let graph = build_seam_graph(&two_island_quad());
        assert_eq!(
            graph.edges.len(),
            1,
            "the cut diagonal must yield exactly one seam edge, got {:?}",
            graph.edges
        );
        let e = graph.edges[0];
        // Canonical endpoint order: A (bit-key 0) first, C second.
        assert_eq!((e.tri_a, e.tri_b), (0, 1));
        assert_eq!(e.uv_a, [[0.0, 0.0], [1.0, 1.0]]);
        assert_eq!(e.uv_b, [[10.0, 10.0], [11.0, 11.0]]);
        assert_eq!(e.mid_3d, Vec3::new(0.5, 0.5, 0.0));
        assert_eq!(graph.tri_touches_seam, vec![true, true]);
        assert!(tri_touches_seam(&graph, 0));
        assert!(tri_touches_seam(&graph, 1));
        assert!(!tri_touches_seam(&graph, 7));
    }

    #[test]
    fn proximity_query_hits_near_and_misses_far() {
        let graph = build_seam_graph(&two_island_quad());
        let mid = Vec3::new(0.5, 0.5, 0.0);
        let near = seam_edges_near(&graph, mid, 0.1);
        assert_eq!(near.len(), 1, "query at the midpoint must find the edge");
        assert_eq!((near[0].tri_a, near[0].tri_b), (0, 1));
        let far = seam_edges_near(&graph, Vec3::new(50.0, 50.0, 50.0), 1.0);
        assert!(
            far.is_empty(),
            "far query must return empty, got {:?}",
            far.iter().map(|e| (e.tri_a, e.tri_b)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_manifold_edge_pairs_first_two_without_panicking() {
        // Three triangles sharing the position edge P0-P1, each with its
        // own UV island so every pairing is a seam.
        let mesh = MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], // tri 0
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0], // tri 1
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0], // tri 2
            ],
            normals: vec![[0.0, 0.0, 1.0]; 9],
            uvs: vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
                [3.0, 2.0],
                [2.0, 3.0],
                [4.0, 4.0],
                [5.0, 4.0],
                [4.0, 5.0],
            ],
            indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            material_names: vec![],
        };
        let graph = build_seam_graph(&mesh);
        assert_eq!(
            graph.edges.len(),
            1,
            "one 3D edge -> one SeamEdge even when non-manifold, got {:?}",
            graph.edges
        );
        // Pairing choice: the first two triangles in index order.
        assert_eq!((graph.edges[0].tri_a, graph.edges[0].tri_b), (0, 1));
        assert_eq!(graph.tri_touches_seam, vec![true, true, false]);
    }

    #[test]
    fn malformed_triangles_are_skipped_without_panicking() {
        // Index 9 is out of bounds; uvs are short; indices not a multiple
        // of 3 (trailing index ignored by chunks_exact).
        let mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![],
            uvs: vec![[0.0, 0.0]],
            indices: vec![0, 1, 9, 0],
            material_names: vec![],
        };
        let graph = build_seam_graph(&mesh);
        assert!(graph.edges.is_empty());
        assert_eq!(graph.tri_touches_seam, vec![false]);
    }
}

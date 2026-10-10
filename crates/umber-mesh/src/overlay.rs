//! Viewport overlay geometry: the wireframe pass's duplicated-vertex
//! buffer.
//!
//! The wireframe overlay shades triangle edges via a barycentric
//! interpolant (see `umber_gpu::shaders::WIREFRAME_SHADER`), which needs a
//! per-vertex barycentric attribute — one corner of the unit simplex per
//! triangle corner. Shared-vertex indexing cannot express that (a vertex
//! shared by N triangles would need N different barycentrics), so the wire
//! pass gets its OWN vertex buffer: 3 duplicated verts per index-triple,
//! no index sharing, built once per mesh load from the index list. The
//! honest v1 the wireframe-grid design prescribes — no provoking-vertex
//! tricks.

use crate::MeshData;

/// One wireframe-pass vertex: object-space position plus the barycentric
/// corner for its slot in the triangle.
///
/// `Pod` so `umber-gpu` can upload the built vec with a single
/// `bytemuck::cast_slice` (same pattern as its `Vertex`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WireVertex {
    /// Object-space position (duplicated per triangle — no sharing).
    pub position: [f32; 3],
    /// Barycentric corner: `(1,0,0)` / `(0,1,0)` / `(0,0,1)` for the
    /// triangle's first / second / third index.
    pub bary: [f32; 3],
}

/// Builds the wireframe vertex buffer for `mesh`: one triangle's worth (3
/// verts) per index-triple, positions duplicated, barycentrics assigned in
/// index order.
///
/// Triangles referencing out-of-range vertices are skipped (same leniency
/// as the GPU upload path's normal computation — a malformed index must
/// not panic the overlay build); a trailing partial triple is ignored via
/// `chunks_exact`.
pub fn wire_vertices_from_indices(mesh: &MeshData) -> Vec<WireVertex> {
    const BARY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut out = Vec::with_capacity(mesh.indices.len());
    for tri in mesh.indices.chunks_exact(3) {
        let mut positions = [[0.0f32; 3]; 3];
        let mut valid = true;
        for (slot, &index) in tri.iter().enumerate() {
            match mesh.positions.get(index as usize) {
                Some(&position) => positions[slot] = position,
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        for (position, bary) in positions.into_iter().zip(BARY) {
            out.push(WireVertex { position, bary });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> MeshData {
        MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_names: vec![],
        }
    }

    #[test]
    fn one_triangle_worth_per_index_triple_with_simplex_bary() {
        let verts = wire_vertices_from_indices(&quad_mesh());
        assert_eq!(verts.len(), 6);
        assert_eq!(
            verts[0].position,
            [0.0, 0.0, 0.0],
            "first vert duplicates the first index's position"
        );
        assert_eq!(verts[0].bary, [1.0, 0.0, 0.0]);
        assert_eq!(verts[1].bary, [0.0, 1.0, 0.0]);
        assert_eq!(verts[2].bary, [0.0, 0.0, 1.0]);
        // Second triangle restarts the simplex; shared vertex 0/2 is
        // duplicated, not shared.
        assert_eq!(verts[3].position, [0.0, 0.0, 0.0]);
        assert_eq!(verts[3].bary, [1.0, 0.0, 0.0]);
        assert_eq!(verts[5].position, [0.0, 1.0, 0.0]);
        assert_eq!(verts[5].bary, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn out_of_range_triangles_are_skipped_not_panicking() {
        let mut mesh = quad_mesh();
        mesh.indices.extend([0, 1, 99]);
        let verts = wire_vertices_from_indices(&mesh);
        assert_eq!(verts.len(), 6);
    }

    #[test]
    fn empty_mesh_builds_empty_buffer() {
        assert!(wire_vertices_from_indices(&MeshData::default()).is_empty());
    }

    #[test]
    fn wire_vertex_is_tightly_packed_for_gpu_upload() {
        assert_eq!(std::mem::size_of::<WireVertex>(), 24);
    }
}

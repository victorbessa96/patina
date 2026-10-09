//! FBX mesh import via the [`ufbx`] crate (official Rust bindings).
//!
//! Every node carrying a mesh contributes its triangulated faces. Vertices
//! are expanded per corner (no index de-duplication): this keeps UV seams
//! intact at the cost of duplicated positions, and is the documented
//! simplification to revisit once a weld pass exists. Missing normals or UVs
//! are filled with zeros so the attribute vectors always stay parallel to
//! `positions`; positions always come from the real vertex data. Node
//! transforms and skinning are ignored (bind pose).

use std::path::Path;

use super::{ImportError, MeshData};

/// Load an `.fbx` file into a single concatenated [`MeshData`].
///
/// Walks every scene node, triangulates each attached mesh, and appends the
/// result with per-primitive index offsets. Returns [`ImportError::Fbx`] when
/// the file cannot be parsed, when index data is internally inconsistent, or
/// when the scene contains no mesh geometry at all.
pub fn load_fbx(path: &Path) -> Result<MeshData, ImportError> {
    let path_str = path.to_string_lossy();
    let scene = ufbx::load_file(&path_str, ufbx::LoadOpts::default())
        .map_err(|e| ImportError::Fbx(format!("{e:?}")))?;
    let mut data = MeshData::default();
    for node in &scene.nodes {
        let Some(mesh) = node.mesh.as_ref() else {
            continue;
        };
        let mesh: &ufbx::Mesh = mesh;
        append_mesh(&mut data, mesh)?;
    }
    if data.positions.is_empty() {
        return Err(ImportError::Fbx("no mesh geometry found".into()));
    }
    Ok(data)
}

/// Append one triangulated [`ufbx::Mesh`] to the accumulated [`MeshData`].
///
/// `corner` values produced by [`ufbx::triangulate_face_vec`] are absolute
/// mesh corner indices, so they index `vertex_indices` and the attribute
/// index buffers directly.
fn append_mesh(data: &mut MeshData, mesh: &ufbx::Mesh) -> Result<(), ImportError> {
    for material in &mesh.materials {
        let name = material.element.name.to_string();
        if !data.material_names.iter().any(|n| n == &name) {
            data.material_names.push(name);
        }
    }

    let positions: &[ufbx::Vec3] = &mesh.vertices;
    let position_indices: &[u32] = &mesh.vertex_indices;
    let uv = &mesh.vertex_uv;
    let normal = &mesh.vertex_normal;
    let has_uv = uv.exists && !uv.values.is_empty() && !uv.indices.is_empty();
    let has_normal = normal.exists && !normal.values.is_empty() && !normal.indices.is_empty();

    let mut corners = Vec::new();
    for face in &mesh.faces {
        corners.clear();
        ufbx::triangulate_face_vec(&mut corners, mesh, *face);
        for corner in &corners {
            let corner = *corner as usize;
            let position_index =
                position_indices.get(corner).copied().ok_or_else(|| {
                    ImportError::Fbx(format!("corner {corner} has no position index"))
                })? as usize;
            let position = positions.get(position_index).ok_or_else(|| {
                ImportError::Fbx(format!("position index {position_index} out of range"))
            })?;
            data.positions
                .push([position.x as f32, position.y as f32, position.z as f32]);

            if has_uv {
                let uv_index =
                    uv.indices.get(corner).copied().ok_or_else(|| {
                        ImportError::Fbx(format!("corner {corner} has no UV index"))
                    })? as usize;
                let tex_coord = uv
                    .values
                    .get(uv_index)
                    .ok_or_else(|| ImportError::Fbx(format!("UV index {uv_index} out of range")))?;
                data.uvs.push([tex_coord.x as f32, tex_coord.y as f32]);
            } else {
                data.uvs.push([0.0; 2]);
            }

            if has_normal {
                let normal_index = normal.indices.get(corner).copied().ok_or_else(|| {
                    ImportError::Fbx(format!("corner {corner} has no normal index"))
                })? as usize;
                let axis = normal.values.get(normal_index).ok_or_else(|| {
                    ImportError::Fbx(format!("normal index {normal_index} out of range"))
                })?;
                data.normals
                    .push([axis.x as f32, axis.y as f32, axis.z as f32]);
            } else {
                data.normals.push([0.0; 3]);
            }

            data.indices.push(data.positions.len() as u32 - 1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real binary FBX fixture (Kaydara FBX 7.4 binary, a Blender-exported
    // cube) from ufbx's own upstream test corpus
    // (data/blender_272_cube_7400_binary.fbx). Binary FBX cannot be built by
    // hand in-test; this is the honest fixture.
    #[test]
    fn loads_fbx_cube_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("assets")
            .join("cube_b7400.fbx");
        let data = crate::load(&path).expect("cube fbx fixture must load");
        assert!(data.vertex_count() >= 8, "cube needs >= 8 verts");
        assert!(data.triangle_count() >= 12, "cube needs >= 12 tris");
        assert_eq!(data.positions.len(), data.normals.len());
        assert_eq!(data.positions.len(), data.uvs.len());
        // Blender 2.72 cube exports carry no material; material presence is
        // not asserted for this fixture. A material-carrying fixture lands
        // with the Wave-2 texture-set work.
    }
}

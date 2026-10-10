//! umber-mesh — mesh container + import (glTF, OBJ, FBX).
//!
//! Wave 1 scope: the `MeshData` container + format detection + OBJ import
//! (tobj). glTF and FBX loaders are the Wave-1 claw work; the half-edge
//! DCEL topology structure lands with seam-aware work (Wave 2+).

use glam::Vec3;

pub mod bvh;
pub mod fbx;
pub mod gltf;
pub mod mesh_maps;
pub mod raycast;
pub mod seam;
pub mod seam_mirror;

pub use fbx::load_fbx;
pub use gltf::load_gltf;
pub use mesh_maps::{
    format_mesh_map, parse_mesh_map, parse_mesh_map_stem, texture_set_name, MeshMapKind,
    MeshMapName,
};
pub use raycast::{ray_intersect, uv_at, RayHit};

/// Interleaved mesh data as imported, before any GPU upload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// Triangle list indices into the vertex attributes.
    pub indices: Vec<u32>,
    /// Materials present on the mesh (one texture set per material in the app).
    pub material_names: Vec<String>,
}

impl MeshData {
    /// Number of vertices (length of the parallel attribute vectors).
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles (one third of the index count).
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned bounds of the vertex positions.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        if self.positions.is_empty() {
            return None;
        }
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for p in &self.positions {
            let p = Vec3::new(p[0], p[1], p[2]);
            min = min.min(p);
            max = max.max(p);
        }
        Some((min, max))
    }
}

/// Import error, uniform across formats.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("unsupported format: {0}")]
    Unsupported(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("obj: {0}")]
    Obj(String),
    #[error("gltf: {0}")]
    Gltf(String),
    #[error("fbx: {0}")]
    Fbx(String),
}

/// Detect the import format from a file extension.
pub fn detect_format(path: &std::path::Path) -> Result<Format, ImportError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("obj") => Ok(Format::Obj),
        Some("gltf" | "glb") => Ok(Format::Gltf),
        Some("fbx") => Ok(Format::Fbx),
        Some(other) => Err(ImportError::Unsupported(other.to_string())),
        None => Err(ImportError::Unsupported("(no extension)".into())),
    }
}

/// Supported import formats (SPEC.md deliverable 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Obj,
    Gltf,
    Fbx,
}

/// Load a mesh from disk, dispatching on file extension.
///
/// Supports `.obj` ([`load_obj`]), `.gltf`/`.glb` ([`load_gltf`]) and `.fbx`
/// ([`load_fbx`]). Returns [`ImportError::Unsupported`] for anything else.
pub fn load(path: &std::path::Path) -> Result<MeshData, ImportError> {
    match detect_format(path)? {
        Format::Obj => load_obj(path),
        Format::Gltf => load_gltf(path),
        Format::Fbx => load_fbx(path),
    }
}

/// OBJ import via tobj (Wave 1 baseline; claw hardening in flight).
///
/// Missing normals/UVs are left as whatever tobj produced (possibly short of
/// `positions`); only the glTF/FBX loaders zero-fill. Returns
/// [`ImportError::Obj`] when the file cannot be parsed.
pub fn load_obj(path: &std::path::Path) -> Result<MeshData, ImportError> {
    let (models, materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    )
    .map_err(|e| ImportError::Obj(e.to_string()))?;

    let material_names = match materials {
        Ok(mats) => mats.into_iter().map(|m| m.name).collect(),
        Err(_) => vec!["DefaultMaterial".to_string()],
    };

    let mut data = MeshData {
        material_names,
        ..Default::default()
    };
    for model in models {
        let mesh = model.mesh;
        // Validate attribute lengths before chunking: tobj does not guarantee
        // exact multiples for malformed files (review #1 — silent-drop risk).
        if mesh.positions.len() % 3 != 0
            || mesh.normals.len() % 3 != 0
            || mesh.texcoords.len() % 2 != 0
        {
            return Err(ImportError::Obj(format!(
                "malformed OBJ '{}': non-multiple attribute lengths (pos {}, nrm {}, uv {})",
                model.name,
                mesh.positions.len(),
                mesh.normals.len(),
                mesh.texcoords.len()
            )));
        }
        data.positions
            .extend(mesh.positions.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
        data.normals
            .extend(mesh.normals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
        data.uvs
            .extend(mesh.texcoords.chunks_exact(2).map(|c| [c[0], c[1]]));
        data.indices.extend(mesh.indices.iter().copied());
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_format_routes_by_extension() {
        assert_eq!(
            detect_format(std::path::Path::new("a.OBJ")).unwrap(),
            Format::Obj
        );
        assert_eq!(
            detect_format(std::path::Path::new("a.glb")).unwrap(),
            Format::Gltf
        );
        assert_eq!(
            detect_format(std::path::Path::new("a.fbx")).unwrap(),
            Format::Fbx
        );
        assert!(detect_format(std::path::Path::new("a.stl")).is_err());
    }

    #[test]
    fn bounds_and_counts_are_consistent() {
        let data = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            normals: vec![[0.0; 3]; 2],
            uvs: vec![[0.0; 2]; 2],
            indices: vec![0, 1, 0],
            material_names: vec!["m".into()],
        };
        assert_eq!(data.vertex_count(), 2);
        assert_eq!(data.triangle_count(), 1);
        let (min, max) = data.bounds().unwrap();
        assert_eq!(min, Vec3::ZERO);
        assert_eq!(max, Vec3::new(1.0, 2.0, 3.0));
        assert!(MeshData::default().bounds().is_none());
    }
}

//! glTF 2.0 / GLB mesh import via the [`gltf`] crate.
//!
//! All meshes and primitives in the document are concatenated into a single
//! [`MeshData`](crate::MeshData); per-primitive indices are offset by the
//! vertex base accumulated so far. Missing normals or `TEXCOORD_0` are filled
//! with zeros so the attribute vectors always stay parallel to `positions`.
//! Non-indexed primitives are treated as sequential triangle lists.

use std::path::Path;

use super::{ImportError, MeshData};

/// Load a `.gltf` or `.glb` file into a single concatenated [`MeshData`].
///
/// Returns [`ImportError::Gltf`] when the document cannot be imported, when a
/// buffer view cannot be resolved, or when the document contains no position
/// data at all.
pub fn load_gltf(path: &Path) -> Result<MeshData, ImportError> {
    let (document, buffers, _) =
        gltf::import(path).map_err(|e| ImportError::Gltf(e.to_string()))?;
    let mut data = MeshData::default();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            append_primitive(&mut data, &primitive, &buffers)?;
        }
    }
    if data.positions.is_empty() {
        return Err(ImportError::Gltf("no mesh geometry found".into()));
    }
    Ok(data)
}

/// Append one glTF primitive to the accumulated [`MeshData`], offsetting its
/// indices by the current vertex base.
///
/// Primitives without `POSITION` data are skipped: there is nothing to keep
/// parallel with the rest of the shape.
fn append_primitive(
    data: &mut MeshData,
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<(), ImportError> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .map(|values| values.collect())
        .unwrap_or_default();
    if positions.is_empty() {
        return Ok(());
    }
    let base = data.positions.len() as u32;
    let vertex_count = positions.len() as u32;
    data.positions.extend(positions.iter().copied());

    match reader.read_normals() {
        Some(normals) => data.normals.extend(normals),
        None => data
            .normals
            .extend(std::iter::repeat_n([0.0; 3], positions.len())),
    }

    match reader.read_tex_coords(0) {
        Some(tex_coords) => data.uvs.extend(tex_coords.into_f32()),
        None => data
            .uvs
            .extend(std::iter::repeat_n([0.0; 2], positions.len())),
    }

    match reader.read_indices() {
        Some(indices) => data.indices.extend(indices.into_u32().map(|i| i + base)),
        None => data.indices.extend(base..base + vertex_count),
    }

    if let Some(name) = primitive.material().name() {
        if !data.material_names.iter().any(|n| n == name) {
            data.material_names.push(name.to_string());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Encode bytes as standard base64 (RFC 4648 §4) for data-URI buffers.
    fn base64_encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let mut block: u32 = 0;
            for (i, byte) in chunk.iter().enumerate() {
                block |= (*byte as u32) << (16 - 8 * i);
            }
            let quantum = match chunk.len() {
                1 => {
                    out.push(ALPHABET[(block >> 18) as usize & 63] as char);
                    out.push(ALPHABET[(block >> 12) as usize & 63] as char);
                    "=="
                }
                2 => {
                    out.push(ALPHABET[(block >> 18) as usize & 63] as char);
                    out.push(ALPHABET[(block >> 12) as usize & 63] as char);
                    out.push(ALPHABET[(block >> 6) as usize & 63] as char);
                    "="
                }
                _ => {
                    for shift in [18, 12, 6, 0] {
                        out.push(ALPHABET[(block >> shift) as usize & 63] as char);
                    }
                    ""
                }
            };
            out.push_str(quantum);
        }
        out
    }

    /// Raw little-endian buffer for the triangle fixture: 3×VEC3 positions,
    /// 3×VEC2 uvs, 3×u16 indices (66 bytes total).
    fn triangle_buffer_bytes() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(66);
        for v in [[0.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0], [0.0f32, 1.0, 0.0]] {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        for v in [[0.0f32, 0.0], [1.0f32, 0.0], [0.0f32, 1.0]] {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        bytes
    }

    /// Minimal valid `.gltf` JSON for one triangle with `POSITION` +
    /// `TEXCOORD_0` (no normals: exercises the zero-fill path), an embedded
    /// data-URI buffer, and one named material.
    fn triangle_gltf_json() -> String {
        let uri = base64_encode(&triangle_buffer_bytes());
        format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0}}],
  "meshes": [{{
    "primitives": [{{
      "attributes": {{"POSITION": 0, "TEXCOORD_0": 1}},
      "indices": 2,
      "material": 0
    }}]
  }}],
  "materials": [{{"name": "TestMat"}}],
  "buffers": [{{
    "byteLength": 66,
    "uri": "data:application/octet-stream;base64,{uri}"
  }}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 24}},
    {{"buffer": 0, "byteOffset": 60, "byteLength": 6}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
      "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}},
    {{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}}
  ]
}}"#
        )
    }

    /// Write `contents` to a uniquely named file under the system temp dir and
    /// return its path. Unique per call so tests can run in parallel.
    fn write_fixture(file_name: &str, contents: &[u8]) -> std::path::PathBuf {
        let id = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "umber-mesh-test-{}-{id}-{file_name}",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("fixture write must succeed");
        path
    }

    #[test]
    fn loads_embedded_gltf_triangle() {
        let path = write_fixture("triangle.gltf", triangle_gltf_json().as_bytes());
        let data = crate::load(&path).expect("triangle gltf must load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(data.vertex_count(), 3);
        assert_eq!(data.triangle_count(), 1);
        assert_eq!(
            data.positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(data.normals, vec![[0.0; 3]; 3]);
        assert_eq!(data.uvs, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(data.indices, vec![0, 1, 2]);
        assert_eq!(data.material_names, vec!["TestMat".to_string()]);
    }

    /// Minimal valid `.glb`: JSON chunk (padded to 4 bytes) plus a BIN chunk
    /// carrying the same triangle buffer as the `.gltf` fixture.
    fn triangle_glb_bytes() -> Vec<u8> {
        let json = triangle_gltf_json_for_glb();
        let bin = triangle_buffer_bytes();
        let json_len = json.len().div_ceil(4) * 4;
        let total = 12 + 8 + json_len + 8 + bin.len();
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json_len as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(json.as_bytes());
        glb.extend_from_slice(&vec![b' '; json_len - json.len()]);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\x00");
        glb.extend_from_slice(&bin);
        glb
    }

    /// Same triangle document as [`triangle_gltf_json`], but with the buffer
    /// referencing the GLB BIN chunk (no `uri`).
    fn triangle_gltf_json_for_glb() -> String {
        r#"{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{
    "primitives": [{
      "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
      "indices": 2,
      "material": 0
    }]
  }],
  "materials": [{"name": "TestMat"}],
  "buffers": [{
    "byteLength": 66
  }],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 24},
    {"buffer": 0, "byteOffset": 60, "byteLength": 6}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
      "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ]
}"#
        .to_string()
    }

    #[test]
    fn loads_glb_triangle() {
        let path = write_fixture("triangle.glb", &triangle_glb_bytes());
        let data = crate::load(&path).expect("triangle glb must load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(data.vertex_count(), 3);
        assert_eq!(data.triangle_count(), 1);
        assert_eq!(data.uvs, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(data.indices, vec![0, 1, 2]);
        assert_eq!(data.material_names, vec!["TestMat".to_string()]);
    }
}

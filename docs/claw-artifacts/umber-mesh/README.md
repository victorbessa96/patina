# Claw artifacts: umber-mesh loaders

Hand-off artifacts for the mesh-import claw (the opencode sandbox cannot read
the cargo registry; these are copies of the exact dependency sources vendored
into the repo for reference — delete this directory when the loaders land and
are cross-reviewed).

- `ufbx_prelude.rs` — ufbx 0.11.5 `src/prelude.rs` verbatim (913 lines): the
  entire safe API surface (LoadOptions, load fns, Node/Mesh/VertexStream types)
- `ufbx_lib.rs` — ufbx 0.11.5 `src/lib.rs` verbatim
- `gltf_README.md` — gltf 1.4.1 README (import entry points)
- `gltf_display_example.rs` — gltf 1.4.1 examples/display/main.rs verbatim
- `gltf_lib_tochi_0.txt` — gltf 1.4.1 src/lib.rs text (module map)

## Verified API facts for the brief (checked against the vendored sources 2026-10-09)

gltf 1.4.1:
- `gltf::import(path) -> Result<Import>` and `gltf::import_slice(slice) -> Result<Import>`
  (src/import.rs:273, :311)
- `Import` derefs to `Document`; `doc.meshes()` → `gltf::Mesh`; `mesh.primitives()`
  → `Primitive`
- `Primitive::read_positions() -> Option<ReadPositions>` (mesh/mod.rs:336),
  `read_normals()` (:343), `read_indices()` (:383), `read_tex_coords(set: u32)` (:418)
  — all with `.into_f32()` style collectors
- `primitive.material().name()` for material names

ufbx 0.11.5 (see ufbx_prelude.rs):
- `ufbx::load_from_path(path, &opts)` / `ufbx::load_from_memory(data, &opts)`
- `LoadOptions::default()`; scene.nodes() → `Node`; node.mesh → `Mesh`
- Vertex data via `mesh.vertex_position(&mesh.vertices, index)`, vertex streams:
  `mesh.vertex_streams` with `VertexStream::position/uv/normal` kinds
- Materials: node.materials / mesh.materials, `material.name`

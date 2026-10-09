# umber-mesh loaders — landing notes (Wave-1 claw pass)

## What was built

- `src/gltf.rs` — real glTF/GLB loader (`pub fn load_gltf`).
  - Uses `gltf::import(path)`, which handles both `.gltf` (external + data-URI
    buffers) and `.glb` (BIN chunk) through one entry point.
  - Iterates `document.meshes()` → `primitives()`, reads `POSITION` / `NORMAL`
    / `TEXCOORD_0` / indices via `Primitive::reader` with
    `read_positions()` / `read_normals()` / `read_tex_coords(0).into_f32()` /
    `read_indices().into_u32()`.
  - All primitives/meshes append into one `MeshData` (attribute vecs
    concatenated, indices offset by the running vertex base).
  - Missing normals / `TEXCOORD_0` zero-filled; non-indexed primitives get
    sequential `base..base+count` indices; position-less primitives skipped.
  - Material names collected from `primitive.material().name()` (deduped,
    insertion order). Requires the `names` feature — enabled on the `gltf`
    dependency in `crates/umber-mesh/Cargo.toml` (additive, no other crate
    affected). Without it `name()` would always be `None`.
  - Errors (import failure, zero position data overall) → `ImportError::Gltf`.
- `src/fbx.rs` — real FBX loader (`pub fn load_fbx`).
  - Uses `ufbx::load_file(path, LoadOpts::default())`, walks `&scene.nodes`,
    takes `node.mesh: Option<Ref<Mesh>>`, triangulates each face with
    `ufbx::triangulate_face_vec`.
  - Positions always real (`vertices[vertex_indices[corner]]`, f64→f32).
    UVs/normals via `vertex_uv` / `vertex_normal` (`values[indices[corner]]`)
    when `.exists` and buffers non-empty, else zeros — keeps `MeshData` shape.
  - Vertices expanded per corner (no weld); indices sequential. Material names
    from `mesh.materials[].element.name`.
  - All failures mapped to `ImportError::Fbx`; no panics (`get` + `copied` +
    `ok_or_else`, no `unwrap` in library code — the single
    `unwrap_or_default` in `gltf.rs` is `Option::unwrap_or_default`, panic-free).
- `src/lib.rs` — `load()` now dispatches to `load_gltf` / `load_fbx`; modules
  re-exported; rustdoc on every public fn (also backfilled `load`, `load_obj`,
  `vertex_count`, `triangle_count`). `MeshData` shape untouched.

## Tests added

- `gltf::tests::loads_embedded_gltf_triangle` — builds a minimal valid `.gltf`
  in-test (triangle, `POSITION`+`TEXCOORD_0`, embedded data-URI buffer via a
  small local base64 encoder, one named material), writes it to a unique path
  under `std::env::temp_dir()`, loads via `crate::load`, asserts 3 verts /
  1 tri / exact positions / zero-filled normals / exact UVs / indices /
  `["TestMat"]`.
- `gltf::tests::loads_glb_triangle` — same triangle as hand-assembled minimal
  `.glb` (JSON chunk + BIN chunk), asserts verts/tris/UVs/indices/material.
- `fbx::tests::loads_fbx_triangle_fixture` — `#[ignore]`d, with TODO naming
  the exact fixture it needs (`crates/umber-mesh/tests/assets/triangle.fbx`,
  still to be checked in). No fake pass.

## Anything unverified / assumptions for the reviewer

1. **FBX loader has never run against a real file** (no fixture exists yet;
   test is `#[ignore]` by design). Highest-risk assumption: `triangulate_face_vec`
   outputs *absolute* mesh corner indices usable directly against
   `vertex_indices` / attribute index arrays (matches ufbx C-example usage).
   Un-ignore the test the moment `tests/assets/triangle.fbx` lands.
2. **Brief vs reality (ufbx 0.11.5):** the hand-off README names
   `ufbx::load_from_path` + `LoadOptions` + `vertex_streams` — none exist.
   Actual API (confirmed via compiler): `ufbx::load_file(&str, LoadOpts)` (opts
   by value), `scene.nodes: RefList<Node>`, `node.mesh: Option<Ref<Mesh>>`,
   `Mesh { vertices: List<Vec3>, faces: List<Face>, vertex_indices: List<u32>,
   vertex_position: VertexVec3, vertex_uv: VertexVec2, vertex_normal:
   VertexVec3, materials: RefList<Material> }`,
   `VertexVec* { values, indices, exists }`, material name at
   `material.element.name`, `ufbx::Error: Debug` but **no** `Display` (errors
   formatted with `{e:?}`). The artifacts README should be corrected.
3. FBX: node transforms and skinning ignored (bind pose); a mesh instanced on
   several nodes is appended once per node; per-face material slots not
   propagated (mesh-level names only). glTF: non-`TRIANGLES` primitive modes
   imported as-is (not reinterpreted); only `TEXCOORD_0` read.
4. Future optimization (noted in `fbx.rs` docs): per-corner expansion
   duplicates positions — add a weld pass when memory matters; UV seams stay
   correct either way.

## Reviewer checklist

- [ ] `cargo fmt --check -p umber-mesh` clean (ran `cargo fmt`).
- [ ] `cargo clippy -p umber-mesh --all-targets -- -D warnings` clean (ran).
- [ ] `cargo test -p umber-mesh` → 4 passed, 1 ignored (ran).
- [ ] Confirm `gltf` `names` feature addition in `crates/umber-mesh/Cargo.toml`
      is acceptable (needed for real material names).
- [ ] Check in `crates/umber-mesh/tests/assets/triangle.fbx`, remove
      `#[ignore]`, run `cargo test -p umber-mesh -- --ignored`.
- [ ] Fix or remove `docs/claw-artifacts/umber-mesh/` (stale ufbx API claims).
- [ ] No other crates touched; no git operations performed.

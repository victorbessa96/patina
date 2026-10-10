// umber plugin wire format v1 — the SHARED constants file.
//
// This file exists in TWO places, byte-identical:
//   crates/umber-wasm/src/wire_consts.rs        (the host, canonical)
//   crates/umber-plugin-sdk/src/wire_consts.rs  (the guest SDK, a copy)
// The SDK cannot depend on umber-wasm (wasmtime is host-only) and a
// cross-crate `#[path]` include breaks packaging, so the file is copied
// verbatim and umber-wasm's test `sdk_wire_consts_are_byte_identical`
// asserts the two copies are equal byte-for-byte. Edit the canonical
// copy, then `cp` it over the SDK's. Layout prose: crates/umber-wasm/WIRE.md.
//
// Plain `pub const` items only: no imports, no std, so the file compiles
// unchanged in the no_std SDK and the std host.

/// Region magic: the ASCII bytes `MBER`, read as a little-endian u32.
pub const WIRE_MAGIC: u32 = u32::from_le_bytes(*b"MBER");

/// Node-definition record magic: the ASCII bytes `MDEF`, little-endian.
pub const NODE_DEF_MAGIC: u32 = u32::from_le_bytes(*b"MDEF");

/// Image region header: `[u32 magic][u32 w][u32 h]`.
pub const IMAGE_HEADER_LEN: usize = 12;

/// Node-definition record header:
/// `[u32 magic][u32 n_inputs][u32 n_params][u32 name_len]`.
pub const NODE_DEF_HEADER_LEN: usize = 16;

/// v1: exactly one image input per plugin node.
pub const MAX_INPUTS: u32 = 1;

/// v1: at most four params per plugin node (the design's boundary).
pub const MAX_PARAMS: u32 = 4;

/// Longest node-def name, in UTF-8 bytes.
pub const MAX_NAME_LEN: u32 = 64;

/// Param kinds (the TLV `kind` field) — mirrors umber-graph's
/// `ParamValue` minus `NodeRef` (references are wiring, sent as the
/// image input, never as a param).
pub const KIND_FLOAT: u32 = 0; // f32
pub const KIND_VEC2: u32 = 1; // [f32; 2]
pub const KIND_VEC3: u32 = 2; // [f32; 3]
pub const KIND_COLOR: u32 = 3; // [f32; 3]
pub const KIND_INT: u32 = 4; // i32
pub const KIND_BOOL: u32 = 5; // u32, 0 or 1
pub const KIND_ASSET: u32 = 6; // UTF-8 bytes

/// `umber_eval` error returns (negative i32; non-negative is out_len).
pub const ERR_BAD_MAGIC: i32 = -1;
pub const ERR_TRUNCATED: i32 = -2;
pub const ERR_OUT_OF_CAPACITY: i32 = -3;
pub const ERR_GUEST_INTERNAL: i32 = -4;

/// Export names the host binds.
pub const EXPORT_NODE_DEF: &str = "umber_node_def";
pub const EXPORT_EVAL: &str = "umber_eval";
/// The canonical linear-memory export name. rustc's wasm32 linker
/// exports memory as `memory` and Rust source cannot rename it, so
/// the host accepts `EXPORT_MEMORY_FALLBACK` too (see WIRE.md).
pub const EXPORT_MEMORY: &str = "umber_mem";
pub const EXPORT_MEMORY_FALLBACK: &str = "memory";

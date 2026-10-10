# umber plugin wire format — v1

The byte contract between the host (`umber-wasm`) and a guest node
plugin (built with `umber-plugin-sdk`). Design:
`docs/specs/wasm-plugins-design.md`. All integers are **little-endian**;
floats are IEEE-754 `f32` bit patterns (NaN payloads survive). The
constants below live in `src/wire_consts.rs`, copied byte-identically to
`crates/umber-plugin-sdk/src/wire_consts.rs` (a test asserts equality —
edit the umber-wasm copy, then `cp` it over).

## Exports the guest provides

| export | signature | purpose |
|---|---|---|
| `umber_node_def` | `() -> i32` | pointer to the node-def record (static guest data) |
| `umber_eval` | `(in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32) -> i32` | one evaluation |
| `umber_mem` | memory | the linear memory both regions live in |

**Memory export name.** `umber_mem` is canonical (the WAT fixtures use
it). rustc's `wasm32-unknown-unknown` linker exports the memory as
`memory`, and Rust source can't rename it, so the host accepts `memory`
as a fallback. A module with neither fails with
`PluginError::MissingExport("umber_mem")`.

**No imports.** A module that imports anything is rejected at load
(`PluginError::BadModule`). v1 plugins are pure functions.

**Arity vs the design doc.** The design sketches a six-argument
`umber_eval(input_ptr, input_len, params_ptr, params_len, out_ptr,
out_cap)`. v1 carries params INSIDE the input region (the README sketch:
one contiguous buffer), so the export takes four arguments.

## Node-def record (`MDEF`)

```
[u32 NODE_DEF_MAGIC = "MDEF"][u32 n_inputs][u32 n_params][u32 name_len][name: name_len bytes]
```

- `n_inputs` must be `1` (v1: one image input).
- `n_params` ≤ 4 (declared, informational v1 — the mtlx slice will use it).
- `name`: 1..=64 bytes of `[A-Za-z0-9_-]`. It becomes the registry key
  and the `.mtlx` nodedef name, so no XML or whitespace can get in.

The host reads the 16-byte header at the returned pointer, then
`name_len` more bytes.

## Input region (`MBER`), written by the host at `in_ptr`

```
[u32 WIRE_MAGIC = "MBER"][u32 w][u32 h][RGBA8: w*h*4 bytes, row-major, top row first]
[u32 param_count]
per param: [u32 kind][u32 len][payload: len bytes]
    payload = [u32 name_len][name UTF-8: name_len bytes][value]
```

The sketch's TLV has no name slot; umber-graph params are NAMED, so the
name rides at the front of the payload (TLV framing unchanged:
`len` covers name + value).

| kind | const | value bytes |
|---|---|---|
| 0 | `KIND_FLOAT` | f32 (4) |
| 1 | `KIND_VEC2` | 2 × f32 (8) |
| 2 | `KIND_VEC3` | 3 × f32 (12) |
| 3 | `KIND_COLOR` | 3 × f32 (12) |
| 4 | `KIND_INT` | i32 (4) |
| 5 | `KIND_BOOL` | u32, 0 or 1 (4) |
| 6 | `KIND_ASSET` | UTF-8 path (len − 4 − name_len) |

- `NodeRef` params never cross. They are wiring, and the engine resolves
  them to the image input before dispatch.
- At most 4 params cross (`MAX_PARAMS`). More than that is
  `EvalError::BadParam` on the host.
- The region must end exactly after the last param. Trailing bytes are
  malformed.

## Out region (`MBER`), written by the guest at `out_ptr`

```
[u32 WIRE_MAGIC][u32 w][u32 h][RGBA8: w*h*4]
```

The host reserves `out_cap = 12 + in_w*in_h*4`, enough for an output
image the input's size. The return value is:

- `>= 0`: `out_len`, the bytes written. The host requires
  `out_len <= out_cap` and `out_len == 12 + w*h*4`, and checks the magic.
- `< 0`: an error code.

| code | const | meaning |
|---|---|---|
| −1 | `ERR_BAD_MAGIC` | input magic wrong |
| −2 | `ERR_TRUNCATED` | input truncated or structurally malformed |
| −3 | `ERR_OUT_OF_CAPACITY` | output would not fit `out_cap` |
| −4 | `ERR_GUEST_INTERNAL` | anything else (mistyped param, plugin-specific) |

Every error code, an unknown negative, a malformed out region, or a trap
becomes `EvalError::Plugin { node, reason }`. Fuel exhaustion becomes
`EvalError::FuelExhausted { node }`.

## Placement and limits (host)

Each eval runs in a fresh store and instance. Before the call the host
grows the guest memory and places the regions past its old end:

```
[guest statics + stack ...][old end = in_ptr: input][pad to 8][out_ptr: out region]
```

A v1 SDK guest has no allocator, so nothing in the guest can collide
with those pages.

- **Memory cap:** guest memory is capped at 64 MiB (`MEMORY_CAP`).
  Input plus output must fit, which is about 2896² for a square image.
- **Fuel:** `FUEL` = 2·10⁹ per call (`umber_node_def` at load,
  `umber_eval` per eval), set with `Store::set_fuel` on each fresh
  store.
- **Fuel can run out before memory does.** A blur-class plugin costs
  about 100 operators per byte, so this budget covers about 2048². That
  is below the memory cap, so a heavy plugin on a near-cap image can
  end as `FuelExhausted`. Cheap per-pixel plugins stay far under the
  budget.

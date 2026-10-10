# Plugins (WASM Node Extensions) — Wave-6 Design

Wave-6's third slice. Written 2026-10-10 08:04 against the tree at
`a6d297f`. The audit's wave-6 list names "plugins" without a shape;
the requirements' §4 row implies the surface ("painter-specific
nodes ship as declared custom nodedefs"). This design pins the
honest v1: **custom node IMPLS in guest WASM**, executed by the
node-graph engine — the same NodeImpl contract the built-in nodes
implement, callable from a sandboxed module.

## The shape

**Host side (umber-graph + a new umber-wasm crate):**
- The engine's `NodeImpl::eval` is already a pure function over
  (inputs, params, ctx) -> NodeOutput. The WASM bridge compiles a
  guest module implementing the same CONTRACT via a C-ABI export
  the host binds:
  - `umber_node_def() -> (name, n_params, n_inputs)` — registration
  - `umber_eval(input_ptr, input_len, params_ptr, params_len,
    out_ptr, out_cap) -> out_len | error_code` — the eval call
- Data crossing the boundary: the host serializes inputs+params
  into a compact binary layout (the design's v1: the
  NodeOutput/ImageBuffer bytes + a params TLV — little-endian,
  documented in the crate's WIRE.md), the guest writes the output
  image into the caller-provided buffer. No host allocations from
  the guest, no panics across (the error codes enumerate: bad
  magic, truncation, out-of-capacity, guest-internal).
- Fuel metering + epoch interruption (wasmtime's Store::fuel) —
  a hung node can't hang the graph; the engine's eval error
  surfaces it. Memory cap on the guest linear memory (64 MiB v1).
- **No host imports v1** — no filesystem, no network, no clock: a
  node plugin is a pure function. (The honest boundary; host
  capabilities arrive with a use-case that demands them, each one
  a deliberate capability grant.)

**Guest side (the SDK):**
- A tiny `umber-plugin-sdk` crate (no_std-ish, wasm32 target) with
  the wire types + a macro: `#[umber_node] fn my_node(inputs:
  NodeInputs) -> Result<NodeOutput, NodeError>` — the macro emits
  the C-ABI shims. Compiled with `--target wasm32-unknown-unknown`
  (or wasm32-wasi if the sdk needs std — v1: no_std, documented).
- Two example plugins in-repo: `examples/plugin-blur5` (a 5px box
  blur — the trivial reference) and `examples/plugin-vignette`
  (a radial darkening — reads a param, writes an image: proves
  the param path).

**Registry integration:** NodeRegistry gains
`register_wasm(bytes) -> Result<String, PluginError>` (the name
from umber_node_def); the eval path dispatches through the same
NodeImpl trait via a WasmNodeImpl adapter. The graph panel's
add-node menu lists WASM nodes beside the built-ins; .mtlx
serialization treats them as custom nodedefs declared at save
(EXACTLY like flood_fill et al — the interchange contract extends
naturally: the plugin's def name + its params).

## The honest v1 boundary

- CPU-image nodes only (the GPU node path is a wave-7 question).
- One image input + up to 4 params v1 (the wire format reserves
  counts; multiple image inputs arrive with the format rev).
- Sync eval in the engine's existing eval path (wasmtime on a
  worker thread is the async-bake pattern — a follow-up if the
  fuel costs show up in the perf numbers).

## Tests (the can-fail cores)

1. The reference plugin round-trip: blur5 compiled to .wasm
   (checked in as a binary fixture + built from source in CI),
   registered, evaluated on the test gradient — output EXACTLY
   the expected 5px box blur (the mirrored-math assert against the
   native blur node's output on the same input: WASM and native
   agree byte-for-byte — the strongest equivalence test).
2. Fuel exhaustion: a plugin with an infinite loop -> the engine
   errors with the fuel code (assert the variant), the graph
   survives, other nodes re-evaluate fine after.
3. Bad magic/truncation: a truncated .wasm and a garbage file ->
  PluginError variants (never a panic).
4. The vignette param path: a param change changes the output
   (before/after differ — the can-fail).
5. mtlx: a graph with a wasm node saves/loads; the nodedef
   declaration carries the plugin's name+params.

## Build slices

1. umber-wasm: the runtime bridge + wire format + WasmNodeImpl
   (wasmtime dep — the one new dependency this wave; vendored
   source into docs/claw-artifacts/ per the dispatch recipe).
2. The SDK + the two example plugins + their fixtures.
3. The registry/panel/mtlx integration + the equivalence suite.

Sizes: 1 is the deep slice; 2 is mostly macro plumbing; 3 rides.

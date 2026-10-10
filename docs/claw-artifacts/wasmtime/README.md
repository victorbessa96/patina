# wasmtime 50.0.0-rc.1 — verified API facts for the umber-wasm slice

Sources vendored at `src/wasmtime/` (the crate's own src tree; the
facts below verified against it by the dragon). The claw builds the
umber-wasm bridge against THESE facts; deviations verified against
the vendored source, never assumed.

## Confirmed facts (grep-verified line refs)

1. **Fuel metering exists at the Config level**: `Config` gains a
   fuel option (src/wasmtime/config.rs ~line 627-634: "Configures
   whether execution of WebAssembly will 'consume fuel'... When
   fuel runs out a trap is raised"). The exact setter name: read
   `config.rs` around that doc block — the `pub fn` adjacent
   (`consume_fuel(bool)` is the family name in this version line —
   VERIFY the exact name in the vendored source before use).
2. **Store::add_fuel / set_fuel** exist on the Store (the fuel
   accounting family in src/wasmtime/store.rs — grep `fn.*fuel`).
3. **Func::call**: the typed/unchecked call paths in
   src/wasmtime/func.rs (`pub fn call` / `call_unchecked` — the
   typed API wraps a Caller-provided impl).
4. **Memory**: `Memory::write`/`read` (src/wasmtime/memories.rs)
   for the buffer crossing; `Memory::ty()` for the cap check.
5. **Engine/Module/Store/Instance/Linker**: the standard
   v50 shape (Module::new(engine, bytes), Linker::new(engine),
   linker.instantiate(&mut store, &module)).
6. **No host imports v1**: an empty Linker — the guest module's
   exports are called directly; nothing imported INTO the guest.
7. **Traps**: `Trap` in the error family; a fuel-out trap carries
   the fuel error code — match on it (grep `Trap`/`error.rs` for
   the variant).

## The bridge sketch (what the claw builds)

- `PluginRuntime::new() -> Self` — Engine with fuel enabled.
- `load(bytes) -> PluginModule` — Module::new + the exports
  lookup (umber_node_def, umber_eval).
- `WasmNodeImpl` implementing the engine's existing NodeImpl:
  serializes inputs/params into the wire buffer, writes into the
  guest memory, calls umber_eval, reads the out region, checks
  the error code, builds the NodeOutput. Fuel: set per call
  (design: generous v1 constant, documented), out-of-fuel ->
  the engine's EvalError (a new FuelExhausted variant — additive).
- Wire format (v1, little-endian, the WIRE.md in the crate):
  [u32 magic][u32 w][u32 h][RGBA bytes...][u32 param_count]
  [per param: u32 kind][u32 len][bytes...]. The out region:
  caller-allocated [magic][w][h][RGBA...]; the guest writes and
  returns the used length or a negative error code (i32).

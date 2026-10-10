//! The wasmtime bridge: [`PluginRuntime`] → [`PluginModule`] →
//! [`WasmNodeImpl`] (an [`umber_graph::NodeImpl`]).
//!
//! API facts verified against the vendored wasmtime 50.0.0-rc.1 source
//! (`docs/claw-artifacts/wasmtime/src/wasmtime/`):
//! * `Config::consume_fuel(bool)` — config.rs:643 (the README's family
//!   name; confirmed exact for this version).
//! * `Store::set_fuel(u64)` / `get_fuel` — store.rs:1016/989. There is
//!   no `add_fuel` in this version; each call gets a fresh store and a
//!   `set_fuel`.
//! * Fuel exhaustion traps with `Trap::OutOfFuel` (vm/libcalls.rs:738),
//!   recovered via `Error::downcast_ref::<Trap>()` (trap.rs:39-69).
//! * `Memory::read`/`write`/`grow`/`data_size` — memory.rs;
//!   `StoreLimitsBuilder::memory_size` + `Store::limiter` — limits.rs,
//!   store.rs:930.
//!
//! # Isolation model
//!
//! The compiled [`Module`] is shared; every eval builds a FRESH
//! [`Store`] + instance. A trapped or fuel-exhausted guest therefore
//! leaves nothing behind — the next eval (of this node or any other)
//! starts clean, which is what lets the graph survive a hung plugin.
//! Instantiation of a small no-import module is cheap next to the image
//! work; pooling is a follow-up if the perf numbers ask for it.
//!
//! # Buffer placement
//!
//! "No host allocations from the guest": the host grows the guest's
//! linear memory by enough pages for `in_len + out_cap` and writes the
//! input region at the old end of memory, the out region right after
//! it. The guest's statics and stack live below the old end, and a v1
//! SDK guest has no allocator, so nothing in the guest can collide with
//! the host's pages. The grow is subject to [`MEMORY_CAP`].

use crate::wire::{self, NodeDef, WireError};
use crate::wire_consts::{
    ERR_BAD_MAGIC, ERR_GUEST_INTERNAL, ERR_OUT_OF_CAPACITY, ERR_TRUNCATED, EXPORT_EVAL,
    EXPORT_MEMORY, EXPORT_MEMORY_FALLBACK, EXPORT_NODE_DEF, NODE_DEF_HEADER_LEN,
};
use crate::PluginError;
use umber_graph::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use wasmtime::{
    Config, Engine, Instance, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, Trap,
    TypedFunc,
};

/// Fuel granted to EACH guest call (`umber_node_def` at registration,
/// `umber_eval` per evaluation). Fuel is roughly one unit per wasm
/// operator, and a hung guest is cut off within a few seconds.
///
/// The honest bound: an unoptimized-looking plugin like the reference
/// blur5 costs on the order of ~100 operators per byte, so 2·10⁹ covers
/// roughly a 2048² RGBA image. That is BELOW the [`MEMORY_CAP`] limit
/// (~2896²): a heavy plugin on a near-cap image can be cut off as
/// `FuelExhausted` despite making progress. Cheap per-pixel plugins
/// (invert, vignette) are far under budget at the cap. Tune per
/// runtime with [`PluginRuntime::with_fuel`]; a size-scaled budget is
/// the follow-up if real plugins hit this.
pub const FUEL: u64 = 2_000_000_000;

/// Cap on a guest's linear memory (the design's 64 MiB v1). The input
/// and out regions live in guest memory, so this bounds the image size
/// a plugin can process: `2 * (12 + w*h*4)` plus the guest's own pages
/// must fit — about 2896² for a square image.
pub const MEMORY_CAP: usize = 64 << 20;

/// wasm page size (the core spec's fixed 64 KiB).
const PAGE: usize = 65_536;

/// `umber_eval(in_ptr, in_len, out_ptr, out_cap) -> out_len | error`.
type EvalFn = TypedFunc<(i32, i32, i32, i32), i32>;

/// A compiled-plugin factory: one wasmtime [`Engine`] (fuel metering
/// on) and the per-call fuel budget.
#[derive(Clone)]
pub struct PluginRuntime {
    engine: Engine,
    fuel: u64,
}

impl Default for PluginRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRuntime {
    /// A runtime with fuel metering enabled and the [`FUEL`] budget.
    ///
    /// # Panics
    ///
    /// Only if wasmtime cannot build an engine for this host at all (an
    /// unsupported target — a build property, not an input). The config
    /// sets nothing but fuel metering.
    #[must_use]
    pub fn new() -> Self {
        Self::with_fuel(FUEL)
    }

    /// [`Self::new`] with a custom per-call fuel budget (tests use a
    /// small one so the infinite-loop case fails fast).
    ///
    /// # Panics
    ///
    /// As [`Self::new`].
    #[must_use]
    pub fn with_fuel(fuel: u64) -> Self {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine =
            Engine::new(&config).expect("wasmtime engine for this host (fuel-only config)");
        Self { engine, fuel }
    }

    /// The per-call fuel budget.
    #[must_use]
    pub fn fuel(&self) -> u64 {
        self.fuel
    }

    /// Compiles `bytes` (wasm binary; WAT text too, via wasmtime's
    /// default `wat` feature), checks the plugin contract, and reads
    /// the node def by calling `umber_node_def` under fuel.
    ///
    /// # Errors
    ///
    /// [`PluginError::BadModule`] — not wasm, fails validation, or
    /// declares imports (v1 plugins get none); [`PluginError::MissingExport`]
    /// — an export is absent or has the wrong type;
    /// [`PluginError::BadWire`] — the node-def record is malformed or
    /// outside v1 limits; [`PluginError::FuelExhausted`] /
    /// [`PluginError::GuestPanic`] — `umber_node_def` ran away / trapped.
    pub fn load(&self, bytes: &[u8]) -> Result<PluginModule, PluginError> {
        let module = Module::new(&self.engine, bytes)
            .map_err(|e| PluginError::BadModule(format!("{e:#}")))?;
        if let Some(import) = module.imports().next() {
            return Err(PluginError::BadModule(format!(
                "v1 plugins take no host imports, found {}::{}",
                import.module(),
                import.name()
            )));
        }
        let mut plugin = PluginModule {
            engine: self.engine.clone(),
            module,
            fuel: self.fuel,
            def: NodeDef {
                name: String::new(),
                n_inputs: 0,
                n_params: 0,
            },
        };
        plugin.def = plugin.read_node_def()?;
        Ok(plugin)
    }
}

/// A live guest: a fresh store + instance + the bound exports.
struct Guest {
    store: Store<StoreLimits>,
    memory: Memory,
    eval: EvalFn,
    node_def: TypedFunc<(), i32>,
}

/// How a guest call went wrong, before mapping to the caller's error.
enum CallFailure {
    Fuel,
    Trap(String),
}

fn classify(err: &wasmtime::Error) -> CallFailure {
    if matches!(err.downcast_ref::<Trap>(), Some(Trap::OutOfFuel)) {
        CallFailure::Fuel
    } else {
        CallFailure::Trap(format!("{err:#}"))
    }
}

/// A compiled plugin whose contract checked out: its [`NodeDef`] is
/// known. Turn it into a registry entry with [`Self::into_node_impl`].
pub struct PluginModule {
    engine: Engine,
    module: Module,
    fuel: u64,
    def: NodeDef,
}

impl PluginModule {
    /// The plugin's self-description.
    #[must_use]
    pub fn def(&self) -> &NodeDef {
        &self.def
    }

    /// The [`NodeImpl`] adapter for the engine's registry.
    #[must_use]
    pub fn into_node_impl(self) -> WasmNodeImpl {
        WasmNodeImpl { plugin: self }
    }

    fn instantiate(&self) -> Result<Guest, PluginError> {
        let limits = StoreLimitsBuilder::new().memory_size(MEMORY_CAP).build();
        let mut store = Store::new(&self.engine, limits);
        // The store.rs:912 doc form (`|state| &mut state.limits`), with
        // the limits AS the store data.
        store.limiter(|limits| limits);
        store
            .set_fuel(self.fuel)
            .map_err(|e| PluginError::BadModule(format!("fuel: {e:#}")))?;
        let instance =
            Instance::new(&mut store, &self.module, &[]).map_err(|e| match classify(&e) {
                // A start function can run (and hang/trap) at instantiation.
                CallFailure::Fuel => PluginError::FuelExhausted,
                CallFailure::Trap(msg) => PluginError::BadModule(msg),
            })?;
        let memory = instance
            .get_memory(&mut store, EXPORT_MEMORY)
            .or_else(|| instance.get_memory(&mut store, EXPORT_MEMORY_FALLBACK))
            .ok_or(PluginError::MissingExport(EXPORT_MEMORY))?;
        let node_def = typed(&instance, &mut store, EXPORT_NODE_DEF)?;
        let eval = typed(&instance, &mut store, EXPORT_EVAL)?;
        Ok(Guest {
            store,
            memory,
            eval,
            node_def,
        })
    }

    fn read_node_def(&self) -> Result<NodeDef, PluginError> {
        let mut g = self.instantiate()?;
        let ptr = g
            .node_def
            .call(&mut g.store, ())
            .map_err(|e| match classify(&e) {
                CallFailure::Fuel => PluginError::FuelExhausted,
                CallFailure::Trap(msg) => PluginError::GuestPanic(msg),
            })?;
        let base = usize::try_from(ptr as u32).expect("u32 fits usize");
        let mut header = [0u8; NODE_DEF_HEADER_LEN];
        g.memory
            .read(&g.store, base, &mut header)
            .map_err(|_| PluginError::BadWire("node def pointer out of bounds"))?;
        let (_, _, name_len) = wire::decode_node_def_header(&header).map_err(wire_static)?;
        let mut record = vec![0u8; NODE_DEF_HEADER_LEN + name_len];
        g.memory
            .read(&g.store, base, &mut record)
            .map_err(|_| PluginError::BadWire("node def name out of bounds"))?;
        wire::decode_node_def(&record).map_err(wire_static)
    }

    /// One guest eval: fresh instance, regions written, `umber_eval`
    /// called under fuel, the out region read back and validated.
    fn call_eval(&self, input: &[u8], out_cap: usize) -> Result<Vec<u8>, EvalError> {
        let plugin_err = |reason: String| EvalError::Plugin { node: 0, reason };
        let mut g = self.instantiate().map_err(|e| match e {
            PluginError::FuelExhausted => EvalError::FuelExhausted { node: 0 },
            other => plugin_err(other.to_string()),
        })?;

        // Place both regions past the guest's current memory (see the
        // module docs): in at `base`, out 8-aligned right after.
        let base = g.memory.data_size(&g.store);
        let out_at = (base + input.len()).next_multiple_of(8);
        let end = out_at + out_cap;
        let pages = (end - base).div_ceil(PAGE) as u64;
        g.memory.grow(&mut g.store, pages).map_err(|_| {
            plugin_err(format!(
                "regions of {} bytes exceed the {} MiB plugin memory cap",
                input.len() + out_cap,
                MEMORY_CAP >> 20
            ))
        })?;
        let as_i32 =
            |v: usize| i32::try_from(v).map_err(|_| plugin_err("region offset beyond i32".into()));
        let (in_ptr, in_len, out_ptr, cap) = (
            as_i32(base)?,
            as_i32(input.len())?,
            as_i32(out_at)?,
            as_i32(out_cap)?,
        );
        g.memory
            .write(&mut g.store, base, input)
            .map_err(|_| plugin_err("input region out of bounds after grow".into()))?;

        let ret = g
            .eval
            .call(&mut g.store, (in_ptr, in_len, out_ptr, cap))
            .map_err(|e| match classify(&e) {
                CallFailure::Fuel => EvalError::FuelExhausted { node: 0 },
                CallFailure::Trap(msg) => plugin_err(format!("guest trapped: {msg}")),
            })?;
        if ret < 0 {
            return Err(plugin_err(guest_code_reason(ret)));
        }
        let out_len = ret as usize;
        if out_len > out_cap {
            return Err(plugin_err(format!(
                "guest reported out_len {out_len} past out_cap {out_cap}"
            )));
        }
        let mut out = vec![0u8; out_len];
        g.memory
            .read(&g.store, out_at, &mut out)
            .map_err(|_| plugin_err("out region out of bounds".into()))?;
        Ok(out)
    }
}

fn typed<P, R>(
    instance: &Instance,
    store: &mut Store<StoreLimits>,
    name: &'static str,
) -> Result<TypedFunc<P, R>, PluginError>
where
    P: wasmtime::WasmParams,
    R: wasmtime::WasmResults,
{
    // Absent and wrongly-typed are the same contract break to a caller:
    // the export the host binds isn't there.
    instance
        .get_typed_func::<P, R>(store, name)
        .map_err(|_| PluginError::MissingExport(name))
}

fn wire_static(e: WireError) -> PluginError {
    PluginError::BadWire(match e {
        WireError::BadMagic(_) => "node def magic is not MDEF",
        WireError::Truncated(_) => "node def record truncated",
        WireError::Trailing(_) => "node def record has trailing bytes",
        WireError::Invalid(what) => what,
    })
}

fn guest_code_reason(code: i32) -> String {
    match code {
        ERR_BAD_MAGIC => "guest rejected the input region: bad magic".into(),
        ERR_TRUNCATED => "guest rejected the input region: truncated".into(),
        ERR_OUT_OF_CAPACITY => "guest output exceeds the out region capacity".into(),
        ERR_GUEST_INTERNAL => "guest internal error".into(),
        other => format!("guest returned unknown error code {other}"),
    }
}

/// [`NodeImpl`] over a WASM plugin. Inputs follow the filter-node
/// convention: exactly one `Image` input (any name). Params: every
/// non-`NodeRef` param crosses the wire by name (at most four).
pub struct WasmNodeImpl {
    plugin: PluginModule,
}

impl WasmNodeImpl {
    /// The plugin's self-description.
    #[must_use]
    pub fn def(&self) -> &NodeDef {
        &self.plugin.def
    }
}

impl NodeImpl for WasmNodeImpl {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let img = match inputs.as_slice() {
            [(_, NodeOutput::Image(buf))] => buf,
            [(_, other)] => {
                return Err(EvalError::TypeMismatch {
                    node: 0,
                    expected: "Image".into(),
                    got: other.kind().into(),
                });
            }
            [] => {
                return Err(EvalError::MissingInput {
                    node: 0,
                    input: "in".into(),
                });
            }
            many => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: format!("expected exactly 1 input, got {}", many.len()),
                });
            }
        };
        let input = wire::encode_input(img, params).map_err(|e| EvalError::BadParam {
            node: 0,
            param: e.to_string(),
        })?;
        let out = self.plugin.call_eval(&input, wire::out_capacity(img))?;
        let buf = wire::decode_output(&out).map_err(|e| EvalError::Plugin {
            node: 0,
            reason: format!("malformed out region: {e}"),
        })?;
        Ok(NodeOutput::Image(buf))
    }
}

//! umber-wasm — WASM plugin nodes for the node-graph engine.
//!
//! Wave-6 plugins slice 1 (`docs/specs/wasm-plugins-design.md`): a
//! guest module implementing the node CONTRACT through two C-ABI
//! exports is compiled by wasmtime and dispatched through the same
//! [`umber_graph::NodeImpl`] trait the built-in nodes implement.
//!
//! * [`wire`] — the v1 byte layout crossing the boundary (`WIRE.md`).
//! * [`runtime`] — [`PluginRuntime`] (fuel-metered engine),
//!   [`PluginModule`] (compiled + contract-checked), [`WasmNodeImpl`].
//! * [`register_wasm`] — compile + check + register in one call.
//!
//! The honest v1 boundary: CPU-image nodes, one image input, up to four
//! params, sync eval, NO host imports (a plugin is a pure function), a
//! fuel budget per call, a 64 MiB guest-memory cap.

pub mod runtime;
pub mod wire;
pub mod wire_consts;

pub use runtime::{PluginModule, PluginRuntime, WasmNodeImpl, FUEL, MEMORY_CAP};
pub use wire::{NodeDef, WireError};

use std::sync::Arc;
use umber_graph::NodeRegistry;

/// Everything that can go wrong loading/registering a plugin. Eval-time
/// failures surface as [`umber_graph::EvalError`] instead
/// (`FuelExhausted` / `Plugin`), never as a panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PluginError {
    /// wasmtime rejected the bytes (not wasm, invalid, truncated) or the
    /// module breaks a v1 rule (declares host imports). Carries
    /// wasmtime's error string.
    #[error("bad module: {0}")]
    BadModule(String),
    /// A required export (`umber_node_def`, `umber_eval`, the linear
    /// memory) is absent or has the wrong type.
    #[error("missing or mistyped export {0:?}")]
    MissingExport(&'static str),
    /// The node-def record the guest returned is malformed or outside
    /// the v1 limits.
    #[error("bad wire data: {0}")]
    BadWire(&'static str),
    /// A guest call ran out of fuel (a hung `umber_node_def`).
    #[error("plugin ran out of fuel")]
    FuelExhausted,
    /// A guest call trapped (a Rust guest's panic is an `unreachable`
    /// trap). Carries wasmtime's trap description.
    #[error("plugin trapped: {0}")]
    GuestPanic(String),
}

/// Compiles `bytes` with a default [`PluginRuntime`], checks the plugin
/// contract, and registers the node under the name its
/// `umber_node_def` record declares. Returns that name.
///
/// Like [`NodeRegistry::register`], a name already present is replaced
/// (a plugin named `blur` shadows the built-in — callers who care can
/// check [`NodeRegistry::get`] against [`PluginModule::def`] first via
/// [`register_module`]).
///
/// # Errors
///
/// Any [`PluginError`] from [`PluginRuntime::load`]; the registry is
/// untouched on error.
pub fn register_wasm(registry: &mut NodeRegistry, bytes: &[u8]) -> Result<String, PluginError> {
    let module = PluginRuntime::new().load(bytes)?;
    Ok(register_module(registry, module))
}

/// Registers an already-loaded [`PluginModule`] (a custom-fuel runtime,
/// or a caller that inspected [`PluginModule::def`] first). Returns the
/// registered name.
pub fn register_module(registry: &mut NodeRegistry, module: PluginModule) -> String {
    let name = module.def().name.clone();
    registry.register(&name, Arc::new(module.into_node_impl()));
    name
}

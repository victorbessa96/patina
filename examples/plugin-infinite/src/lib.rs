//! `infinite`: a deliberately hung plugin — the fuel-exhaustion test
//! fixture (umber-wasm `tests/fixtures.rs`; its hand-written WAT twin
//! runs ungated in `tests/runtime.rs`). The host must cut it off with
//! `EvalError::FuelExhausted` and keep the graph usable.

#![no_std]

use umber_plugin_sdk::{umber_node, Image, ImageMut, NodeError, Params};

fn spin(
    _input: &Image<'_>,
    _params: &Params<'_>,
    _out: &mut ImageMut<'_>,
) -> Result<(), NodeError> {
    // `black_box` keeps the optimizer from reasoning about the loop:
    // it must stay a real, fuel-consuming loop in the compiled wasm.
    let mut i: u32 = 0;
    loop {
        i = core::hint::black_box(i.wrapping_add(1));
    }
}

umber_node!(name: "infinite", params: 0, eval: spin);

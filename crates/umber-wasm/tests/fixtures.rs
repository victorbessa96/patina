//! FIXTURE-GATED tests: these need the Rust example plugins compiled to
//! wasm32 (`tests/fixtures/build.sh` — the gate's build step). When a
//! fixture is absent they print `SKIP: …` and return WITHOUT asserting
//! (see `common::fixture`); with `UMBER_REQUIRE_WASM_FIXTURES=1` an absent
//! fixture fails instead.

mod common;

use common::{fixture, image_of, noise_image};
use std::collections::HashMap;
use umber_graph::nodes::{register_filter_nodes, register_gradient_nodes, BlurNode};
use umber_graph::{
    eval_graph, Edge, EvalContext, EvalError, Graph, ImageBuffer, Node, NodeImpl, NodeOutput,
    NodeRegistry, ParamValue,
};
use umber_wasm::{register_module, register_wasm, PluginRuntime};

fn node(id: u64, def: &str, params: Vec<(String, ParamValue)>) -> Node {
    Node {
        id,
        node_def: def.into(),
        params,
        canvas: None,
    }
}

fn edge(from: u64, to: u64) -> Edge {
    Edge {
        from,
        to,
        input: "in".into(),
    }
}

/// (d) THE equivalence test: the blur5 plugin and the native `blur`
/// node at radius 2 (a 5-px box) agree byte-for-byte.
#[test]
fn blur5_plugin_matches_native_blur_byte_for_byte() {
    let Some(bytes) = fixture("plugin_blur5") else {
        return;
    };
    let mut registry = NodeRegistry::seeded();
    register_gradient_nodes(&mut registry);
    register_filter_nodes(&mut registry);
    let name = register_wasm(&mut registry, &bytes).expect("blur5 registers");
    assert_eq!(name, "blur5");

    // The test gradient, through the real engine: radial (varies in x
    // AND y), odd non-square size (catches w/h swaps and edge clamps).
    let mut g = Graph::new();
    g.add_node(node(
        1,
        "gradient",
        vec![("type".into(), ParamValue::Int(1))],
    ));
    g.add_node(node(2, "blur", vec![("radius".into(), ParamValue::Int(2))]));
    g.add_node(node(3, "blur5", vec![]));
    g.add_edge(edge(1, 2));
    g.add_edge(edge(1, 3));
    let out = eval_graph(&g, &registry, HashMap::new(), &EvalContext::new(33, 17))
        .expect("graph evaluates");
    let (native, wasm) = (image_of(&out[&2]), image_of(&out[&3]));
    assert_ne!(
        native,
        image_of(&out[&1]),
        "the blur must actually change the gradient"
    );
    assert_eq!(wasm, native, "blur5 (wasm) != blur radius 2 (native)");

    // Varying alpha + noise (the gradient's alpha is constant 255), and
    // the degenerate 1-px-wide / 1-px-tall edges.
    let plugin = registry.get("blur5").unwrap();
    let ctx = EvalContext::new(1, 1);
    let radius2 = [("radius".to_string(), ParamValue::Int(2))];
    for (w, h, seed) in [
        (19, 23, 7),
        (1, 9, 11),
        (9, 1, 13),
        (1, 1, 17),
        (64, 64, 19),
    ] {
        let src = NodeOutput::Image(noise_image(w, h, seed));
        let native = BlurNode
            .eval(vec![("in".into(), src.clone())], &radius2, &ctx)
            .unwrap();
        let wasm = plugin.eval(vec![("in".into(), src)], &[], &ctx).unwrap();
        assert_eq!(wasm, native, "blur5 diverges on {w}x{h} noise");
    }
}

/// (e) The param path: changing `strength` changes the output.
#[test]
fn vignette_param_change_changes_the_output() {
    let Some(bytes) = fixture("plugin_vignette") else {
        return;
    };
    let plugin = PluginRuntime::new().load(&bytes).expect("vignette loads");
    assert_eq!(plugin.def().name, "vignette");
    assert_eq!(plugin.def().n_params, 1);
    let plugin = plugin.into_node_impl();
    let ctx = EvalContext::new(1, 1);
    let gray = || {
        vec![(
            "in".to_string(),
            NodeOutput::Image(ImageBuffer::filled(32, 24, [200, 200, 200, 128]).unwrap()),
        )]
    };
    let run = |params: &[(String, ParamValue)]| {
        image_of(&plugin.eval(gray(), params, &ctx).unwrap()).clone()
    };
    let weak = run(&[("strength".into(), ParamValue::Float(0.2))]);
    let strong = run(&[("strength".into(), ParamValue::Float(0.9))]);
    let none = run(&[("strength".into(), ParamValue::Float(0.0))]);
    let default = run(&[]);

    assert_ne!(weak, strong, "the param must reach the guest");
    // Direction, not just difference: stronger darkens corners more.
    let corner = |img: &ImageBuffer| img.pixel(0, 0).unwrap()[0];
    assert!(
        corner(&strong) < corner(&weak),
        "{} !< {}",
        corner(&strong),
        corner(&weak)
    );
    // strength 0 is the identity; alpha is never touched.
    assert_eq!(
        none.data,
        ImageBuffer::filled(32, 24, [200, 200, 200, 128])
            .unwrap()
            .data
    );
    assert!(strong.data.chunks_exact(4).all(|px| px[3] == 128));
    // Absent param = the documented default (0.5): between weak and strong.
    assert!(corner(&strong) < corner(&default) && corner(&default) < corner(&weak));
    // A mistyped param is the guest's ERR_GUEST_INTERNAL → Plugin error.
    let err = plugin
        .eval(gray(), &[("strength".into(), ParamValue::Int(1))], &ctx)
        .unwrap_err();
    assert!(matches!(err, EvalError::Plugin { .. }), "got {err:?}");
}

/// (b, Rust-built twin) The examples/plugin-infinite crate, compiled by
/// rustc, is cut off by fuel exactly like the WAT twin in runtime.rs.
#[test]
fn rust_infinite_plugin_exhausts_fuel() {
    let Some(bytes) = fixture("plugin_infinite") else {
        return;
    };
    let rt = PluginRuntime::with_fuel(500_000);
    let mut registry = NodeRegistry::seeded();
    assert_eq!(
        register_module(&mut registry, rt.load(&bytes).expect("loads")),
        "infinite"
    );
    let mut g = Graph::new();
    g.add_node(node(
        1,
        "uniform",
        vec![("color".into(), ParamValue::Color([1.0, 1.0, 1.0]))],
    ));
    g.add_node(node(2, "infinite", vec![]));
    g.add_edge(edge(1, 2));
    let err = eval_graph(&g, &registry, HashMap::new(), &EvalContext::new(4, 4)).unwrap_err();
    assert!(
        matches!(err, EvalError::FuelExhausted { node: 2 }),
        "got {err:?}"
    );
}

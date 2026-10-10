//! Ungated runtime tests: every module here is hand-written WAT
//! (wasmtime's default `wat` feature compiles it), so these run on any
//! host without the wasm32 build step.

mod common;

use common::{
    def_record, image_of, infinite_wat, invert_wat, noise_image, wat_plugin, INVERT_EVAL,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use umber_graph::{
    eval_graph, Edge, EvalContext, EvalError, Graph, ImageBuffer, Node, NodeImpl, NodeOutput,
    NodeRegistry, ParamValue,
};
use umber_wasm::{register_module, register_wasm, PluginError, PluginRuntime};

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

/// uniform(color) at 4×4 → `def`.
fn fill_then(def: &str) -> Graph {
    let mut g = Graph::new();
    g.add_node(node(
        1,
        "uniform",
        vec![("color".into(), ParamValue::Color([0.2, 0.4, 0.6]))],
    ));
    g.add_node(node(2, def, vec![]));
    g.add_edge(edge(1, 2));
    g
}

fn load_err(bytes: impl AsRef<[u8]>) -> PluginError {
    match PluginRuntime::new().load(bytes.as_ref()) {
        Ok(m) => panic!(
            "expected a load error, got a module named {:?}",
            m.def().name
        ),
        Err(e) => e,
    }
}

// ── the full wire path ────────────────────────────────────────────────

#[test]
fn wat_invert_roundtrips_through_the_engine() {
    let mut registry = NodeRegistry::seeded();
    let name = register_wasm(&mut registry, invert_wat().as_bytes()).expect("registers");
    assert_eq!(name, "wat_invert");
    let out = eval_graph(
        &fill_then("wat_invert"),
        &registry,
        HashMap::new(),
        &EvalContext::new(4, 4),
    )
    .expect("plugin evaluates");
    let img = image_of(&out[&2]);
    assert_eq!((img.width, img.height), (4, 4));
    // uniform rounds 0.2/0.4/0.6 → 51/102/153; inverted → 204/153/102.
    assert!(img
        .data
        .chunks_exact(4)
        .all(|px| px == [204, 153, 102, 255]));
}

#[test]
fn wat_invert_matches_a_host_reference_on_noise() {
    // Odd, non-square, alpha varying: the whole region crosses intact.
    let plugin = PluginRuntime::new()
        .load(invert_wat().as_bytes())
        .unwrap()
        .into_node_impl();
    let src = noise_image(13, 7, 0xC0FFEE);
    let out = plugin
        .eval(
            vec![("in".into(), NodeOutput::Image(src.clone()))],
            &[("ignored".into(), ParamValue::Float(1.0))],
            &EvalContext::new(1, 1),
        )
        .expect("evaluates");
    let mut expected = src.data.clone();
    for px in expected.chunks_exact_mut(4) {
        for c in &mut px[..3] {
            *c = 255 - *c;
        }
    }
    assert_eq!(image_of(&out).data, expected);
    assert_eq!((image_of(&out).width, image_of(&out).height), (13, 7));
}

#[test]
fn memory_export_fallback_name_is_accepted() {
    // rustc names the memory export `memory`; the host accepts it.
    let wat = wat_plugin("memory", &def_record("rustish", 1, 0), INVERT_EVAL);
    let m = PluginRuntime::new().load(wat.as_bytes()).expect("loads");
    assert_eq!(m.def().name, "rustish");
}

// ── (b) fuel: a hung plugin can't hang the graph ─────────────────────

#[test]
fn infinite_loop_exhausts_fuel_and_the_graph_survives() {
    // Small budget: the loop must be cut off fast (the default FUEL is
    // sized for real images and would spin for seconds).
    let rt = PluginRuntime::with_fuel(200_000);
    let mut registry = NodeRegistry::seeded();
    assert_eq!(
        register_module(&mut registry, rt.load(infinite_wat().as_bytes()).unwrap()),
        "wat_infinite"
    );
    assert_eq!(
        register_module(&mut registry, rt.load(invert_wat().as_bytes()).unwrap()),
        "wat_invert"
    );
    let ctx = EvalContext::new(4, 4);

    let started = Instant::now();
    let err = eval_graph(&fill_then("wat_infinite"), &registry, HashMap::new(), &ctx)
        .expect_err("a hung plugin must error, not hang");
    assert!(
        matches!(err, EvalError::FuelExhausted { node: 2 }),
        "expected FuelExhausted on node 2, got {err:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "fuel cut-off too slow"
    );

    // Survival: the same registry evaluates another plugin node fine…
    let out = eval_graph(&fill_then("wat_invert"), &registry, HashMap::new(), &ctx)
        .expect("a second plugin node evaluates after the failure");
    assert!(image_of(&out[&2])
        .data
        .chunks_exact(4)
        .all(|px| px == [204, 153, 102, 255]));
    // …and the hung node fails the same way again (fresh store per eval:
    // no poisoned or fuel-drained state carries over).
    let again = eval_graph(&fill_then("wat_infinite"), &registry, HashMap::new(), &ctx);
    assert!(matches!(again, Err(EvalError::FuelExhausted { node: 2 })));
}

#[test]
fn a_hung_node_def_is_fuel_exhausted_at_load() {
    let wat = r#"(module
      (memory (export "umber_mem") 1)
      (func (export "umber_node_def") (result i32) (loop $l (br $l)) (unreachable))
      (func (export "umber_eval") (param i32 i32 i32 i32) (result i32) (i32.const 0)))"#;
    let err = PluginRuntime::with_fuel(100_000).load(wat.as_bytes()).err();
    assert_eq!(err, Some(PluginError::FuelExhausted));
}

#[test]
fn a_hung_start_function_is_fuel_exhausted_at_load() {
    let wat = r#"(module
      (memory (export "umber_mem") 1)
      (func $spin (loop $l (br $l)))
      (start $spin)
      (func (export "umber_node_def") (result i32) (i32.const 0))
      (func (export "umber_eval") (param i32 i32 i32 i32) (result i32) (i32.const 0)))"#;
    let err = PluginRuntime::with_fuel(100_000).load(wat.as_bytes()).err();
    assert_eq!(err, Some(PluginError::FuelExhausted));
}

// ── (c) bad modules / missing exports ────────────────────────────────

#[test]
fn garbage_and_truncated_bytes_are_bad_module() {
    // Not wasm (and not WAT either — the text parser rejects it).
    assert!(matches!(
        load_err(b"definitely not wasm"),
        PluginError::BadModule(_)
    ));
    // The truncate trick: the 8-byte empty module is valid; cut it to 6.
    assert!(matches!(
        load_err(b"\0asm\x01\0"),
        PluginError::BadModule(_)
    ));
    // A valid header with a bogus section id.
    assert!(matches!(
        load_err(b"\0asm\x01\0\0\0\x7f\x00"),
        PluginError::BadModule(_)
    ));
    // (Empty input is deliberately not asserted: wasmtime's WAT front
    // end may read zero bytes as an empty inline module, which is then
    // MissingExport — the contract is "an error, never a panic".)
    assert!(PluginRuntime::new().load(b"").is_err());
}

#[test]
fn valid_modules_missing_an_export_name_it() {
    // The minimal hand-wasm: magic + version, nothing else — valid,
    // exports nothing.
    assert_eq!(
        load_err(b"\0asm\x01\0\0\0"),
        PluginError::MissingExport("umber_mem")
    );
    // Memory + def, no eval.
    let no_eval = wat_plugin("umber_mem", &def_record("x", 1, 0), "");
    assert_eq!(load_err(no_eval), PluginError::MissingExport("umber_eval"));
    // Eval with the wrong signature counts as missing.
    let wrong_sig = wat_plugin(
        "umber_mem",
        &def_record("x", 1, 0),
        r#"(func (export "umber_eval") (param i32) (result i32) (i32.const 0))"#,
    );
    assert_eq!(
        load_err(wrong_sig),
        PluginError::MissingExport("umber_eval")
    );
    // No def.
    let no_def = r#"(module (memory (export "umber_mem") 1))"#;
    assert_eq!(
        load_err(no_def),
        PluginError::MissingExport("umber_node_def")
    );
    // Memory under neither accepted name.
    let wrong_mem = wat_plugin("heap", &def_record("x", 1, 0), INVERT_EVAL);
    assert_eq!(load_err(wrong_mem), PluginError::MissingExport("umber_mem"));
}

#[test]
fn host_imports_are_refused() {
    let wat = r#"(module
      (import "env" "clock" (func (result i64)))
      (memory (export "umber_mem") 1))"#;
    match load_err(wat) {
        PluginError::BadModule(msg) => assert!(msg.contains("env::clock"), "{msg}"),
        other => panic!("expected BadModule, got {other:?}"),
    }
}

#[test]
fn malformed_node_defs_are_bad_wire() {
    let with = |rec: &[u8]| load_err(wat_plugin("umber_mem", rec, INVERT_EVAL));
    let mut bad_magic = def_record("x", 1, 0);
    bad_magic[0] = b'X';
    assert!(matches!(with(&bad_magic), PluginError::BadWire(_)));
    assert!(matches!(
        with(&def_record("x", 2, 0)),
        PluginError::BadWire(_)
    ));
    assert!(matches!(
        with(&def_record("x", 1, 5)),
        PluginError::BadWire(_)
    ));
    assert!(matches!(
        with(&def_record("", 1, 0)),
        PluginError::BadWire(_)
    ));
    assert!(matches!(
        with(&def_record("a<b", 1, 0)),
        PluginError::BadWire(_)
    ));
    // name_len = 64 KiB: past the limit, never read.
    let mut huge = def_record("x", 1, 0);
    huge[12..16].copy_from_slice(&65_536u32.to_le_bytes());
    assert!(matches!(with(&huge), PluginError::BadWire(_)));

    // A pointer off the end of memory.
    let oob = r#"(module
      (memory (export "umber_mem") 1)
      (func (export "umber_node_def") (result i32) (i32.const 65530))
      (func (export "umber_eval") (param i32 i32 i32 i32) (result i32) (i32.const 0)))"#;
    assert!(matches!(load_err(oob), PluginError::BadWire(_)));
}

#[test]
fn a_trapping_node_def_is_guest_panic() {
    let wat = r#"(module
      (memory (export "umber_mem") 1)
      (func (export "umber_node_def") (result i32) (unreachable))
      (func (export "umber_eval") (param i32 i32 i32 i32) (result i32) (i32.const 0)))"#;
    assert!(matches!(load_err(wat), PluginError::GuestPanic(_)));
}

// ── eval-time failures are typed errors, never panics ────────────────

fn eval_with(eval_func: &str, img: ImageBuffer) -> Result<NodeOutput, EvalError> {
    let wat = wat_plugin("umber_mem", &def_record("probe", 1, 0), eval_func);
    PluginRuntime::new()
        .load(wat.as_bytes())
        .expect("loads")
        .into_node_impl()
        .eval(
            vec![("in".into(), NodeOutput::Image(img))],
            &[],
            &EvalContext::new(1, 1),
        )
}

#[test]
fn guest_error_codes_and_bad_outputs_map_to_plugin_errors() {
    let img = || ImageBuffer::filled(2, 2, [1, 2, 3, 4]).unwrap();
    let returns = |v: i32| {
        format!(
            r#"(func (export "umber_eval") (param i32 i32 i32 i32) (result i32) (i32.const {v}))"#
        )
    };
    for (code, needle) in [
        (-1, "bad magic"),
        (-2, "truncated"),
        (-3, "capacity"),
        (-4, "internal"),
        (-77, "unknown error code -77"),
        (5, "malformed out region"),  // shorter than a header
        (28, "malformed out region"), // right length, zero magic
        (1_000_000, "past out_cap"),  // claims more than reserved
    ] {
        match eval_with(&returns(code), img()) {
            Err(EvalError::Plugin { reason, .. }) => {
                assert!(
                    reason.contains(needle),
                    "code {code}: {reason:?} lacks {needle:?}"
                )
            }
            other => panic!("code {code}: expected Plugin error, got {other:?}"),
        }
    }
    let trap = r#"(func (export "umber_eval") (param i32 i32 i32 i32) (result i32) (unreachable))"#;
    assert!(matches!(
        eval_with(trap, img()),
        Err(EvalError::Plugin { reason, .. }) if reason.contains("trapped")
    ));
}

#[test]
fn input_arity_and_kind_follow_the_filter_convention() {
    let plugin = PluginRuntime::new()
        .load(invert_wat().as_bytes())
        .unwrap()
        .into_node_impl();
    let ctx = EvalContext::new(1, 1);
    assert!(matches!(
        plugin.eval(vec![], &[], &ctx),
        Err(EvalError::MissingInput { .. })
    ));
    assert!(matches!(
        plugin.eval(
            vec![("in".into(), NodeOutput::Uniform(ParamValue::Float(1.0)))],
            &[],
            &ctx
        ),
        Err(EvalError::TypeMismatch { .. })
    ));
    let img = NodeOutput::Image(ImageBuffer::filled(1, 1, [0; 4]).unwrap());
    assert!(matches!(
        plugin.eval(
            vec![("a".into(), img.clone()), ("b".into(), img.clone())],
            &[],
            &ctx
        ),
        Err(EvalError::BadParam { .. })
    ));
    // Five wire params: over the v1 limit.
    let five: Vec<(String, ParamValue)> = (0..5)
        .map(|i| (format!("p{i}"), ParamValue::Int(i)))
        .collect();
    assert!(matches!(
        plugin.eval(vec![("in".into(), img)], &five, &ctx),
        Err(EvalError::BadParam { .. })
    ));
}

#[test]
fn images_past_the_memory_cap_are_refused_not_crashed() {
    // 3000² RGBA = 36 MB in + 36 MB out > the 64 MiB cap.
    let big = ImageBuffer::filled(3000, 3000, [9, 9, 9, 255]).unwrap();
    match eval_with(INVERT_EVAL, big) {
        Err(EvalError::Plugin { reason, .. }) => assert!(reason.contains("memory cap"), "{reason}"),
        other => panic!(
            "expected the memory-cap Plugin error, got {:?}",
            other.map(|_| ())
        ),
    }
    // Just under: 2048² (16 MiB each way) fits.
    let ok = ImageBuffer::filled(2048, 2048, [9, 9, 9, 255]).unwrap();
    let out = eval_with(INVERT_EVAL, ok).expect("fits under the cap");
    assert_eq!(image_of(&out).pixel(2047, 2047), Some([246, 246, 246, 255]));
}

#[test]
fn registered_plugins_list_beside_builtins() {
    let mut registry = NodeRegistry::seeded();
    register_wasm(&mut registry, invert_wat().as_bytes()).unwrap();
    assert!(registry.node_defs().contains(&"wat_invert"));
    assert!(registry.get("wat_invert").is_some());
    // A failed registration leaves the registry untouched.
    let before = registry.node_defs().len();
    assert!(register_wasm(&mut registry, b"junk").is_err());
    assert_eq!(registry.node_defs().len(), before);
    let _: Arc<dyn NodeImpl> = registry.get("wat_invert").unwrap();
}

//! Shared test helpers: the prebuilt-fixture loader (with the honest
//! skip) and hand-written WAT plugins for the ungated tests.

#![allow(dead_code)] // each test binary uses a different subset

use std::path::PathBuf;
use umber_graph::{ImageBuffer, NodeOutput};

/// Env var the gate sets AFTER running the fixture build step: with it,
/// a missing fixture FAILS instead of skipping.
pub const REQUIRE_FIXTURES_ENV: &str = "UMBER_REQUIRE_WASM_FIXTURES";

/// Where the prebuilt `.wasm` fixtures live (populated by
/// `tests/fixtures/build.sh` — the wasm32 build is the gate's step).
pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.wasm"))
}

/// The prebuilt fixture `name` (e.g. `plugin_blur5`), or `None` after
/// printing a `SKIP:` line when it hasn't been built.
///
/// Rust has no runtime skip, so a skipping test still reports `ok`. The
/// `SKIP:` line is the honest signal: the test did NOT run its assertions.
/// Setting `UMBER_REQUIRE_WASM_FIXTURES=1` turns the absence into a panic.
/// The gate sets it once the build step has run, so a present-but-unbuilt
/// fixture can never pass silently there.
pub fn fixture(name: &str) -> Option<Vec<u8>> {
    let path = fixture_path(name);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            let required = std::env::var(REQUIRE_FIXTURES_ENV).is_ok_and(|v| v == "1");
            assert!(
                !required,
                "{REQUIRE_FIXTURES_ENV}=1 but fixture {} is unreadable ({err}); \
                 run crates/umber-wasm/tests/fixtures/build.sh",
                path.display()
            );
            eprintln!(
                "SKIP: fixture {} absent ({err}) — the wasm32 build step \
                 (crates/umber-wasm/tests/fixtures/build.sh) has not run; this test's \
                 assertions did NOT execute. Set {REQUIRE_FIXTURES_ENV}=1 to fail instead.",
                path.display()
            );
            None
        }
    }
}

/// WAT string-literal escape of raw bytes (`\hh` per byte).
pub fn wat_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("\\{b:02x}")).collect()
}

/// A node-def record, built by hand (independent of `wire.rs`):
/// `[MDEF][n_inputs][n_params][name_len][name]`.
pub fn def_record(name: &str, n_inputs: u32, n_params: u32) -> Vec<u8> {
    let mut r = b"MDEF".to_vec();
    r.extend_from_slice(&n_inputs.to_le_bytes());
    r.extend_from_slice(&n_params.to_le_bytes());
    r.extend_from_slice(&(name.len() as u32).to_le_bytes());
    r.extend_from_slice(name.as_bytes());
    r
}

/// A plugin module in WAT: memory export `mem_name`, a def record at
/// offset 16, `umber_node_def` returning 16, plus `extra` (usually the
/// `umber_eval` func).
pub fn wat_plugin(mem_name: &str, record: &[u8], extra: &str) -> String {
    format!(
        r#"(module
  (memory (export "{mem_name}") 1)
  (data (i32.const 16) "{rec}")
  (func (export "umber_node_def") (result i32) (i32.const 16))
  {extra})"#,
        rec = wat_bytes(record)
    )
}

/// `umber_eval` that inverts RGB (alpha kept) — the full wire path in
/// raw wasm: checks magic/length/capacity, writes the out header and
/// pixels, returns `12 + w*h*4`. Error codes per WIRE.md.
pub const INVERT_EVAL: &str = r#"
  (func (export "umber_eval") (param $in i32) (param $in_len i32) (param $out i32) (param $cap i32) (result i32)
    (local $n i32) (local $i i32) (local $b i32)
    (if (i32.lt_u (local.get $in_len) (i32.const 16)) (then (return (i32.const -2))))
    (if (i32.ne (i32.load (local.get $in)) (i32.const 0x5245424D)) (then (return (i32.const -1))))
    (local.set $n (i32.mul (i32.mul (i32.load offset=4 (local.get $in))
                                    (i32.load offset=8 (local.get $in)))
                           (i32.const 4)))
    (if (i32.lt_u (local.get $in_len) (i32.add (local.get $n) (i32.const 16)))
      (then (return (i32.const -2))))
    (if (i32.lt_u (local.get $cap) (i32.add (local.get $n) (i32.const 12)))
      (then (return (i32.const -3))))
    (i32.store (local.get $out) (i32.const 0x5245424D))
    (i32.store offset=4 (local.get $out) (i32.load offset=4 (local.get $in)))
    (i32.store offset=8 (local.get $out) (i32.load offset=8 (local.get $in)))
    (block $done
      (loop $px
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $b (i32.load8_u offset=12 (i32.add (local.get $in) (local.get $i))))
        (if (i32.ne (i32.and (local.get $i) (i32.const 3)) (i32.const 3))
          (then (local.set $b (i32.sub (i32.const 255) (local.get $b)))))
        (i32.store8 offset=12 (i32.add (local.get $out) (local.get $i)) (local.get $b))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $px)))
    (i32.add (local.get $n) (i32.const 12)))
"#;

/// `umber_eval` that never returns (the WAT twin of
/// examples/plugin-infinite).
pub const INFINITE_EVAL: &str = r#"
  (func (export "umber_eval") (param i32 i32 i32 i32) (result i32)
    (loop $spin (br $spin))
    (unreachable))
"#;

/// The WAT invert plugin, registered as `wat_invert`.
pub fn invert_wat() -> String {
    wat_plugin("umber_mem", &def_record("wat_invert", 1, 0), INVERT_EVAL)
}

/// The WAT infinite-loop plugin, registered as `wat_infinite`.
pub fn infinite_wat() -> String {
    wat_plugin(
        "umber_mem",
        &def_record("wat_infinite", 1, 0),
        INFINITE_EVAL,
    )
}

/// Deterministic xorshift RGBA noise, alpha varying too (catches a
/// plugin that skips or mishandles the alpha channel).
pub fn noise_image(w: u32, h: u32, seed: u32) -> ImageBuffer {
    let mut s = seed | 1;
    let data = (0..w * h * 4)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 24) as u8
        })
        .collect();
    ImageBuffer::new(w, h, data).expect("w*h*4 bytes")
}

pub fn image_of(out: &NodeOutput) -> &ImageBuffer {
    match out {
        NodeOutput::Image(buf) => buf,
        other => panic!("expected an Image, got {other:?}"),
    }
}

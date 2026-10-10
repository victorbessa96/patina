# umber-wasm prebuilt plugin fixtures

These are the checked-in `.wasm` binaries of the example plugins. The
fixture-gated tests in `../fixtures.rs` read them:

| file | source | test |
|---|---|---|
| `plugin_blur5.wasm` | `examples/plugin-blur5` | (d) blur5 == native `blur` radius 2, byte-for-byte |
| `plugin_vignette.wasm` | `examples/plugin-vignette` | (e) the `strength` param changes the output |
| `plugin_infinite.wasm` | `examples/plugin-infinite` | (b) the Rust-built hang is fuel-exhausted |

To (re)build them, run `sh build.sh`. It needs the `wasm32-unknown-unknown`
target and copies the release builds here. Commit the results so the
fixtures stay checked in. Rebuild whenever the SDK or an example changes.

## When a fixture is missing

A test whose fixture is missing prints `SKIP: …` and returns without
asserting anything. Rust has no runtime skip, so the harness still
reports it as `ok`, and the `SKIP:` line is the only sign that nothing
ran. After building, run the tests with `UMBER_REQUIRE_WASM_FIXTURES=1`;
with that set, a missing fixture fails instead of skipping.

## Ungated coverage

Everything else runs ungated on hand-written WAT in `../runtime.rs`,
including:

- the full wire round-trip (a WAT invert plugin);
- the fuel test (a WAT infinite loop);
- bad and missing-export modules;
- guest error codes;
- the memory cap.

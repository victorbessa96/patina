#!/usr/bin/env sh
# Builds the example plugins (examples/, its own workspace) for
# wasm32-unknown-unknown and copies the .wasm files here, where
# umber-wasm's fixture-gated tests (tests/fixtures.rs) read them.
#
# The gate's step, run from anywhere:
#   sh crates/umber-wasm/tests/fixtures/build.sh
#   UMBER_REQUIRE_WASM_FIXTURES=1 cargo test -p umber-wasm
# (the env var turns a missing fixture from SKIP into a failure).
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
# rustup resolves rust-toolchain.toml from the CWD, not --manifest-path:
# run from the repo root so the pinned toolchain gets the target.
cd "$root"

rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown \
    --manifest-path "$root/examples/Cargo.toml"

out="$root/examples/target/wasm32-unknown-unknown/release"
for name in plugin_blur5 plugin_vignette plugin_infinite; do
    cp "$out/$name.wasm" "$here/$name.wasm"
    echo "fixture: $here/$name.wasm ($(wc -c < "$here/$name.wasm") bytes)"
done

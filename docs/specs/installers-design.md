# Installers / Packaging — Wave-6 Design

Wave-6's fourth slice. Written 2026-10-10 08:58 against the tree at
`263df1f`. The requirements' §10 row: cross-platform installers,
the §12 budget: < 150 MB tracked per release. The honest v1 for a
Rust workspace app: **cargo-dist** (the axo.dev standard — the
Rust ecosystem's release tooling: per-target archives, installers,
checksums, the GitHub release flow).

## The shape

- **cargo-dist config** in the workspace root's `Cargo.toml`
  (`[dist]`): the bins (umber-app's `umber` binary + umber-cli's
  `umber-cli`), the targets (x86_64-unknown-linux-gnu,
  x86_64-pc-windows-msvc — the two CI platforms already proven),
  the artifacts (executables + their installers: the shell
  installer for Linux, the PowerShell for Windows), the checksums.
- **The GitHub workflow**: cargo-dist's generated
  `release-dry-run` + `release` actions — a TAG (v0.x.y) triggers
  the build matrix, the artifacts land on the GitHub release with
  sha256s. The workflow generation is `cargo dist init`'s job (run
  once, the generated file committed).
- **The size budget**: the §12 protocol tracks the artifact sizes
  per release (the perf/baseline table grows a release column).
  If umber-app's binary + the wgpu/wasmtime trees exceed 150 MB
  unstripped: the release profile strips + the dist config's
  `strip = true` (the honest v1 — measure first, strip if needed;
  the perf protocol's rule: no claim without a number).
- **The wasmtime dependency**: statically linked (its default
  Cranelift path) — no runtime .so shipping concerns. wgpu:
  the Vulkan loader is SYSTEM-provided (libvulkan) on both
  platforms — the installer carries no driver code, documented.
- **The examples/plugins**: the three .wasm fixtures ship beside
  the binary in the archive (the dist config's extra-files or a
  package step — verify cargo-dist's extra-files support in its
  docs; the fallback: a small build script appends them to the
  archive).

## What this deliberately is NOT

- No MSIs/AppImages/debs v1 — the cargo-dist archives + shell/PS
  installers are the release surface (the native package formats
  are a follow-up with real distribution needs).
- No code signing (the self-hosted release has no certs; the
  installers print the checksum so the user can verify — the
  honest v1).

## Steps

1. Install cargo-dist locally (cargo install cargo-dist — the
   dragon's prep step), run `cargo dist init` against the
   workspace, commit the generated config + workflow.
2. `cargo dist build` (the dry-run) on the remote — the artifact
   sizes measured, recorded in the §12 table, the strip decision
   made on the number.
3. The first real release tagged when Bessa says ship (the
   public-posting rule: releases are his call, the same gate as
   PRs).

## Tests

The packaging surface is config + workflow files; the gate is the
dry-run's success + the size numbers. The runtime tests are the
existing suites (unchanged — this slice touches no crate code
except the release profile if stripping).

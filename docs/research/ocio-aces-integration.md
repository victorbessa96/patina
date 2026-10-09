# §9 Color Management — OCIO/ACES Integration Survey (2026-10-09)

Research pass for the wave-3 §9 P0: "Scene-linear working space, OCIO v2
configs (ocio-rs, vendored build) incl. ACES 2.0 CG/Studio built-ins."
Sources: crates.io/docs.rs (ocio-rs 0.2.1), OCIO 2.5 release notes
(readthedocs), ASWF OpenColorIO-Config-ACES repo, aces-core CHANGELOG.

## The conclusion first

**ocio-rs 0.2.1 targets OpenColorIO v2.5.2 — the exact OCIO release that
ships built-in ACES 2.0 configs.** OCIO 2.5.0 (Sept 2025, VFX Reference
Platform CY2026) provides built-in ACES 2.0 versions of the Studio and
CG configs natively: the requirement's "ACES 2.0 CG/Studio built-ins"
maps to OCIO's own built-in config registry, not to vendored config
files. `CreateFromBuiltin("CG Config for ACES 2.0")` (or Studio) gives
the whole config in-memory with zero external LUT files.

## ocio-rs 0.2.1 (crates.io, 2 months old, BSD-3)

- API: `Config::from_file(path)` / built-in configs, `config.processor(from, to)`,
  `default_cpu_processor().apply_rgba(&mut pixel)` — the CPU reference
  path umber-color's display.rs is the spec for; GPU: `GpuShaderDesc`,
  `GpuTexture2D`/`GpuTexture3D`, `GpuUniform` — **shader extraction with
  3D-texture payloads**, the §9 "OCIO GPU shader extraction → 3D texture"
  display-LUT path, already wrapped.
- Build modes (the critical decision):
  - **Stub (default)**: `--no-default-features`. Compiles + tests with NO
    OCIO install — safe API-shape returns, no real color management.
    Windows-CI-safe with zero build cost.
  - **Bundled**: `--features bundled`. Vendors the upstream OCIO C++ tree
    + transitive deps, static link. `cargo build --features bundled
    --offline` is their validated release path. Cost: a heavy C++ build
    (minutes; Imath/Half/OCIO core) added to our workspace build.
  - **Pre-installed**: `OCIO_RS_ENABLE_REAL=1` + `OCIO_INSTALL_DIR`.
    Links a system OCIO; packagers' path, not ours.
- Status per docs: core 2.5 surface broadly in place; remaining work is
  "release hardening and longer-tail behavioral validation" — pin a known
  version and probe the paths we use before trusting the rest.

## Recommended architecture for umber

1. **umber-color**: `ocio` optional dependency behind a `real-ocio`
  feature (default OFF). Stub mode runs in CI everywhere; the CPU
  reference tests (display.rs) stay pure-Rust and gate the OCIO path:
  the same anchor values must come out of `apply_display` and the OCIO
  sRGB display processor.
2. **Config source**: OCIO built-in ACES 2.0 CG (default) / Studio
  (opt-in) via the built-in registry — no config files to vendor or
  ship. Scene-linear = ACES2065-1 (AP0) as the working space per the
  CG config's roles.
3. **umber-gpu display LUT**: extract the display transform's GPU
  shader (`GpuShaderDesc`) on config load; upload the `GpuTexture3D`
  payloads as wgpu 3D textures; the viewport's display pass samples
  the LUT instead of the fixed sRGB curve. CPU path remains the spec.
4. **CI strategy**: stub mode for the default matrix (both platforms);
  the `bundled` build only on a manual/dispatch job (the gentle-load
  directive — the C++ build is minutes of full-core compile; never in
  the per-commit path). Linux `OCIO_INSTALL_DIR` if a system OCIO
  exists in the runner image.

## Open questions (probe before wiring)

- ~~Does ocio-rs 0.2.1 expose the built-in config creation surface?~~
  **ANSWERED by stub-mode probe (2026-10-09, scratch ocio-probe):**
  - `Config::create_from_builtin_config(name)` — exists, compiles.
  - `BuiltinConfigRegistry::get() / num_builtin_configs() /
    config_name(i) / config_ui_name(i)` — the full enumeration API.
  - `GpuShaderDesc::create()` — constructs even in stub mode (Ok).
  - Stub mode behavior confirmed: handle-allocating calls error with
    "OpenColorIO handle allocation failed" — clean, documented,
    CI-safe. Real-mode verification needs the bundled build (manual
    dispatch only, per the gentle-load directive).
- The `GpuTexture3D` payload's edge/interpolation metadata vs wgpu's
  `SamplerDescriptor` — one mapping shim, needs the real types.
- ACES 2.0 config's display-color-space set (Rec.709/sRGB, P3 variants,
  HDR) → the display panel's dropdown list.

## Effort map

| Slice | Size | Load |
|---|---|---|
| ocio stub dep + built-in probe test | small | zero (stub mode) |
| CPU processor bridge in umber-color | small | zero (stub) |
| Display panel: config/look/view pickers | medium | UI only |
| GPU LUT path in umber-gpu | medium | one gated test |
| Bundled OCIO build job | config-only | heavy — manual dispatch only |

The whole §9 CPU+UI surface is stub-mode cheap; only the bundled C++
build is heavy, and it never needs to sit in the per-commit path.

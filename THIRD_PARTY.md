# Third-Party Licenses — Dependency Manifest

> Wave-1 merge gate (SPEC.md Constraints): every dependency's license is
> recorded here before first claw-authored merge to master. Updated
> whenever `Cargo.lock` changes materially. Generated + verified with
> `cargo license` / `cargo about`; hand-audited for the load-bearing ones.

## Application license

Umber is GPL-3.0-or-later (LICENSE). Standalone library crates may be MIT
where upstream licenses permit relicensing.

## Direct dependencies (Wave 1)

| Crate | Version | License | GPL-3.0 compatible | Notes |
|---|---|---|---|---|
| wgpu | 30.x | MIT OR Apache-2.0 | ✅ | GPU abstraction |
| naga | 30.x | MIT OR Apache-2.0 | ✅ | shader IR |
| eframe / egui | 0.36 | MIT OR Apache-2.0 | ✅ | app shell (bundles rfd: MIT/Apache) |
| egui_dock | 0.21 | MIT OR Apache-2.0 | ✅ | dockable panels |
| glam | 0.34 | MIT OR Apache-2.0 | ✅ | math |
| gltf | 1.x | MIT OR Apache-2.0 | ✅ | glTF import |
| tobj | 4.x | MIT | ✅ | OBJ import |
| ufbx | 0.11 | MIT | ✅ | FBX import (C lib, official bindings) |
| thiserror / anyhow | 2 / 1 | MIT OR Apache-2.0 | ✅ | errors |
| rfd | 0.15.4 | MIT | ✅ | native file dialogs (umber-app; added Wave 1 — recorded after the fact, gate audit caught it) |
| egui-wgpu / epaint | 0.36 | MIT OR Apache-2.0 | ✅ | egui wgpu bindings (umber-gpu, claw pass) |
| windows | 0.62 (optional, target-gated to Windows) | MIT OR Apache-2.0 | ✅ | Win32 WM_POINTER/pen API for the stylus Windows Ink backend |
| bytemuck | 1.x | MIT OR Zlib OR Apache-2.0 | ✅ | pod casting for GPU buffers |
| pollster | 1.x | MIT OR Apache-2.0 | ✅ | adapter blocking in gpu-feature tests (optional dep) |
| log / env_logger | 0.4 / 0.11 | MIT OR Apache-2.0 | ✅ | logging |
| rayon / parking_lot / crossbeam | (W2+) | MIT OR Apache-2.0 | ✅ | concurrency |

## Reimplementation (not derivative) sources

| Upstream | License | Relationship |
|---|---|---|
| libmypaint | LGPL-2.1-or-later | **Documented-semantics reimplementation only** — we port the published brush model (inputs/curves/dab spacing) from its documentation and papers; no code is copied. SPEC.md Constraints. |
| OpenPBR spec / openpbr-bsdf | Apache-2.0 (ASWF) | Reference used for the WGSL über-shader port. |

## Deferred dependencies (later waves — pre-audited)

| Crate | Wave | License | Risk note |
|---|---|---|---|
| ocio-rs / ocio-sys (vendored OCIO 2.5) | 2 | BSD-3-Clause (OCIO) | single-maintainer bindings; LUT fallback pre-designed |
| image / exr / half | 2 | MIT OR Apache-2.0 | imaging |
| wasmtime + WASI 0.3 | 6 | Apache-2.0 WITH LLVM-exception | plugin runtime |
| openusd | 5 | MIT (crate) — verify against Pixar USD (Apache-2.0) at adoption | pre-1.0 |
| octotablet (fork/absorb) | 2 | MIT OR Apache-2.0 — **verify at absorption; relicensing path requires it** | stylus layer |
| quick-xml | 4 | MIT | .mtlx parsing |

## Hard-walled (never)

- Adobe .sbsar / Materials SDK — closed terms, no FOSS path
- Autodesk FBX SDK — proprietary, non-redistributable (ufbx is the route)

# OCIO Adoption Probe — 2026-10-10 (the real-OCIO row's ecosystem check)

The audit's §9 row: "real-OCIO build (manual dispatch)" as the
display-transform gap. Today's probe result, recorded for the
adoption decision:

## The finding

**`ocio 0.1.0` — a pure-Rust OCIO port** exists on crates.io
(no C++ build, no FFI): config (Config::load, environment),
processor (from_ops, apply paths), the full transforms set, ops,
GPU shader-gen, baker, 20+ LUT fileformats (CDL, ICC, cube, spi,
3dl, ctf, vf...), dynamic properties. Substantial — this is a
serious port, not a placeholder.

The alternatives: ocio-rs/ocio-sys 0.2.1 (bindings to the C++
lib — the "real build" the audit named, with the C++ build cost);
the app's built-in sRGB/linear curves (the current §9 reality).

## The honest disposition

- The app's v1 color contract (sRGB source → linear working →
  sRGB display, per-map transfer) is SERVED by the built-in
  curves; the display-transform panel + GPU LUT are the missing
  UX rows, not the color engine.
- The pure-Rust port makes OCIO adoption a REAL option now (no
  C build — the audit's "manual dispatch" cost mostly evaporates
  for the CPU paths), but 0.1.0 maturity + the GPU shader-gen
  surface are unverified beyond the file listing.
- Decision rule (recorded): when the display-transform panel
  work starts, probe ocio 0.1.x's processor path against the
  app's existing ICC/transfer tests first — if the round-trip
  holds, adopt for the panel's custom-transform loading; if not,
  the C++ bindings remain the manual-dispatch fallback. No
  adoption before the panel work needs it (the §9 P0 display
  rows are about UX, not engine).

## The stale claim retired

The audit line "real-OCIO build (manual dispatch)" implied the
C++ lib was the only route. The pure-Rust port is now on the
table — the row's cost estimate dropped from "C++ build +
dispatch" to "vendored Rust dep + verification".

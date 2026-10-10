# Brush Presets — Wave-4 Design (§1 P0)

Wave-4 item 2 of the renewed roadmap (audit doc). Written 2026-10-09
against the tree at `49b2d09`; every referenced API exists. This is
the implementation contract for the preset system + the starter
library + the properties panel's data model.

## What exists today (the surface to compose)

umber-brush already exposes the complete parameter set a preset
binds:
- `StrokeEvent` (pos/pressure/tilt/time/proximity/contact) — the input
  vocabulary.
- `BrushParams { color, alpha, hardness, pressure_gamma }` — the dab
  shape + response.
- `OneEuroParams { min_cutoff, beta, d_cutoff }` — the stabilizer.
- `LazyMouse` — the string-tension smoother.
- `SpacingAccumulator` / `DabPlan` — the spacing engine (residual
  partial-dab spacing, dabs-per-radius).

What does NOT exist: any way to name, save, load, or organize a set
of these parameters — no `BrushPreset` type, no file format, no
library, no UI. That is this slice.

## The preset model

```rust
// crates/umber-brush/src/preset.rs (new)
pub struct BrushPreset {
    pub name: String,
    pub params: BrushParams,       // color/alpha/hardness/gamma
    pub one_euro: OneEuroParams,   // stabilizer
    pub lazy_mouse: LazyMouseConfig, // NEW: extract LazyMouse's config
                                    // into a plain data struct (it
                                    // holds radius+strength today —
                                    // verify, then expose)
    pub spacing: SpacingConfig,     // NEW: same extraction for
                                    // spacing (spacing %, min dpr)
    // Curve mappings (the libmypaint inputs→dab semantics — §1's
    // "input→curve" row): pressure→alpha, pressure→radius. Stored as
    // 4-point control curves (libmypaint's YAML format is the
    // interop precedent; our own JSON for v1).
    pub alpha_curve: ControlCurve,
    pub radius_curve: ControlCurve,
}
```

`ControlCurve`: 4 control points, monotonic-X, linear interpolation
first (smoothstep can come later — Substance's curves are also
piecewise-linear under the hood). `pressure_gamma` stays a shortcut
for the identity-curve case.

## The file format

`preset.umberbrush` — a directory convention mirrors the project
format's per-layer JSON precedent:

```json
{
  "format": "umber-brush-preset",
  "version": 1,
  "name": "Soft Round",
  "params": { "color": [0.8,0.2,0.1,1.0], "alpha": 0.85,
              "hardness": 0.25, "pressure_gamma": 1.0 },
  "one_euro": { "min_cutoff": 1.2, "beta": 0.02, "d_cutoff": 1.0 },
  "lazy_mouse": { "radius": 12.0, "strength": 0.75 },
  "spacing": { "spacing": 0.12, "min_dpr": 1.0 },
  "alpha_curve": [[0,0],[0.33,0.4],[0.66,0.7],[1,1]],
  "radius_curve": [[0,0.6],[0.5,0.8],[1,1]]
}
```

Serialization: serde + serde_json (already workspace deps for the
project format — no new deps). Round-trip determinism test required
(the project-format precedent: write→read→write, byte-identical).

## The starter library (the ship blocker — artists judge the library)

Twelve presets, three families, committed as data:

| Family | Presets |
|---|---|
| Round | Hard Round, Soft Round, Pressure Round |
| Detail | Fine Detail, Chisel, Spackwerfer-style Spatter (random jitter via existing spacing engine's jitter — verify presence, else defer spatter to wave 4's jitter slice) |
| Utility | Fill Flat, Eraser Hard, Eraser Soft (eraser = alpha-over with color [0,0,0,0] + blend mode from §2's stack — verify the erase path composites to transparency, not black) |
| Texture | Dry Brush (low alpha + high hardness variance), Grunge (spatter + soft) |

Each lands in `assets/brushes/*.umberbrush`. CI loads all twelve in a
test (they are part of the shipped surface). The families map 1:1 to
Substance's default shelf so a migrating artist finds their muscle
memory intact.

| Coverage |
|---|---|
| Hard Round | alpha 0.95, hardness 1.0, identity curves |
| Soft Round | alpha 0.7, hardness 0.15, pressure→alpha curve |
| Pressure Round | radius curve steep (0.3→1.0) |
| Fine Detail | radius 2px equiv, spacing 0.08 |
| Chisel | tilt-driven — DEFER tilt input until the stylus slice lands (mark the preset data with the tilt curve but the engine ignores it; the data model carries it) |
| Spatter | jitter 0.35 — DEFER if spacing engine lacks jitter; the wave-4 jitter slice feeds this |
| Fill Flat | alpha 1.0, hardness 1.0, spacing 1.0, radius 512px equiv |
| Eraser Hard/Soft | target-alpha compositing — §11 wave-2 row |
| Dry Brush | alpha 0.4, hardness 0.8, jitter 0.15 (deferred as spatter) |
| Grunge | Spatter + Soft Round composition |

## The properties panel data model (§11 P0 — the paired UX slice)

The panel edits a `BrushPreset` live: engine params bind directly
(sliders), curves bind to a 4-point curve editor widget (egui,
drag the control points; the curve renders as the interpolation
between them). The preset in the panel is the ACTIVE brush; changes
write back to the preset file on explicit save (auto-save on stroke
end is a P2 nicety). New presets: "Save As" into the user library
(`~/.local/share/umber/brushes/` Linux, `%APPDATA%/umber/brushes/`
Windows — the XDG convention both platforms honor).

Library search path: user dir first, then `assets/brushes/`. Name
collisions: user copy wins (the artist's edited Hard Round overrides
the shipped one — the Substance behavior).

## Build order (claw-dispatchable)

1. `ControlCurve` + `BrushPreset` + JSON round-trip (umber-brush,
   pure CPU) — the data model. Failable tests: curve interpolation
   exactness (identity curve at 0/0.5/1), round-trip byte-identical,
   malformed-JSON error path.
2. Config extraction: `LazyMouseConfig`/`SpacingConfig` structs +
   accessor wiring (small, lands with 1).
3. The twelve starter presets as data + the CI-load test.
4. The properties panel (umber-app; egui) — rides the data model.

Slices 1-2 are one claw dispatch (the data model is self-contained,
testable headless). Slice 3 is data + one test. Slice 4 is app UI —
dispatch after 1-3 merge.

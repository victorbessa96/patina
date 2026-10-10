# Display panel (Wave 5, last named item) — landing notes

The display half of wave-5's "display panel + real OCIO" item: the viewer
chain vocabulary in `umber-color`, the Display panel in the egui shell, and
`.umber` persistence. The real-OCIO build stays gated per its design: the
`ocio` feature's stub-mode bridge (`umber-color/src/ocio.rs`) is unchanged,
and its API shapes are still the contract its gated tests verify.

Written cargo-free (headless session; rustfmt/cargo not run locally). The
remote gate is the first compile.

## Reconciliation: extend `DisplayTransform`, don't invent `ViewTransform`

`umber-color::display` already had `DisplayTransform { None, Srgb }` with
`apply_display` / `invert_display` (the CPU reference path). The brief's
`ViewTransform { Raw, SRGB, Rec709 }` is that enum plus one variant, so the
enum was **extended**:

- `Rec709` added (ITU-R BT.709 OETF, spec-rounded constants 1.099 / 0.099 /
  0.018 / 4.5 / 0.45), with forward and inverse arms in both functions.
- `None` **is** Raw. It keeps its Rust name (no churn for existing tests);
  the UI label (`DisplayTransform::label`) and the serde name are `"Raw"`.
- `DisplayTransform::default()` stays `Srgb` (pinned by
  `default_display_is_srgb`). `DisplaySettings::default()` is written by hand
  as Raw / 0 EV / gamma 1, so a project with no display key loads as "no
  viewer adjustment". The two defaults are different on purpose.

## What was built

**`umber-color/src/display.rs`**
- `DisplaySettings { view: DisplayTransform, exposure: f32, gamma: f32 }`:
  `serde(default)`, `EXPOSURE_RANGE` (-5..=5), `GAMMA_RANGE` (0.5..=2.5),
  `sanitized()` (clamps to the UI ranges and resets non-finite values to
  their defaults; used on project load), and `is_identity()`.
- `apply_display_chain(linear, &settings)`: exposure (`× 2^ev`), then
  `apply_display(view)`, then display gamma `v^(1/gamma)`. Gamma follows the
  viewer convention, so values above 1 brighten. Exposure 0 and gamma 1
  skip the arithmetic, which makes the identity settings bit-exact. Before
  the gamma pow, negatives are clamped to 0 (Raw does no clamping, and pow
  on a negative would give NaN). A non-positive or non-finite gamma is
  treated as 1.
- Deps: `serde` (workspace; derive), plus `serde_json` as a dev-dependency.
  Neither is a new crate in the lockfile, since umber-core already used
  both. umber-color still has no GPU or wgpu dependency.

**`umber-core/src/project.rs`**
- `ProjectSettings` gains `#[serde(default)] display: DisplaySettings`
  (umber-core now depends on umber-color). The change is additive:
  pre-display `project.json` files load as the identity chain.
- One struct literal had to change: the test fixture (`..Default::default()`).

**`umber-app/src/display_panel.rs`** (new) + `main.rs` wiring
- `show(ui, &mut DisplaySettings)` draws:
  - a View combo (Raw / sRGB / Rec.709);
  - an Exposure (EV) slider and a Gamma slider;
  - a Reset button;
  - an 8-swatch preview strip;
  - a status line naming the active chain.
- The preview is the pure chain applied to a synthetic gray ramp
  `PREVIEW_RAMP = 0.2·2^(k−7)`, one EV per swatch. The output is quantized
  straight into `Color32`, which is already display-encoded. It never goes
  through `egui::Rgba`, because that would encode a second time.
- `Panel::Display` ("Display" tab) is docked in the left column.
  `AppState::display` holds the settings, and `PanelViewer` borrows them.
- Save writes `ProjectSettings::display`. Open restores it through
  `sanitized()`.

## The v1 boundary (no faked GPU effect)

The viewport's mesh pass and the UV view's paint-target display
(`umber_gpu::texture_display`) both render on the GPU. A CPU chain can't
reach those pixels, so **neither view changes with these settings yet**.
The panel's status line says this ("Preview only — the viewport and UV
view are unaffected until the GPU display LUT lands").

What is ready for that step:
- the settings are stored and persisted;
- the chain is pure, tested, and the reference the GPU path has to match.

**Named follow-up: the GPU display LUT.** Bake `apply_display_chain` into a
1D/3D LUT in `umber-gpu`, sample it in the mesh pass and in
`texture_display`, and validate it against the CPU chain. That matches the
§9 row "Display transform via GPU LUT". Real OCIO views (ACES 2.0 CG/Studio)
join the combo once the bundled OCIO build lands.

## Tests

- **umber-color:**
  - Rec.709 known points (0, 0.01 linear segment, mid-gray ≈ 0.409, 1) and
    a roundtrip across the 0.018 join.
  - Identity chain is bit-exact.
  - +1 EV doubles exactly.
  - sRGB + gamma 2.2: a mirrored f32 chain (within 1e-6, because the dev
    profile's opt-level 1 can const-fold `powf` at one call site and not
    the other), with the hand derivation 0.461 356^(1/2.2) = 0.703 541 →
    byte 179.
  - Chain order: exposure runs before the view's clamp.
  - Raw + gamma gives no NaN.
  - Degenerate gamma, `sanitized`, and serde (default bytes pinned, modified
    values bit-exact, missing keys filled with defaults).
- **umber-core:** an old-format `project.json` without a display key loads
  as defaults. Default settings and modified settings (Rec709 / 1.37 / 2.2)
  each round-trip save→load→save byte-identical.
- **umber-app:**
  - The ramp is one stop per swatch and stays below the clamp at +2 EV.
  - **Monotonicity property:** +2 > +1 > 0 EV is strict at all 8 stops ×
    3 views × 3 gammas, asserted on f32 values rather than quantized bytes.
  - Identity swatches equal the raw ramp bytes (top swatch = 51).
  - sRGB is brighter than Raw at every swatch.
  - `Color32` clamping, including NaN.
  - The status-line text.

# Umber UI strings — en-US, the source of truth (docs/specs/i18n-design.md).
#
# Key scheme: the code calls `tr("group.name")`; Fluent ids cannot contain
# dots, so `group.name` resolves to message `group`, attribute `.name`.
# Every key the code calls must exist here (the completeness test in
# crates/umber-app/src/i18n.rs fails otherwise). Other locales may omit
# keys: a missing key falls back to this file, then to the key itself.

## Top menu bar

menu =
    .file = File
    .view = View
    .help = Help
    .open-mesh = Open Mesh…
    .open-project = Open Project…
    .save-project = Save Project…
    .export-png = Export Painted Map (PNG)…
    .load-environment = Load Environment…
    .perf-hud = Show Perf HUD
    .perf-hud-unavailable = Perf HUD needs --features perf
    .wireframe = Show Wireframe (W)
    .grid = Show Grid (G)
    .about = About Umber
    .about-text = Umber v0.1.0 — Wave 2 in progress

## Dock panel titles

panel =
    .viewport = Viewport
    .uv-view = 2D UV
    .layers = Layers
    .properties = Properties
    .assets = Assets
    .history = History
    .texture-sets = Texture Sets
    .bakes = Bakes
    .export = Export
    .graph = Graph
    .display = Display

## Primary buttons

button =
    .bake = Bake
    .export = Export
    .evaluate = Evaluate
    .save = Save
    .save-as = Save As…
    .reset = Reset

## Status phrases

status =
    .baking = { $count ->
        [one] Baking { $count } map…
       *[other] Baking { $count } maps…
    }
    .plugins = { $loaded } plugin(s) loaded, { $failed } failed

## Settings

settings =
    .language = Language

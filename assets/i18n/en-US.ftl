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
    .choose = Choose…
    .add = Add

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

## Shared fragments (several panels)

common =
    .none = none
    .out-dir = Out: { $dir }

## App shell: placeholder panels, file-dialog filters

shell =
    .assets-placeholder = Assets / shelf (Wave 4+)
    .texture-sets-placeholder = Texture sets (Wave 2)
    .filter-environment = Environment
    .filter-meshes = Meshes

## Bakes panel

bakes =
    .map-ao = Ambient occlusion
    .map-curvature = Curvature
    .map-thickness = Thickness
    .map-position = Position
    .no-mesh = No mesh loaded — open a mesh to enable baking.
    .no-gpu = No GPU device — baking needs the wgpu device.
    .resolution = Resolution
    .rays = Rays
    .dilation = Dilation
    .select-map = Select at least one map to bake.
    .select-tile = Select at least one tile to bake.
    .no-bake-yet = No bake yet.
    .done = { $count ->
        [one] Baked { $count } map in { $ms } ms: { $files }
       *[other] Baked { $count } maps in { $ms } ms: { $files }
    }
    .skipped-note = skipped (v1 bakes only AO per tile; other maps tile 1001 only): { $pairs }
    .failed = Bake failed: { $error }
    .worker-failed = Bake failed: could not start the bake worker: { $error }

## Export panel

export =
    .no-mesh = No mesh loaded — open a mesh to enable export.
    .no-gpu = No GPU device — export needs the wgpu device.
    .preset = Preset
    .size = Size
    .materialx = MaterialX (.mtlx)
    .select-tile = Select at least one tile to export.
    .tile-badge = { $count ->
        [one] { $count } tile
       *[other] { $count } tiles
    }
    .source-painted = Base Color source: painted{ $tiles } — what you painted is what exports.
    .source-graph = Base Color source: graph node { $node }{ $tiles } — the panel's evaluated output.
    .source-flat-size = Base Color source: flat placeholder — the paint target is { $width } × { $height }, not the { $size } × { $size } export size.
    .source-flat-none = Base Color source: flat placeholder — no paint session or graph output live.
    .base-painted = painted
    .base-graph = graph node { $node }
    .base-flat = flat placeholder
    .base-none = Base Color: none
    .base = Base Color: { $source }
    .base-tiles = Base Color: { $source } ({ $tiles })
    .no-export-yet = No export yet.
    .done = Exported { $count } outputs ({ $base }; bake { $bake-ms } ms, write { $write-ms } ms): { $files }
    .skipped-output = { $file } (missing { $maps })
    .skipped = skipped: { $outputs }
    .tiles-skipped = tiles skipped (no mesh geometry): { $tiles }
    .failed = Export failed: { $error }

## Graph panel (list view, editor, status line)

graph =
    .view-canvas = Canvas
    .view-list = List
    .canvas-help = Drag empty space to pan, wheel to zoom, drag an output port onto an input to connect, right-click to add, Delete removes the selected edge.
    .output = Output: { $node }
    .nodes = Nodes
    .no-nodes = No nodes yet — add one below.
    .add-node = Add Node
    .remove-selected = Remove selected
    .select-node = Select a node to edit its params.
    .node-title = Node { $id }: { $def }
    .set-output = Set as output
    .edges = Edges
    .edge = { $input } ← node { $from }
    .edge-input = input
    .edge-from = from
    .add-edge = Add edge
    .no-output = No output yet — press Evaluate.
    .uniform-output = Uniform output: { $value }
    .no-eval-yet = No evaluation yet.
    .loaded = Graph loaded — press Evaluate.
    .loaded-unknown = Graph loaded with { $count } unknown node type(s) — press Evaluate.
    .load-failed = Graph load failed: { $error }
    .empty = Graph is empty — add a node first.
    .no-output-node = No output node — select one first.
    .evaluated = Evaluated { $count } node(s) at { $width }x{ $height }.
    .eval-failed = Eval failed: { $error }

## Graph canvas (noodle editor)

canvas =
    .empty-hint = Right-click to add a node
    .add-node = Add node

## Display panel

display =
    .view = View
    .exposure = Exposure (EV)
    .gamma = Gamma
    .preview = Preview (linear ramp, 1 EV per swatch):
    .chain-identity = identity (Raw, 0 EV, gamma 1)
    .chain = exposure { $exposure } EV → { $view } → gamma { $gamma }
    .live = Live in the 3D viewport (mesh pass) and the UV view (paint display) via the GPU display LUT.
    .rebuilding = Rebuilding the GPU display LUT for the 3D viewport and the UV view.
    .active-chain = Active chain: { $chain }. { $status }

## Brush properties panel

brush =
    .no-preset = No preset selected
    .preset = Preset
    .preset-none = None
    .broken = { $count } broken
    .overridden = { $count } overridden
    .name = Name
    .color = Color
    .alpha = Alpha
    .hardness = Hardness
    .pressure-gamma = Pressure gamma
    .stabilizer = Stabilizer (one-euro)
    .min-cutoff = Min cutoff
    .beta = Beta
    .d-cutoff = d cutoff
    .lazy-mouse = Lazy mouse
    .radius-px = Radius px
    .strength = Strength
    .spacing = Spacing
    .dabs-per-radius = Dabs per radius
    .pressure-alpha = Pressure → alpha
    .pressure-radius = Pressure → radius
    .save-as-label = Save as
    .save-as-hint = New preset name
    .save-failed = Save failed: { $error }
    .save-as-needs-name = Save As needs a name
    .save-as-needs-preset = Save As needs an active preset
    .save-as-no-dir = Save As failed: no user preset dir
    .save-as-failed = Save As failed: { $error }
    .reset-failed = Reset failed: { $error }

## Layers panel

layers =
    .add-paint = ＋ Add paint layer
    .default-paint-name = Paint { $n }
    .kind-paint = paint
    .kind-fill = fill
    .kind-folder = folder

## History panel

history =
    .undo = ↶ Undo
    .redo = ↷ Redo
    .journal = { $count ->
        [one] { $count } entry in journal
       *[other] { $count } entries in journal
    }

## UDIM tile selector (Bakes + Export)

tiles =
    .label = Tiles:
    .tile = Tile { $tile }

## Size combo (Bakes + Export)

size =
    .vram-note = { $size } × { $size } (~{ $mib } MB/texel-buffer)

## 2D UV view

uv =
    .no-mesh = no mesh loaded

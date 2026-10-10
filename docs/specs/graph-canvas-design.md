# Graph Canvas (Noodle) Editor — Wave-6 Design

Wave-6's first slice. Written 2026-10-10 06:45 against the tree at
`723e29f`. The requirements' §11 row: "Node-canvas widget (custom,
for graph editor) — P0" — the audit's re-prioritization moved
painting first; the graph's v1 shipped the list+params panel
(abf04b6) with this canvas explicitly named as the follow-up. This
note contracts it.

## What exists (the v1 panel's contract to keep)

GraphPanel: the node list, param editors, edge wiring via combos,
the output-node marker, cached eval, the three-way export source,
.umber persistence via the mtlx-string carry. The canvas REPLACES
the list/interaction surface, NOT the eval/persistence logic —
GraphPanel's data (graph, registry, cache, dirty, output_node) is
the canvas's model; the panel file grows the widget, the model
stays.

## The canvas (egui custom painter — the wireframe-grid precedent)

The umber-app already has two custom egui painters: the curve
editor (brush_panel — draggable control points) and the grid
overlay (viewport — inverse-VP world math). The node canvas is
their combination at panel scale:

1. **The view**: an egui `Ui` rect, pannable (drag on empty space)
   + zoomable (wheel, clamped 0.25..2.5). Node positions live in
   graph-space; the view transform is pan+scale (a 2x3 affine —
   pure math, headless-testable: `to_screen(pos, view)` /
   `to_graph(screen, view)` round-trip exactly).
2. **Nodes**: rounded rects (egui's painter.rect), the node_def
   name + id, one input port per DISTINCT input the node_def
   declares (the mtlx nodedef input lists ARE the port lists —
   reuse `painter_nodedefs()` for the customs, the standard-node
   heuristics for the rest), the output port on the right. Node
   positions persist in the graph data: `Node` gains
   `#[serde(default)] pub canvas: Option<[f32; 2]>` (default: the
   id-hash scatter — deterministic, so old files load with sane
   layout; serde-default keeps the .umber additive rule).
3. **Edges (the noodles)**: cubic beziers from output port to input
   port (horizontal-tangent standard: out-tangent +x, in-tangent
   -x, control length = clamp(|dx|, 40, 160)). Selected edge:
   brighter + click to select + Delete removes. Edge hit-testing:
   distance to the sampled polyline < 6px (16 samples — the math
   in a pure fn, headless-testable).
4. **Interactions**:
   - Drag node body: move (graph-space delta from the view
     transform).
   - Drag from output port: a pending edge follows the cursor;
     drop on an input port = connect (the graph's add_edge — the
     SAME dirty-marking as the combo path); drop elsewhere =
     cancel. Drop on empty = the node-creation menu AT that point.
   - Click node: select (the params editor shows below the canvas
     — the v1 panel's editor, reused verbatim).
   - Right-click empty: the add-node menu (defs from
     registry.node_defs()).
   - Wheel on node: not zoom (event guard: zoom only on empty
     space — the classic canvas UX trap, pinned here).
5. **The output marker**: the output node's rect gets the accent
   border.
6. **Persistence**: node canvas positions in the graph's mtlx
   carry — Node.canvas serializes as a Vec2 param named
   __canvas_pos (the mtlx layer keeps unknown params; VERIFY the
   round-trip preserves it in the claw's tests, else a sidecar
   map keyed by node id — pick what the merged mtlx tests prove,
   document).

## Tests (the pure-math core, all headless)

1. View round-trip: to_screen(to_graph(p)) == p exactly, several
   zooms/pans; the clamped zoom bounds enforced.
2. Bezier sample + hit-test: the pure fns; an edge's own polyline
   hits (< 6px), a distant point misses; the sampled points
   monotonic in t.
3. Node default layout: the id-hash scatter is deterministic
   (same ids -> same positions) and collision-light (20 nodes ->
   min pairwise distance > node width — a property assert).
4. The edge-pending state machine: pure transitions
   (None→Pending(out)→Connected(in)|Cancelled), assert all.
5. .umber round-trip: canvas positions survive (load == saved; old
   files: None -> the deterministic scatter).

The egui-side painting stays in the widget (not testable
headless); ALL logic it consumes is the pure fns above.

## Build (one lean claw slice)

The view math + beziers + hit-tests (pure, tested), the widget
(painter + interactions), the persistence, the panel swap (list
view retained behind a toggle — canvas default, list fallback).
No new deps.

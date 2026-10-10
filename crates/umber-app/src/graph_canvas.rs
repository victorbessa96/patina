//! The graph canvas ("noodle") editor — wave-6's first slice
//! (`docs/specs/graph-canvas-design.md`).
//!
//! The canvas replaces the v1 node list as the Graph panel's interaction
//! surface; the model (graph, registry, cache, dirty set, output node)
//! stays [`GraphPanel`]'s. Every piece of logic the widget consumes is a
//! pure fn here — no egui types in their signatures — so it is
//! headless-tested; [`canvas_ui`] only wraps them in painting and egui
//! input plumbing.
//!
//! # Coordinate spaces
//!
//! * **graph space** — where [`umber_graph::Node::canvas`] positions live
//!   (unbounded, y down).
//! * **screen space** — canvas-local pixels: `(0, 0)` is the canvas
//!   rect's top-left. The widget adds the rect origin when painting.
//!
//! [`CanvasView`] is the pan+zoom affine between them
//! (`screen = graph * zoom + pan`).
//!
//! # Node geometry (v1: fixed)
//!
//! Every node is a fixed [`NODE_SIZE`] (140x60) rect whose top-left is its
//! canvas position. The def name is NOT measured: the longest registered
//! name (`mesh_map_generator`) fits at the default zoom, and longer
//! (foreign) names are clipped to the rect. A header strip
//! ([`NODE_HEADER`]) carries the title; ports sit below it.

use umber_graph::mtlx::NodedefDecl;
use umber_graph::Graph;

use crate::graph_panel::GraphPanel;

/// Zoom clamp, lower bound (the design's 0.25..2.5).
pub const ZOOM_MIN: f32 = 0.25;
/// Zoom clamp, upper bound.
pub const ZOOM_MAX: f32 = 2.5;
/// Fixed node size in graph units (width, height).
pub const NODE_SIZE: [f32; 2] = [140.0, 60.0];
/// Title strip height in graph units; ports distribute below it.
pub const NODE_HEADER: f32 = 22.0;
/// Bezier sample count per edge (the design's 16).
pub const EDGE_SAMPLES: usize = 16;
/// Edge hit radius in screen pixels (the design's 6px).
pub const EDGE_HIT_RADIUS: f32 = 6.0;
/// Port hit radius in screen pixels (converted to graph units per zoom).
const PORT_HIT_PX: f32 = 9.0;
/// Drawn port radius in graph units.
const PORT_DRAW_RADIUS: f32 = 5.0;
/// Bezier control length clamp (the horizontal-tangent standard).
const CONTROL_MIN: f32 = 40.0;
const CONTROL_MAX: f32 = 160.0;
/// Background dot-grid pitch in graph units.
const GRID_SPACING: f32 = 32.0;
/// Wheel zoom rate: factor = exp(scroll_points * rate) (~1.1x per notch
/// at egui's ~50pt smooth-scroll step).
const WHEEL_ZOOM_RATE: f32 = 0.002;
/// Canvas widget height in pixels (the width fills the panel).
const CANVAS_HEIGHT: f32 = 360.0;
/// The output node's accent (border, pending noodle).
const OUTPUT_ACCENT: egui::Color32 = egui::Color32::from_rgb(232, 160, 64);

/// Scatter multiplier: Knuth's multiplicative-hash constant
/// (`floor(2^32 / phi)`). It is odd mod 64 (`2654435761 % 64 == 49`), so
/// `id * MUL mod 64` is a BIJECTION on id residues: any 64 consecutive
/// ids land in 64 distinct slots — collision-free for the common
/// sequential-id case, merely collision-light for arbitrary ids.
const SCATTER_MUL: u64 = 2_654_435_761;
/// The sparse scatter grid: 8x8 = 64 slots (a power of two — required by
/// the bijection argument above).
const SCATTER_COLS: u64 = 8;
const SCATTER_ROWS: u64 = 8;
/// Slot pitch in graph units. BOTH axes exceed the node width (140), so
/// any two distinct slots sit more than a node width apart.
const SCATTER_CELL: [f32; 2] = [200.0, 160.0];

// ---------------------------------------------------------------------------
// The view (pure).
// ---------------------------------------------------------------------------

/// Clamps a zoom to `ZOOM_MIN..=ZOOM_MAX` (non-finite → 1.0).
pub fn clamp_zoom(zoom: f32) -> f32 {
    if zoom.is_finite() {
        zoom.clamp(ZOOM_MIN, ZOOM_MAX)
    } else {
        1.0
    }
}

/// The canvas pan+zoom: `screen = graph * zoom + pan`.
///
/// Round-trip exactness: IEEE `f32` affine maps are bit-exact both ways
/// only where the products/quotients are representable — power-of-two
/// zooms with dyadic coordinates (pinned exactly in the tests). Arbitrary
/// zooms round-trip within a float ulp or so (pinned with a 1e-3 bound);
/// no f32 affine can promise more.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanvasView {
    /// Screen-space offset of the graph origin.
    pub pan: [f32; 2],
    /// Screen pixels per graph unit (always within the zoom clamp).
    pub zoom: f32,
}

impl Default for CanvasView {
    fn default() -> Self {
        Self {
            pan: [24.0, 24.0],
            zoom: 1.0,
        }
    }
}

impl CanvasView {
    /// Graph → screen.
    pub fn to_screen(self, p: [f32; 2]) -> [f32; 2] {
        [
            p[0] * self.zoom + self.pan[0],
            p[1] * self.zoom + self.pan[1],
        ]
    }

    /// Screen → graph (the inverse of [`Self::to_screen`]).
    pub fn to_graph(self, s: [f32; 2]) -> [f32; 2] {
        [
            (s[0] - self.pan[0]) / self.zoom,
            (s[1] - self.pan[1]) / self.zoom,
        ]
    }

    /// The view panned by a screen-space delta.
    pub fn pan_by(&self, delta: [f32; 2]) -> Self {
        Self {
            pan: [self.pan[0] + delta[0], self.pan[1] + delta[1]],
            zoom: self.zoom,
        }
    }

    /// The view zoomed by `factor` (clamped) about a screen anchor: the
    /// graph point under `anchor` stays under it.
    pub fn zoom_about(&self, anchor: [f32; 2], factor: f32) -> Self {
        let pinned = self.to_graph(anchor);
        let zoom = clamp_zoom(self.zoom * factor);
        Self {
            pan: [anchor[0] - pinned[0] * zoom, anchor[1] - pinned[1] * zoom],
            zoom,
        }
    }
}

/// Wheel zoom with THE guard: the view zooms only when the wheel is over
/// empty space ([`Hit::Empty`]); over a node or port the view is returned
/// unchanged (the classic canvas trap — scrolling a node list must not
/// zoom). `scroll` is egui's smooth-scroll delta in points (+ = zoom in).
pub fn wheel_zoom(view: CanvasView, hit: &Hit, anchor: [f32; 2], scroll: f32) -> CanvasView {
    if scroll == 0.0 || *hit != Hit::Empty {
        return view;
    }
    view.zoom_about(anchor, (scroll * WHEEL_ZOOM_RATE).exp())
}

/// Dot-grid line positions (screen space) covering a canvas of `extent`
/// pixels at `spacing` graph units: `(xs, ys)`; dots sit at the cross
/// product.
pub fn grid_lines(view: &CanvasView, extent: [f32; 2], spacing: f32) -> (Vec<f32>, Vec<f32>) {
    let step = spacing * view.zoom;
    let along = |axis: usize| {
        let mut out = Vec::new();
        if step <= 0.0 || !step.is_finite() {
            return out;
        }
        let first = (view.to_graph([0.0, 0.0])[axis] / spacing).floor() * spacing;
        let mut at = first * view.zoom + view.pan[axis];
        while at <= extent[axis] {
            if at >= 0.0 {
                out.push(at);
            }
            at += step;
        }
        out
    };
    (along(0), along(1))
}

// ---------------------------------------------------------------------------
// Node geometry + layout (pure).
// ---------------------------------------------------------------------------

/// An axis-aligned rect in graph space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphRect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl GraphRect {
    /// Inclusive point containment.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        (self.min[0]..=self.max[0]).contains(&p[0]) && (self.min[1]..=self.max[1]).contains(&p[1])
    }
}

/// The deterministic id-hash scatter: where a never-placed node
/// (`canvas == None`) sits. `slot = (id * SCATTER_MUL) % 64` on the 8x8
/// sparse grid of [`SCATTER_CELL`] pitch — see the constants for why
/// sequential ids never collide.
pub fn default_position(node_id: u64) -> [f32; 2] {
    let slot = node_id.wrapping_mul(SCATTER_MUL) % (SCATTER_COLS * SCATTER_ROWS);
    let x = (slot % SCATTER_COLS) as f32 * SCATTER_CELL[0];
    let y = (slot / SCATTER_COLS) as f32 * SCATTER_CELL[1];
    [x, y]
}

/// A node's rect in graph space: top-left at its canvas position (or the
/// scatter when unplaced), fixed [`NODE_SIZE`] (see module docs).
pub fn node_rect(node_id: u64, canvas: Option<[f32; 2]>) -> GraphRect {
    let min = canvas.unwrap_or_else(|| default_position(node_id));
    GraphRect {
        min,
        max: [min[0] + NODE_SIZE[0], min[1] + NODE_SIZE[1]],
    }
}

/// Which port of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortSide {
    /// The `index`-th of `count` input ports (left edge).
    Input { index: usize, count: usize },
    /// The single output port (right edge).
    Output,
}

/// A port's graph-space center: inputs spread evenly down the left edge
/// below the header, the output centered on the right edge's body.
pub fn port_position(rect: &GraphRect, port: PortSide) -> [f32; 2] {
    let body_top = rect.min[1] + NODE_HEADER;
    let body_h = rect.max[1] - body_top;
    match port {
        PortSide::Input { index, count } => [
            rect.min[0],
            body_top + body_h * (index as f32 + 1.0) / (count.max(1) as f32 + 1.0),
        ],
        PortSide::Output => [rect.max[0], body_top + body_h * 0.5],
    }
}

/// mtlx nodedef input types that carry images (edges always emit as
/// `color3`; `vector3` is `direction_warp`'s per-pixel vector map).
/// Scalar/vector2/string inputs are params, not ports.
const IMAGE_INPUT_TYPES: &[&str] = &["color3", "vector3"];

/// Registered generators without a painter nodedef: no input ports.
const GENERATORS: &[&str] = &[
    "uniform",
    "image_asset",
    "noise_perlin",
    "noise_value",
    "noise_worley",
    "gradient",
    "checkerboard",
    "dots",
];

/// The standard-node port heuristic (nodes without a painter nodedef):
/// the input names each engine impl actually reads.
fn standard_inputs(node_def: &str) -> &'static [&'static str] {
    if GENERATORS.contains(&node_def) {
        return &[];
    }
    match node_def {
        // Named multi-input consumers.
        "mix" => &["fg", "bg"],
        "triplanar_blend" => &["x", "y", "z"],
        // Single-input filters (and passthrough / unknown defs): the
        // engine accepts any one name; "in" is the panel's convention.
        _ => &["in"],
    }
}

/// A node's input ports: the painter nodedef's image-typed inputs (the
/// mtlx input lists ARE the port lists) or the standard heuristic, then
/// any already-wired input names not covered (so loaded graphs and
/// v1-combo edges always have a port to land on). Deduped, ordered.
pub fn input_ports(node_def: &str, customs: &[NodedefDecl], wired: &[&str]) -> Vec<String> {
    let mut ports: Vec<String> = match customs.iter().find(|d| d.name == node_def) {
        Some(decl) => decl
            .inputs
            .iter()
            .filter(|(_, ty)| IMAGE_INPUT_TYPES.contains(&ty.as_str()))
            .map(|(name, _)| name.clone())
            .collect(),
        None => standard_inputs(node_def)
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
    };
    for name in wired {
        if !ports.iter().any(|p| p == name) {
            ports.push((*name).to_string());
        }
    }
    ports
}

/// One node's resolved canvas geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeLayout {
    pub id: u64,
    pub node_def: String,
    pub rect: GraphRect,
    pub inputs: Vec<String>,
}

/// Lays out every node (graph order = paint order; later is on top).
pub fn node_layouts(graph: &Graph, customs: &[NodedefDecl]) -> Vec<NodeLayout> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let wired: Vec<&str> = graph
                .edges
                .iter()
                .filter(|e| e.to == node.id)
                .map(|e| e.input.as_str())
                .collect();
            NodeLayout {
                id: node.id,
                node_def: node.node_def.clone(),
                rect: node_rect(node.id, node.canvas),
                inputs: input_ports(&node.node_def, customs, &wired),
            }
        })
        .collect()
}

/// What a graph-space point lands on.
#[derive(Debug, Clone, PartialEq)]
pub enum Hit {
    Empty,
    Body(u64),
    Output(u64),
    Input { node: u64, input: String },
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    dx * dx + dy * dy
}

/// Hit-tests a graph-space point: topmost node first (reverse paint
/// order); ports (within `port_radius` graph units — they straddle the
/// rect edge) before the body.
pub fn hit_test(layouts: &[NodeLayout], p: [f32; 2], port_radius: f32) -> Hit {
    let r2 = port_radius * port_radius;
    for layout in layouts.iter().rev() {
        if dist2(port_position(&layout.rect, PortSide::Output), p) <= r2 {
            return Hit::Output(layout.id);
        }
        let count = layout.inputs.len();
        for (index, name) in layout.inputs.iter().enumerate() {
            let at = port_position(&layout.rect, PortSide::Input { index, count });
            if dist2(at, p) <= r2 {
                return Hit::Input {
                    node: layout.id,
                    input: name.clone(),
                };
            }
        }
        if layout.rect.contains(p) {
            return Hit::Body(layout.id);
        }
    }
    Hit::Empty
}

/// Hit-test at a screen point under `view` (port radius scales so it
/// stays [`PORT_HIT_PX`] on screen).
fn hit_screen(view: &CanvasView, layouts: &[NodeLayout], screen: [f32; 2]) -> Hit {
    hit_test(layouts, view.to_graph(screen), PORT_HIT_PX / view.zoom)
}

// ---------------------------------------------------------------------------
// Edges (pure).
// ---------------------------------------------------------------------------

/// The noodle: a cubic bezier from an output port to an input port,
/// horizontal tangents (out +x, in -x), control length
/// `clamp(|dx|, 40, 160)`, sampled at `t = i / 15` for `i in 0..16`.
/// Endpoints are exact (`[0] == out_pos`, `[15] == in_pos`). Affine
/// invariant, so callers may sample in either space.
pub fn bezier_points(out_pos: [f32; 2], in_pos: [f32; 2]) -> [[f32; 2]; EDGE_SAMPLES] {
    let c = (in_pos[0] - out_pos[0])
        .abs()
        .clamp(CONTROL_MIN, CONTROL_MAX);
    let p1 = [out_pos[0] + c, out_pos[1]];
    let p2 = [in_pos[0] - c, in_pos[1]];
    let mut out = [[0.0; 2]; EDGE_SAMPLES];
    for (i, sample) in out.iter_mut().enumerate() {
        let t = i as f32 / (EDGE_SAMPLES - 1) as f32;
        let u = 1.0 - t;
        let (b0, b1, b2, b3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
        let blend = |a: usize| b0 * out_pos[a] + b1 * p1[a] + b2 * p2[a] + b3 * in_pos[a];
        *sample = [blend(0), blend(1)];
    }
    out
}

fn segment_dist2(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if len2 > 0.0 {
        let along = (p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1];
        (along / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist2(p, [a[0] + ab[0] * t, a[1] + ab[1] * t])
}

/// Whether `point` lies strictly within `radius` of the sampled polyline.
pub fn edge_hit(sampled: &[[f32; 2]], point: [f32; 2], radius: f32) -> bool {
    let r2 = radius * radius;
    match sampled {
        [] => false,
        [only] => dist2(*only, point) < r2,
        _ => sampled
            .windows(2)
            .any(|w| segment_dist2(point, w[0], w[1]) < r2),
    }
}

/// One edge's screen-space noodle, keyed `(to, input)` (an input takes
/// one edge — the selection/delete key).
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeCurve {
    pub to: u64,
    pub input: String,
    pub points: [[f32; 2]; EDGE_SAMPLES],
}

/// Every drawable edge's screen-space noodle (edges whose endpoints are
/// missing from `layouts` are skipped).
pub fn edge_curves(graph: &Graph, layouts: &[NodeLayout], view: &CanvasView) -> Vec<EdgeCurve> {
    graph
        .edges
        .iter()
        .filter_map(|edge| {
            let from = layouts.iter().find(|l| l.id == edge.from)?;
            let to = layouts.iter().find(|l| l.id == edge.to)?;
            let index = to.inputs.iter().position(|n| *n == edge.input)?;
            let count = to.inputs.len();
            let out_pos = view.to_screen(port_position(&from.rect, PortSide::Output));
            let in_port = port_position(&to.rect, PortSide::Input { index, count });
            let in_pos = view.to_screen(in_port);
            Some(EdgeCurve {
                to: edge.to,
                input: edge.input.clone(),
                points: bezier_points(out_pos, in_pos),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The drag state machine (pure).
// ---------------------------------------------------------------------------

/// The canvas drag state.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DragState {
    #[default]
    Idle,
    /// Panning the view; carries the last screen pointer position (the
    /// origin of the next frame's pan delta).
    Pan([f32; 2]),
    /// Moving a node; carries its id and the grab offset (pointer minus
    /// node origin, graph space) so the node doesn't jump to the cursor.
    MoveNode(u64, [f32; 2]),
    /// Dragging a noodle out of `from`'s output port; carries the cursor
    /// in graph space.
    PendingEdge(u64, [f32; 2]),
}

/// What a continuing drag asks the widget to apply this frame.
#[derive(Debug, Clone, PartialEq)]
pub enum DragEffect {
    None,
    /// Pan the view by a screen delta.
    PanBy([f32; 2]),
    /// Place a node's origin at a graph position.
    MoveNodeTo(u64, [f32; 2]),
}

/// What a finished drag asks the widget to do.
#[derive(Debug, Clone, PartialEq)]
pub enum CanvasAction {
    None,
    /// Wire `from`'s output into `to`'s `input` (the panel's add_edge).
    Connect {
        from: u64,
        to: u64,
        input: String,
    },
    /// A pending edge dropped on nothing connectable.
    Cancel,
    /// A pending edge dropped on empty space: open the node-creation menu
    /// at this screen point.
    Menu([f32; 2]),
}

/// Starts a drag from what the press landed on: empty → [`DragState::Pan`];
/// node body → [`DragState::MoveNode`]; output port →
/// [`DragState::PendingEdge`]; input port → [`DragState::Idle`] (v1:
/// inputs are drop targets, not drag sources).
pub fn begin_drag(
    hit: &Hit,
    layouts: &[NodeLayout],
    pointer: [f32; 2],
    view: &CanvasView,
) -> DragState {
    let at = view.to_graph(pointer);
    match hit {
        Hit::Empty => DragState::Pan(pointer),
        Hit::Body(id) => match layouts.iter().find(|l| l.id == *id) {
            Some(l) => DragState::MoveNode(*id, [at[0] - l.rect.min[0], at[1] - l.rect.min[1]]),
            None => DragState::Idle,
        },
        Hit::Output(id) => DragState::PendingEdge(*id, at),
        Hit::Input { .. } => DragState::Idle,
    }
}

/// Advances a drag to the pointer's new screen position.
pub fn continue_drag(
    state: &DragState,
    pointer: [f32; 2],
    view: &CanvasView,
) -> (DragState, DragEffect) {
    match *state {
        DragState::Idle => (DragState::Idle, DragEffect::None),
        DragState::Pan(last) => (
            DragState::Pan(pointer),
            DragEffect::PanBy([pointer[0] - last[0], pointer[1] - last[1]]),
        ),
        DragState::MoveNode(id, offset) => {
            let at = view.to_graph(pointer);
            (
                DragState::MoveNode(id, offset),
                DragEffect::MoveNodeTo(id, [at[0] - offset[0], at[1] - offset[1]]),
            )
        }
        DragState::PendingEdge(from, _) => (
            DragState::PendingEdge(from, view.to_graph(pointer)),
            DragEffect::None,
        ),
    }
}

/// Ends a drag on `drop` (what the release point hit; `pointer` its
/// screen position). Always returns to [`DragState::Idle`]. A pending
/// edge: on another node's input → [`CanvasAction::Connect`]; on empty
/// space → [`CanvasAction::Menu`]; anywhere else (a body, an output, its
/// own input) → [`CanvasAction::Cancel`]. Other drags → `None`.
pub fn end_drag(state: &DragState, drop: &Hit, pointer: [f32; 2]) -> (DragState, CanvasAction) {
    let action = match state {
        DragState::PendingEdge(from, _) => match drop {
            Hit::Input { node, input } if node != from => CanvasAction::Connect {
                from: *from,
                to: *node,
                input: input.clone(),
            },
            Hit::Empty => CanvasAction::Menu(pointer),
            _ => CanvasAction::Cancel,
        },
        DragState::Idle | DragState::Pan(_) | DragState::MoveNode(..) => CanvasAction::None,
    };
    (DragState::Idle, action)
}

// ---------------------------------------------------------------------------
// Widget state + the widget.
// ---------------------------------------------------------------------------

/// The open node-creation popup.
#[derive(Debug, Clone, PartialEq)]
pub struct CanvasMenu {
    /// Screen position (canvas-local) the popup opens at.
    pub at: [f32; 2],
    /// Graph position the created node is placed at.
    pub graph: [f32; 2],
    /// A pending edge's source (drop-on-empty): the new node's first
    /// input gets wired from it.
    pub connect_from: Option<u64>,
    /// Suppresses the click-elsewhere close on the frame that opened it.
    pub just_opened: bool,
}

/// The canvas's editor-session state (not persisted — positions live on
/// the nodes themselves).
#[derive(Debug, Clone, Default)]
pub struct CanvasState {
    pub view: CanvasView,
    pub drag: DragState,
    /// The selected edge, keyed `(to, input)`.
    pub selected_edge: Option<(u64, String)>,
    pub menu: Option<CanvasMenu>,
}

/// Draws the canvas and applies its interactions to `panel` (see the
/// module docs + the design's interaction list). Untested by
/// construction; every decision it makes is a pure fn above.
pub fn canvas_ui(ui: &mut egui::Ui, panel: &mut GraphPanel) {
    let width = ui.available_width().max(160.0);
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width, CANVAS_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    let local = |p: egui::Pos2| [p.x - rect.min.x, p.y - rect.min.y];
    let to_pos = |p: [f32; 2]| egui::pos2(rect.min.x + p[0], rect.min.y + p[1]);
    let customs = umber_graph::mtlx::painter_nodedefs();

    // --- Interactions (against the frame-start layout). ---
    let layouts = node_layouts(panel.graph(), &customs);
    let curves = edge_curves(panel.graph(), &layouts, &panel.canvas.view);

    // Wheel zoom — only over empty space (the guard lives in wheel_zoom).
    if let Some(hover) = response.hover_pos() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let anchor = local(hover);
            let hit = hit_screen(&panel.canvas.view, &layouts, anchor);
            let zoomed = wheel_zoom(panel.canvas.view, &hit, anchor, scroll);
            if zoomed != panel.canvas.view {
                panel.canvas.view = zoomed;
                // Consumed: the hosting scroll area must not also scroll.
                ui.ctx()
                    .input_mut(|i| i.smooth_scroll_delta = egui::Vec2::ZERO);
            }
        }
    }

    // Middle-drag pans anywhere (body drags included).
    if response.dragged_by(egui::PointerButton::Middle) {
        let d = response.drag_delta();
        panel.canvas.view = panel.canvas.view.pan_by([d.x, d.y]);
    }

    if response.drag_started_by(egui::PointerButton::Primary) {
        // Hit-test at the PRESS origin: drag_started fires past the drag
        // threshold, by which time a fast flick has left the port.
        let origin = ui
            .input(|i| i.pointer.press_origin())
            .or_else(|| response.interact_pointer_pos());
        if let Some(origin) = origin {
            let at = local(origin);
            let hit = hit_screen(&panel.canvas.view, &layouts, at);
            if let Hit::Body(id) = hit {
                panel.select(Some(id));
            }
            panel.canvas.drag = begin_drag(&hit, &layouts, at, &panel.canvas.view);
        }
    }
    if response.dragged_by(egui::PointerButton::Primary) {
        if let Some(pos) = response.interact_pointer_pos() {
            let (next, effect) = continue_drag(&panel.canvas.drag, local(pos), &panel.canvas.view);
            panel.canvas.drag = next;
            match effect {
                DragEffect::None => {}
                DragEffect::PanBy(delta) => panel.canvas.view = panel.canvas.view.pan_by(delta),
                DragEffect::MoveNodeTo(id, at) => {
                    panel.set_node_canvas(id, at);
                }
            }
        }
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        let state = std::mem::take(&mut panel.canvas.drag);
        let pending_from = match state {
            DragState::PendingEdge(from, _) => Some(from),
            _ => None,
        };
        let pos = response
            .interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.latest_pos()));
        if let Some(pos) = pos {
            let at = local(pos);
            let drop = hit_screen(&panel.canvas.view, &layouts, at);
            let (next, action) = end_drag(&state, &drop, at);
            panel.canvas.drag = next;
            match action {
                CanvasAction::Connect { from, to, input } => {
                    panel.connect(from, to, &input);
                }
                CanvasAction::Menu(at) => {
                    panel.canvas.menu = Some(CanvasMenu {
                        at,
                        graph: panel.canvas.view.to_graph(at),
                        connect_from: pending_from,
                        just_opened: true,
                    });
                }
                CanvasAction::Cancel | CanvasAction::None => {}
            }
        }
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let at = local(pos);
            match hit_screen(&panel.canvas.view, &layouts, at) {
                Hit::Body(id) | Hit::Output(id) | Hit::Input { node: id, .. } => {
                    panel.select(Some(id));
                    panel.canvas.selected_edge = None;
                }
                Hit::Empty => {
                    let picked = curves
                        .iter()
                        .find(|c| edge_hit(&c.points, at, EDGE_HIT_RADIUS))
                        .map(|c| (c.to, c.input.clone()));
                    if picked.is_none() {
                        panel.select(None);
                    }
                    panel.canvas.selected_edge = picked;
                }
            }
        }
    }
    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let at = local(pos);
            if hit_screen(&panel.canvas.view, &layouts, at) == Hit::Empty {
                panel.canvas.menu = Some(CanvasMenu {
                    at,
                    graph: panel.canvas.view.to_graph(at),
                    connect_from: None,
                    just_opened: true,
                });
            }
        }
    }
    if panel.canvas.selected_edge.is_some()
        && !ui.ctx().text_edit_focused()
        && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
    {
        if let Some((to, input)) = panel.canvas.selected_edge.take() {
            panel.remove_edge(to, &input);
        }
    }

    // --- Painting (against the post-interaction layout). ---
    let view = panel.canvas.view;
    let layouts = node_layouts(panel.graph(), &customs);
    let curves = edge_curves(panel.graph(), &layouts, &view);
    let visuals = ui.visuals().clone();
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 0.0, visuals.extreme_bg_color);
    let dot = visuals.weak_text_color().gamma_multiply(0.35);
    let (xs, ys) = grid_lines(&view, [rect.width(), rect.height()], GRID_SPACING);
    for &x in &xs {
        for &y in &ys {
            painter.circle_filled(to_pos([x, y]), 1.0, dot);
        }
    }

    for curve in &curves {
        let selected = panel
            .canvas
            .selected_edge
            .as_ref()
            .is_some_and(|(to, input)| *to == curve.to && *input == curve.input);
        let stroke = if selected {
            egui::Stroke::new(3.0, visuals.selection.stroke.color)
        } else {
            egui::Stroke::new(1.5, visuals.widgets.inactive.fg_stroke.color)
        };
        let points: Vec<egui::Pos2> = curve.points.iter().map(|p| to_pos(*p)).collect();
        painter.add(egui::Shape::line(points, stroke));
    }
    if let DragState::PendingEdge(from, cursor) = panel.canvas.drag {
        if let Some(source) = layouts.iter().find(|l| l.id == from) {
            let out_pos = view.to_screen(port_position(&source.rect, PortSide::Output));
            let points: Vec<egui::Pos2> = bezier_points(out_pos, view.to_screen(cursor))
                .iter()
                .map(|p| to_pos(*p))
                .collect();
            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(2.0, OUTPUT_ACCENT),
            ));
        }
    }

    let port_r = (PORT_DRAW_RADIUS * view.zoom).max(2.0);
    let title_font = egui::FontId::proportional((13.0 * view.zoom).max(5.0));
    let port_font = egui::FontId::proportional((10.0 * view.zoom).max(4.0));
    for layout in &layouts {
        let r = egui::Rect::from_min_max(
            to_pos(view.to_screen(layout.rect.min)),
            to_pos(view.to_screen(layout.rect.max)),
        );
        let selected = panel.selected() == Some(layout.id);
        let is_output = panel.output_node() == Some(layout.id);
        let fill = if selected {
            visuals.selection.bg_fill.gamma_multiply(0.45)
        } else {
            visuals.widgets.inactive.bg_fill
        };
        let stroke = if is_output {
            egui::Stroke::new(if selected { 3.0 } else { 2.0 }, OUTPUT_ACCENT)
        } else if selected {
            egui::Stroke::new(2.0, visuals.selection.stroke.color)
        } else {
            visuals.widgets.noninteractive.bg_stroke
        };
        painter.rect(r, 6.0 * view.zoom, fill, stroke, egui::StrokeKind::Inside);

        let text = painter.with_clip_rect(r.intersect(rect));
        text.text(
            r.min + egui::vec2(8.0, 4.0) * view.zoom,
            egui::Align2::LEFT_TOP,
            format!("{}  #{}", layout.node_def, layout.id),
            title_font.clone(),
            visuals.text_color(),
        );
        let count = layout.inputs.len();
        for (index, name) in layout.inputs.iter().enumerate() {
            let at = to_pos(view.to_screen(port_position(
                &layout.rect,
                PortSide::Input { index, count },
            )));
            painter.circle_filled(at, port_r, visuals.widgets.active.bg_fill);
            if view.zoom >= 0.75 {
                text.text(
                    at + egui::vec2(port_r + 3.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    name,
                    port_font.clone(),
                    visuals.weak_text_color(),
                );
            }
        }
        let out_at = to_pos(view.to_screen(port_position(&layout.rect, PortSide::Output)));
        let out_fill = if is_output {
            OUTPUT_ACCENT
        } else {
            visuals.widgets.active.bg_fill
        };
        painter.circle_filled(out_at, port_r, out_fill);
    }
    if layouts.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Right-click to add a node",
            egui::FontId::proportional(13.0),
            visuals.weak_text_color(),
        );
    }

    // --- The node-creation popup. ---
    if let Some(menu) = panel.canvas.menu.clone() {
        let defs: Vec<String> = panel
            .registry()
            .node_defs()
            .iter()
            .map(|d| (*d).to_string())
            .collect();
        let mut picked: Option<String> = None;
        let area = egui::Area::new(ui.id().with("graph-canvas-add-menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(to_pos(menu.at))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.strong("Add node");
                    egui::ScrollArea::vertical()
                        .max_height(220.0)
                        .show(ui, |ui| {
                            for def in &defs {
                                if ui.button(def.as_str()).clicked() {
                                    picked = Some(def.clone());
                                }
                            }
                        });
                });
            });
        let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
        if let Some(def) = picked {
            let id = panel.add_node_at(&def, menu.graph);
            if let Some(from) = menu.connect_from {
                if let Some(input) = input_ports(&def, &customs, &[]).first() {
                    panel.connect(from, id, input);
                }
            }
            panel.select(Some(id));
            panel.canvas.menu = None;
        } else if escape || (!menu.just_opened && area.response.clicked_elsewhere()) {
            panel.canvas.menu = None;
        } else if let Some(open) = panel.canvas.menu.as_mut() {
            open.just_opened = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umber_graph::{Edge, Node};

    fn layout(id: u64, min: [f32; 2], inputs: &[&str]) -> NodeLayout {
        NodeLayout {
            id,
            node_def: "test".into(),
            rect: node_rect(id, Some(min)),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    // --- a. View round-trip + clamp. ---

    #[test]
    fn view_round_trip_is_exact_on_representable_lattice() {
        // Power-of-two zooms with dyadic pans/points: every product and
        // quotient is representable, so both directions are bit-exact.
        for zoom in [0.25_f32, 0.5, 1.0, 2.0] {
            for pan in [[0.0_f32, 0.0], [24.0, -13.5], [-640.25, 1024.0]] {
                let view = CanvasView { pan, zoom };
                for p in [[0.0_f32, 0.0], [140.0, 60.0], [-333.5, 87.25], [1e4, -2e3]] {
                    assert_eq!(view.to_graph(view.to_screen(p)), p, "g→s→g {zoom} {pan:?}");
                    assert_eq!(view.to_screen(view.to_graph(p)), p, "s→g→s {zoom} {pan:?}");
                }
            }
        }
    }

    #[test]
    fn view_round_trip_holds_tightly_at_arbitrary_zooms() {
        // Non-dyadic zooms cannot be bit-exact in f32; pin a tight bound.
        for zoom in [0.3_f32, 1.7, 2.5, 0.37] {
            let view = CanvasView {
                pan: [17.3, -91.7],
                zoom,
            };
            for p in [[0.0_f32, 0.0], [123.456, -78.9], [-1000.1, 999.9]] {
                for (a, b) in [
                    (view.to_graph(view.to_screen(p)), p),
                    (view.to_screen(view.to_graph(p)), p),
                ] {
                    assert!(
                        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3,
                        "zoom {zoom}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn zoom_clamp_is_enforced() {
        assert_eq!(clamp_zoom(0.01), ZOOM_MIN);
        assert_eq!(clamp_zoom(99.0), ZOOM_MAX);
        assert_eq!(clamp_zoom(1.3), 1.3);
        assert_eq!(clamp_zoom(f32::NAN), 1.0);
        let mut view = CanvasView::default();
        for _ in 0..100 {
            view = view.zoom_about([50.0, 50.0], 1.5);
        }
        assert_eq!(view.zoom, ZOOM_MAX, "repeated zoom-in stops at the max");
        for _ in 0..100 {
            view = view.zoom_about([50.0, 50.0], 0.5);
        }
        assert_eq!(view.zoom, ZOOM_MIN, "repeated zoom-out stops at the min");
    }

    #[test]
    fn zoom_about_keeps_the_anchor_point_fixed() {
        let view = CanvasView {
            pan: [30.0, -10.0],
            zoom: 1.0,
        };
        let anchor = [200.0, 120.0];
        let before = view.to_graph(anchor);
        let zoomed = view.zoom_about(anchor, 2.0);
        assert_eq!(zoomed.zoom, 2.0);
        assert_eq!(zoomed.to_graph(anchor), before, "anchor stays put");
    }

    #[test]
    fn wheel_zooms_only_over_empty_space() {
        let view = CanvasView::default();
        let anchor = [100.0, 100.0];
        assert_ne!(wheel_zoom(view, &Hit::Empty, anchor, 50.0).zoom, view.zoom);
        assert!(wheel_zoom(view, &Hit::Empty, anchor, 50.0).zoom > view.zoom);
        assert!(wheel_zoom(view, &Hit::Empty, anchor, -50.0).zoom < view.zoom);
        for guarded in [
            Hit::Body(1),
            Hit::Output(1),
            Hit::Input {
                node: 1,
                input: "in".into(),
            },
        ] {
            assert_eq!(
                wheel_zoom(view, &guarded, anchor, 50.0),
                view,
                "wheel over {guarded:?} must not zoom"
            );
        }
        assert_eq!(wheel_zoom(view, &Hit::Empty, anchor, 0.0), view);
    }

    // --- b. Bezier sampling + hit-testing. ---

    #[test]
    fn bezier_endpoints_exact_and_tangents_horizontal() {
        let (a, b) = ([10.0, 20.0], [300.0, 140.0]);
        let pts = bezier_points(a, b);
        assert_eq!(pts.len(), EDGE_SAMPLES);
        assert_eq!(pts[0], a);
        assert_eq!(pts[EDGE_SAMPLES - 1], b);
        // Horizontal tangents: the first step leaves rightward, the last
        // arrives rightward, both nearly flat.
        assert!(pts[1][0] > a[0] && (pts[1][1] - a[1]).abs() < (pts[1][0] - a[0]));
        assert!(pts[14][0] < b[0] && (b[1] - pts[14][1]).abs() < (b[0] - pts[14][0]));
    }

    #[test]
    fn bezier_samples_are_monotonic_in_t() {
        // y is monotone for EVERY edge (controls share the endpoint ys).
        for (a, b) in [
            ([0.0, 0.0], [400.0, 200.0]),
            ([0.0, 300.0], [20.0, -50.0]),
            ([500.0, 10.0], [0.0, 90.0]), // backwards edge
        ] {
            let pts = bezier_points(a, b);
            let rising = b[1] >= a[1];
            for w in pts.windows(2) {
                if rising {
                    assert!(w[1][1] >= w[0][1], "y must not fall: {w:?}");
                } else {
                    assert!(w[1][1] <= w[0][1], "y must not rise: {w:?}");
                }
            }
        }
        // x is strictly increasing for a forward edge wider than two
        // control lengths.
        let pts = bezier_points([0.0, 0.0], [400.0, 120.0]);
        for w in pts.windows(2) {
            assert!(w[1][0] > w[0][0], "x must advance with t: {w:?}");
        }
    }

    #[test]
    fn edge_hit_on_own_polyline_and_misses_far_points() {
        let pts = bezier_points([0.0, 0.0], [300.0, 150.0]);
        for p in pts {
            assert!(edge_hit(&pts, p, EDGE_HIT_RADIUS), "own sample {p:?}");
        }
        // Segment midpoints are on the polyline too.
        for w in pts.windows(2) {
            let mid = [(w[0][0] + w[1][0]) * 0.5, (w[0][1] + w[1][1]) * 0.5];
            assert!(edge_hit(&pts, mid, EDGE_HIT_RADIUS));
            // 5px off a segment end still hits.
            assert!(edge_hit(&pts, [w[0][0], w[0][1] + 5.0], EDGE_HIT_RADIUS));
        }
        assert!(!edge_hit(&pts, [0.0, 150.0], EDGE_HIT_RADIUS), "far corner");
        assert!(
            !edge_hit(&pts, [-20.0, 0.0], EDGE_HIT_RADIUS),
            "beyond start"
        );
        assert!(!edge_hit(&pts, [150.0, 75.0 + 80.0], EDGE_HIT_RADIUS));
        assert!(!edge_hit(&[], [0.0, 0.0], EDGE_HIT_RADIUS));
    }

    #[test]
    fn control_length_is_clamped() {
        // |dx| = 10 → control 40: the first interior sample pushes right
        // past the 10px span (the S-curve), proving the 40 floor.
        let short = bezier_points([0.0, 0.0], [10.0, 100.0]);
        assert!(short.iter().any(|p| p[0] > 10.0), "40px floor applies");
        // |dx| = 1000 → control 160, not 1000: the sample at t=1/15 sits
        // well inside a 1000px-control curve's reach.
        let long = bezier_points([0.0, 0.0], [1000.0, 0.0]);
        let t = 1.0_f32 / 15.0;
        let u = 1.0 - t;
        let expect_x = 3.0 * u * u * t * 160.0 + 3.0 * u * t * t * 840.0 + t * t * t * 1000.0;
        assert!(
            (long[1][0] - expect_x).abs() < 1e-3,
            "{} vs {expect_x}",
            long[1][0]
        );
    }

    // --- c. The scatter. ---

    #[test]
    fn scatter_is_deterministic() {
        for id in [0_u64, 1, 2, 7, 63, 64, 1000, u64::MAX] {
            assert_eq!(default_position(id), default_position(id));
            assert_eq!(node_rect(id, None).min, default_position(id));
        }
        // Pinned values (the constants are a contract: changing them
        // re-lays-out every old file).
        assert_eq!(default_position(0), [0.0, 0.0]);
        // 1 * 2654435761 % 64 = 49 → col 1, row 6.
        assert_eq!(default_position(1), [200.0, 960.0]);
    }

    #[test]
    fn scatter_is_collision_light_for_20_nodes() {
        let positions: Vec<[f32; 2]> = (1..=20).map(default_position).collect();
        let mut min = f32::INFINITY;
        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                min = min.min(dist2(*a, *b).sqrt());
            }
        }
        assert!(
            min > NODE_SIZE[0],
            "min pairwise distance {min} must exceed the node width"
        );
        // Bijection: 64 consecutive ids fill all 64 slots.
        let mut slots: Vec<[u32; 2]> = (100..164)
            .map(default_position)
            .map(|p| [p[0] as u32, p[1] as u32])
            .collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 64);
    }

    #[test]
    fn placed_nodes_use_their_canvas_position() {
        let r = node_rect(5, Some([-12.5, 40.0]));
        assert_eq!(r.min, [-12.5, 40.0]);
        assert_eq!(r.max, [127.5, 100.0]);
    }

    // --- Ports + hit-testing. ---

    #[test]
    fn port_lists_follow_nodedefs_heuristics_and_wiring() {
        let customs = umber_graph::mtlx::painter_nodedefs();
        assert_eq!(input_ports("edge_detect", &customs, &[]), ["mask"]);
        assert_eq!(
            input_ports("direction_warp", &customs, &[]),
            ["input", "vector"]
        );
        assert!(input_ports("brick_pattern", &customs, &[]).is_empty());
        assert!(input_ports("mesh_map_generator", &customs, &[]).is_empty());
        assert_eq!(input_ports("mix", &customs, &[]), ["fg", "bg"]);
        assert_eq!(
            input_ports("triplanar_blend", &customs, &[]),
            ["x", "y", "z"]
        );
        assert_eq!(input_ports("blur", &customs, &[]), ["in"]);
        assert!(input_ports("noise_perlin", &customs, &[]).is_empty());
        // Wired names not covered are appended, deduped.
        assert_eq!(
            input_ports("mix", &customs, &["in", "fg", "in"]),
            ["fg", "bg", "in"]
        );
    }

    #[test]
    fn port_positions_sit_on_the_rect_edges() {
        let r = node_rect(1, Some([0.0, 0.0]));
        assert_eq!(port_position(&r, PortSide::Output), [140.0, 41.0]);
        assert_eq!(
            port_position(&r, PortSide::Input { index: 0, count: 1 }),
            [0.0, 41.0]
        );
        let two: Vec<[f32; 2]> = (0..2)
            .map(|index| port_position(&r, PortSide::Input { index, count: 2 }))
            .collect();
        assert!(two[0][1] < two[1][1], "inputs go top to bottom");
        assert!(two
            .iter()
            .all(|p| p[1] > NODE_HEADER && p[1] < NODE_SIZE[1]));
    }

    #[test]
    fn hit_test_resolves_ports_bodies_and_empty() {
        let layouts = vec![
            layout(1, [0.0, 0.0], &[]),
            layout(2, [300.0, 0.0], &["fg", "bg"]),
        ];
        assert_eq!(hit_test(&layouts, [140.0, 41.0], 6.0), Hit::Output(1));
        let fg = port_position(&layouts[1].rect, PortSide::Input { index: 0, count: 2 });
        assert_eq!(
            hit_test(&layouts, fg, 6.0),
            Hit::Input {
                node: 2,
                input: "fg".into()
            }
        );
        assert_eq!(hit_test(&layouts, [70.0, 10.0], 6.0), Hit::Body(1));
        assert_eq!(hit_test(&layouts, [220.0, 30.0], 6.0), Hit::Empty);
        // Overlap: the later (topmost) node wins.
        let stacked = vec![layout(1, [0.0, 0.0], &[]), layout(2, [10.0, 10.0], &[])];
        assert_eq!(hit_test(&stacked, [50.0, 20.0], 6.0), Hit::Body(2));
    }

    // --- d. The drag state machine. ---

    #[test]
    fn drag_from_empty_pans() {
        let view = CanvasView::default();
        let s = begin_drag(&Hit::Empty, &[], [10.0, 10.0], &view);
        assert_eq!(s, DragState::Pan([10.0, 10.0]));
        let (s, fx) = continue_drag(&s, [25.0, 4.0], &view);
        assert_eq!(s, DragState::Pan([25.0, 4.0]));
        assert_eq!(fx, DragEffect::PanBy([15.0, -6.0]));
        assert_eq!(
            end_drag(&s, &Hit::Empty, [25.0, 4.0]),
            (DragState::Idle, CanvasAction::None)
        );
    }

    #[test]
    fn drag_on_body_moves_node_keeping_grab_offset() {
        let view = CanvasView {
            pan: [0.0, 0.0],
            zoom: 2.0,
        };
        let layouts = vec![layout(4, [10.0, 20.0], &[])];
        // Screen (40, 60) = graph (20, 30): offset (10, 10) into the node.
        let s = begin_drag(&Hit::Body(4), &layouts, [40.0, 60.0], &view);
        assert_eq!(s, DragState::MoveNode(4, [10.0, 10.0]));
        let (s2, fx) = continue_drag(&s, [100.0, 60.0], &view);
        assert_eq!(s2, s, "MoveNode state is stable");
        assert_eq!(fx, DragEffect::MoveNodeTo(4, [40.0, 20.0]));
        assert_eq!(
            end_drag(&s2, &Hit::Body(4), [100.0, 60.0]),
            (DragState::Idle, CanvasAction::None)
        );
        // Unknown node id → nothing to move.
        assert_eq!(
            begin_drag(&Hit::Body(9), &layouts, [0.0, 0.0], &view),
            DragState::Idle
        );
    }

    #[test]
    fn drag_from_output_is_a_pending_edge_that_tracks_the_cursor() {
        let view = CanvasView {
            pan: [10.0, 10.0],
            zoom: 1.0,
        };
        let s = begin_drag(&Hit::Output(1), &[], [150.0, 51.0], &view);
        assert_eq!(s, DragState::PendingEdge(1, [140.0, 41.0]));
        let (s, fx) = continue_drag(&s, [210.0, 110.0], &view);
        assert_eq!(s, DragState::PendingEdge(1, [200.0, 100.0]));
        assert_eq!(fx, DragEffect::None);
    }

    #[test]
    fn pending_edge_drop_outcomes() {
        let pending = DragState::PendingEdge(1, [0.0, 0.0]);
        // Drop on another node's input → Connect.
        assert_eq!(
            end_drag(
                &pending,
                &Hit::Input {
                    node: 2,
                    input: "fg".into()
                },
                [5.0, 5.0]
            ),
            (
                DragState::Idle,
                CanvasAction::Connect {
                    from: 1,
                    to: 2,
                    input: "fg".into()
                }
            )
        );
        // Drop on empty → the creation menu at the drop point.
        assert_eq!(
            end_drag(&pending, &Hit::Empty, [33.0, 44.0]),
            (DragState::Idle, CanvasAction::Menu([33.0, 44.0]))
        );
        // Drop nowhere connectable → Cancel.
        for nowhere in [
            Hit::Body(2),
            Hit::Output(2),
            Hit::Input {
                node: 1,
                input: "in".into(),
            }, // its own input: no self-loops
        ] {
            assert_eq!(
                end_drag(&pending, &nowhere, [0.0, 0.0]),
                (DragState::Idle, CanvasAction::Cancel),
                "drop on {nowhere:?}"
            );
        }
    }

    #[test]
    fn input_ports_and_idle_are_inert() {
        let view = CanvasView::default();
        let input = Hit::Input {
            node: 1,
            input: "in".into(),
        };
        assert_eq!(begin_drag(&input, &[], [0.0, 0.0], &view), DragState::Idle);
        assert_eq!(
            continue_drag(&DragState::Idle, [9.0, 9.0], &view),
            (DragState::Idle, DragEffect::None)
        );
        assert_eq!(
            end_drag(&DragState::Idle, &Hit::Empty, [0.0, 0.0]),
            (DragState::Idle, CanvasAction::None)
        );
    }

    // --- Layout + edge curves over a real graph. ---

    #[test]
    fn edge_curves_connect_output_to_the_named_input_port() {
        let mut g = Graph::new();
        for (id, def, at) in [
            (1, "noise_perlin", [0.0, 0.0]),
            (2, "uniform", [0.0, 200.0]),
            (3, "mix", [400.0, 100.0]),
        ] {
            g.add_node(Node {
                id,
                node_def: def.into(),
                params: vec![],
                canvas: Some(at),
            });
        }
        g.add_edge(Edge {
            from: 1,
            to: 3,
            input: "fg".into(),
        });
        g.add_edge(Edge {
            from: 2,
            to: 3,
            input: "bg".into(),
        });
        let customs = umber_graph::mtlx::painter_nodedefs();
        let layouts = node_layouts(&g, &customs);
        let view = CanvasView::default();
        let curves = edge_curves(&g, &layouts, &view);
        assert_eq!(curves.len(), 2);
        let bg = curves.iter().find(|c| c.input == "bg").expect("bg edge");
        let mix_rect = node_rect(3, Some([400.0, 100.0]));
        assert_eq!(
            bg.points[0],
            view.to_screen(port_position(
                &node_rect(2, Some([0.0, 200.0])),
                PortSide::Output
            ))
        );
        assert_eq!(
            bg.points[EDGE_SAMPLES - 1],
            view.to_screen(port_position(
                &mix_rect,
                PortSide::Input { index: 1, count: 2 }
            ))
        );
    }

    #[test]
    fn grid_lines_cover_the_canvas_at_zoomed_pitch() {
        let view = CanvasView {
            pan: [10.0, 0.0],
            zoom: 0.5,
        };
        let (xs, ys) = grid_lines(&view, [100.0, 48.0], 32.0);
        assert_eq!(xs, [10.0, 26.0, 42.0, 58.0, 74.0, 90.0]);
        assert_eq!(ys, [0.0, 16.0, 32.0, 48.0]);
    }
}

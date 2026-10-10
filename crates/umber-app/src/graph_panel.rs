//! The Graph panel: the wave-5 slice-5 app bridge for the procedural
//! node graph (`docs/specs/node-graph-design.md`, "The app bridge").
//!
//! v1 is a node list + param editors (the [`crate::brush_panel`] slider
//! patterns), NOT the noodle canvas — the canvas editor is wave-6, and
//! the design is explicit that the P0 is the engine + interchange. The
//! panel owns a [`umber_graph::Graph`], a fully-seeded
//! [`umber_graph::NodeRegistry`], an [`umber_graph::EvalCache`], and the
//! dirty set the cached-eval API consumes; edits mark nodes dirty, the
//! Evaluate button re-runs only dirty nodes + their downstream, and the
//! output node's image displays in the panel via an egui texture.
//!
//! Failures are honest: eval errors land on the status line, never a
//! panic. Mesh-map externals (`mesh_map:AO`, `mesh_map:Normal`) are flat
//! placeholders in v1 — the bake bridge does not expose live in-memory
//! bakes yet, so `mesh_map_generator` nodes evaluate against documented
//! stand-ins until it does.
//!
//! Wave-6: the noodle canvas ([`crate::graph_canvas`]) is the default
//! interaction surface; the v1 list stays behind a Canvas/List toggle.
//! Both views share the selected-node param editor below them.
//!
//! # Canvas-position persistence (the mtlx carry)
//!
//! [`Node::canvas`] rides the panel's `.mtlx` string as a `vector2` input
//! named [`CANVAS_POS_PARAM`] — chosen because the mtlx layer round-trips
//! Vec2 params exactly (shortest-round-trip `f32` formatting; pinned by
//! umber-graph's `round_trip_preserves_data`), so no sidecar map is
//! needed. [`GraphPanel::to_mtlx`] appends it only for placed nodes (never
//! touching the live graph), and [`GraphPanel::load_graph`] lifts it back
//! out of the params into `Node::canvas`. Files without it — every
//! pre-canvas `.umber` — load with `None`, which the canvas lays out with
//! its deterministic id-hash scatter.
//!
//! # WASM plugins (wave-6 plugins slice 3)
//!
//! [`GraphPanel::load_plugins`] registers plugin nodes AFTER the
//! built-ins (see [`crate::plugins`] for the dirs, the per-file
//! tolerance, and the no-shadowing collision rule). They list in the
//! Add-Node menu beside the built-ins (the registry's `node_defs` is the
//! one listing), dispatch through the same `eval_graph_cached`, and save
//! as custom nodedefs declared after [`umber_graph::mtlx::painter_nodedefs`].

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use umber_graph::{
    eval_graph_cached, EvalCache, EvalContext, Graph, Node, NodeOutput, NodeRegistry, ParamValue,
};

use crate::graph_canvas::CanvasState;
use crate::i18n::tr;
use crate::plugins::{self, PluginLoadReport};

/// Default raster resolution (square) for a fresh panel.
const DEFAULT_RESOLUTION: (u32, u32) = (512, 512);

/// The reserved param name carrying [`Node::canvas`] through `.mtlx`
/// (see module docs). Never shown in the param editor.
const CANVAS_POS_PARAM: &str = "__canvas_pos";

/// Which interaction surface the panel shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphView {
    /// The noodle canvas (default).
    Canvas,
    /// The v1 node list (fallback).
    List,
}

/// Sensible starting params for an added node, by `node_def`. Nodes not
/// listed here start with no params — evaluation then reports the honest
/// missing-param error on the status line rather than silently inventing
/// values.
fn default_params(node_def: &str) -> Vec<(String, ParamValue)> {
    match node_def {
        "uniform" => vec![("color".into(), ParamValue::Color([1.0, 1.0, 1.0]))],
        "noise_perlin" | "noise_value" | "noise_worley" => vec![
            ("scale".into(), ParamValue::Float(4.0)),
            ("seed".into(), ParamValue::Int(1)),
        ],
        "image_asset" => vec![("path".into(), ParamValue::Asset(String::new()))],
        "mesh_map_generator" => vec![("map_name".into(), ParamValue::Asset("AO".into()))],
        _ => Vec::new(),
    }
}

/// Builds the panel's registry: the engine seed plus every merged node
/// family (slices 2a/2b/2c).
fn full_registry() -> NodeRegistry {
    let mut registry = NodeRegistry::seeded();
    umber_graph::nodes::register_generator_nodes(&mut registry);
    umber_graph::nodes::register_filter_color_nodes(&mut registry);
    umber_graph::nodes::register_mask_spatial_nodes(&mut registry);
    registry
}

/// The Graph panel state: the editable graph, its eval cache, and the
/// egui session around them.
///
/// The egui drawing itself (`show`) is untested by construction; every
/// other method is pure logic with no UI dependency.
pub struct GraphPanel {
    graph: Graph,
    registry: NodeRegistry,
    cache: EvalCache,
    dirty: HashSet<u64>,
    resolution: (u32, u32),
    last_output: Option<NodeOutput>,
    selected: Option<u64>,
    status: String,
    output_node: Option<u64>,
    next_id: u64,
    /// Add-Node combo selection (a `node_def` name).
    add_def: String,
    /// Edge editor: the input name new edges attach to.
    edge_input: String,
    /// Edge editor: the upstream node new edges come from.
    edge_from: Option<u64>,
    /// Displayed output texture + the key (dims + content hash) it was
    /// built from, so the texture only re-uploads when the bytes change.
    texture: Option<egui::TextureHandle>,
    texture_key: Option<(u32, u32, u64)>,
    /// Canvas or list.
    view: GraphView,
    /// The canvas's session state (view, drag, edge selection, menu).
    pub(crate) canvas: CanvasState,
    /// `.mtlx` declarations for the loaded plugin nodes (sorted by name).
    plugin_decls: Vec<umber_graph::mtlx::NodedefDecl>,
    /// The last plugin scan's summary line (kept apart from `status`,
    /// which every evaluate overwrites).
    plugin_status: Option<String>,
}

impl GraphPanel {
    /// Creates the panel: empty graph, fully-seeded registry, 512x512
    /// resolution, no selection, nothing evaluated yet.
    pub fn new() -> Self {
        let registry = full_registry();
        let add_def = registry
            .node_defs()
            .first()
            .copied()
            .unwrap_or("passthrough")
            .to_string();
        Self {
            graph: Graph::new(),
            registry,
            cache: EvalCache::new(),
            dirty: HashSet::new(),
            resolution: DEFAULT_RESOLUTION,
            last_output: None,
            selected: None,
            status: String::from("No evaluation yet."),
            output_node: None,
            next_id: 1,
            add_def,
            edge_input: String::from("in"),
            edge_from: None,
            texture: None,
            texture_key: None,
            view: GraphView::Canvas,
            canvas: CanvasState::default(),
            plugin_decls: Vec::new(),
            plugin_status: None,
        }
    }

    /// Scans `dirs` for `*.wasm` plugins and registers them after the
    /// built-ins ([`crate::plugins::load_plugins`]): failures are
    /// warnings on the plugin status line, never fatal. Returns the
    /// scan's report.
    pub fn load_plugins(&mut self, dirs: &[PathBuf]) -> PluginLoadReport {
        let report = plugins::load_plugins(&mut self.registry, dirs);
        self.plugin_decls.extend(
            report
                .loaded
                .iter()
                .map(|name| plugins::plugin_nodedef(name)),
        );
        self.plugin_decls.sort_by(|a, b| a.name.cmp(&b.name));
        self.plugin_status = report.status_line();
        report
    }

    /// The last plugin scan's summary (`None` when no `.wasm` was found).
    pub fn plugin_status(&self) -> Option<&str> {
        self.plugin_status.as_deref()
    }

    /// The editable graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The seeded registry (all merged node families).
    pub fn registry(&self) -> &NodeRegistry {
        &self.registry
    }

    /// The dirty set (nodes awaiting re-evaluation).
    pub fn dirty_set(&self) -> &HashSet<u64> {
        &self.dirty
    }

    /// The selected node id, if any.
    pub fn selected(&self) -> Option<u64> {
        self.selected
    }

    /// The output node id (the node Evaluate displays/exports).
    pub fn output_node(&self) -> Option<u64> {
        self.output_node
    }

    /// The last evaluation's status line.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// The output node's last evaluated value, if any.
    pub fn last_output(&self) -> Option<&NodeOutput> {
        self.last_output.as_ref()
    }

    /// The shared raster resolution evaluations render at.
    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    /// Retargets the raster resolution (marks everything dirty — every
    /// cached image is now the wrong size).
    pub fn set_resolution(&mut self, resolution: (u32, u32)) {
        let (w, h) = (resolution.0.max(1), resolution.1.max(1));
        if (w, h) != self.resolution {
            self.resolution = (w, h);
            self.mark_all_dirty();
        }
    }

    /// Selects a node for param editing (`None` clears).
    pub fn select(&mut self, id: Option<u64>) {
        self.selected = id.filter(|id| self.graph.nodes.iter().any(|n| n.id == *id));
    }

    /// Edits (or adds) one param on a node and marks it dirty. Returns
    /// `false` when the node id is unknown.
    pub fn set_param(&mut self, node_id: u64, name: &str, value: ParamValue) -> bool {
        let Some(node) = self.graph.nodes.iter_mut().find(|n| n.id == node_id) else {
            return false;
        };
        match node.params.iter_mut().find(|(n, _)| n == name) {
            Some((_, v)) => *v = value,
            None => node.params.push((name.to_string(), value)),
        }
        self.dirty.insert(node_id);
        true
    }

    /// Appends a node with a fresh unique id and starter params, marks
    /// it dirty, and adopts it as the output node when the graph had
    /// none. Returns the new id.
    pub fn add_node(&mut self, node_def: &str) -> u64 {
        while self.graph.nodes.iter().any(|n| n.id == self.next_id) {
            self.next_id += 1;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.graph.add_node(Node {
            id,
            node_def: node_def.to_string(),
            params: default_params(node_def),
            canvas: None,
        });
        self.dirty.insert(id);
        if self.output_node.is_none() {
            self.output_node = Some(id);
        }
        id
    }

    /// [`Self::add_node`], placed at a graph-space canvas position (the
    /// canvas creation menu's path).
    pub fn add_node_at(&mut self, node_def: &str, pos: [f32; 2]) -> u64 {
        let id = self.add_node(node_def);
        self.set_node_canvas(id, pos);
        id
    }

    /// Moves a node on the canvas. Position is editor-only data: nothing
    /// is marked dirty. Returns `false` when the id is unknown.
    pub fn set_node_canvas(&mut self, node_id: u64, pos: [f32; 2]) -> bool {
        match self.graph.nodes.iter_mut().find(|n| n.id == node_id) {
            Some(node) => {
                node.canvas = Some(pos);
                true
            }
            None => false,
        }
    }

    /// The canvas connect: replaces whatever already feeds `to`'s
    /// `input` (an input takes one edge), then wires `from` through
    /// [`Self::add_edge`] — the v1 combo's add/dirty path. Self-loops and
    /// unknown endpoints return `false` and change nothing.
    pub fn connect(&mut self, from: u64, to: u64, input: &str) -> bool {
        let known = |id: u64| self.graph.nodes.iter().any(|n| n.id == id);
        if from == to || !known(from) || !known(to) {
            return false;
        }
        self.graph
            .edges
            .retain(|e| !(e.to == to && e.input == input));
        self.add_edge(from, to, input)
    }

    /// Removes the edge feeding `to`'s `input` and marks `to` dirty.
    /// Returns `false` when no such edge exists.
    pub fn remove_edge(&mut self, to: u64, input: &str) -> bool {
        let before = self.graph.edges.len();
        self.graph
            .edges
            .retain(|e| !(e.to == to && e.input == input));
        if self.graph.edges.len() == before {
            return false;
        }
        self.dirty.insert(to);
        true
    }

    /// Removes a node and every edge touching it. Returns `false` when
    /// the id is unknown. Structural edits invalidate the whole cache
    /// (downstream wiring changed), so everything still standing is
    /// marked dirty.
    pub fn remove_node(&mut self, node_id: u64) -> bool {
        if !self.graph.nodes.iter().any(|n| n.id == node_id) {
            return false;
        }
        self.graph.nodes.retain(|n| n.id != node_id);
        self.graph
            .edges
            .retain(|e| e.from != node_id && e.to != node_id);
        if self.selected == Some(node_id) {
            self.selected = None;
        }
        if self.output_node == Some(node_id) {
            self.output_node = self.graph.nodes.iter().map(|n| n.id).max();
        }
        self.mark_all_dirty();
        true
    }

    /// Adds an edge (`from` output → `to`'s `input`) and marks the
    /// consumer dirty. Returns `false` when either endpoint is unknown.
    pub fn add_edge(&mut self, from: u64, to: u64, input: &str) -> bool {
        let known = |id: u64| self.graph.nodes.iter().any(|n| n.id == id);
        if !known(from) || !known(to) {
            return false;
        }
        self.graph.edges.push(umber_graph::Edge {
            from,
            to,
            input: input.to_string(),
        });
        self.dirty.insert(to);
        true
    }

    /// Marks a node as the panel's output (displayed + exported).
    /// Returns `false` when the id is unknown.
    pub fn set_output_node(&mut self, node_id: u64) -> bool {
        if !self.graph.nodes.iter().any(|n| n.id == node_id) {
            return false;
        }
        self.output_node = Some(node_id);
        true
    }

    /// Replaces the graph wholesale (the Open-Project path): cache
    /// cleared, everything marked dirty, output defaulting to the
    /// highest id, selection cleared. Carried canvas positions
    /// ([`CANVAS_POS_PARAM`]) are lifted out of the params into
    /// [`Node::canvas`].
    pub fn load_graph(&mut self, mut graph: Graph) {
        lift_canvas_positions(&mut graph);
        self.canvas.drag = Default::default();
        self.canvas.selected_edge = None;
        self.canvas.menu = None;
        self.next_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        self.output_node = graph.nodes.iter().map(|n| n.id).max();
        self.graph = graph;
        self.cache = EvalCache::new();
        self.mark_all_dirty();
        self.selected = None;
        self.last_output = None;
        self.texture = None;
        self.texture_key = None;
        self.status = String::from("Graph loaded — press Evaluate.");
    }

    /// Serializes the graph to a `.mtlx` document string (the
    /// project-persistence carry). Placed nodes carry their canvas
    /// position as a trailing [`CANVAS_POS_PARAM`] `vector2` input (see
    /// module docs); unplaced nodes emit exactly as before.
    pub fn to_mtlx(&self) -> String {
        let mut carried = self.graph.clone();
        for node in &mut carried.nodes {
            if let Some(pos) = node.canvas {
                node.params
                    .push((CANVAS_POS_PARAM.to_string(), ParamValue::Vec2(pos)));
            }
        }
        let mut decls = umber_graph::mtlx::painter_nodedefs();
        decls.extend(self.plugin_decls.iter().cloned());
        umber_graph::mtlx::to_mtlx(&carried, &decls)
    }

    /// Parses a `.mtlx` document string into the panel (the
    /// project-restore path). Unknown node types survive as opaque defs
    /// per the mtlx forward-compat contract; the warning count rides
    /// the status line. Returns `false` (keeping the current graph)
    /// when the string does not parse.
    pub fn load_mtlx(&mut self, doc: &str) -> bool {
        match umber_graph::mtlx::from_mtlx(doc) {
            Ok((graph, _, warnings)) => {
                self.load_graph(graph);
                if warnings.is_empty() {
                    self.status = String::from("Graph loaded — press Evaluate.");
                } else {
                    self.status = format!(
                        "Graph loaded with {} unknown node type(s) — press Evaluate.",
                        warnings.len()
                    );
                }
                true
            }
            Err(err) => {
                self.status = format!("Graph load failed: {err}");
                false
            }
        }
    }

    /// Re-runs dirty nodes (+ downstream) through the cache and refreshes
    /// the output display. Errors land on the status line; the previous
    /// output (if any) is kept. Never panics.
    pub fn evaluate(&mut self) {
        if self.graph.nodes.is_empty() {
            self.status = String::from("Graph is empty — add a node first.");
            self.last_output = None;
            return;
        }
        let Some(output_id) = self.output_node else {
            self.status = String::from("No output node — select one first.");
            self.last_output = None;
            return;
        };
        let ctx = EvalContext {
            resolution: self.resolution,
            external: mesh_map_externals(self.resolution),
        };
        match eval_graph_cached(
            &self.graph,
            &self.registry,
            HashMap::new(),
            &ctx,
            &mut self.cache,
            &self.dirty,
        ) {
            Ok(outputs) => {
                self.dirty.clear();
                self.last_output = outputs.get(&output_id).cloned();
                let n = outputs.len();
                self.status = format!(
                    "Evaluated {n} node(s) at {}x{}.",
                    self.resolution.0, self.resolution.1
                );
            }
            Err(err) => {
                self.status = format!("Eval failed: {err}");
            }
        }
    }

    /// Whether the output node currently has exportable bytes (the cheap
    /// probe the Export badge reads each frame — no allocation).
    pub fn has_exportable_output(&self) -> bool {
        self.export_output_dims().is_some()
    }

    /// The dims the export bytes would have (`Image` dims, or the panel
    /// resolution for a `Uniform(Color)` fill); `None` when nothing
    /// exportable is cached.
    pub fn export_output_dims(&self) -> Option<(u32, u32)> {
        match self.export_output()? {
            NodeOutput::Image(buf) => Some((buf.width, buf.height)),
            NodeOutput::Uniform(ParamValue::Color(_)) => Some(self.resolution),
            NodeOutput::Uniform(_) => None,
        }
    }

    /// The output node's image as tightly-packed RGBA8 bytes at the
    /// panel resolution: `Image` outputs clone their bytes;
    /// `Uniform(Color)` fills a buffer; anything else (other uniforms,
    /// unevaluated, empty graph) is `None`.
    pub fn export_base_color(&self) -> Option<Vec<u8>> {
        match self.export_output()? {
            NodeOutput::Image(buf) => {
                if (buf.width, buf.height) == self.resolution {
                    Some(buf.data.clone())
                } else {
                    None
                }
            }
            NodeOutput::Uniform(ParamValue::Color(rgb)) => {
                let px = [
                    float_to_u8(rgb[0]),
                    float_to_u8(rgb[1]),
                    float_to_u8(rgb[2]),
                    255,
                ];
                let (w, h) = self.resolution;
                let mut out = Vec::with_capacity(w as usize * h as usize * 4);
                for _ in 0..(w as usize * h as usize) {
                    out.extend_from_slice(&px);
                }
                Some(out)
            }
            NodeOutput::Uniform(_) => None,
        }
    }

    /// The cached value of the output node, if evaluated.
    fn export_output(&self) -> Option<&NodeOutput> {
        let id = self.output_node?;
        self.cache.get(id)
    }

    /// Marks every node dirty (structural edits, resolution changes).
    fn mark_all_dirty(&mut self) {
        self.dirty = self.graph.nodes.iter().map(|n| n.id).collect();
    }

    /// Draws the panel: the Canvas/List toggle, the chosen view (noodle
    /// canvas or v1 node list), the Add-Node combo, the selected node's
    /// param editors, edge wiring, output marker, Evaluate + status, and
    /// the output image.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view, GraphView::Canvas, "Canvas");
            ui.selectable_value(&mut self.view, GraphView::List, "List");
        });
        match self.view {
            GraphView::Canvas => {
                crate::graph_canvas::canvas_ui(ui, self);
                ui.weak(
                    "Drag empty space to pan, wheel to zoom, drag an output port \
                     onto an input to connect, right-click to add, Delete removes \
                     the selected edge.",
                );
            }
            GraphView::List => self.node_list(ui),
        }
        self.add_remove_row(ui);
        if let Some(line) = self.plugin_status() {
            ui.weak(line);
        }
        self.editor_section(ui);

        ui.add_space(4.0);
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui.button(tr("button.evaluate")).clicked() {
                self.evaluate();
            }
            ui.label(format!(
                "Output: {}",
                self.output_node
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "none".into())
            ));
        });
        let status = self.status.clone();
        if status.starts_with("Eval failed") || status.starts_with("Graph load failed") {
            ui.label(egui::RichText::new(status).color(ui.visuals().error_fg_color));
        } else {
            ui.label(status);
        }

        ui.add_space(4.0);
        self.output_image(ui);
    }

    /// The v1 node list (the List view).
    fn node_list(&mut self, ui: &mut egui::Ui) {
        ui.strong("Nodes");
        let mut ids: Vec<u64> = self.graph.nodes.iter().map(|n| n.id).collect();
        ids.sort_unstable();
        egui::ScrollArea::vertical()
            .max_height(140.0)
            .show(ui, |ui| {
                for id in &ids {
                    let def = self
                        .graph
                        .nodes
                        .iter()
                        .find(|n| n.id == *id)
                        .map(|n| n.node_def.clone())
                        .unwrap_or_default();
                    let marker = if self.output_node == Some(*id) {
                        " ●"
                    } else {
                        ""
                    };
                    let label = format!("{id}: {def}{marker}");
                    if ui
                        .selectable_label(self.selected == Some(*id), label)
                        .clicked()
                    {
                        self.selected = Some(*id);
                    }
                }
                if ids.is_empty() {
                    ui.weak("No nodes yet — add one below.");
                }
            });
    }

    /// The Add-Node combo + Remove-selected row (both views).
    fn add_remove_row(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let defs = self.registry.node_defs();
            egui::ComboBox::from_label("Add Node")
                .selected_text(&self.add_def)
                .show_ui(ui, |ui| {
                    for def in &defs {
                        ui.selectable_value(&mut self.add_def, (*def).to_string(), *def);
                    }
                });
            if ui.button("Add").clicked() {
                let id = self.add_node(&self.add_def.clone());
                self.selected = Some(id);
            }
            if ui
                .add_enabled(
                    self.selected.is_some(),
                    egui::Button::new("Remove selected"),
                )
                .clicked()
            {
                if let Some(id) = self.selected {
                    self.remove_node(id);
                }
            }
        });
    }

    /// The selected node's editor section — the v1 editor verbatim, shared
    /// by both views (the canvas selects, this edits).
    fn editor_section(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.separator();
        if let Some(id) = self.selected {
            self.selected_editor(ui, id);
        } else {
            ui.weak("Select a node to edit its params.");
        }
    }

    /// Param editors + edge wiring + output marker for the selected node.
    fn selected_editor(&mut self, ui: &mut egui::Ui, id: u64) {
        let Some(pos) = self.graph.nodes.iter().position(|n| n.id == id) else {
            self.selected = None;
            return;
        };
        let def = self.graph.nodes[pos].node_def.clone();
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Node {id}: {def}"));
            if self.output_node != Some(id) && ui.small_button("Set as output").clicked() {
                self.set_output_node(id);
            }
        });

        // Snapshot params so UI edits below can't borrow-conflict with
        // the write-back (E0502): each editor produces an optional
        // replacement routed through set_param (which marks dirty).
        let params = self.graph.nodes[pos].params.clone();
        for (name, value) in &params {
            let replacement = param_editor(ui, name, value, self);
            if let Some(new_value) = replacement {
                self.set_param(id, name, new_value);
            }
        }

        ui.add_space(4.0);
        ui.strong("Edges");
        for (from, input) in self
            .graph
            .edges
            .iter()
            .filter(|e| e.to == id)
            .map(|e| (e.from, e.input.clone()))
            .collect::<Vec<_>>()
        {
            ui.horizontal(|ui| {
                ui.label(format!("{input} ← node {from}"));
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("input");
            ui.add(
                egui::TextEdit::singleline(&mut self.edge_input)
                    .hint_text("in")
                    .desired_width(80.0),
            );
            let others: Vec<u64> = self
                .graph
                .nodes
                .iter()
                .map(|n| n.id)
                .filter(|nid| *nid != id)
                .collect();
            if self.edge_from.is_none() {
                self.edge_from = others.first().copied();
            }
            let from_label = self
                .edge_from
                .map(|f| f.to_string())
                .unwrap_or_else(|| "none".into());
            egui::ComboBox::from_label("from")
                .selected_text(from_label)
                .show_ui(ui, |ui| {
                    for candidate in &others {
                        ui.selectable_value(
                            &mut self.edge_from,
                            Some(*candidate),
                            candidate.to_string(),
                        );
                    }
                });
            if ui.button("Add edge").clicked() {
                if let Some(from) = self.edge_from {
                    let input = self.edge_input.clone();
                    let input = if input.trim().is_empty() {
                        "in".to_string()
                    } else {
                        input
                    };
                    self.add_edge(from, id, &input);
                }
            }
        });
    }

    /// The output image display: the evaluated `Image` as an egui
    /// texture (re-uploaded only when the bytes change), a swatch line
    /// for `Uniform` values, or a hint when nothing evaluated yet.
    fn output_image(&mut self, ui: &mut egui::Ui) {
        let Some(output) = self.last_output.clone() else {
            ui.weak("No output yet — press Evaluate.");
            return;
        };
        match output {
            NodeOutput::Image(buf) => {
                let key = (buf.width, buf.height, content_hash(&buf.data));
                if self.texture_key != Some(key) {
                    let image = egui::ColorImage::from_rgba_unmultiplied(
                        [buf.width as usize, buf.height as usize],
                        &buf.data,
                    );
                    self.texture = Some(ui.ctx().load_texture(
                        "graph-output",
                        image,
                        Default::default(),
                    ));
                    self.texture_key = Some(key);
                }
                if let Some(texture) = &self.texture {
                    let max_side = 256.0;
                    let (w, h) = (buf.width as f32, buf.height as f32);
                    let scale = (max_side / w.max(h)).min(1.0);
                    ui.image((texture.id(), egui::vec2(w * scale, h * scale)));
                }
            }
            NodeOutput::Uniform(value) => {
                ui.label(format!("Uniform output: {value:?}"));
            }
        }
    }
}

impl Default for GraphPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// One param editor row. Returns the replacement value when the user
/// edited it this frame (`None` = untouched). `panel` is threaded
/// through read-only for the `NodeRef` combo's node listing.
fn param_editor(
    ui: &mut egui::Ui,
    name: &str,
    value: &ParamValue,
    panel: &GraphPanel,
) -> Option<ParamValue> {
    match value {
        ParamValue::Float(v) => {
            let mut edit = *v;
            ui.horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut edit).speed(0.01));
            });
            (edit != *v).then_some(ParamValue::Float(edit))
        }
        ParamValue::Int(v) => {
            let mut edit = *v;
            ui.horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut edit));
            });
            (edit != *v).then_some(ParamValue::Int(edit))
        }
        ParamValue::Vec2(v) => {
            let mut edit = *v;
            ui.horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut edit[0]).speed(0.01));
                ui.add(egui::DragValue::new(&mut edit[1]).speed(0.01));
            });
            (edit != *v).then_some(ParamValue::Vec2(edit))
        }
        ParamValue::Vec3(v) => {
            let mut edit = *v;
            ui.horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut edit[0]).speed(0.01));
                ui.add(egui::DragValue::new(&mut edit[1]).speed(0.01));
                ui.add(egui::DragValue::new(&mut edit[2]).speed(0.01));
            });
            (edit != *v).then_some(ParamValue::Vec3(edit))
        }
        ParamValue::Color(v) => {
            let mut edit = *v;
            ui.horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut edit[0]).speed(0.01));
                ui.add(egui::DragValue::new(&mut edit[1]).speed(0.01));
                ui.add(egui::DragValue::new(&mut edit[2]).speed(0.01));
            });
            (edit != *v).then_some(ParamValue::Color(edit))
        }
        ParamValue::Bool(v) => {
            let mut edit = *v;
            ui.checkbox(&mut edit, name);
            (edit != *v).then_some(ParamValue::Bool(edit))
        }
        ParamValue::Asset(v) => {
            let mut edit = v.clone();
            ui.horizontal(|ui| {
                ui.label(name);
                ui.add(egui::TextEdit::singleline(&mut edit).desired_width(160.0));
            });
            (edit != *v).then_some(ParamValue::Asset(edit))
        }
        ParamValue::NodeRef(v) => {
            let mut edit = *v;
            let others: Vec<u64> = panel.graph.nodes.iter().map(|n| n.id).collect();
            egui::ComboBox::from_label(name)
                .selected_text(edit.to_string())
                .show_ui(ui, |ui| {
                    for candidate in &others {
                        ui.selectable_value(&mut edit, *candidate, candidate.to_string());
                    }
                });
            (edit != *v).then_some(ParamValue::NodeRef(edit))
        }
    }
}

/// Lifts every carried [`CANVAS_POS_PARAM`] out of the node params into
/// [`Node::canvas`] (the load half of the mtlx carry). The param is always
/// removed; only a finite `Vec2` becomes a position — anything else
/// (a foreign-typed or NaN value) leaves the node unplaced (the scatter).
fn lift_canvas_positions(graph: &mut Graph) {
    for node in &mut graph.nodes {
        let mut lifted = None;
        node.params.retain(|(name, value)| {
            if name != CANVAS_POS_PARAM {
                return true;
            }
            if let ParamValue::Vec2(pos) = value {
                if pos.iter().all(|c| c.is_finite()) {
                    lifted = Some(*pos);
                }
            }
            false
        });
        if lifted.is_some() {
            node.canvas = lifted;
        }
    }
}

/// The v1 mesh-map externals: flat stand-ins at the panel resolution
/// (see module docs — the bake bridge does not expose live in-memory
/// bakes yet). AO-flat is opaque white (unoccluded); Normal-flat is
/// tangent-space up, matching `bake_sources::flat_normal_rgba8`.
fn mesh_map_externals(resolution: (u32, u32)) -> HashMap<String, NodeOutput> {
    let (w, h) = (resolution.0.max(1), resolution.1.max(1));
    let texels = w as usize * h as usize;
    let mut externals = HashMap::new();
    if let Ok(ao) = umber_graph::ImageBuffer::filled(w, h, [255, 255, 255, 255]) {
        externals.insert("mesh_map:AO".to_string(), NodeOutput::Image(ao));
    }
    let mut normal = Vec::with_capacity(texels * 4);
    for _ in 0..texels {
        normal.extend_from_slice(&[128, 128, 255, 255]);
    }
    if let Ok(buf) = umber_graph::ImageBuffer::new(w, h, normal) {
        externals.insert("mesh_map:Normal".to_string(), NodeOutput::Image(buf));
    }
    externals
}

/// The engine's float→u8 quantizer, restated (umber-graph keeps the
/// function private; the export path pins the same math).
fn float_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// FNV-1a64 over the bytes (the texture re-upload key — std-only,
/// no new deps).
fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8482_2225;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A noise→passthrough graph built through the panel's own pub API.
    fn noise_chain(panel: &mut GraphPanel) -> (u64, u64) {
        let head = panel.add_node("noise_perlin");
        assert!(panel.set_param(head, "scale", ParamValue::Float(4.0)));
        assert!(panel.set_param(head, "seed", ParamValue::Int(5)));
        let tail = panel.add_node("passthrough");
        assert!(panel.add_edge(head, tail, "in"));
        assert!(panel.set_output_node(tail));
        (head, tail)
    }

    #[test]
    fn panel_eval_matches_direct_engine_eval_byte_exact() {
        // THE CAN-FAIL CORE: the panel's cached eval must equal a direct
        // engine eval of the same graph — any panel-side input mangling
        // (wrong resolution, dropped externals, cache staleness) breaks
        // byte equality here.
        let mut panel = GraphPanel::new();
        panel.set_resolution((8, 8));
        let (_head, tail) = noise_chain(&mut panel);
        panel.evaluate();
        assert!(
            !panel.status().starts_with("Eval failed"),
            "eval must succeed, got {:?}",
            panel.status()
        );
        let shown = panel.last_output().expect("output node evaluated").clone();

        let ctx = EvalContext {
            resolution: (8, 8),
            external: mesh_map_externals((8, 8)),
        };
        let direct = umber_graph::eval_graph(panel.graph(), panel.registry(), HashMap::new(), &ctx)
            .expect("direct engine eval works");
        assert_eq!(
            shown, direct[&tail],
            "panel display must equal the direct engine eval"
        );
        assert!(
            matches!(shown, NodeOutput::Image(_)),
            "noise chain must produce an Image, got {shown:?}"
        );
    }

    #[test]
    fn param_edit_dirties_and_changes_noise_output() {
        let mut panel = GraphPanel::new();
        panel.set_resolution((8, 8));
        let (head, _tail) = noise_chain(&mut panel);
        panel.evaluate();
        let before = panel.last_output().cloned().expect("evaluated");

        // Editing a param marks that node dirty…
        assert!(panel.dirty_set().is_empty(), "eval must clear dirty");
        assert!(panel.set_param(head, "scale", ParamValue::Float(4.0)));
        assert!(
            panel.dirty_set().contains(&head),
            "edited node must be dirty"
        );
        // …and a scale change MUST change noise bytes (same value is a
        // no-op write; a new scale re-renders).
        assert!(panel.set_param(head, "scale", ParamValue::Float(9.0)));
        panel.evaluate();
        let after = panel.last_output().cloned().expect("re-evaluated");
        assert_ne!(
            before, after,
            "noise scale 4.0 → 9.0 must change the output bytes"
        );
    }

    #[test]
    fn export_base_color_fills_uniform_color_exactly_and_empty_is_none() {
        let mut panel = GraphPanel::new();
        assert_eq!(
            panel.export_base_color(),
            None,
            "empty graph exports nothing"
        );

        let id = panel.add_node("uniform");
        assert!(panel.set_param(id, "color", ParamValue::Color([1.0, 0.0, 0.25])));
        assert!(panel.set_output_node(id));
        panel.evaluate();
        assert!(
            !panel.status().starts_with("Eval failed"),
            "uniform fill must evaluate, got {:?}",
            panel.status()
        );
        let bytes = panel
            .export_base_color()
            .expect("uniform color graph exports");
        let (w, h) = panel.resolution();
        assert_eq!(bytes.len(), w as usize * h as usize * 4);
        // [1.0, 0.0, 0.25] → [255, 0, 64, 255] under the pinned quantizer.
        for texel in bytes.chunks_exact(4) {
            assert_eq!(texel, [255, 0, 64, 255]);
        }
    }

    #[test]
    fn project_round_trip_preserves_graph_nodes_and_edges() {
        // graph → mtlx string → project dir → back → PartialEq on the
        // rebuilt graph (the additive mtlx-string carry path).
        let mut panel = GraphPanel::new();
        let (_head, _tail) = noise_chain(&mut panel);
        let doc = panel.to_mtlx();
        assert!(!doc.is_empty());

        let model = umber_core::project::ProjectModel::new(vec![], vec![], Default::default())
            .with_graphs_mtlx(vec![doc]);
        let dir = std::env::temp_dir().join(format!(
            "umber-graph-panel-roundtrip-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        umber_core::project::save_to_dir(&model, &dir).expect("save works");
        let back = umber_core::project::load_from_dir(&dir).expect("old-shape load works");
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.graphs_mtlx.len(), 1);

        let (graph, _, _) =
            umber_graph::mtlx::from_mtlx(&back.graphs_mtlx[0]).expect("mtlx parses");
        assert_eq!(
            graph,
            *panel.graph(),
            "nodes AND edges must survive the project round-trip"
        );
    }

    /// Saves `doc` through the real project dir path and reads it back.
    fn project_round_trip(doc: String, tag: &str) -> String {
        let model = umber_core::project::ProjectModel::new(vec![], vec![], Default::default())
            .with_graphs_mtlx(vec![doc]);
        let dir =
            std::env::temp_dir().join(format!("umber-graph-panel-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        umber_core::project::save_to_dir(&model, &dir).expect("save works");
        let back = umber_core::project::load_from_dir(&dir).expect("load works");
        std::fs::remove_dir_all(&dir).ok();
        back.graphs_mtlx.into_iter().next().expect("one graph")
    }

    #[test]
    fn canvas_positions_survive_project_save_load_exactly() {
        let mut saved = GraphPanel::new();
        let (head, tail) = noise_chain(&mut saved);
        // Awkward floats: negative, fractional, large, tiny.
        assert!(saved.set_node_canvas(head, [-123.456, 0.1]));
        assert!(saved.set_node_canvas(tail, [98765.43, -0.000_123]));
        let placed = saved.add_node_at("mix", [3.0e-7, 1.0e6 + 0.5]);
        let unplaced = saved.add_node("blur");
        assert!(saved.connect(tail, placed, "fg"));

        let doc = project_round_trip(saved.to_mtlx(), "canvas-roundtrip");
        assert!(doc.contains(CANVAS_POS_PARAM), "positions ride the mtlx");

        let mut loaded = GraphPanel::new();
        assert!(loaded.load_mtlx(&doc), "status: {}", loaded.status());
        assert_eq!(
            loaded.graph(),
            saved.graph(),
            "load == saved: params, edges AND canvas positions"
        );
        let canvas_of = |p: &GraphPanel, id: u64| {
            p.graph()
                .nodes
                .iter()
                .find(|n| n.id == id)
                .and_then(|n| n.canvas)
        };
        assert_eq!(canvas_of(&loaded, head), Some([-123.456, 0.1]));
        assert_eq!(canvas_of(&loaded, tail), Some([98765.43, -0.000_123]));
        assert_eq!(canvas_of(&loaded, placed), Some([3.0e-7, 1.0e6 + 0.5]));
        assert_eq!(canvas_of(&loaded, unplaced), None, "unplaced stays None");
        // The carry param never leaks into the live params (the editor).
        assert!(loaded
            .graph()
            .nodes
            .iter()
            .all(|n| n.params.iter().all(|(name, _)| name != CANVAS_POS_PARAM)));
        // Save → load → save is byte-stable (no duplicate carry params).
        assert_eq!(loaded.to_mtlx(), saved.to_mtlx());
    }

    #[test]
    fn pre_canvas_files_load_unplaced_onto_the_scatter() {
        // An old-format document: no __canvas_pos anywhere.
        let mut old = GraphPanel::new();
        let (head, tail) = noise_chain(&mut old);
        let doc = old.to_mtlx();
        assert!(!doc.contains(CANVAS_POS_PARAM), "unplaced nodes emit as v1");

        let mut loaded = GraphPanel::new();
        assert!(loaded.load_mtlx(&project_round_trip(doc, "canvas-old")));
        for id in [head, tail] {
            let node = loaded
                .graph()
                .nodes
                .iter()
                .find(|n| n.id == id)
                .expect("node survives");
            assert_eq!(node.canvas, None);
            assert_eq!(
                crate::graph_canvas::node_rect(id, node.canvas).min,
                crate::graph_canvas::default_position(id),
                "old files land on the deterministic scatter"
            );
        }
    }

    #[test]
    fn non_vec2_or_nan_carry_is_dropped_not_placed() {
        let doc = r#"<?xml version="1.0" encoding="UTF-8"?>
<materialx version="1.38">
  <node name="N1" type="uniform">
    <input name="__canvas_pos" type="string" value="garbage" />
  </node>
  <node name="N2" type="uniform">
    <input name="__canvas_pos" type="vector2" value="NaN, 4" />
  </node>
</materialx>
"#;
        let mut panel = GraphPanel::new();
        assert!(panel.load_mtlx(doc), "status: {}", panel.status());
        for node in &panel.graph().nodes {
            assert_eq!(node.canvas, None, "node {} must stay unplaced", node.id);
            assert!(node.params.is_empty(), "carry param stripped: {node:?}");
        }
    }

    #[test]
    fn moving_a_node_dirties_nothing_and_connect_rewires_the_input() {
        let mut panel = GraphPanel::new();
        panel.set_resolution((4, 4));
        let a = panel.add_node("uniform");
        let b = panel.add_node("uniform");
        let p = panel.add_node("passthrough");
        assert!(panel.connect(a, p, "in"));
        panel.evaluate();
        assert!(panel.dirty_set().is_empty(), "status: {}", panel.status());

        assert!(panel.set_node_canvas(a, [10.0, 20.0]));
        assert!(panel.dirty_set().is_empty(), "position is editor-only");
        assert!(!panel.set_node_canvas(999, [0.0, 0.0]));

        assert!(panel.connect(b, p, "in"), "re-wiring an input");
        assert!(
            panel.dirty_set().contains(&p),
            "same dirty path as add_edge"
        );
        let feeding: Vec<u64> = panel
            .graph()
            .edges
            .iter()
            .filter(|e| e.to == p && e.input == "in")
            .map(|e| e.from)
            .collect();
        assert_eq!(feeding, [b], "one edge per input: the new one replaces");
        assert!(!panel.connect(p, p, "in"), "no self-loops");
        assert!(!panel.connect(a, 999, "in"));

        panel.evaluate();
        assert!(panel.dirty_set().is_empty(), "status: {}", panel.status());
        assert!(panel.remove_edge(p, "in"));
        assert!(panel.dirty_set().contains(&p));
        assert!(!panel.remove_edge(p, "in"), "already gone");
    }

    #[test]
    fn registry_listing_covers_all_merged_families() {
        // 3 engine seeds + 7 slice-2a + 8 slice-2b + 6 slice-2c = 24.
        let panel = GraphPanel::new();
        let defs = panel.registry().node_defs();
        assert_eq!(defs.len(), 24, "unexpected registry size: {defs:?}");
        for expected in [
            "uniform",
            "image_asset",
            "passthrough",
            "noise_perlin",
            "noise_value",
            "noise_worley",
            "gradient",
            "checkerboard",
            "dots",
            "brick_pattern",
            "blur",
            "sharpen",
            "levels",
            "curves",
            "invert",
            "mix",
            "color_correct",
            "hsv_adjust",
            "flood_fill",
            "edge_detect",
            "histogram_match",
            "direction_warp",
            "triplanar_blend",
            "mesh_map_generator",
        ] {
            assert!(
                defs.contains(&expected),
                "registry must list {expected:?}, got {defs:?}"
            );
        }
        let mut sorted = defs.clone();
        sorted.sort_unstable();
        assert_eq!(defs, sorted, "listing must be sorted");
    }

    #[test]
    fn structural_edits_keep_output_marker_honest() {
        let mut panel = GraphPanel::new();
        let a = panel.add_node("uniform");
        let b = panel.add_node("passthrough");
        assert_eq!(panel.output_node(), Some(a), "first node adopts output");
        assert!(panel.set_output_node(b));
        assert!(panel.remove_node(b));
        assert_eq!(
            panel.output_node(),
            Some(a),
            "removing the output node falls back to the highest id"
        );
        assert!(!panel.remove_node(999), "unknown remove returns false");
        assert!(!panel.set_output_node(999), "unknown output returns false");
        assert!(!panel.set_param(999, "x", ParamValue::Float(1.0)));
        assert!(!panel.add_edge(a, 999, "in"));
    }

    /// A panel with the repo's example plugins loaded (blur5, infinite,
    /// vignette — git-tracked, so no skip path).
    fn plugin_panel() -> GraphPanel {
        let mut panel = GraphPanel::new();
        let report = panel.load_plugins(&[crate::plugins::dev_plugin_dir()]);
        assert!(report.failed.is_empty(), "examples load: {report:?}");
        panel
    }

    #[test]
    fn panel_registry_lists_and_dispatches_a_wasm_node() {
        let mut panel = plugin_panel();
        let defs = panel.registry().node_defs();
        assert!(defs.contains(&"blur5"), "Add-Node lists blur5: {defs:?}");
        assert!(defs.contains(&"blur"), "built-ins beside it: {defs:?}");
        assert_eq!(panel.plugin_status(), Some("3 plugin(s) loaded, 0 failed"));

        // Small odd raster (catches w/h swaps; keeps the guest cheap).
        panel.set_resolution((33, 17));
        let grad = panel.add_node("gradient");
        panel.set_param(grad, "type", ParamValue::Int(1));
        let wasm = panel.add_node("blur5");
        let native = panel.add_node("blur");
        panel.set_param(native, "radius", ParamValue::Int(2));
        assert!(panel.connect(grad, wasm, "in"));
        assert!(panel.connect(grad, native, "in"));

        assert!(panel.set_output_node(wasm));
        panel.evaluate();
        assert!(
            !panel.status().starts_with("Eval failed"),
            "status: {}",
            panel.status()
        );
        let wasm_out = panel.last_output().cloned().expect("blur5 output");
        assert!(panel.set_output_node(native));
        panel.evaluate();
        let native_out = panel.last_output().cloned().expect("blur output");
        assert!(matches!(wasm_out, NodeOutput::Image(_)));
        assert_eq!(wasm_out, native_out, "panel-dispatched blur5 == blur r2");
    }

    #[test]
    fn wasm_node_saves_as_declared_nodedef_and_round_trips() {
        let mut saved = plugin_panel();
        let grad = saved.add_node("gradient");
        let vig = saved.add_node("vignette");
        saved.set_param(vig, "strength", ParamValue::Float(0.3));
        assert!(saved.connect(grad, vig, "in"));
        let doc = saved.to_mtlx();
        assert!(
            doc.contains("<nodedef name=\"vignette\" node=\"vignette\">"),
            "plugin nodedef declared: {doc}"
        );
        assert!(
            doc.find("<nodedef name=\"vignette\"").unwrap() < doc.find("<node ").unwrap(),
            "declarations precede nodes"
        );

        let (graph, decls, warnings) = umber_graph::mtlx::from_mtlx(&doc).expect("parses");
        assert!(
            warnings.is_empty(),
            "declared type must not warn: {warnings:?}"
        );
        assert!(decls.iter().any(|d| d.name == "vignette"));
        let node = graph.nodes.iter().find(|n| n.id == vig).expect("node kept");
        assert_eq!(node.node_def, "vignette");
        assert!(node
            .params
            .contains(&("strength".to_string(), ParamValue::Float(0.3))));
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from == grad && e.to == vig && e.input == "in"));

        let mut loaded = plugin_panel();
        assert!(loaded.load_mtlx(&doc), "status: {}", loaded.status());
        assert_eq!(loaded.status(), "Graph loaded — press Evaluate.");
        assert_eq!(loaded.to_mtlx(), doc, "save/load/save is byte-stable");
    }
}

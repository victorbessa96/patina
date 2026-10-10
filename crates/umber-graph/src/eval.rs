//! Typed graph evaluation: [`NodeImpl`], [`NodeRegistry`], [`eval_graph`].
//!
//! Wave-5 slice 1 (docs/specs/node-graph-design.md §"The eval engine").
//! Pipeline: [`Graph::validate`] first (dangling/duplicate/cycle
//! checks from [`crate::topo`]), then a deterministic Kahn sweep, then
//! per node: resolve inputs (edges + [`ParamValue::NodeRef`] params
//! read upstream outputs), dispatch through the registry, store.
//!
//! This module does NOT reuse [`Graph::evaluate`]'s generic closure
//! API: that API boxes errors (`Box<dyn Error>`), which would force an
//! `EvalError` round-trip through downcasting. The Kahn sweep here is
//! small, sorted-deterministic, and keeps typed errors intact; the
//! old API is untouched so its callers keep compiling.
//!
//! [`Graph::validate`]: crate::Graph::validate
//! [`Graph::evaluate`]: crate::Graph::evaluate
//! [`ParamValue::NodeRef`]: crate::ParamValue::NodeRef

use crate::topo::{Graph, GraphError};
use crate::value::{ImageBuffer, ImageError, NodeOutput};
use crate::{Node, ParamValue};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// Everything that can go wrong during typed evaluation.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// No impl registered for this `node_def`.
    #[error("unknown node_def {node_def:?}")]
    UnknownNode {
        /// The unregistered definition name.
        node_def: String,
    },
    /// An input (edge or `NodeRef` param) names an upstream output
    /// that isn't available.
    #[error("node {node} is missing input {input:?}")]
    MissingInput {
        /// The consumer's node id.
        node: u64,
        /// The input name (edge `input` or param name).
        input: String,
    },
    /// An input's runtime kind isn't what the node consumes
    /// (the Uniform/Image confusion, made loud).
    #[error("node {node} type mismatch: expected {expected}, got {got}")]
    TypeMismatch {
        /// The consumer's node id.
        node: u64,
        /// What the node wanted (`"Image"`, …).
        expected: String,
        /// What it received ([`NodeOutput::kind`]).
        got: String,
    },
    /// A param is absent, mistyped, or out of range.
    #[error("node {node} has a bad param {param:?}")]
    BadParam {
        /// The node's id.
        node: u64,
        /// The offending param name ( MAY carry detail after a space).
        param: String,
    },
    /// Topology rejection (dangling refs, duplicates, cycles) —
    /// validation runs before any node executes.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Image-buffer / PNG failures.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// Filesystem failures (the `image_asset` load).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl EvalError {
    /// Stamps the dispatching node's id onto node-scoped variants.
    ///
    /// [`NodeImpl::eval`] doesn't receive its node id (the trait is a
    /// pure value function, per the design), so impls construct
    /// [`EvalError::MissingInput`]/[`TypeMismatch`]/[`BadParam`] with
    /// `node: 0` and [`eval_graph`] stamps the real id on dispatch.
    /// Variants without a node field pass through untouched.
    ///
    /// [`TypeMismatch`]: EvalError::TypeMismatch
    /// [`BadParam`]: EvalError::BadParam
    #[must_use]
    pub fn with_node(self, id: u64) -> Self {
        match self {
            Self::MissingInput { input, .. } => Self::MissingInput { node: id, input },
            Self::TypeMismatch { expected, got, .. } => Self::TypeMismatch {
                node: id,
                expected,
                got,
            },
            Self::BadParam { param, .. } => Self::BadParam { node: id, param },
            other => other,
        }
    }
}

/// Evaluation context: the shared raster resolution plus the graph's
/// external inputs (texture-set channels, mesh maps — bound by name by
/// generator nodes; slice-2's `mesh_map` consumes these).
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// The graph-wide raster resolution (v1: one shared resolution —
    /// the target's; per-node negotiation is wave-6).
    pub resolution: (u32, u32),
    /// External inputs by name, available to name-binding nodes.
    pub external: HashMap<String, NodeOutput>,
}

impl EvalContext {
    /// A context for `width`×`height` rasters with no externals.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resolution: (width, height),
            external: HashMap::new(),
        }
    }
}

/// A node's pure evaluation function: named inputs + raw params +
/// context in, one typed output out. No globals, no interior cache —
/// dirty-tracking lives in [`eval_graph_cached`], outside the impls.
pub trait NodeImpl: Send + Sync {
    /// Evaluates the node.
    ///
    /// # Errors
    ///
    /// [`EvalError::MissingInput`] when an expected input is absent;
    /// [`EvalError::TypeMismatch`] on kind confusion;
    /// [`EvalError::BadParam`] on absent/mistyped params (construct
    /// with `node: 0` — the dispatcher stamps the id; see
    /// [`EvalError::with_node`]).
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError>;
}

/// Dispatch table: `node_def` name → shared impl.
#[derive(Default)]
pub struct NodeRegistry {
    impls: HashMap<String, Arc<dyn NodeImpl>>,
}

impl NodeRegistry {
    /// An empty registry (lookups fail until [`Self::register`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The v1 seed: generator nodes ONLY (this slice) —
    /// `uniform`, `image_asset`, `passthrough`.
    #[must_use]
    pub fn seeded() -> Self {
        let mut r = Self::new();
        r.register("uniform", Arc::new(UniformNode));
        r.register("image_asset", Arc::new(ImageAssetNode));
        r.register("passthrough", Arc::new(PassthroughNode));
        r
    }

    /// Registers (or replaces) the impl for `node_def`.
    pub fn register(&mut self, node_def: &str, imp: Arc<dyn NodeImpl>) {
        self.impls.insert(node_def.to_string(), imp);
    }

    /// The impl for `node_def`, if registered.
    #[must_use]
    pub fn get(&self, node_def: &str) -> Option<Arc<dyn NodeImpl>> {
        self.impls.get(node_def).cloned()
    }

    /// The registered `node_def` names, sorted (the app bridge's
    /// Add-Node combo reads this — additive slice-5 API; the registry
    /// itself stays the dispatch table, this is just its listing).
    #[must_use]
    pub fn node_defs(&self) -> Vec<&str> {
        let mut defs: Vec<&str> = self.impls.keys().map(String::as_str).collect();
        defs.sort_unstable();
        defs
    }
}

fn param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

fn f32_to_u8(v: f32) -> u8 {
    // NaN normalizes to 0 through clamp's max-then-min chain.
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// `uniform`: `value` param → [`NodeOutput::Uniform`]; `color` param →
/// solid [`NodeOutput::Image`] fill at the `resolution` param
/// (`Vec2([w, h])`) or, when absent, the context resolution.
struct UniformNode;

impl NodeImpl for UniformNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        if let Some(color) = param(params, "color") {
            let rgb = match color {
                ParamValue::Color(c) => *c,
                _ => {
                    return Err(EvalError::BadParam {
                        node: 0,
                        param: "color".into(),
                    });
                }
            };
            let (w, h) = match param(params, "resolution") {
                None => ctx.resolution,
                Some(ParamValue::Vec2([w, h])) => {
                    if !w.is_finite() || !h.is_finite() {
                        return Err(EvalError::BadParam {
                            node: 0,
                            param: "resolution".into(),
                        });
                    }
                    let (w, h) = (w.round(), h.round());
                    if w < 1.0 || h < 1.0 || w > 8192.0 || h > 8192.0 {
                        return Err(EvalError::BadParam {
                            node: 0,
                            param: "resolution".into(),
                        });
                    }
                    (w as u32, h as u32)
                }
                Some(_) => {
                    return Err(EvalError::BadParam {
                        node: 0,
                        param: "resolution".into(),
                    });
                }
            };
            let px = [f32_to_u8(rgb[0]), f32_to_u8(rgb[1]), f32_to_u8(rgb[2]), 255];
            return Ok(NodeOutput::Image(ImageBuffer::filled(w, h, px)?));
        }
        match param(params, "value") {
            None => Err(EvalError::BadParam {
                node: 0,
                param: "value".into(),
            }),
            Some(ParamValue::NodeRef(_)) => Err(EvalError::BadParam {
                node: 0,
                param: "value (NodeRef must be wired via edges, not passed as a param)".into(),
            }),
            Some(v) => Ok(NodeOutput::Uniform(v.clone())),
        }
    }
}

/// `image_asset`: `path: Asset(filesystem-path)` → PNG file loaded via
/// `std::fs` and decoded to [`NodeOutput::Image`].
///
/// The string is a plain filesystem path to PNG bytes (typically
/// resolved from the content-hash store's sharded
/// `<root>/assets/xx/<hash>.png` layout by the caller — `umber_core`
/// owns that layout, and umber-graph stays dependency-minimal by NOT
/// depending on it; see the choice log in the module docs of `value`).
struct ImageAssetNode;

impl NodeImpl for ImageAssetNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let path = match param(params, "path") {
            Some(ParamValue::Asset(p)) => p,
            _ => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "path".into(),
                });
            }
        };
        let bytes = std::fs::read(path)?;
        Ok(NodeOutput::Image(ImageBuffer::from_png_bytes(&bytes)?))
    }
}

/// `passthrough`: exactly one input, echoed unchanged (the test
/// workhorse — chains prove value flow without transformation).
struct PassthroughNode;

impl NodeImpl for PassthroughNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        _params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        match inputs.len() {
            1 => Ok(inputs
                .into_iter()
                .next()
                .map(|(_, v)| v)
                .expect("len checked")),
            0 => Err(EvalError::MissingInput {
                node: 0,
                input: "in".into(),
            }),
            n => Err(EvalError::BadParam {
                node: 0,
                param: format!("expected exactly 1 input, got {n}"),
            }),
        }
    }
}

/// Deterministic Kahn order (ascending ids first — same convention as
/// [`Graph::evaluate`]). Precondition: the graph validated.
fn topo_order(graph: &Graph) -> Result<Vec<u64>, GraphError> {
    let mut indegree: HashMap<u64, usize> = graph.nodes.iter().map(|n| (n.id, 0)).collect();
    let mut outs: HashMap<u64, Vec<u64>> = HashMap::new();
    for e in &graph.edges {
        *indegree.get_mut(&e.to).expect("validated") += 1;
        outs.entry(e.from).or_default().push(e.to);
    }
    let mut queue: Vec<u64> = indegree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();
    queue.sort_unstable();
    let mut order = Vec::with_capacity(graph.nodes.len());
    while let Some(&id) = queue.first() {
        queue.remove(0);
        order.push(id);
        if let Some(next) = outs.get(&id) {
            for &to in next {
                let d = indegree.get_mut(&to).expect("validated");
                *d -= 1;
                if *d == 0 {
                    queue.push(to);
                    queue.sort_unstable();
                }
            }
        }
    }
    if order.len() != graph.nodes.len() {
        let cyclic: Vec<u64> = indegree
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(&id, _)| id)
            .collect();
        return Err(GraphError::Cycle(cyclic));
    }
    Ok(order)
}

/// Resolves a node's inputs, sorted by name for determinism:
/// `NodeRef` params first (in param order), then edges (in
/// [`Graph::edges`] order, overlaying same-named entries — explicit
/// wiring wins).
fn resolve_inputs(
    graph: &Graph,
    node: &Node,
    outputs: &HashMap<u64, NodeOutput>,
) -> Result<Vec<(String, NodeOutput)>, EvalError> {
    resolve_inputs_cached(graph, node, outputs, &HashMap::new())
}

/// [`resolve_inputs`] with a cache fallback for clean upstreams
/// (see [`eval_graph_cached`]): fresh outputs win, cached fill gaps.
fn resolve_inputs_cached(
    graph: &Graph,
    node: &Node,
    fresh: &HashMap<u64, NodeOutput>,
    cached: &HashMap<u64, NodeOutput>,
) -> Result<Vec<(String, NodeOutput)>, EvalError> {
    let lookup = |id: &u64| fresh.get(id).or_else(|| cached.get(id));
    let mut map: BTreeMap<String, NodeOutput> = BTreeMap::new();
    for (name, pv) in &node.params {
        if let ParamValue::NodeRef(up) = pv {
            let v = lookup(up).ok_or_else(|| EvalError::MissingInput {
                node: node.id,
                input: name.clone(),
            })?;
            map.insert(name.clone(), v.clone());
        }
    }
    for e in &graph.edges {
        if e.to == node.id {
            let v = lookup(&e.from).ok_or_else(|| EvalError::MissingInput {
                node: node.id,
                input: e.input.clone(),
            })?;
            map.insert(e.input.clone(), v.clone());
        }
    }
    Ok(map.into_iter().collect())
}

fn node_by_id(graph: &Graph, id: u64) -> &Node {
    graph
        .nodes
        .iter()
        .find(|n| n.id == id)
        .expect("topo order only contains graph nodes")
}

/// Evaluates the whole graph: validate → topo order → resolve →
/// registry dispatch. Returns node id → output.
///
/// The `inputs` map seeds the effective context's externals
/// (overriding [`EvalContext::external` entries of the same name) for
/// name-binding generator nodes; edge/`NodeRef` wiring carries the
/// per-node data flow.
pub fn eval_graph(
    graph: &Graph,
    registry: &NodeRegistry,
    inputs: HashMap<String, NodeOutput>,
    ctx: &EvalContext,
) -> Result<HashMap<u64, NodeOutput>, EvalError> {
    graph.validate()?;
    let order = topo_order(graph)?;
    let mut external = ctx.external.clone();
    external.extend(inputs);
    let eff_ctx = EvalContext {
        resolution: ctx.resolution,
        external,
    };
    let mut outputs: HashMap<u64, NodeOutput> = HashMap::with_capacity(graph.nodes.len());
    for id in order {
        let node = node_by_id(graph, id);
        let in_vec = resolve_inputs(graph, node, &outputs)?;
        let imp = registry
            .get(&node.node_def)
            .ok_or_else(|| EvalError::UnknownNode {
                node_def: node.node_def.clone(),
            })?;
        let out = imp
            .eval(in_vec, &node.params, &eff_ctx)
            .map_err(|e| e.with_node(node.id))?;
        outputs.insert(id, out);
    }
    Ok(outputs)
}

/// Whole-graph output cache for dirty re-evaluation (the design's
/// slice-4 surface, implemented now): node id → last output.
#[derive(Debug, Clone, Default)]
pub struct EvalCache {
    outputs: HashMap<u64, NodeOutput>,
}

impl EvalCache {
    /// An empty cache (the first [`eval_graph_cached`] call with any
    /// `dirty` set evaluates everything, like [`eval_graph`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The cached output for `id`, if evaluated.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&NodeOutput> {
        self.outputs.get(&id)
    }
}

/// Dirty re-evaluation: re-runs ONLY `dirty` nodes plus everything
/// downstream of them (one topo sweep: `dirty` seeds the run set,
/// edges propagate it forward), reusing cached outputs for clean
/// nodes. Results merge into `cache`; the returned map is the FULL
/// output set (fresh + clean-cached), equal to a fresh [`eval_graph`].
///
/// Cache misses count as dirty (a node with no cached output always
/// runs, and its downstream re-runs — so a cold cache with an empty
/// `dirty` set still evaluates the reachable graph, and partial
/// caches can't serve stale data past a re-run).
pub fn eval_graph_cached(
    graph: &Graph,
    registry: &NodeRegistry,
    inputs: HashMap<String, NodeOutput>,
    ctx: &EvalContext,
    cache: &mut EvalCache,
    dirty: &HashSet<u64>,
) -> Result<HashMap<u64, NodeOutput>, EvalError> {
    graph.validate()?;
    let order = topo_order(graph)?;
    let mut downstream: HashMap<u64, Vec<u64>> = HashMap::new();
    for e in &graph.edges {
        downstream.entry(e.from).or_default().push(e.to);
    }
    // Seed: dirty ids + their transitive downstream (one sweep).
    let mut run: HashSet<u64> = dirty.clone();
    let mut stack: Vec<u64> = dirty.iter().copied().collect();
    while let Some(id) = stack.pop() {
        if let Some(next) = downstream.get(&id) {
            for &to in next {
                if run.insert(to) {
                    stack.push(to);
                }
            }
        }
    }
    let mut external = ctx.external.clone();
    external.extend(inputs);
    let eff_ctx = EvalContext {
        resolution: ctx.resolution,
        external,
    };
    let mut outputs: HashMap<u64, NodeOutput> = HashMap::with_capacity(graph.nodes.len());
    for id in order {
        let node = node_by_id(graph, id);
        if !run.contains(&id) {
            if let Some(v) = cache.outputs.get(&id) {
                outputs.insert(id, v.clone());
                continue;
            }
            // Cache miss on a "clean" node: fall through and run it;
            // propagation below re-runs its downstream, so no stale
            // consumer survives a re-run.
        }
        let in_vec = resolve_inputs_cached(graph, node, &outputs, &cache.outputs)?;
        let imp = registry
            .get(&node.node_def)
            .ok_or_else(|| EvalError::UnknownNode {
                node_def: node.node_def.clone(),
            })?;
        let out = imp
            .eval(in_vec, &node.params, &eff_ctx)
            .map_err(|e| e.with_node(node.id))?;
        outputs.insert(id, out.clone());
        cache.outputs.insert(id, out);
        if let Some(next) = downstream.get(&id) {
            for &to in next {
                run.insert(to);
            }
        }
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::Edge;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn red_uniform(id: u64) -> Node {
        Node {
            id,
            node_def: "uniform".into(),
            params: vec![("value".into(), ParamValue::Color([1.0, 0.0, 0.0]))],
            canvas: None,
        }
    }

    fn passthrough(id: u64) -> Node {
        Node {
            id,
            node_def: "passthrough".into(),
            params: vec![],
            canvas: None,
        }
    }

    fn edge(from: u64, to: u64) -> Edge {
        Edge {
            from,
            to,
            input: "in".into(),
        }
    }

    #[test]
    fn uniform_chain_echoes_exactly() {
        // uniform(red) -> passthrough x2: full PartialEq on the tail.
        let mut g = Graph::new();
        g.add_node(red_uniform(1));
        g.add_node(passthrough(2));
        g.add_node(passthrough(3));
        g.add_edge(edge(1, 2));
        g.add_edge(edge(2, 3));
        let out = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect("chain evaluates");
        assert_eq!(
            out[&3],
            NodeOutput::Uniform(ParamValue::Color([1.0, 0.0, 0.0]))
        );
    }

    #[test]
    fn uniform_as_image_fills_every_texel() {
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "uniform".into(),
            params: vec![("color".into(), ParamValue::Color([1.0, 0.0, 0.0]))],
            canvas: None,
        });
        let out = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(4, 4),
        )
        .expect("fill evaluates");
        match &out[&1] {
            NodeOutput::Image(buf) => {
                assert_eq!((buf.width, buf.height), (4, 4));
                assert_eq!(buf.data.len(), 64);
                assert!(buf.data.chunks_exact(4).all(|px| px == [255, 0, 0, 255]));
            }
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    #[test]
    fn resolution_param_overrides_context() {
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "uniform".into(),
            params: vec![
                ("color".into(), ParamValue::Color([0.0, 1.0, 0.0])),
                ("resolution".into(), ParamValue::Vec2([2.0, 3.0])),
            ],
            canvas: None,
        });
        let out = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(9, 9),
        )
        .expect("fill evaluates");
        match &out[&1] {
            NodeOutput::Image(buf) => assert_eq!((buf.width, buf.height), (2, 3)),
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    /// A node that consumes ONLY images — the TypeMismatch workhorse.
    struct ExpectImage;
    impl NodeImpl for ExpectImage {
        fn eval(
            &self,
            inputs: Vec<(String, NodeOutput)>,
            _params: &[(String, ParamValue)],
            _ctx: &EvalContext,
        ) -> Result<NodeOutput, EvalError> {
            match inputs.first() {
                Some((_, NodeOutput::Image(_))) => Ok(inputs
                    .into_iter()
                    .next()
                    .map(|(_, v)| v)
                    .expect("first checked")),
                Some((_, other)) => Err(EvalError::TypeMismatch {
                    node: 0,
                    expected: "Image".into(),
                    got: other.kind().into(),
                }),
                None => Err(EvalError::MissingInput {
                    node: 0,
                    input: "in".into(),
                }),
            }
        }
    }

    #[test]
    fn image_consumer_rejects_uniform_with_variant() {
        let mut g = Graph::new();
        g.add_node(red_uniform(1)); // Uniform output…
        g.add_node(Node {
            id: 2,
            node_def: "expect_image".into(),
            params: vec![],
            canvas: None,
        });
        g.add_edge(edge(1, 2)); // …fed into an Image consumer.
        let mut registry = NodeRegistry::seeded();
        registry.register("expect_image", Arc::new(ExpectImage));
        let err = eval_graph(&g, &registry, HashMap::new(), &EvalContext::new(1, 1))
            .expect_err("kind confusion must fail");
        match err {
            EvalError::TypeMismatch {
                node,
                expected,
                got,
            } => {
                assert_eq!(node, 2, "dispatcher stamps the consumer id");
                assert_eq!(expected, "Image");
                assert_eq!(got, "Uniform");
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unregistered_node_def_is_unknown() {
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "nope_missing".into(),
            params: vec![],
            canvas: None,
        });
        let err = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("unregistered def must fail");
        match err {
            EvalError::UnknownNode { node_def } => assert_eq!(node_def, "nope_missing"),
            other => panic!("expected UnknownNode, got {other:?}"),
        }
    }

    #[test]
    fn cycle_and_dangling_propagate_from_validate() {
        let mut cyclic = Graph::new();
        cyclic.add_node(passthrough(1));
        cyclic.add_node(passthrough(2));
        cyclic.add_edge(edge(1, 2));
        cyclic.add_edge(Edge {
            from: 2,
            to: 1,
            input: "in".into(),
        });
        let err = eval_graph(
            &cyclic,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("cycle must error, not hang");
        assert!(
            matches!(err, EvalError::Graph(GraphError::Cycle(_))),
            "got {err:?}"
        );

        let mut dangling = Graph::new();
        dangling.add_node(passthrough(1));
        dangling.add_edge(Edge {
            from: 1,
            to: 99,
            input: "in".into(),
        });
        let err = eval_graph(
            &dangling,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("dangling ref must error");
        assert!(
            matches!(err, EvalError::Graph(GraphError::DanglingNodeRef(99))),
            "got {err:?}"
        );
    }

    #[test]
    fn passthrough_enforces_arity() {
        // No inputs at all: MissingInput, not a silent default.
        let mut g = Graph::new();
        g.add_node(passthrough(1));
        let err = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("unwired passthrough must fail");
        assert!(
            matches!(err, EvalError::MissingInput { node: 1, .. }),
            "got {err:?}"
        );

        // Two inputs: BadParam (echo is only defined for one).
        let mut g = Graph::new();
        g.add_node(red_uniform(1));
        g.add_node(red_uniform(2));
        g.add_node(passthrough(3));
        g.add_edge(Edge {
            from: 1,
            to: 3,
            input: "a".into(),
        });
        g.add_edge(Edge {
            from: 2,
            to: 3,
            input: "b".into(),
        });
        let err = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("two inputs must fail");
        assert!(
            matches!(err, EvalError::BadParam { node: 3, .. }),
            "got {err:?}"
        );
    }

    /// Run-counting impl: the slice-4 falsifiability probe. Sources
    /// (no inputs) emit their tag; consumers echo.
    struct CounterNode {
        count: Arc<AtomicUsize>,
        tag: i32,
    }
    impl NodeImpl for CounterNode {
        fn eval(
            &self,
            inputs: Vec<(String, NodeOutput)>,
            _params: &[(String, ParamValue)],
            _ctx: &EvalContext,
        ) -> Result<NodeOutput, EvalError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            if inputs.is_empty() {
                Ok(NodeOutput::Uniform(ParamValue::Int(self.tag)))
            } else {
                Ok(inputs.into_iter().next().map(|(_, v)| v).expect("nonempty"))
            }
        }
    }

    fn counter_chain() -> (Graph, NodeRegistry, [Arc<AtomicUsize>; 3]) {
        let counts = [
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        ];
        let mut registry = NodeRegistry::new();
        for (i, def) in ["counter_a", "counter_b", "counter_c"].iter().enumerate() {
            registry.register(
                def,
                Arc::new(CounterNode {
                    count: counts[i].clone(),
                    tag: (i + 1) as i32,
                }),
            );
        }
        let mut g = Graph::new();
        for (i, def) in ["counter_a", "counter_b", "counter_c"].iter().enumerate() {
            g.add_node(Node {
                id: (i + 1) as u64,
                node_def: def.to_string(),
                params: vec![],
                canvas: None,
            });
        }
        g.add_edge(edge(1, 2));
        g.add_edge(edge(2, 3));
        (g, registry, counts)
    }

    fn counts(counts: &[Arc<AtomicUsize>; 3]) -> [usize; 3] {
        counts
            .iter()
            .map(|c| c.load(Ordering::SeqCst))
            .collect::<Vec<_>>()
            .try_into()
            .expect("three counters")
    }

    #[test]
    fn cached_eval_reruns_only_dirty_and_downstream() {
        let (g, registry, counters) = counter_chain();
        let ctx = EvalContext::new(1, 1);
        let mut cache = EvalCache::new();

        // Phase 1: everything dirty on a cold cache — all three run.
        let out1 = eval_graph_cached(
            &g,
            &registry,
            HashMap::new(),
            &ctx,
            &mut cache,
            &[1, 2, 3].into_iter().collect::<HashSet<_>>(),
        )
        .expect("phase 1 evaluates");
        assert_eq!(counts(&counters), [1, 1, 1]);
        assert_eq!(out1[&3], NodeOutput::Uniform(ParamValue::Int(1)));

        // Phase 2: only C dirty — A and B MUST NOT re-run.
        let out2 = eval_graph_cached(
            &g,
            &registry,
            HashMap::new(),
            &ctx,
            &mut cache,
            &[3].into_iter().collect::<HashSet<_>>(),
        )
        .expect("phase 2 evaluates");
        assert_eq!(counts(&counters), [1, 1, 2]);
        assert_eq!(out2, out1, "same values, fewer runs");

        // Phase 3: B dirty — B and downstream C re-run, A stays.
        let out3 = eval_graph_cached(
            &g,
            &registry,
            HashMap::new(),
            &ctx,
            &mut cache,
            &[2].into_iter().collect::<HashSet<_>>(),
        )
        .expect("phase 3 evaluates");
        assert_eq!(counts(&counters), [1, 2, 3]);
        assert_eq!(out3, out1);

        // Cache correctness: a fresh full eval agrees exactly.
        let fresh = eval_graph(&g, &registry, HashMap::new(), &ctx).expect("fresh evaluates");
        assert_eq!(out3, fresh, "no stale data survives cached re-evals");
    }

    #[test]
    fn image_asset_loads_a_png_from_disk() {
        use crate::value::encode_png_rgba8;
        let data: Vec<u8> = vec![
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 200,
        ];
        let bytes = encode_png_rgba8(2, 2, &data).unwrap();
        let path =
            std::env::temp_dir().join(format!("umber-graph-asset-{}.png", std::process::id()));
        std::fs::write(&path, &bytes).expect("temp png writes");
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "image_asset".into(),
            params: vec![(
                "path".into(),
                ParamValue::Asset(path.to_string_lossy().into_owned()),
            )],
            canvas: None,
        });
        let out = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(2, 2),
        );
        let _ = std::fs::remove_file(&path);
        let out = out.expect("asset loads");
        match &out[&1] {
            NodeOutput::Image(buf) => {
                assert_eq!((buf.width, buf.height), (2, 2));
                assert_eq!(buf.data, data);
            }
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    #[test]
    fn image_asset_rejects_missing_file_and_bad_param() {
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "image_asset".into(),
            params: vec![("path".into(), ParamValue::Asset("/no/such/file.png".into()))],
            canvas: None,
        });
        let err = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("missing file must fail");
        assert!(matches!(err, EvalError::Io(_)), "got {err:?}");

        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "image_asset".into(),
            params: vec![("path".into(), ParamValue::Float(1.0))],
            canvas: None,
        });
        let err = eval_graph(
            &g,
            &NodeRegistry::seeded(),
            HashMap::new(),
            &EvalContext::new(1, 1),
        )
        .expect_err("mistyped path must fail");
        assert!(
            matches!(err, EvalError::BadParam { node: 1, .. }),
            "got {err:?}"
        );
    }
}

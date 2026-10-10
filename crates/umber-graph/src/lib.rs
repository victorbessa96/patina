//! umber-graph — the original procedural node-graph engine.
//!
//! Wave 4 scope (docs/specs/requirements.md §4): DAG, topological eval,
//! dirty-region propagation; graphs serialize as .mtlx documents where
//! standard MaterialX nodes exist, with DECLARED CUSTOM NODEDEFS for
//! painter-specific nodes (flood fill, edge detect, mesh-map generators).
//! No automatic full-fidelity interchange is claimed (adversarial
//! review finding #2, docs/research/06).
//!
//! Wave 1 scope: the node vocabulary skeleton the workspace compiles on.
//! Wave 3 (this slice): the DAG core — [`topo`] with validation
//! (dangling refs, duplicate ids) and topological evaluation (Kahn's
//! algorithm, cycle rejection, inputs resolved before consumers run).
//! Wave 5 slice 1: the typed eval engine — [`value`] (Uniform/Image
//! value space, RGBA8 buffers) and [`eval`] (registry-dispatched
//! evaluation, dirty-aware cached re-evaluation).

pub mod eval;
pub mod mtlx;
pub mod nodes;
pub mod topo;
pub mod value;

pub use eval::{
    eval_graph, eval_graph_cached, EvalCache, EvalContext, EvalError, NodeImpl, NodeRegistry,
};
pub use topo::{Edge, Graph, GraphError};
pub use value::{decode_png_rgba8, encode_png_rgba8, ImageBuffer, ImageError, NodeOutput};

/// A node in a graph. The Wave-4 engine evaluates these topologically.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: u64,
    /// MaterialX standard node name, or our custom nodedef name.
    pub node_def: String,
    pub params: Vec<(String, ParamValue)>,
    /// The node's graph-space position on the editor canvas (wave-6,
    /// `docs/specs/graph-canvas-design.md`). `None` = never placed: the
    /// canvas lays it out with its deterministic id-hash scatter.
    ///
    /// Editor-only data — evaluation never reads it, so moving a node
    /// dirties nothing. The crate has no serde, so the design's
    /// `#[serde(default)]` additive rule is enforced by the persistence
    /// carry instead: the app's Graph panel emits it as a `__canvas_pos`
    /// `vector2` input in its `.mtlx` string (Vec2 values round-trip
    /// exactly — see `mtlx::tests::round_trip_preserves_data`) and lifts
    /// it back out on load; documents without one load as `None`. The
    /// mtlx layer itself neither emits nor parses this field.
    pub canvas: Option<[f32; 2]>,
}

/// Parameter values, matching the MaterialX value space we target.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Color([f32; 3]),
    Int(i32),
    Bool(bool),
    /// Reference to another node's output.
    NodeRef(u64),
    /// Reference to an image asset (content-hash path).
    Asset(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_vocabulary_compiles() {
        let n = Node {
            id: 1,
            node_def: "noise_fractal3d".into(),
            params: vec![
                ("scale".into(), ParamValue::Float(8.0)),
                ("octaves".into(), ParamValue::Int(5)),
            ],
            canvas: None,
        };
        assert_eq!(n.params.len(), 2);
    }
}

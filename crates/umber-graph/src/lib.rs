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

pub mod topo;

pub use topo::{Edge, Graph, GraphError};

/// A node in a graph. The Wave-4 engine evaluates these topologically.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: u64,
    /// MaterialX standard node name, or our custom nodedef name.
    pub node_def: String,
    pub params: Vec<(String, ParamValue)>,
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
        };
        assert_eq!(n.params.len(), 2);
    }
}

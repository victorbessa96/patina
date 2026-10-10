//! Graph topology: the DAG structure + topological evaluation.
//!
//! Wave-4 engine core. Nodes carry MaterialX-compatible params
//! (see [`crate::Node`]); edges connect one node's output to another's
//! input. Evaluation is topological with cycle rejection — a graph that
//! can't be ordered doesn't evaluate, it errors (no silent partial
//! results).

use crate::Node;
use std::collections::HashMap;
/// Errors from graph construction and evaluation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GraphError {
    /// A node input references a node id not in the graph.
    #[error("dangling node reference {0}")]
    DanglingNodeRef(u64),
    /// The graph contains a cycle (listed nodes participate).
    #[error("cycle detected among nodes {0:?}")]
    Cycle(Vec<u64>),
}

/// A directed edge: `from`'s single output feeds `to`'s input named
/// `input`. Every node has exactly one output (MaterialX convention for
/// this graph level; multi-output nodedefs are a later concern).
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    /// The input name on `to` (e.g. "mask", "base").
    pub input: String,
}

/// A node graph: nodes + edges, evaluated topologically.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node (id must be unique — duplicates are the caller's bug,
    /// checked in [`Self::validate`]).
    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// Adds an edge (`from` output → `to`'s `input`).
    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Validates: unique ids, no dangling references, no cycles.
    ///
    /// # Errors
    ///
    /// [`GraphError::DanglingNodeRef`] on unknown node ids;
    /// [`GraphError::Cycle`] on any cycle (all cycle participants
    /// listed, order unspecified).
    pub fn validate(&self) -> Result<(), GraphError> {
        let ids: std::collections::HashSet<u64> = self.nodes.iter().map(|n| n.id).collect();
        if ids.len() != self.nodes.len() {
            // Duplicate ids: report via a dangling-style probe (find the
            // first id that appears twice — deterministic).
            let mut seen = std::collections::HashSet::new();
            for n in &self.nodes {
                if !seen.insert(n.id) {
                    return Err(GraphError::DanglingNodeRef(n.id));
                }
            }
        }
        for e in &self.edges {
            if !ids.contains(&e.from) || !ids.contains(&e.to) {
                return Err(GraphError::DanglingNodeRef(if !ids.contains(&e.from) {
                    e.from
                } else {
                    e.to
                }));
            }
        }
        // Kahn's algorithm; leftover nodes at the end = cycle.
        let mut indegree: HashMap<u64, usize> = self.nodes.iter().map(|n| (n.id, 0)).collect();
        let mut outs: HashMap<u64, Vec<u64>> = HashMap::new();
        for e in &self.edges {
            *indegree.get_mut(&e.to).expect("validated above") += 1;
            outs.entry(e.from).or_default().push(e.to);
        }
        let mut queue: Vec<u64> = indegree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop() {
            order.push(id);
            if let Some(next) = outs.get(&id) {
                for &to in next {
                    let d = indegree.get_mut(&to).expect("validated above");
                    *d -= 1;
                    if *d == 0 {
                        queue.push(to);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            let cyclic: Vec<u64> = indegree
                .iter()
                .filter(|(_, &d)| d > 0)
                .map(|(&id, _)| id)
                .collect();
            return Err(GraphError::Cycle(cyclic));
        }
        Ok(())
    }

    /// Topological evaluation: visits nodes so every input is ready
    /// before its consumer runs. `eval` receives each node and a
    /// resolved map of its input values (by input name).
    ///
    /// # Errors
    ///
    /// Propagates [`Self::validate`] failures.
    pub fn evaluate<T: Clone>(
        &self,
        mut eval: impl FnMut(&Node, &HashMap<String, T>) -> Result<T, Box<dyn std::error::Error>>,
    ) -> Result<HashMap<u64, T>, Box<dyn std::error::Error>> {
        self.validate()?;

        // Kahn order (validate proved acyclic).
        let mut indegree: HashMap<u64, usize> = self.nodes.iter().map(|n| (n.id, 0)).collect();
        let mut outs: HashMap<u64, Vec<(u64, &str)>> = HashMap::new();
        for e in &self.edges {
            *indegree.get_mut(&e.to).expect("validated") += 1;
            outs.entry(e.from)
                .or_default()
                .push((e.to, e.input.as_str()));
        }
        let mut queue: Vec<u64> = indegree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        // Deterministic queue order for reproducible evaluation.
        queue.sort_unstable();

        let mut values: HashMap<u64, T> = HashMap::new();
        let mut order = Vec::new();
        while let Some(&id) = queue.first() {
            queue.remove(0);
            order.push(id);
            let node = self.nodes.iter().find(|n| n.id == id).expect("validated");
            let mut inputs: HashMap<String, T> = HashMap::new();
            for e in &self.edges {
                if e.to == id {
                    if let Some(v) = values.get(&e.from) {
                        inputs.insert(e.input.clone(), v.clone());
                    }
                }
            }
            let out = eval(node, &inputs)?;
            values.insert(id, out);
            if let Some(next) = outs.get(&id) {
                for &(to, _) in next {
                    let d = indegree.get_mut(&to).expect("validated");
                    *d -= 1;
                    if *d == 0 {
                        queue.push(to);
                        queue.sort_unstable();
                    }
                }
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParamValue;

    fn noise(id: u64) -> Node {
        Node {
            id,
            node_def: "noise_fractal3d".into(),
            params: vec![("scale".into(), ParamValue::Float(8.0))],
            canvas: None,
        }
    }

    fn blend(id: u64) -> Node {
        Node {
            id,
            node_def: "mix".into(),
            params: vec![("amount".into(), ParamValue::Float(0.5))],
            canvas: None,
        }
    }

    #[test]
    fn linear_graph_evaluates_in_order() {
        let mut g = Graph::new();
        g.add_node(noise(1));
        g.add_node(blend(2));
        g.add_edge(Edge {
            from: 1,
            to: 2,
            input: "fg".into(),
        });
        let order: std::cell::RefCell<Vec<u64>> = std::cell::RefCell::new(vec![]);
        let values = g
            .evaluate(|node, _| {
                order.borrow_mut().push(node.id);
                Ok::<_, Box<dyn std::error::Error>>(node.id as f32)
            })
            .expect("linear graph evaluates");
        assert_eq!(order.into_inner(), vec![1, 2]);
        assert_eq!(values[&2], 2.0);
    }

    #[test]
    fn diamond_evaluates_each_node_once() {
        // 1 -> {2, 3} -> 4: 2 and 3 both consume 1; 4 consumes both.
        let mut g = Graph::new();
        for id in 1..=4 {
            g.add_node(if id == 1 { noise(id) } else { blend(id) });
        }
        for (from, to, input) in [(1, 2, "fg"), (1, 3, "fg"), (2, 4, "fg"), (3, 4, "bg")] {
            g.add_edge(Edge {
                from,
                to,
                input: input.into(),
            });
        }
        let visits: std::cell::RefCell<std::collections::HashMap<u64, usize>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
        g.evaluate(|node, _| {
            *visits.borrow_mut().entry(node.id).or_insert(0) += 1;
            Ok::<_, Box<dyn std::error::Error>>(0.0)
        })
        .expect("diamond evaluates");
        let v = visits.into_inner();
        assert_eq!(v[&1], 1);
        assert_eq!(v[&4], 1);
    }

    #[test]
    fn cycle_is_rejected_with_participants() {
        let mut g = Graph::new();
        g.add_node(noise(1));
        g.add_node(blend(2));
        g.add_edge(Edge {
            from: 1,
            to: 2,
            input: "fg".into(),
        });
        g.add_edge(Edge {
            from: 2,
            to: 1,
            input: "fg".into(),
        });
        match g.validate() {
            Err(GraphError::Cycle(nodes)) => {
                assert_eq!(nodes.len(), 2);
            }
            other => panic!("cycle must be rejected, got {other:?}"),
        }
    }

    #[test]
    fn dangling_edge_is_rejected() {
        let mut g = Graph::new();
        g.add_node(noise(1));
        g.add_edge(Edge {
            from: 1,
            to: 99,
            input: "fg".into(),
        });
        assert!(matches!(g.validate(), Err(GraphError::DanglingNodeRef(99))));
    }

    #[test]
    fn inputs_are_resolved_before_consumer_runs() {
        let mut g = Graph::new();
        g.add_node(noise(1));
        g.add_node(blend(2));
        g.add_edge(Edge {
            from: 1,
            to: 2,
            input: "fg".into(),
        });
        let values = g
            .evaluate(|node, inputs| {
                if node.id == 2 {
                    let fg: &f32 = inputs.get("fg").expect("blend must see its resolved input");
                    assert!((*fg - 1.0).abs() < 1e-6, "fg carries node 1's output");
                }
                Ok::<_, Box<dyn std::error::Error>>(node.id as f32)
            })
            .expect("evaluates");
        assert_eq!(values.len(), 2);
    }
}

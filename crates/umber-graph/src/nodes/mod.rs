//! Wave-5 slice 2a generator nodes: noise + gradient + pattern.
//!
//! Seven node_defs: `noise_perlin`, `noise_value`, `noise_worley`
//! ([`noise`]), `gradient` ([`gradient`]), `checkerboard`, `dots`,
//! `brick_pattern` ([`pattern`]). Every image-producing node renders at
//! [`EvalContext::resolution`](crate::EvalContext::resolution).

pub mod gradient;
pub mod noise;
pub mod pattern;

pub use gradient::{register_gradient_nodes, GradientNode};
pub use noise::{build_perm, register_noise_nodes, PerlinNode, ValueNode, WorleyNode};
pub use pattern::{
    register_pattern_nodes, BrickNode, CheckerNode, DotsNode, BRICK_FACE, BRICK_MORTAR,
};

/// Registers all seven slice-2a generator node_defs on `registry`.
/// The [`crate::NodeRegistry::seeded`] set is untouched; callers add
/// these on a fresh (or seeded) registry.
pub fn register_generator_nodes(registry: &mut crate::NodeRegistry) {
    noise::register_noise_nodes(registry);
    gradient::register_gradient_nodes(registry);
    pattern::register_pattern_nodes(registry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{eval_graph, EvalContext, Graph, Node, NodeOutput, ParamValue};
    use std::collections::HashMap;

    #[test]
    fn engine_end_to_end_noise_through_passthrough() {
        // Graph: noise_perlin(1) -> passthrough(2). The tail must echo
        // the head, and the head must equal a direct registry eval with
        // identical params/context (proves engine wiring, not just echo).
        let params = vec![
            ("scale".into(), ParamValue::Float(4.0)),
            ("seed".into(), ParamValue::Int(5)),
        ];
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "noise_perlin".into(),
            params: params.clone(),
        });
        g.add_node(Node {
            id: 2,
            node_def: "passthrough".into(),
            params: vec![],
        });
        g.add_edge(crate::Edge {
            from: 1,
            to: 2,
            input: "in".into(),
        });
        let mut registry = crate::NodeRegistry::seeded();
        register_generator_nodes(&mut registry);
        let ctx = EvalContext::new(8, 8);
        let out = eval_graph(&g, &registry, HashMap::new(), &ctx).expect("graph evaluates");
        assert_eq!(out[&2], out[&1], "passthrough must echo the noise output");
        let direct = registry
            .get("noise_perlin")
            .expect("registered")
            .eval(vec![], &params, &ctx)
            .expect("direct eval works");
        assert_eq!(out[&1], direct, "graph eval == direct registry eval");
        assert!(
            matches!(direct, NodeOutput::Image(_)),
            "noise produces an Image, got {direct:?}"
        );
    }
}

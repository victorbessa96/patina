# Procedural Node Graph — Wave-5 Design (§4)

Wave-5's centerpiece. Written 2026-10-10 00:05 against the tree at
`4cfbb76`. umber-graph holds the DAG skeleton (topo validation, Kahn's
eval order, typed ParamValue with NodeRef/Asset variants — 364 lines,
all green). This note contracts the engine build: typed evaluation,
the v1 node set, .mtlx round-trip, dirty-region propagation, and the
GPU bridge the viewport ultimately needs.

## The eval engine (slice 1 — the core)

**Typed values flow down the DAG.** `Node::params` carries
`ParamValue`; evaluation produces per-node **outputs** in a typed
value space. Two kinds of value, kept distinct (the Substance/Mari
precedent — confusing them is the classic procedural-engine bug):

```rust
pub enum NodeOutput {
    /// Uniform value — a constant for this evaluation.
    Uniform(ParamValue),
    /// A spatially-varying image — WxH RGBA, the graph's raster
    /// resolution. v1: one shared resolution for the whole graph
    /// (the target's); per-node resolution negotiation is wave-6.
    Image(ImageBuffer),
}
```

`eval_graph(graph, inputs) -> Result<Vec<(u64, NodeOutput)>, EvalError>`:
Kahn's order (topo.rs already computes it); each node's inputs
resolve from upstream outputs (NodeRef edges) or the graph's
external inputs (the texture-set's channels, the mesh maps — the
`inputs` map); a node's `eval` fn consumes `Vec<(String, NodeOutput)>`
+ its params and produces its output. **Nodes are pure functions** —
no globals, no caches inside the engine v1 (dirty-tracking is
external, below).

**ImageBuffer**: v1 is CPU RGBA8 (the existing image plumbing's
byte format). GPU compute nodes are wave-6 (the node graph renders
for export/viewport composite in v1 — CPU raster at 512-1024 is
fast enough for procedural masks; the perf protocol's numbers will
tell us when to move).

## The node set (slice 2 — ~40 nodes in families)

Per requirements §4's row, the v1 families with their nodes:

| Family | Nodes |
|---|---|
| Noise | perlin, value, worley (each with scale/offset/seed params — SHARED impl via a trait, three tiling functions) |
| Gradient | linear, radial, angular (type param + repeat + center/angle) |
| Pattern | checker, dots, brick (the three Substance staples) |
| Filter | blur (separable box v1 — gaussian wave-6), sharpen (unsharp via blur), levels (in-low/in-high/gamma/out), curves (the 4-point ControlCurve from umber-brush — REUSE, it's merged), invert |
| Color ops | mix (two inputs + factor), color_correct (hue/sat/brightness/contrast), hsv_adjust |
| Mask ops | flood_fill (from a mask input: label connected components — 4-neighborhood, deterministic labeling), edge_detect (sobel magnitude from a mask), histogram (per-channel CDF matching to a target — v1: auto-levels), levels-as-mask |
| Spatial | direction_warp (UV offset by a vector input), triplanar_blend (three inputs + weights) |
| Generators | mesh_map (binds a baked map by name — the bridge to umber-bake's outputs), uniform (color/float constant), image_asset (loads via the content-hash store) |

Each node: `node_def` name (MaterialX-standard where one exists —
noise/gradient/checker/mix/triplanar have MTLX equivalents; flood_fill
/edge_detect/histogram/brick ship as DECLARED custom nodedefs per the
requirements' no-false-interchange rule), params, and an `eval`
implementation. ~40 nodes is three claw slices by family (noise+gen,
filter+color, mask+spatial).

## .mtlx serialization (slice 3)

`to_mtlx(graph) -> String` / `from_mtlx(&str) -> Result<Graph, _>`:
XML, MaterialX 1.38 document shape — `<nodedef>` for our custom
nodes (declared BEFORE use, the spec's requirement),
`<node name=... type=...>` instances, `<input>`/`<output>` with
value or `nodename=` references. Round-trip determinism test (the
project-format precedent): write→read→write byte-identical. Unknown
nodes on read: kept as opaque `node_def` strings with a warning
list (forward compat — no data loss, no silent eval). NO full
MaterialX fidelity claim — the custom nodedefs make the painter
nodes legible to consuming DCCs that implement them; the standard
nodes interoperate for real.

## Dirty-region propagation (slice 4)

`DirtySet`: on param change / structural edit, mark the node; the
propagator marks all downstream consumers (topo order makes this one
sweep). Re-eval re-runs ONLY dirty nodes, reusing cached outputs of
clean upstreams (the cache: `HashMap<u64, NodeOutput>` — v1's
whole-graph cache keyed by node; the memory budget scales with
graph size × image res, watched by the perf protocol's undo-RAM
row). The engine API: `eval_graph_cached(graph, inputs, cache,
dirty) -> outputs` — dirty nodes + everything downstream re-runs;
clean subtrees read the cache. Falsifiable test: a two-node chain
where only the tail's param changes — the head MUST NOT re-run
(a run-counter test node proves it; the counter asserts are the
failability).

## The app bridge (slice 5, last)

The Properties/Assets panel gains a Graph tab v1: a node list +
param editors (the egui patterns the properties panel established);
a graph's output image displays in the UV view; the export dialog
accepts a graph output as a map source (the painted-bridge's
BaseColorSource::Graphed — the same honest-source badge pattern).
The canvas editor (nodes-and-wires UI) is wave-6 — v1 is list+params,
not the noodle UI ( Substance's full graph editor is a quarter of
work on its own; the requirements' P0 is the ENGINE + interchange).

## Build order (claw-shaped)

1. Eval engine + Uniform/Image value space + the cache API (umber-
   graph, pure CPU — the everything-else unblocks here).
2. Node set by family (three slices, each self-testable: noise+gen,
   filter+color, mask+spatial) — each node's eval gets exact-value
   tests on tiny buffers (mirrored-math where procedural, golden
   where textural).
3. .mtlx round-trip.
4. Dirty propagation (the run-counter test).
5. App bridge (list + param UI + export source).

Sizes: 1 is a solid claw evening-slice; 2 is three; 3-5 one each.
The engine slice dispatches first — everything lands on its API.

//! `.mtlx` (MaterialX 1.38) document serialization for the node graph.
//!
//! Wave-5 slice 3 (`docs/specs/node-graph-design.md`, ".mtlx serialization").
//! [`to_mtlx`] emits the MaterialX document shape for a [`Graph`]; [`from_mtlx`]
//! parses back the subset we emit. This is NOT a general MaterialX parser:
//! unknown elements are skipped (tolerated, never fatal) and unknown node
//! types are preserved as opaque `node_def` strings with a warning entry
//! (forward compat — no data loss, no silent eval; the caller decides).
//!
//! # Emission rules (the determinism contract)
//!
//! * Header `<?xml version="1.0" encoding="UTF-8"?>`, root
//!   `<materialx version="1.38">`. No timestamps, no paths, nothing
//!   nondeterministic.
//! * DECLARED custom nodedefs FIRST (the spec's ordering rule / the
//!   no-false-interchange contract): every [`NodedefDecl`] emits a
//!   `<nodedef name=... node=...>` block before any `<node>`.
//! * Nodes emit as `<node name="N{id}" type="{node_def}">`, sorted by id.
//!   MaterialX standard nodes (`noise`, `checker`, `mix`, `triplanar`, …)
//!   map 1:1 by `node_def` name; custom nodes reference their declared
//!   nodedef.
//! * Params emit as `<input name type value />`. [`ParamValue`] mapping:
//!   `Float → float`, `Int → integer`, `Vec2 → vector2`, `Vec3 → vector3`,
//!   `Color → color3`, `Bool → boolean`, `Asset → string` (content-hash
//!   path). `NodeRef` is NOT a value input — it is a connection (below).
//! * Edges emit painter-idiomatically: the connected input carries
//!   `output="N{from}"`, always typed `color3` (single-output nodes emit
//!   images, so connections carry image data). Per node, connected inputs
//!   come first (sorted by name), then value params in their original
//!   order. Edges are sorted by `(to, input)` — byte-identical output for
//!   byte-identical graphs regardless of edge insertion order.
//! * A stray `NodeRef` param with no matching [`Edge`] is normalized to a
//!   connection on emit (documented; canonical graphs carry connections
//!   as edges only). A value param shadowed by an edge on the same input
//!   name is dropped in favour of the edge (duplicate input names cannot
//!   round-trip; canonical graphs never do this).
//! * 2-space indentation. Floats use [`f32::to_string`] (shortest
//!   round-trip form); vectors are `", "`-joined. Params are assumed
//!   finite (NaN never compares equal, so it cannot round-trip).
//!
//! # Parse rules (the subset we emit)
//!
//! * `<nodedef>` blocks rebuild [`NodedefDecl`]; `<node>` blocks rebuild
//!   [`Node`]s (`N{id}` → id); `output=` inputs rebuild [`Edge`]s.
//! * Unknown input value types fall back to `Asset(raw)` so no data is
//!   lost (the re-emit then types them `string` — values survive, the
//!   spelling may not be byte-identical for such foreign inputs).
//! * Node types that are neither declared by a `<nodedef>` in this
//!   document nor in the small [`KNOWN_STANDARD_NODES`] list are appended
//!   (deduped) to the warnings vec. They still parse fully — params
//!   intact.
//! * `<nodegraph>` wrappers are descended into; anything else unknown
//!   (`<look>`, `<material>`, stray `<output>`, …) is skipped.
//!
//! # Painter-specific v1 nodedefs ([`painter_nodedefs`])
//!
//! | name | inputs |
//! |---|
//! | `flood_fill` | `mask: color3` — label connected components of a mask (4-neighbourhood, deterministic) |
//! | `edge_detect` | `mask: color3`, `radius: float` — Sobel magnitude of a mask |
//! | `histogram_match` | `input: color3`, `target: color3` — per-channel CDF match (v1: auto-levels) |
//! | `brick_pattern` | `scale: vector2`, `mortar: float` — Substance-style brick |
//! | `direction_warp` | `input: color3`, `vector: vector3` — UV offset by a vector input |
//! | `mesh_map_generator` | `map_name: string` — binds a baked map by name |

use crate::{Edge, Graph, Node, ParamValue};

/// A declared custom nodedef: `name` is the `<nodedef name=…>`, `category`
/// the `<nodedef node=…>` (the MaterialX node category it instantiates),
/// `inputs` the `(name, type-string)` parameter list.
#[derive(Debug, Clone, PartialEq)]
pub struct NodedefDecl {
    pub name: String,
    pub category: String,
    pub inputs: Vec<(String, String)>,
}

/// The painter-specific v1 custom nodedefs (see module docs for semantics).
pub fn painter_nodedefs() -> Vec<NodedefDecl> {
    vec![
        NodedefDecl {
            name: "flood_fill".into(),
            category: "flood_fill".into(),
            inputs: vec![("mask".into(), "color3".into())],
        },
        NodedefDecl {
            name: "edge_detect".into(),
            category: "edge_detect".into(),
            inputs: vec![
                ("mask".into(), "color3".into()),
                ("radius".into(), "float".into()),
            ],
        },
        NodedefDecl {
            name: "histogram_match".into(),
            category: "histogram_match".into(),
            inputs: vec![
                ("input".into(), "color3".into()),
                ("target".into(), "color3".into()),
            ],
        },
        NodedefDecl {
            name: "brick_pattern".into(),
            category: "brick_pattern".into(),
            inputs: vec![
                ("scale".into(), "vector2".into()),
                ("mortar".into(), "float".into()),
            ],
        },
        NodedefDecl {
            name: "direction_warp".into(),
            category: "direction_warp".into(),
            inputs: vec![
                ("input".into(), "color3".into()),
                ("vector".into(), "vector3".into()),
            ],
        },
        NodedefDecl {
            name: "mesh_map_generator".into(),
            category: "mesh_map_generator".into(),
            inputs: vec![("map_name".into(), "string".into())],
        },
    ]
}

/// MaterialX standard node types we recognise without a local `<nodedef>`
/// (heuristic v1 list: the stdlib names the design doc's node table maps
/// to, plus the design's own family names). Anything else that is not
/// declared in-document lands in the [`from_mtlx`] warnings vec —
// preserved, never dropped.
pub const KNOWN_STANDARD_NODES: &[&str] = &[
    "noise2d",
    "noise3d",
    "fractal3d",
    "cellnoise2d",
    "cellnoise3d",
    "turbulence2d",
    "turbulence3d",
    "checker",
    "mix",
    "triplanar",
    "constant",
    "image",
    "blur",
    "invert",
    "contrast",
    "hsvadjust",
    "levels",
    "range",
    "smoothstep",
    "warp",
    "place2d",
    "rotate2d",
    "separate2d",
    "combine2d",
    "separate3d",
    "combine3d",
    "separate4d",
    "combine4d",
    "convert",
    "clamp",
    "remap",
    "perlin",
    "value",
    "worley",
    "linear",
    "radial",
    "angular",
    "dots",
    "brick",
    "sharpen",
    "curves",
    "color_correct",
    "uniform",
    "image_asset",
    "mesh_map",
    "levels_as_mask",
    "histogram",
    "triplanar_blend",
    "noise",
];

/// Errors from `.mtlx` parsing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MtlxError {
    /// Malformed XML or an unparseable value, with 1-based line number.
    #[error("mtlx parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
    /// A node name / output reference that is not `N<digits>`.
    #[error("bad node id '{found}': expected N<digits>")]
    BadId { found: String },
    /// An element with a shape we cannot map (missing / conflicting attrs).
    #[error("element '{element}': expected {expected}")]
    TypeMismatch { element: String, expected: String },
    /// Two `<node>` elements with the same id.
    #[error("duplicate node id {id}")]
    DuplicateId { id: u64 },
    /// An `output=` reference to a node id not present in the document.
    #[error("node '{node}' input '{input}' references missing node '{target}'")]
    DanglingRef {
        node: String,
        input: String,
        target: String,
    },
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            _ => o.push(c),
        }
    }
    o
}

fn unesc(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn fmt_float(v: f32) -> String {
    v.to_string()
}

fn fmt_vec(vs: &[f32]) -> String {
    vs.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Serialize `graph` to a MaterialX 1.38 document string.
///
/// `custom_nodedefs` are declared FIRST (before any `<node>`), per the
/// spec's ordering rule. Deterministic: nodes by id, edges by
/// `(to, input)`, connected inputs by name; value params keep graph order.
pub fn to_mtlx(graph: &Graph, custom_nodedefs: &[NodedefDecl]) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<materialx version=\"1.38\">\n");
    for nd in custom_nodedefs {
        out.push_str(&format!(
            "  <nodedef name=\"{}\" node=\"{}\">\n",
            esc(&nd.name),
            esc(&nd.category)
        ));
        for (name, ty) in &nd.inputs {
            out.push_str(&format!(
                "    <input name=\"{}\" type=\"{}\" />\n",
                esc(name),
                esc(ty)
            ));
        }
        out.push_str("  </nodedef>\n");
    }
    let mut nodes: Vec<&Node> = graph.nodes.iter().collect();
    nodes.sort_by_key(|n| n.id);
    for node in nodes {
        out.push_str(&format!(
            "  <node name=\"N{}\" type=\"{}\">\n",
            node.id,
            esc(&node.node_def)
        ));
        // Connections: union of edges + stray NodeRef params, by input name.
        let mut conns: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        for e in &graph.edges {
            if e.to == node.id {
                conns.insert(e.input.as_str(), e.from);
            }
        }
        for (name, p) in &node.params {
            if let ParamValue::NodeRef(id) = p {
                conns.entry(name.as_str()).or_insert(*id);
            }
        }
        for (input, from) in &conns {
            out.push_str(&format!(
                "    <input name=\"{}\" type=\"color3\" output=\"N{}\" />\n",
                esc(input),
                from
            ));
        }
        for (name, p) in &node.params {
            if conns.contains_key(name.as_str()) {
                continue; // edge wins over a same-named value (see docs).
            }
            match p {
                ParamValue::Float(v) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"float\" value=\"{}\" />\n",
                    esc(name),
                    esc(&fmt_float(*v))
                )),
                ParamValue::Int(v) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"integer\" value=\"{}\" />\n",
                    esc(name),
                    v
                )),
                ParamValue::Vec2(v) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"vector2\" value=\"{}\" />\n",
                    esc(name),
                    esc(&fmt_vec(v))
                )),
                ParamValue::Vec3(v) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"vector3\" value=\"{}\" />\n",
                    esc(name),
                    esc(&fmt_vec(v))
                )),
                ParamValue::Color(v) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"color3\" value=\"{}\" />\n",
                    esc(name),
                    esc(&fmt_vec(v))
                )),
                ParamValue::Bool(v) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"boolean\" value=\"{}\" />\n",
                    esc(name),
                    if *v { "true" } else { "false" }
                )),
                ParamValue::Asset(s) => out.push_str(&format!(
                    "    <input name=\"{}\" type=\"string\" value=\"{}\" />\n",
                    esc(name),
                    esc(s)
                )),
                ParamValue::NodeRef(_) => {} // emitted as a connection above.
            }
        }
        out.push_str("  </node>\n");
    }
    out.push_str("</materialx>\n");
    out
}

// ---------------------------------------------------------------------------
// Minimal tolerant XML reader (the subset we emit; documented, not general).
// ---------------------------------------------------------------------------

struct Elem {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<Elem>,
    line: usize,
}

impl Elem {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
    line: usize, // 1-based line of `pos`.
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            b: s.as_bytes(),
            pos: 0,
            line: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.b.get(self.pos).copied()?;
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
        }
        Some(c)
    }

    fn starts_with(&self, s: &str) -> bool {
        self.b[self.pos..].starts_with(s.as_bytes())
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.bump();
        }
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T, MtlxError> {
        Err(MtlxError::Parse {
            line: self.line,
            message: msg.into(),
        })
    }

    fn expect(&mut self, s: &str, what: &str) -> Result<(), MtlxError> {
        if self.starts_with(s) {
            for _ in 0..s.len() {
                self.bump();
            }
            Ok(())
        } else {
            self.err(format!("expected {what}"))
        }
    }

    fn read_name(&mut self) -> Result<String, MtlxError> {
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' | b'.' | b':')
        ) {
            self.bump();
        }
        if self.pos == start {
            return self.err("expected a name");
        }
        Ok(String::from_utf8_lossy(&self.b[start..self.pos]).into_owned())
    }

    fn read_attr_value(&mut self) -> Result<String, MtlxError> {
        let q = self.peek();
        if q != Some(b'"') && q != Some(b'\'') {
            return self.err("expected quoted attribute value");
        }
        self.bump();
        let quote = q.unwrap();
        let start = self.pos;
        loop {
            match self.bump() {
                None => return self.err("unterminated attribute value"),
                Some(c) if c == quote => break,
                Some(_) => {}
            }
        }
        let raw = String::from_utf8_lossy(&self.b[start..self.pos - 1]).into_owned();
        Ok(unesc(&raw))
    }

    /// Skip `<? … ?>`.
    fn skip_pi(&mut self) -> Result<(), MtlxError> {
        loop {
            match self.bump() {
                None => return self.err("unterminated processing instruction"),
                Some(b'?') if self.peek() == Some(b'>') => {
                    self.bump();
                    return Ok(());
                }
                Some(_) => {}
            }
        }
    }

    /// Skip `<!-- … -->` or generic `<! … >`.
    fn skip_bang(&mut self) -> Result<(), MtlxError> {
        if self.starts_with("!--") {
            for _ in 0..3 {
                self.bump();
            }
            loop {
                match self.bump() {
                    None => return self.err("unterminated comment"),
                    Some(b'-') if self.starts_with("->") => {
                        for _ in 0..2 {
                            self.bump();
                        }
                        return Ok(());
                    }
                    Some(_) => {}
                }
            }
        }
        loop {
            match self.bump() {
                None => return self.err("unterminated <! … >"),
                Some(b'>') => return Ok(()),
                Some(_) => {}
            }
        }
    }

    fn parse_elem(&mut self) -> Result<Elem, MtlxError> {
        let line = self.line;
        self.expect("<", "'<'")?;
        if self.peek() == Some(b'/') {
            return self.err("unexpected closing tag");
        }
        let name = self.read_name()?;
        let mut attrs = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return self.err(format!("unterminated <{name}>")),
                Some(b'/') => {
                    self.bump();
                    match self.peek() {
                        Some(b'>') => {
                            self.bump();
                            return Ok(Elem {
                                name,
                                attrs,
                                children: Vec::new(),
                                line,
                            });
                        }
                        _ => return self.err("expected '/>'"),
                    }
                }
                Some(b'>') => {
                    self.bump();
                    break;
                }
                Some(_) => {
                    let key = self.read_name()?;
                    self.skip_ws();
                    self.expect("=", "'='")?;
                    self.skip_ws();
                    let val = self.read_attr_value()?;
                    attrs.push((key, val));
                }
            }
        }
        let mut children = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => {
                    return self.err(format!("unclosed <{name}> (reached end of input)"));
                }
                Some(b'<') if self.starts_with("</") => {
                    self.bump();
                    self.bump();
                    let close = self.read_name()?;
                    self.skip_ws();
                    self.expect(">", "'>'")?;
                    if close != name {
                        return self.err(format!("mismatched close </{close}> for <{name}>"));
                    }
                    return Ok(Elem {
                        name,
                        attrs,
                        children,
                        line,
                    });
                }
                Some(b'<') if self.starts_with("<?") => {
                    for _ in 0..2 {
                        self.bump();
                    }
                    self.skip_pi()?;
                }
                Some(b'<') if self.starts_with("<!") => {
                    for _ in 0..2 {
                        self.bump();
                    }
                    self.skip_bang()?;
                }
                Some(b'<') => children.push(self.parse_elem()?),
                Some(_) => {
                    // Text: skip to next tag (we carry no text content).
                    while !matches!(self.peek(), Some(b'<') | None) {
                        self.bump();
                    }
                }
            }
        }
    }
}

fn parse_doc(s: &str) -> Result<Elem, MtlxError> {
    let mut c = Cursor::new(s);
    let mut root: Option<Elem> = None;
    loop {
        c.skip_ws();
        match c.peek() {
            None => break,
            Some(b'<') if c.starts_with("<?") => {
                c.bump();
                c.bump();
                c.skip_pi()?;
            }
            Some(b'<') if c.starts_with("<!--") => {
                c.bump();
                c.bump();
                c.skip_bang()?;
            }
            Some(b'<') if c.starts_with("<!") => {
                c.bump();
                c.bump();
                c.skip_bang()?;
            }
            Some(b'<') => {
                if root.is_some() {
                    return c.err("multiple root elements");
                }
                root = Some(c.parse_elem()?);
            }
            Some(_) => return c.err("unexpected text outside the root element"),
        }
    }
    match root {
        Some(r) if r.name == "materialx" => Ok(r),
        Some(r) => Err(MtlxError::Parse {
            line: r.line,
            message: format!("expected <materialx> root, found <{}>", r.name),
        }),
        None => Err(MtlxError::Parse {
            line: 1,
            message: "empty document".into(),
        }),
    }
}

fn parse_node_id(s: &str) -> Result<u64, MtlxError> {
    s.strip_prefix('N')
        .and_then(|d| d.parse::<u64>().ok())
        .filter(|_| !s[1..].is_empty() && s[1..].bytes().all(|b| b.is_ascii_digit()))
        .ok_or_else(|| MtlxError::BadId { found: s.into() })
}

fn parse_floats(value: &str, line: usize, elem: &str, n: usize) -> Result<Vec<f32>, MtlxError> {
    let parts: Vec<&str> = value.split(',').map(str::trim).collect();
    if parts.len() != n || parts.iter().any(|p| p.is_empty()) {
        return Err(MtlxError::Parse {
            line,
            message: format!("element '{elem}': expected {n} comma-separated floats"),
        });
    }
    parts
        .iter()
        .map(|p| {
            p.parse::<f32>().map_err(|_| MtlxError::Parse {
                line,
                message: format!("element '{elem}': value '{p}' is not a float"),
            })
        })
        .collect()
}

fn value_from_typed(
    ty: &str,
    value: &str,
    line: usize,
    elem: &str,
) -> Result<ParamValue, MtlxError> {
    match ty {
        "float" => value
            .parse::<f32>()
            .map(ParamValue::Float)
            .map_err(|_| MtlxError::Parse {
                line,
                message: format!("element '{elem}': value '{value}' is not a float"),
            }),
        "integer" => value
            .parse::<i32>()
            .map(ParamValue::Int)
            .map_err(|_| MtlxError::Parse {
                line,
                message: format!("element '{elem}': value '{value}' is not an integer"),
            }),
        "vector2" => parse_floats(value, line, elem, 2).map(|v| ParamValue::Vec2([v[0], v[1]])),
        "vector3" => {
            parse_floats(value, line, elem, 3).map(|v| ParamValue::Vec3([v[0], v[1], v[2]]))
        }
        "color3" => {
            parse_floats(value, line, elem, 3).map(|v| ParamValue::Color([v[0], v[1], v[2]]))
        }
        "boolean" => match value.trim() {
            "true" | "1" => Ok(ParamValue::Bool(true)),
            "false" | "0" => Ok(ParamValue::Bool(false)),
            _ => Err(MtlxError::Parse {
                line,
                message: format!("element '{elem}': value '{value}' is not a boolean"),
            }),
        },
        "string" => Ok(ParamValue::Asset(value.into())),
        // Unknown value type: preserve the raw string as an Asset so no
        // data is lost (re-emits as `string`; see module docs).
        _ => Ok(ParamValue::Asset(value.into())),
    }
}

/// Parse the subset of MaterialX 1.38 that [`to_mtlx`] emits.
///
/// Returns `(graph, declared_nodedefs, warnings)` where `warnings` holds
/// the (deduped, first-seen-order) type names of nodes that are neither
/// declared in-document nor in [`KNOWN_STANDARD_NODES`]. Such nodes are
/// still rebuilt with params intact. Unknown non-node elements are
/// skipped, never fatal.
pub fn from_mtlx(s: &str) -> Result<(Graph, Vec<NodedefDecl>, Vec<String>), MtlxError> {
    let root = parse_doc(s)?;
    let mut nodedefs = Vec::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    fn walk_node_list(
        elem: &Elem,
        nodedefs: &mut Vec<NodedefDecl>,
        nodes: &mut Vec<Node>,
        edges: &mut Vec<Edge>,
        warnings: &mut Vec<String>,
    ) -> Result<(), MtlxError> {
        for child in &elem.children {
            match child.name.as_str() {
                "nodedef" => {
                    let name = child.attr("name").ok_or_else(|| MtlxError::Parse {
                        line: child.line,
                        message: "<nodedef> misses required 'name'".into(),
                    })?;
                    let category = child.attr("node").unwrap_or(name).to_string();
                    let mut inputs = Vec::new();
                    for inp in &child.children {
                        if inp.name != "input" {
                            continue;
                        }
                        let iname = inp.attr("name").ok_or_else(|| MtlxError::Parse {
                            line: inp.line,
                            message: "<nodedef> input misses 'name'".into(),
                        })?;
                        let ity = inp.attr("type").ok_or_else(|| MtlxError::TypeMismatch {
                            element: format!("{}/{}", name, iname),
                            expected: "a 'type' on every nodedef input".into(),
                        })?;
                        inputs.push((iname.to_string(), ity.to_string()));
                    }
                    nodedefs.push(NodedefDecl {
                        name: name.to_string(),
                        category,
                        inputs,
                    });
                }
                "node" => {
                    let raw_name = child.attr("name").ok_or_else(|| MtlxError::Parse {
                        line: child.line,
                        message: "<node> misses required 'name'".into(),
                    })?;
                    let ty = child.attr("type").ok_or_else(|| MtlxError::Parse {
                        line: child.line,
                        message: format!("<node {raw_name}> misses required 'type'"),
                    })?;
                    let id = parse_node_id(raw_name)?;
                    if !nodedefs.iter().any(|d| d.name == ty)
                        && !KNOWN_STANDARD_NODES.contains(&ty)
                        && !warnings.iter().any(|w| w == ty)
                    {
                        warnings.push(ty.to_string());
                    }
                    let mut params = Vec::new();
                    for inp in &child.children {
                        if inp.name != "input" {
                            continue; // tolerated, never fatal.
                        }
                        let iname = inp.attr("name").ok_or_else(|| MtlxError::Parse {
                            line: inp.line,
                            message: "<input> misses required 'name'".into(),
                        })?;
                        let has_value = inp.attr("value").is_some();
                        let has_output = inp.attr("output").is_some();
                        match (has_value, has_output) {
                            (true, true) => {
                                return Err(MtlxError::TypeMismatch {
                                    element: iname.to_string(),
                                    expected: "exactly one of 'value' / 'output'".into(),
                                });
                            }
                            (false, false) => {
                                return Err(MtlxError::TypeMismatch {
                                    element: iname.to_string(),
                                    expected: "one of 'value' / 'output'".into(),
                                });
                            }
                            (false, true) => {
                                let target = inp.attr("output").unwrap_or_default();
                                let from = parse_node_id(target)?;
                                edges.push(Edge {
                                    from,
                                    to: id,
                                    input: iname.to_string(),
                                });
                            }
                            (true, false) => {
                                let ty =
                                    inp.attr("type").ok_or_else(|| MtlxError::TypeMismatch {
                                        element: iname.to_string(),
                                        expected: "'type' alongside 'value'".into(),
                                    })?;
                                let value = inp.attr("value").unwrap_or_default();
                                params.push((
                                    iname.to_string(),
                                    value_from_typed(ty, value, inp.line, iname)?,
                                ));
                            }
                        }
                    }
                    nodes.push(Node {
                        id,
                        node_def: ty.to_string(),
                        params,
                    });
                }
                "nodegraph" => walk_node_list(child, nodedefs, nodes, edges, warnings)?,
                _ => {} // unknown elements: skipped, never fatal.
            }
        }
        Ok(())
    }

    walk_node_list(&root, &mut nodedefs, &mut nodes, &mut edges, &mut warnings)?;

    let mut seen = std::collections::HashSet::new();
    for n in &nodes {
        if !seen.insert(n.id) {
            return Err(MtlxError::DuplicateId { id: n.id });
        }
    }
    let ids: std::collections::HashSet<u64> = seen;
    for e in &edges {
        if !ids.contains(&e.from) {
            let to_name = format!("N{}", e.to);
            return Err(MtlxError::DanglingRef {
                node: to_name,
                input: e.input.clone(),
                target: format!("N{}", e.from),
            });
        }
    }
    nodes.sort_by_key(|n| n.id);
    edges.sort_by(|a, b| (a.to, &a.input).cmp(&(b.to, &b.input)));
    Ok((Graph { nodes, edges }, nodedefs, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative chain covering every value-carrying ParamValue
    /// variant: standard `noise` → custom `flood_fill` → `mix`.
    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "noise".into(),
            params: vec![
                ("scale".into(), ParamValue::Float(8.5)),
                ("octaves".into(), ParamValue::Int(5)),
                ("offset".into(), ParamValue::Vec2([0.25, -1.5])),
            ],
        });
        g.add_node(Node {
            id: 2,
            node_def: "flood_fill".into(),
            params: vec![
                ("tolerance".into(), ParamValue::Float(0.1)),
                ("tint".into(), ParamValue::Color([1.0, 0.5, 0.25])),
                ("enabled".into(), ParamValue::Bool(true)),
            ],
        });
        g.add_node(Node {
            id: 3,
            node_def: "mix".into(),
            params: vec![
                ("amount".into(), ParamValue::Float(0.5)),
                ("albedo".into(), ParamValue::Asset("b3:deadbeefcafe".into())),
                ("flag".into(), ParamValue::Bool(false)),
            ],
        });
        g.add_edge(Edge {
            from: 1,
            to: 2,
            input: "mask".into(),
        });
        g.add_edge(Edge {
            from: 2,
            to: 3,
            input: "fg".into(),
        });
        g
    }

    #[test]
    fn double_round_trip_is_byte_identical() {
        let nodedefs = painter_nodedefs();
        let g = sample_graph();
        let first = to_mtlx(&g, &nodedefs);
        assert!(first.contains("<nodedef"), "must declare custom nodedefs");
        let (g2, nd2, _) = from_mtlx(&first).expect("sample must parse");
        let second = to_mtlx(&g2, &nd2);
        assert_eq!(first, second, "write→read→write must be byte-identical");
    }

    #[test]
    fn round_trip_preserves_data() {
        // Canonical order (nodes by id, edges by (to, input)) round-trips
        // to equal Graph + NodedefDecl data.
        let nodedefs: Vec<NodedefDecl> = painter_nodedefs()
            .into_iter()
            .filter(|d| d.name == "flood_fill")
            .collect();
        let g = sample_graph();
        let s = to_mtlx(&g, &nodedefs);
        let (g2, nd2, warnings) = from_mtlx(&s).expect("sample must parse");
        assert_eq!(g, g2, "graph data must survive the round-trip");
        assert_eq!(nodedefs, nd2, "nodedef data must survive the round-trip");
        assert!(
            warnings.is_empty(),
            "known types must not warn, got {warnings:?}"
        );
    }

    #[test]
    fn unknown_node_type_is_preserved_with_warning() {
        let doc = r#"<?xml version="1.0" encoding="UTF-8"?>
<materialx version="1.38">
  <node name="N7" type="quantum_flux_capacitor">
    <input name="gigawatts" type="float" value="1.21" />
    <input name="shade" type="color3" value="0.1, 0.2, 0.3" />
  </node>
</materialx>
"#;
        let (g, _, warnings) = from_mtlx(doc).expect("unknown types must parse");
        assert!(
            warnings.contains(&"quantum_flux_capacitor".to_string()),
            "unknown type must be warned, got {warnings:?}"
        );
        assert_eq!(g.nodes.len(), 1);
        let n = &g.nodes[0];
        assert_eq!(n.id, 7);
        assert_eq!(n.node_def, "quantum_flux_capacitor");
        assert_eq!(
            n.params,
            vec![
                ("gigawatts".to_string(), ParamValue::Float(1.21)),
                ("shade".to_string(), ParamValue::Color([0.1, 0.2, 0.3])),
            ],
            "unknown-node params must survive intact"
        );
        // And the survivor re-emits losslessly.
        let again = to_mtlx(&g, &[]);
        let (g3, _, _) = from_mtlx(&again).expect("re-emit must parse");
        assert_eq!(g, g3);
    }

    #[test]
    fn malformed_xml_errors_with_line_number() {
        let nodedefs = painter_nodedefs();
        let valid = to_mtlx(&sample_graph(), &nodedefs);
        // Truncate mid-element on a known line (inside node N2's block).
        let cut_at = valid
            .find("tolerance")
            .expect("fixture must contain tolerance");
        let line_of_cut = valid[..cut_at].chars().filter(|&c| c == '\n').count() + 1;
        let truncated = &valid[..cut_at];
        match from_mtlx(truncated) {
            Err(MtlxError::Parse { line, .. }) => {
                assert!(line >= 1, "line numbers are 1-based");
                assert!(
                    (line as i64 - line_of_cut as i64).abs() <= 2,
                    "error line {line} must be near truncation line {line_of_cut}"
                );
            }
            other => panic!("truncated doc must fail with Parse, got {other:?}"),
        }
    }

    #[test]
    fn nodedefs_emit_before_nodes() {
        let s = to_mtlx(&sample_graph(), &painter_nodedefs());
        let first_node = s.find("<node ").expect("must contain nodes");
        let mut nodedef_positions = Vec::new();
        let mut search_from = 0;
        while let Some(i) = s[search_from..].find("<nodedef") {
            nodedef_positions.push(search_from + i);
            search_from += i + 1;
        }
        assert!(!nodedef_positions.is_empty(), "must declare nodedefs");
        for pos in nodedef_positions {
            assert!(
                pos < first_node,
                "<nodedef> at {pos} must precede first <node> at {first_node}"
            );
        }
    }

    #[test]
    fn edge_insertion_order_does_not_change_bytes() {
        let mut a = sample_graph();
        let mut b = sample_graph();
        a.edges.clear();
        b.edges.clear();
        a.add_edge(Edge {
            from: 1,
            to: 2,
            input: "mask".into(),
        });
        a.add_edge(Edge {
            from: 2,
            to: 3,
            input: "fg".into(),
        });
        b.add_edge(Edge {
            from: 2,
            to: 3,
            input: "fg".into(),
        });
        b.add_edge(Edge {
            from: 1,
            to: 2,
            input: "mask".into(),
        });
        assert_ne!(a.edges, b.edges, "fixture must differ in insertion order");
        let nd = painter_nodedefs();
        assert_eq!(to_mtlx(&a, &nd), to_mtlx(&b, &nd));
    }
}

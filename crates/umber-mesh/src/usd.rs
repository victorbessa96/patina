//! USD ASCII (`.usda`) mesh import via a small self-contained parser.
//!
//! No USD dependency: the text format (Sdf text syntax) is parsed by hand.
//! The parser understands the full statement structure — layer metadata,
//! nested `def`/`over`/`class` prims, attributes with metadata, `rel`s,
//! `variantSet`s, `.timeSamples`/`.connect`, paths and asset paths — but only
//! extracts geometry from the **first `def Mesh` prim** in document order
//! (pre-order, so a Mesh nested in an Xform counts). Everything else is
//! syntax-checked and skipped. Binary crate files (`usdc`, or a `.usd` whose
//! bytes start with `PXR-USDC`) return [`ImportError::UsdBinary`].
//!
//! Geometry extraction:
//! - `points` → positions; `faceVertexCounts` + `faceVertexIndices` → faces.
//! - Vertices are expanded per face corner (one output vertex per entry of
//!   `faceVertexIndices`), so `faceVarying`/`uniform` primvars and flat
//!   normals stay exact. As in the FBX loader, there is no weld pass.
//! - Each face is fan-triangulated over its corners: `(0,1,2), (0,2,3), …`.
//!   This is exact for convex n-gons; concave n-gons and `holeIndices` are the
//!   follow-up (holes are rejected with an error rather than ignored).
//! - `orientation = "leftHanded"` flips the fan winding so output triangles
//!   are always counter-clockwise front-facing, like glTF.
//! - Normals come from `primvars:normals`, else `normals`. When neither is
//!   authored, flat per-face normals are computed (Newell's method). This
//!   deliberately differs from the glTF/FBX loaders, which zero-fill.
//! - UVs come from `primvars:st`, else the first `texCoord2*` primvar (e.g.
//!   Blender's `primvars:UVMap`); indexed primvars (`<name>:indices`) are
//!   resolved. Missing UVs are zero-filled, matching the glTF/FBX loaders.
//! - Primvar interpolation (`constant`/`uniform`/`vertex`/`varying`/
//!   `faceVarying`) is read from attribute metadata; when absent it is
//!   inferred from the element count.
//!
//! Not supported in v1 (error or ignored, documented): prim transforms
//! (local-space points, like FBX bind pose), time-sampled points (error),
//! `elementSize` > 1 (error), variants and references/payloads (not composed),
//! subdivision (the cage is imported as-is). `material_names` stays empty.

use std::iter::Peekable;
use std::path::Path;

use glam::Vec3;

use super::{ImportError, MeshData};

/// Magic bytes opening a binary crate (`usdc`) file.
const USDC_MAGIC: &[u8] = b"PXR-USDC";

/// Prim specifiers that open a prim statement.
const SPECIFIERS: [&str; 3] = ["def", "over", "class"];

/// List-op keywords that may prefix a property statement.
const LIST_OPS: [&str; 5] = ["prepend", "append", "add", "delete", "reorder"];

/// Load a `.usda` (or text `.usd`) file: the first `def Mesh` prim becomes
/// a fan-triangulated [`MeshData`].
///
/// Returns [`ImportError::Usd`] (message prefixed `usd:`, with a `line N:`
/// hint where one applies) when the text is malformed, truncated, has no
/// Mesh prim, or the Mesh topology is inconsistent; [`ImportError::UsdBinary`]
/// for binary crate content; [`ImportError::Io`] when the file can't be read.
pub fn load_usda(path: &Path) -> Result<MeshData, ImportError> {
    let bytes = std::fs::read(path)?;
    parse_usda_bytes(&bytes)
}

/// Parse in-memory `.usda` bytes (the testable core of [`load_usda`]).
fn parse_usda_bytes(bytes: &[u8]) -> Result<MeshData, ImportError> {
    if bytes.starts_with(USDC_MAGIC) {
        return Err(ImportError::UsdBinary);
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|e| ImportError::Usd(format!("file is not UTF-8 text: {e}")))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if !text.starts_with("#usda") {
        return Err(usd_error(
            1,
            "missing '#usda 1.0' header; not a USD ASCII layer",
        ));
    }
    let mut parser = Parser {
        tokens: lex(text)?.into_iter().peekable(),
        last_line: 1,
    };
    let raw = parser
        .parse_layer()?
        .ok_or_else(|| ImportError::Usd("no 'def Mesh' prim found in the layer".into()))?;
    build_mesh(&raw)
}

/// A [`ImportError::Usd`] carrying a 1-based line hint.
fn usd_error(line: usize, message: impl std::fmt::Display) -> ImportError {
    ImportError::Usd(format!("line {line}: {message}"))
}

// ---------------------------------------------------------------- lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// Identifiers, keywords, namespaced names (`primvars:st`) and numbers.
    Word(String),
    /// Quoted string (single, double, or triple-quoted).
    Str(String),
    /// `</prim/path.property>`.
    Path(String),
    /// `@asset/path@` or `@@@asset@@@`.
    Asset(String),
    /// One of `[ ] ( ) { } = , ;`.
    Punct(char),
}

impl Tok {
    /// Short human description for error messages.
    fn describe(&self) -> String {
        match self {
            Tok::Word(w) => format!("'{w}'"),
            Tok::Str(_) => "a string".into(),
            Tok::Path(p) => format!("path <{p}>"),
            Tok::Asset(a) => format!("asset @{a}@"),
            Tok::Punct(c) => format!("'{c}'"),
        }
    }
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
}

/// Bytes that may appear in a [`Tok::Word`] (identifiers and numbers).
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b':' | b'.' | b'-' | b'+')
}

/// Tokenize the layer text. `#` comments (including the `#usda` header line)
/// are dropped; every token records its starting line.
fn lex(text: &str) -> Result<Vec<Token>, ImportError> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while let Some(&b) = bytes.get(i) {
        match b {
            b'\n' => {
                line += 1;
                i += 1;
            }
            b' ' | b'\t' | b'\r' => i += 1,
            b'#' => {
                while bytes.get(i).is_some_and(|&c| c != b'\n') {
                    i += 1;
                }
            }
            b'"' | b'\'' => {
                let (value, end, newlines) = lex_string(bytes, i, line)?;
                tokens.push(Token {
                    tok: Tok::Str(value),
                    line,
                });
                line += newlines;
                i = end;
            }
            b'<' => {
                let start = i + 1;
                let close = bytes
                    .get(start..)
                    .and_then(|rest| rest.iter().position(|&c| c == b'>' || c == b'\n'));
                match close {
                    Some(n) if bytes.get(start + n) == Some(&b'>') => {
                        let path = String::from_utf8_lossy(&bytes[start..start + n]).into_owned();
                        tokens.push(Token {
                            tok: Tok::Path(path),
                            line,
                        });
                        i = start + n + 1;
                    }
                    _ => return Err(usd_error(line, "unterminated path '<...>'")),
                }
            }
            b'@' => {
                let triple = bytes.get(i..i + 3) == Some(&b"@@@"[..]);
                let (delimiter, start): (&[u8], usize) = if triple {
                    (b"@@@", i + 3)
                } else {
                    (b"@", i + 1)
                };
                let close = bytes
                    .get(start..)
                    .and_then(|rest| rest.windows(delimiter.len()).position(|w| w == delimiter));
                match close {
                    Some(n) if !bytes[start..start + n].contains(&b'\n') => {
                        let asset = String::from_utf8_lossy(&bytes[start..start + n]).into_owned();
                        tokens.push(Token {
                            tok: Tok::Asset(asset),
                            line,
                        });
                        i = start + n + delimiter.len();
                    }
                    _ => return Err(usd_error(line, "unterminated asset path '@...@'")),
                }
            }
            b'[' | b']' | b'(' | b')' | b'{' | b'}' | b'=' | b',' | b';' => {
                tokens.push(Token {
                    tok: Tok::Punct(b as char),
                    line,
                });
                i += 1;
            }
            c if is_word_byte(c) => {
                let start = i;
                while bytes.get(i).is_some_and(|&c| is_word_byte(c)) {
                    i += 1;
                }
                tokens.push(Token {
                    tok: Tok::Word(text[start..i].to_string()),
                    line,
                });
            }
            other => {
                return Err(usd_error(
                    line,
                    format!("unexpected character '{}'", other.escape_ascii()),
                ))
            }
        }
    }
    Ok(tokens)
}

/// Lex a quoted string starting at `bytes[start]` (a `"` or `'`). Returns
/// the unescaped value, the index just past the closing quote, and the number
/// of newlines consumed (triple-quoted strings may span lines).
fn lex_string(
    bytes: &[u8],
    start: usize,
    line: usize,
) -> Result<(String, usize, usize), ImportError> {
    let quote = bytes[start];
    let triple = bytes.get(start + 1) == Some(&quote) && bytes.get(start + 2) == Some(&quote);
    let mut i = start + if triple { 3 } else { 1 };
    let mut out = Vec::new();
    let mut newlines = 0;
    loop {
        let Some(&c) = bytes.get(i) else {
            return Err(usd_error(line, "unterminated string"));
        };
        if c == b'\\' {
            let Some(&escaped) = bytes.get(i + 1) else {
                return Err(usd_error(line, "unterminated string"));
            };
            if escaped == b'\n' {
                newlines += 1;
            }
            out.push(escaped);
            i += 2;
            continue;
        }
        if c == quote {
            if !triple {
                return Ok((String::from_utf8_lossy(&out).into_owned(), i + 1, newlines));
            }
            if bytes.get(i + 1) == Some(&quote) && bytes.get(i + 2) == Some(&quote) {
                return Ok((String::from_utf8_lossy(&out).into_owned(), i + 3, newlines));
            }
        } else if c == b'\n' {
            if !triple {
                return Err(usd_error(line, "unterminated string"));
            }
            newlines += 1;
        }
        out.push(c);
        i += 1;
    }
}

// ---------------------------------------------------------------- parser

/// A parsed attribute value. Only the shapes geometry extraction needs are
/// kept; dictionaries, time samples, paths and assets collapse to `Opaque`.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    Num(f64),
    /// Bare identifier, e.g. `None`.
    Word(String),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Opaque,
}

/// The attribute metadata the importer reads; everything else is skipped.
#[derive(Debug, Default)]
struct PropertyMeta {
    interpolation: Option<String>,
    element_size: Option<f64>,
}

#[derive(Debug)]
struct Property {
    type_name: String,
    value: Option<Value>,
    meta: PropertyMeta,
    line: usize,
}

impl Property {
    /// The authored value, treating a declaration without `=` and an
    /// explicit `None` as unauthored.
    fn authored_value(&self) -> Option<&Value> {
        match &self.value {
            Some(Value::Word(w)) if w == "None" => None,
            other => other.as_ref(),
        }
    }
}

/// The first `def Mesh` prim's properties, in document order.
#[derive(Debug)]
struct RawMesh {
    name: String,
    line: usize,
    props: Vec<(String, Property)>,
}

impl RawMesh {
    /// The last declaration of `name` (a later opinion in the same prim wins).
    fn prop(&self, name: &str) -> Option<&Property> {
        self.props
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
    }
}

struct Parser {
    tokens: Peekable<std::vec::IntoIter<Token>>,
    /// Line of the most recently consumed token (the EOF error hint).
    last_line: usize,
}

impl Parser {
    fn peek_punct(&mut self, c: char) -> bool {
        matches!(self.tokens.peek(), Some(Token { tok: Tok::Punct(p), .. }) if *p == c)
    }

    fn eat_punct(&mut self, c: char) -> bool {
        if self.peek_punct(c) {
            if let Some(token) = self.tokens.next() {
                self.last_line = token.line;
            }
            true
        } else {
            false
        }
    }

    /// Consume the next token; end of input is a (truncation) error.
    fn next(&mut self, context: &str) -> Result<Token, ImportError> {
        match self.tokens.next() {
            Some(token) => {
                self.last_line = token.line;
                Ok(token)
            }
            None => Err(usd_error(
                self.last_line,
                format!("unexpected end of file {context} (truncated file?)"),
            )),
        }
    }

    fn expect_punct(&mut self, c: char, context: &str) -> Result<(), ImportError> {
        let token = self.next(context)?;
        if token.tok == Tok::Punct(c) {
            Ok(())
        } else {
            Err(usd_error(
                token.line,
                format!("expected '{c}' {context}, found {}", token.tok.describe()),
            ))
        }
    }

    fn expect_word(&mut self, context: &str) -> Result<String, ImportError> {
        let token = self.next(context)?;
        match token.tok {
            Tok::Word(w) => Ok(w),
            other => Err(usd_error(
                token.line,
                format!(
                    "expected an identifier {context}, found {}",
                    other.describe()
                ),
            )),
        }
    }

    /// Skip tokens up to the bracket closing one already consumed on
    /// `open_line`. Bracket kinds are counted together, not matched.
    fn skip_balanced(&mut self, open_line: usize) -> Result<(), ImportError> {
        let mut depth = 1usize;
        while depth > 0 {
            let token = self.tokens.next().ok_or_else(|| {
                usd_error(
                    self.last_line,
                    format!(
                        "unexpected end of file: bracket opened on line {open_line} is never closed (truncated file?)"
                    ),
                )
            })?;
            self.last_line = token.line;
            match token.tok {
                Tok::Punct('(' | '[' | '{') => depth += 1,
                Tok::Punct(')' | ']' | '}') => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    /// Parse the whole layer; returns the first `def Mesh` prim, if any.
    fn parse_layer(&mut self) -> Result<Option<RawMesh>, ImportError> {
        if self.eat_punct('(') {
            self.skip_balanced(self.last_line)?;
        }
        let mut found = None;
        while self.tokens.peek().is_some() {
            let token = self.next("")?;
            match token.tok {
                Tok::Word(w) if SPECIFIERS.contains(&w.as_str()) => {
                    self.parse_prim(&w, token.line, &mut found)?
                }
                other => {
                    return Err(usd_error(
                        token.line,
                        format!(
                            "expected a prim ('def', 'over' or 'class'), found {}",
                            other.describe()
                        ),
                    ))
                }
            }
        }
        Ok(found)
    }

    /// Parse one prim after its specifier: `[Type] "name" [( meta )] { body }`.
    fn parse_prim(
        &mut self,
        specifier: &str,
        line: usize,
        found: &mut Option<RawMesh>,
    ) -> Result<(), ImportError> {
        let token = self.next("after a prim specifier")?;
        let (type_name, name) = match token.tok {
            Tok::Str(name) => (None, name),
            Tok::Word(type_name) => {
                let token = self.next("after a prim type")?;
                match token.tok {
                    Tok::Str(name) => (Some(type_name), name),
                    other => {
                        return Err(usd_error(
                            token.line,
                            format!("expected a quoted prim name, found {}", other.describe()),
                        ))
                    }
                }
            }
            other => {
                return Err(usd_error(
                    token.line,
                    format!(
                        "expected a prim type or quoted name, found {}",
                        other.describe()
                    ),
                ))
            }
        };
        if self.eat_punct('(') {
            self.skip_balanced(self.last_line)?;
        }
        self.expect_punct('{', "to open the prim body")?;
        let capture = specifier == "def" && type_name.as_deref() == Some("Mesh") && found.is_none();
        let mut props = capture.then(Vec::new);
        self.parse_body(&mut props, found)?;
        if let Some(props) = props {
            *found = Some(RawMesh { name, line, props });
        }
        Ok(())
    }

    /// Parse prim body statements up to and including the closing `}`.
    /// Properties are recorded only when `props` is `Some` (the captured Mesh).
    fn parse_body(
        &mut self,
        props: &mut Option<Vec<(String, Property)>>,
        found: &mut Option<RawMesh>,
    ) -> Result<(), ImportError> {
        loop {
            let token = self.next("inside a prim body (missing '}')")?;
            match token.tok {
                Tok::Punct('}') => return Ok(()),
                Tok::Punct(';') => {}
                Tok::Word(w) if SPECIFIERS.contains(&w.as_str()) => {
                    self.parse_prim(&w, token.line, found)?
                }
                Tok::Word(w) if w == "variantSet" => {
                    // `variantSet "name" = { ... }` — variants are not composed in v1.
                    self.next("after 'variantSet'")?;
                    self.expect_punct('=', "after the variantSet name")?;
                    self.expect_punct('{', "to open the variantSet")?;
                    self.skip_balanced(self.last_line)?;
                }
                Tok::Word(w) => self.parse_property(w, token.line, props)?,
                other => {
                    return Err(usd_error(
                        token.line,
                        format!("unexpected {} in a prim body", other.describe()),
                    ))
                }
            }
        }
    }

    /// Parse one property statement whose first word was already consumed:
    /// `[listop] [custom] [uniform|varying|config] type[[]] name [= value] [( meta )]`
    /// or `[listop] [custom] rel name [= target] [( meta )]`.
    fn parse_property(
        &mut self,
        first: String,
        line: usize,
        props: &mut Option<Vec<(String, Property)>>,
    ) -> Result<(), ImportError> {
        let mut word = first;
        if LIST_OPS.contains(&word.as_str()) {
            word = self.expect_word("after a list-op keyword")?;
        }
        if word == "custom" {
            word = self.expect_word("after 'custom'")?;
        }
        if word == "rel" {
            self.expect_word("for the relationship name")?;
            if self.eat_punct('=') {
                self.parse_value()?;
            }
            if self.eat_punct('(') {
                self.skip_balanced(self.last_line)?;
            }
            return Ok(());
        }
        if matches!(word.as_str(), "uniform" | "varying" | "config") {
            word = self.expect_word("after a variability keyword")?;
        }
        let type_name = word;
        if self.eat_punct('[') {
            self.expect_punct(']', "to close the array type")?;
        }
        // Two-word statements (`reorder nameChildren = [...]`) have no type.
        let name = if self.peek_punct('=') {
            type_name.clone()
        } else {
            self.expect_word("for the property name")?
        };
        let value = if self.eat_punct('=') {
            Some(self.parse_value()?)
        } else {
            None
        };
        let meta = if self.eat_punct('(') {
            self.parse_property_metadata()?
        } else {
            PropertyMeta::default()
        };
        if let Some(props) = props {
            props.push((
                name,
                Property {
                    type_name,
                    value,
                    meta,
                    line,
                },
            ));
        }
        Ok(())
    }

    fn parse_value(&mut self) -> Result<Value, ImportError> {
        let token = self.next("while reading a value")?;
        Ok(match token.tok {
            Tok::Punct('[') => Value::List(self.parse_sequence(']')?),
            Tok::Punct('(') => Value::Tuple(self.parse_sequence(')')?),
            Tok::Punct('{') => {
                self.skip_balanced(token.line)?;
                Value::Opaque
            }
            Tok::Word(w) => match w.parse::<f64>() {
                Ok(n) => Value::Num(n),
                Err(_) => Value::Word(w),
            },
            Tok::Str(s) => Value::Str(s),
            Tok::Path(_) | Tok::Asset(_) => Value::Opaque,
            other => {
                return Err(usd_error(
                    token.line,
                    format!("expected a value, found {}", other.describe()),
                ))
            }
        })
    }

    /// Comma-separated values up to `close` (the opener is already consumed);
    /// a trailing comma is allowed.
    fn parse_sequence(&mut self, close: char) -> Result<Vec<Value>, ImportError> {
        let mut items = Vec::new();
        loop {
            if self.eat_punct(close) {
                return Ok(items);
            }
            items.push(self.parse_value()?);
            if !self.eat_punct(',') {
                self.expect_punct(close, "to close the list (or ',' between elements)")?;
                return Ok(items);
            }
        }
    }

    /// Attribute metadata after its `(`: reads `interpolation` and
    /// `elementSize`, skips every other entry.
    fn parse_property_metadata(&mut self) -> Result<PropertyMeta, ImportError> {
        let mut meta = PropertyMeta::default();
        loop {
            if self.eat_punct(')') {
                return Ok(meta);
            }
            let token = self.next("inside attribute metadata (missing ')')")?;
            match token.tok {
                Tok::Word(key) if self.eat_punct('=') => {
                    let value = self.parse_value()?;
                    match (key.as_str(), value) {
                        ("interpolation", Value::Str(s)) => meta.interpolation = Some(s),
                        ("elementSize", Value::Num(n)) => meta.element_size = Some(n),
                        _ => {}
                    }
                }
                // List-op prefixes, `dictionary` type keywords, doc shorthand.
                Tok::Word(_) | Tok::Str(_) | Tok::Punct(',' | ';') => {}
                Tok::Punct('(' | '[' | '{') => self.skip_balanced(token.line)?,
                other => {
                    return Err(usd_error(
                        token.line,
                        format!("unexpected {} in attribute metadata", other.describe()),
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------- mesh build

/// Element counts a primvar's length is checked against.
struct Topology {
    points: usize,
    faces: usize,
    corners: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Interp {
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
}

impl Interp {
    fn usd_name(self) -> &'static str {
        match self {
            Interp::Constant => "constant",
            Interp::Uniform => "uniform",
            Interp::Vertex => "vertex",
            Interp::FaceVarying => "faceVarying",
        }
    }

    fn expected_len(self, topo: &Topology) -> usize {
        match self {
            Interp::Constant => 1,
            Interp::Uniform => topo.faces,
            Interp::Vertex => topo.points,
            Interp::FaceVarying => topo.corners,
        }
    }
}

/// A resolved primvar: values, optional index indirection, interpolation.
struct Primvar<const N: usize> {
    values: Vec<[f32; N]>,
    indices: Option<Vec<usize>>,
    interp: Interp,
}

impl<const N: usize> Primvar<N> {
    /// The value at one face corner (`point` is that corner's point index).
    fn at(&self, face: usize, point: usize, corner: usize) -> Result<[f32; N], ImportError> {
        let k = match self.interp {
            Interp::Constant => 0,
            Interp::Uniform => face,
            Interp::Vertex => point,
            Interp::FaceVarying => corner,
        };
        let k = match &self.indices {
            Some(indices) => indices.get(k).copied(),
            None => Some(k),
        };
        k.and_then(|k| self.values.get(k).copied())
            .ok_or_else(|| ImportError::Usd(format!("primvar lookup out of range at face {face}")))
    }
}

/// An `N`-tuple array attribute (`point3f[]`, `texCoord2f[]`, ...); `None`
/// when the property has no authored value.
fn tuples<const N: usize>(
    name: &str,
    prop: &Property,
) -> Result<Option<Vec<[f32; N]>>, ImportError> {
    let bad = || {
        usd_error(
            prop.line,
            format!("'{name}' must be an array of {N}-component tuples"),
        )
    };
    let Some(value) = prop.authored_value() else {
        return Ok(None);
    };
    let Value::List(items) = value else {
        return Err(bad());
    };
    items
        .iter()
        .map(|item| {
            let Value::Tuple(components) = item else {
                return Err(bad());
            };
            if components.len() != N {
                return Err(bad());
            }
            let mut out = [0.0f32; N];
            for (slot, component) in out.iter_mut().zip(components) {
                let Value::Num(n) = component else {
                    return Err(bad());
                };
                *slot = *n as f32;
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// An `int[]` attribute; `None` when the property has no authored value.
fn ints(name: &str, prop: &Property) -> Result<Option<Vec<i64>>, ImportError> {
    let bad = || usd_error(prop.line, format!("'{name}' must be an array of integers"));
    let Some(value) = prop.authored_value() else {
        return Ok(None);
    };
    let Value::List(items) = value else {
        return Err(bad());
    };
    items
        .iter()
        .map(|item| match item {
            Value::Num(n) if n.fract() == 0.0 => Ok(*n as i64),
            _ => Err(bad()),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Resolve and validate a primvar's interpolation against its element count
/// (after index indirection).
fn resolve_interp(
    name: &str,
    prop: &Property,
    len: usize,
    topo: &Topology,
) -> Result<Interp, ImportError> {
    let interp = match prop.meta.interpolation.as_deref() {
        Some("constant") => Interp::Constant,
        Some("uniform") => Interp::Uniform,
        Some("vertex" | "varying") => Interp::Vertex,
        Some("faceVarying") => Interp::FaceVarying,
        Some(other) => {
            return Err(usd_error(
                prop.line,
                format!("'{name}' has unknown interpolation \"{other}\""),
            ))
        }
        None if len == topo.points => Interp::Vertex,
        None if len == topo.corners => Interp::FaceVarying,
        None if len == topo.faces => Interp::Uniform,
        None if len == 1 => Interp::Constant,
        None => {
            return Err(usd_error(
                prop.line,
                format!(
                    "'{name}' has {len} elements, matching no interpolation \
                     ({} points, {} faces, {} face corners)",
                    topo.points, topo.faces, topo.corners
                ),
            ))
        }
    };
    let expected = interp.expected_len(topo);
    if len != expected {
        return Err(usd_error(
            prop.line,
            format!(
                "'{name}' has {len} elements but {} interpolation needs {expected}",
                interp.usd_name()
            ),
        ));
    }
    Ok(interp)
}

/// Load primvar `name` (with optional `<name>:indices`); `None` when the
/// attribute is absent or unauthored.
fn load_primvar<const N: usize>(
    raw: &RawMesh,
    name: &str,
    topo: &Topology,
) -> Result<Option<Primvar<N>>, ImportError> {
    let Some(prop) = raw.prop(name) else {
        return Ok(None);
    };
    let Some(values) = tuples::<N>(name, prop)? else {
        return Ok(None);
    };
    if let Some(size) = prop.meta.element_size {
        if size != 1.0 {
            return Err(usd_error(
                prop.line,
                format!("'{name}' has elementSize {size}; only 1 is supported in v1"),
            ));
        }
    }
    let indices_name = format!("{name}:indices");
    let indices = match raw.prop(&indices_name) {
        Some(index_prop) => match ints(&indices_name, index_prop)? {
            Some(raw_indices) => Some(
                raw_indices
                    .into_iter()
                    .map(|i| {
                        usize::try_from(i)
                            .ok()
                            .filter(|&i| i < values.len())
                            .ok_or_else(|| {
                                usd_error(
                                    index_prop.line,
                                    format!(
                                        "'{indices_name}' entry {i} is out of range for {} values",
                                        values.len()
                                    ),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        },
        None => None,
    };
    let len = indices.as_ref().map_or(values.len(), Vec::len);
    let interp = resolve_interp(name, prop, len, topo)?;
    Ok(Some(Primvar {
        values,
        indices,
        interp,
    }))
}

/// Flat normal of a polygon via Newell's method (robust for n-gons);
/// degenerate faces fall back to +Z instead of producing NaN.
fn face_normal(polygon: &[[f32; 3]]) -> [f32; 3] {
    let mut n = Vec3::ZERO;
    for (i, a) in polygon.iter().enumerate() {
        let a = Vec3::from(*a);
        let b = Vec3::from(polygon[(i + 1) % polygon.len()]);
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    n.try_normalize().unwrap_or(Vec3::Z).to_array()
}

/// Turn the captured Mesh prim into a corner-expanded, fan-triangulated
/// [`MeshData`], validating topology and primvar sizes along the way.
fn build_mesh(raw: &RawMesh) -> Result<MeshData, ImportError> {
    let mesh = &raw.name;
    let required = |attr: &str| {
        raw.prop(attr).ok_or_else(|| {
            usd_error(
                raw.line,
                format!("Mesh \"{mesh}\" has no '{attr}' attribute"),
            )
        })
    };
    let unauthored = |attr: &str, prop: &Property| {
        usd_error(
            prop.line,
            format!(
                "Mesh \"{mesh}\": '{attr}' has no authored default value \
                 (time-sampled geometry is not supported in v1)"
            ),
        )
    };

    if raw.prop("points").is_none() {
        if let Some(sampled) = raw.prop("points.timeSamples") {
            return Err(unauthored("points", sampled));
        }
    }
    let points_prop = required("points")?;
    let points =
        tuples::<3>("points", points_prop)?.ok_or_else(|| unauthored("points", points_prop))?;
    let counts_prop = required("faceVertexCounts")?;
    let counts = ints("faceVertexCounts", counts_prop)?
        .ok_or_else(|| unauthored("faceVertexCounts", counts_prop))?;
    let indices_prop = required("faceVertexIndices")?;
    let face_indices = ints("faceVertexIndices", indices_prop)?
        .ok_or_else(|| unauthored("faceVertexIndices", indices_prop))?;

    if let Some(holes_prop) = raw.prop("holeIndices") {
        if ints("holeIndices", holes_prop)?.is_some_and(|h| !h.is_empty()) {
            return Err(usd_error(
                holes_prop.line,
                format!("Mesh \"{mesh}\": holeIndices are not supported in v1"),
            ));
        }
    }

    let counts = counts
        .iter()
        .enumerate()
        .map(|(face, &n)| {
            if n < 3 {
                Err(usd_error(
                    counts_prop.line,
                    format!(
                        "faceVertexCounts: face {face} has {n} vertices; faces need at least 3"
                    ),
                ))
            } else if n as u64 > face_indices.len() as u64 {
                // Bounding each count also keeps the sum below from overflowing.
                Err(usd_error(
                    counts_prop.line,
                    format!(
                        "faceVertexCounts: face {face} has {n} vertices but faceVertexIndices \
                         has only {} entries",
                        face_indices.len()
                    ),
                ))
            } else {
                Ok(n as usize)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total = counts
        .iter()
        .try_fold(0usize, |acc, &n| acc.checked_add(n))
        .ok_or_else(|| usd_error(counts_prop.line, "faceVertexCounts sum overflows"))?;
    if total != face_indices.len() {
        return Err(usd_error(
            counts_prop.line,
            format!(
                "faceVertexCounts sum to {total} but faceVertexIndices has {} entries",
                face_indices.len()
            ),
        ));
    }
    if total > u32::MAX as usize {
        return Err(usd_error(
            indices_prop.line,
            format!("{total} face corners exceed the 32-bit index range"),
        ));
    }
    let face_indices = face_indices
        .iter()
        .map(|&i| {
            usize::try_from(i)
                .ok()
                .filter(|&i| i < points.len())
                .ok_or_else(|| {
                    usd_error(
                        indices_prop.line,
                        format!(
                            "faceVertexIndices entry {i} is out of range for {} points",
                            points.len()
                        ),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let topo = Topology {
        points: points.len(),
        faces: counts.len(),
        corners: total,
    };
    let normals = match load_primvar::<3>(raw, "primvars:normals", &topo)? {
        Some(normals) => Some(normals),
        None => load_primvar::<3>(raw, "normals", &topo)?,
    };
    let st_name = if raw.prop("primvars:st").is_some() {
        Some("primvars:st")
    } else {
        raw.props
            .iter()
            .find(|(n, p)| {
                n.starts_with("primvars:")
                    && !n.ends_with(":indices")
                    && p.type_name.starts_with("texCoord2")
            })
            .map(|(n, _)| n.as_str())
    };
    let uvs = match st_name {
        Some(name) => load_primvar::<2>(raw, name, &topo)?,
        None => None,
    };
    let left_handed = raw
        .prop("orientation")
        .and_then(Property::authored_value)
        .is_some_and(|v| matches!(v, Value::Str(s) if s == "leftHanded"));

    let mut data = MeshData {
        positions: Vec::with_capacity(total),
        normals: Vec::with_capacity(total),
        uvs: Vec::with_capacity(total),
        indices: Vec::with_capacity((total - 2 * counts.len()) * 3),
        material_names: Vec::new(),
    };
    let mut corner = 0usize;
    for (face, &count) in counts.iter().enumerate() {
        let base = data.positions.len() as u32;
        let face_points = face_indices
            .get(corner..corner + count)
            .ok_or_else(|| ImportError::Usd(format!("face {face} runs past faceVertexIndices")))?;
        for (offset, &point) in face_points.iter().enumerate() {
            let c = corner + offset;
            let position = points.get(point).copied().ok_or_else(|| {
                ImportError::Usd(format!("point {point} out of range at face {face}"))
            })?;
            data.positions.push(position);
            data.uvs.push(match &uvs {
                Some(st) => st.at(face, point, c)?,
                None => [0.0; 2],
            });
            if let Some(normals) = &normals {
                data.normals.push(normals.at(face, point, c)?);
            }
        }
        if normals.is_none() {
            let mut n = face_normal(&data.positions[base as usize..]);
            if left_handed {
                n = [-n[0], -n[1], -n[2]];
            }
            data.normals.extend(std::iter::repeat_n(n, count));
        }
        // Fan triangulation over this face's corners.
        for i in 1..count as u32 - 1 {
            let (b, c) = (base + i, base + i + 1);
            data.indices.extend(if left_handed {
                [base, c, b]
            } else {
                [base, b, c]
            });
        }
        corner += count;
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Write `contents` to a uniquely named file under the system temp dir and
    /// return its path. Unique per call so tests can run in parallel.
    fn write_fixture(file_name: &str, contents: &[u8]) -> std::path::PathBuf {
        let id = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "umber-mesh-usd-test-{}-{id}-{file_name}",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("fixture write must succeed");
        path
    }

    /// Single unit quad in z=0, nested in an Xform, with vertex normals and
    /// faceVarying st.
    const QUAD_USDA: &str = r#"#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Y"
)

def Xform "Root"
{
    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "faceVarying"
        )
        uniform token subdivisionScheme = "none"
    }
}
"#;

    /// Unit cube points (±0.5), indexed by the face lists below.
    const CUBE_POINTS: [[f32; 3]; 8] = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];

    /// Six quads, each counter-clockwise seen from outside (right-handed).
    const CUBE_FACES: [[usize; 4]; 6] = [
        [0, 3, 2, 1], // -Z
        [4, 5, 6, 7], // +Z
        [0, 1, 5, 4], // -Y
        [3, 7, 6, 2], // +Y
        [0, 4, 7, 3], // -X
        [1, 2, 6, 5], // +X
    ];

    /// Outward normal of each entry of [`CUBE_FACES`].
    const CUBE_NORMALS: [[f32; 3]; 6] = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];

    /// The classic 8-point cube; `with_normals` adds uniform (per-face) normals.
    fn cube_usda(with_normals: bool) -> String {
        let normals = if with_normals {
            r#"normal3f[] normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)] (
            interpolation = "uniform"
        )"#
        } else {
            ""
        };
        format!(
            r#"#usda 1.0

def Mesh "Cube"
{{
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
        {normals}
        point3f[] points = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
}}
"#
        )
    }

    /// Syntax the importer must parse and skip: layer doc strings, comments,
    /// prim metadata, xformOps, timeSamples, connections, assets, rels,
    /// variantSets, `None`, an indexed non-`st` UV primvar, and a second Mesh
    /// that must not win.
    const RICH_USDA: &str = r#"#usda 1.0
(
    doc = """A "rich" layer: exercises the syntax the
    importer must skip without understanding it."""
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
)

# A comment line; the parser must ignore it.
def Xform "World" (
    kind = "component"
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    double3 xformOp:translate = (0, 1.5, -2)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    float3 xformOp:scale.timeSamples = {
        0: (1, 1, 1),
        24: (2, 2, 2),
    }
    variantSet "lod" = {
        "high" {
            def Scope "Hi"
            {
            }
        }
    }

    def Material "Mat"
    {
        token outputs:surface.connect = </World/Mat/Shader.outputs:surface>

        def Shader "Shader"
        {
            uniform token info:id = "UsdPreviewSurface"
            asset inputs:file = @./textures/base color.png@
            color3f inputs:diffuseColor = (0.8, 0.1, 0.1)
        }
    }

    def Mesh "Tri" (
        variants = {
            string lod = "high"
        }
    )
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        rel material:binding = </World/Mat>
        point3f[] points = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
        texCoord2f[] primvars:UVMap = [(0, 0), (1, 0)] (
            interpolation = "faceVarying"
        )
        int[] primvars:UVMap:indices = [0, 1, 0]
        uniform token subdivisionScheme = "none"
        custom string userNote = 'single-quoted; with a # not-a-comment'
        float[] primvars:unused = None
    }

    def Mesh "SecondMesh"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(5, 5, 5), (6, 5, 5), (5, 6, 5)]
    }
}
"#;

    fn assert_close(a: [f32; 3], b: [f32; 3]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    fn usd_message(result: Result<MeshData, ImportError>) -> String {
        match result {
            Err(e @ ImportError::Usd(_)) => {
                let message = e.to_string();
                assert!(message.contains("usd"), "message must name usd: {message}");
                message
            }
            other => panic!("expected ImportError::Usd, got {other:?}"),
        }
    }

    #[test]
    fn loads_quad_as_two_fan_triangles() {
        let path = write_fixture("quad.usda", QUAD_USDA.as_bytes());
        let data = load_usda(&path).expect("quad usda must load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(data.vertex_count(), 4);
        assert_eq!(data.triangle_count(), 2);
        assert_eq!(
            data.positions,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0]
            ]
        );
        assert_eq!(data.normals, vec![[0.0, 0.0, 1.0]; 4]);
        assert_eq!(
            data.uvs,
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        );
        assert_eq!(data.indices, vec![0, 1, 2, 0, 2, 3]);
        assert!(data.material_names.is_empty());
    }

    #[test]
    fn loads_cube_with_uniform_normals() {
        let data = parse_usda_bytes(cube_usda(true).as_bytes()).expect("cube must load");

        assert_eq!(data.vertex_count(), 24, "one vertex per face corner");
        assert_eq!(data.triangle_count(), 12);
        assert_eq!(data.normals.len(), 24);
        assert_eq!(data.uvs, vec![[0.0; 2]; 24], "missing st zero-fills");
        for (face, corners) in CUBE_FACES.iter().enumerate() {
            for (offset, &point) in corners.iter().enumerate() {
                let k = face * 4 + offset;
                assert_eq!(data.positions[k], CUBE_POINTS[point]);
                assert_eq!(data.normals[k], CUBE_NORMALS[face]);
            }
            let base = face as u32 * 4;
            assert_eq!(
                data.indices[face * 6..face * 6 + 6],
                [base, base + 1, base + 2, base, base + 2, base + 3]
            );
        }
        assert!(data
            .indices
            .iter()
            .all(|&i| (i as usize) < data.vertex_count()));
    }

    #[test]
    fn missing_normals_fall_back_to_outward_face_normals() {
        let data = parse_usda_bytes(cube_usda(false).as_bytes()).expect("cube must load");

        assert_eq!(data.triangle_count(), 12);
        assert_eq!(data.normals.len(), data.positions.len());
        for (face, corners) in CUBE_FACES.iter().enumerate() {
            let centroid = corners
                .iter()
                .fold(Vec3::ZERO, |acc, &p| acc + Vec3::from(CUBE_POINTS[p]))
                / 4.0;
            for offset in 0..4 {
                let n = data.normals[face * 4 + offset];
                assert!(n.iter().all(|c| c.is_finite()));
                assert!((Vec3::from(n).length() - 1.0).abs() < 1e-5, "unit length");
                assert!(Vec3::from(n).dot(centroid) > 0.0, "outward-facing");
                assert_close(n, CUBE_NORMALS[face]);
            }
        }

        // The quad without its normals attribute: flat +Z.
        let start = QUAD_USDA.find("        normal3f[]").unwrap();
        let end = QUAD_USDA.find("        point3f[]").unwrap();
        let no_normals = format!("{}{}", &QUAD_USDA[..start], &QUAD_USDA[end..]);
        let data = parse_usda_bytes(no_normals.as_bytes()).expect("quad must load");
        for n in &data.normals {
            assert_close(*n, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn left_handed_orientation_flips_winding_and_normals() {
        let start = QUAD_USDA.find("        normal3f[]").unwrap();
        let end = QUAD_USDA.find("        point3f[]").unwrap();
        let left = format!(
            "{}        uniform token orientation = \"leftHanded\"\n{}",
            &QUAD_USDA[..start],
            &QUAD_USDA[end..]
        );
        let data = parse_usda_bytes(left.as_bytes()).expect("quad must load");
        assert_eq!(data.indices, vec![0, 2, 1, 0, 3, 2]);
        for n in &data.normals {
            assert_close(*n, [0.0, 0.0, -1.0]);
        }
    }

    #[test]
    fn skips_unrelated_syntax_and_takes_first_mesh() {
        let data = parse_usda_bytes(RICH_USDA.as_bytes()).expect("rich layer must load");

        assert_eq!(
            data.positions,
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            "first Mesh (Tri) wins over SecondMesh"
        );
        assert_eq!(data.indices, vec![0, 1, 2]);
        assert_eq!(
            data.uvs,
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "indexed texCoord2f primvar used when st is absent"
        );
        for n in &data.normals {
            assert_close(*n, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn truncated_file_errors_with_line_hint() {
        let cut = QUAD_USDA.find("(1, 0, 0), (1, 1").unwrap() + 5;
        let message = usd_message(parse_usda_bytes(&QUAD_USDA.as_bytes()[..cut]));
        assert!(message.contains("line 16"), "{message}");
        assert!(message.contains("end of file"), "{message}");

        let path = write_fixture("truncated.usda", &QUAD_USDA.as_bytes()[..cut]);
        let result = load_usda(&path);
        let _ = std::fs::remove_file(&path);
        assert!(usd_message(result).contains("line"));
    }

    #[test]
    fn bad_syntax_errors_with_line_hint() {
        let missing_comma =
            "#usda 1.0\ndef Mesh \"Bad\"\n{\n    int[] faceVertexCounts = [4 4]\n}\n";
        let message = usd_message(parse_usda_bytes(missing_comma.as_bytes()));
        assert!(message.contains("line 4"), "{message}");

        let stray_char = "#usda 1.0\ndef Mesh \"Bad\" {\n    int[] faceVertexCounts = [3] $\n}\n";
        let message = usd_message(parse_usda_bytes(stray_char.as_bytes()));
        assert!(
            message.contains("line 3") && message.contains('$'),
            "{message}"
        );

        let unterminated = "#usda 1.0\ndef Mesh \"Bad\n{\n}\n";
        let message = usd_message(parse_usda_bytes(unterminated.as_bytes()));
        assert!(message.contains("line 2"), "{message}");

        let no_header = "def Mesh \"M\" {\n}\n";
        let message = usd_message(parse_usda_bytes(no_header.as_bytes()));
        assert!(message.contains("#usda"), "{message}");

        let no_mesh = "#usda 1.0\ndef Xform \"Root\"\n{\n}\n";
        let message = usd_message(parse_usda_bytes(no_mesh.as_bytes()));
        assert!(message.contains("Mesh"), "{message}");
    }

    #[test]
    fn inconsistent_topology_errors() {
        let mesh = |body: &str| format!("#usda 1.0\ndef Mesh \"M\"\n{{\n{body}\n}}\n");
        let points = "point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]";
        let cases = [
            (
                format!("int[] faceVertexCounts = [4]\nint[] faceVertexIndices = [0, 1, 2]\n{points}"),
                "faceVertexCounts",
            ),
            (
                format!("int[] faceVertexCounts = [3]\nint[] faceVertexIndices = [0, 1, 9]\n{points}"),
                "out of range",
            ),
            (
                format!("int[] faceVertexCounts = [2]\nint[] faceVertexIndices = [0, 1]\n{points}"),
                "at least 3",
            ),
            (
                format!(
                    "int[] faceVertexCounts = [9000000000000000000, 9000000000000000000, \
                     9000000000000000000]\nint[] faceVertexIndices = [0, 1, 2]\n{points}"
                ),
                "faceVertexCounts",
            ),
            (
                "int[] faceVertexCounts = [3]\nint[] faceVertexIndices = [0, 1, 2]".to_string(),
                "points",
            ),
            (
                "int[] faceVertexCounts = [3]\nint[] faceVertexIndices = [0, 1, 2]\n\
                 point3f[] points.timeSamples = {\n 0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],\n}"
                    .to_string(),
                "time-sampled",
            ),
            (
                format!(
                    "int[] faceVertexCounts = [3]\nint[] faceVertexIndices = [0, 1, 2]\n{points}\n\
                     texCoord2f[] primvars:st = [(0, 0), (1, 0)] (\n interpolation = \"faceVarying\"\n)"
                ),
                "primvars:st",
            ),
            (
                format!(
                    "int[] faceVertexCounts = [4]\nint[] faceVertexIndices = [0, 1, 2, 3]\n{points}\n\
                     int[] holeIndices = [0]"
                ),
                "holeIndices",
            ),
        ];
        for (body, needle) in cases {
            let message = usd_message(parse_usda_bytes(mesh(&body).as_bytes()));
            assert!(message.contains(needle), "{needle}: {message}");
            assert!(message.contains("line"), "{message}");
        }
    }

    #[test]
    fn every_truncation_returns_without_panicking() {
        for fixture in [QUAD_USDA, RICH_USDA, cube_usda(true).as_str()] {
            for cut in 0..fixture.len() {
                let _ = parse_usda_bytes(&fixture.as_bytes()[..cut]);
            }
        }
        let _ = parse_usda_bytes(&[0xff, 0xfe, 0x00]);
        let _ = parse_usda_bytes(b"#usda 1.0\n\xc3\xa9");
    }

    #[test]
    fn load_dispatches_usd_extensions() {
        let usda = write_fixture("quad.usda", QUAD_USDA.as_bytes());
        let via_load = crate::load(&usda).expect(".usda must route to load_usda");
        let direct = load_usda(&usda).expect("quad usda must load");
        let _ = std::fs::remove_file(&usda);
        assert_eq!(via_load, direct);
        assert_eq!(via_load.triangle_count(), 2);

        // Text `.usd` is sniffed and parsed.
        let text_usd = write_fixture("quad.usd", QUAD_USDA.as_bytes());
        let result = crate::load(&text_usd);
        let _ = std::fs::remove_file(&text_usd);
        assert_eq!(result.expect("text .usd must load").triangle_count(), 2);

        // `.usdc` errors honestly before touching the file.
        let err = crate::load(Path::new("does-not-exist.usdc")).unwrap_err();
        assert!(matches!(err, ImportError::UsdBinary), "{err:?}");
        assert!(err.to_string().contains("usdc (binary)"), "{err}");

        // A binary crate behind a `.usd` extension gets the same error.
        let binary_usd = write_fixture("crate.usd", b"PXR-USDC\x00\x07\x00\x00");
        let result = crate::load(&binary_usd);
        let _ = std::fs::remove_file(&binary_usd);
        assert!(matches!(result, Err(ImportError::UsdBinary)), "{result:?}");
    }
}

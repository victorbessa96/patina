//! The v1 host↔guest wire format (layout prose: `WIRE.md`).
//!
//! Pure, allocation-only encode/decode in both directions — the host
//! encodes the input region and decodes the out region at eval time;
//! the opposite directions exist so tests (and future tooling) can
//! round-trip every region exactly. Constants live in
//! [`crate::wire_consts`], shared byte-identically with the guest SDK.
//!
//! All integers are little-endian. Regions:
//!
//! * **input**: `[u32 MAGIC][u32 w][u32 h][RGBA w*h*4][u32 param_count]`
//!   then per param `[u32 kind][u32 len][len bytes]`, where the payload
//!   is `[u32 name_len][name UTF-8][value]`.
//! * **out**: `[u32 MAGIC][u32 w][u32 h][RGBA w*h*4]`.
//! * **node def**: `[u32 NODE_DEF_MAGIC][u32 n_inputs][u32 n_params]
//!   [u32 name_len][name UTF-8]`.

use crate::wire_consts::{
    IMAGE_HEADER_LEN, KIND_ASSET, KIND_BOOL, KIND_COLOR, KIND_FLOAT, KIND_INT, KIND_VEC2,
    KIND_VEC3, MAX_INPUTS, MAX_NAME_LEN, MAX_PARAMS, NODE_DEF_HEADER_LEN, NODE_DEF_MAGIC,
    WIRE_MAGIC,
};
use umber_graph::{ImageBuffer, ParamValue};

/// Everything a region can be wrong about. The payload names the
/// offending field (static — wire errors are structural, not data).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// The leading magic isn't `MBER` / `MDEF`.
    #[error("bad magic in {0}")]
    BadMagic(&'static str),
    /// The region ends before the field it promises.
    #[error("truncated {0}")]
    Truncated(&'static str),
    /// Bytes left over after the last field.
    #[error("trailing bytes after {0}")]
    Trailing(&'static str),
    /// A field holds a value v1 rejects (bad kind, too many params,
    /// non-UTF-8 name, zero dims, …).
    #[error("invalid {0}")]
    Invalid(&'static str),
}

/// A plugin's self-description, read from its `umber_node_def` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDef {
    /// The `node_def` name the plugin registers under.
    pub name: String,
    /// Image inputs consumed (v1: exactly [`MAX_INPUTS`]).
    pub n_inputs: u32,
    /// Params declared (v1: at most [`MAX_PARAMS`]).
    pub n_params: u32,
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// A bounds-checked little-endian reader over one region.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError::Truncated(what))?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or(WireError::Truncated(what))?;
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, WireError> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn f32(&mut self, what: &'static str) -> Result<f32, WireError> {
        self.u32(what).map(f32::from_bits)
    }

    fn finish(&self, what: &'static str) -> Result<(), WireError> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(WireError::Trailing(what))
        }
    }
}

/// `w*h*4`, rejecting zero dims and overflow (a hostile header must not
/// size an allocation).
fn rgba_len(w: u32, h: u32) -> Result<usize, WireError> {
    if w == 0 || h == 0 {
        return Err(WireError::Invalid("image dimensions"));
    }
    (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(WireError::Invalid("image dimensions"))
}

fn encode_image(out: &mut Vec<u8>, img: &ImageBuffer) {
    put_u32(out, WIRE_MAGIC);
    put_u32(out, img.width);
    put_u32(out, img.height);
    out.extend_from_slice(&img.data);
}

fn decode_image(r: &mut Reader<'_>, what: &'static str) -> Result<ImageBuffer, WireError> {
    if r.u32(what)? != WIRE_MAGIC {
        return Err(WireError::BadMagic(what));
    }
    let w = r.u32(what)?;
    let h = r.u32(what)?;
    let len = rgba_len(w, h)?;
    let data = r.take(len, what)?.to_vec();
    ImageBuffer::new(w, h, data).map_err(|_| WireError::Invalid("image dimensions"))
}

/// The params that cross the wire: everything except
/// [`ParamValue::NodeRef`] (references are wiring — the engine resolves
/// them to inputs before dispatch).
pub fn wire_params(params: &[(String, ParamValue)]) -> Vec<&(String, ParamValue)> {
    params
        .iter()
        .filter(|(_, v)| !matches!(v, ParamValue::NodeRef(_)))
        .collect()
}

fn encode_value(out: &mut Vec<u8>, v: &ParamValue) -> Result<u32, WireError> {
    let floats = |out: &mut Vec<u8>, fs: &[f32]| {
        for f in fs {
            put_u32(out, f.to_bits());
        }
    };
    Ok(match v {
        ParamValue::Float(f) => {
            floats(out, &[*f]);
            KIND_FLOAT
        }
        ParamValue::Vec2(a) => {
            floats(out, a);
            KIND_VEC2
        }
        ParamValue::Vec3(a) => {
            floats(out, a);
            KIND_VEC3
        }
        ParamValue::Color(a) => {
            floats(out, a);
            KIND_COLOR
        }
        ParamValue::Int(i) => {
            out.extend_from_slice(&i.to_le_bytes());
            KIND_INT
        }
        ParamValue::Bool(b) => {
            put_u32(out, u32::from(*b));
            KIND_BOOL
        }
        ParamValue::Asset(s) => {
            out.extend_from_slice(s.as_bytes());
            KIND_ASSET
        }
        ParamValue::NodeRef(_) => return Err(WireError::Invalid("param kind (NodeRef)")),
    })
}

/// Encodes the input region: one image + its non-`NodeRef` params.
///
/// # Errors
///
/// [`WireError::Invalid`] for more than [`MAX_PARAMS`] wire params or
/// a name/value longer than a u32 can describe.
pub fn encode_input(
    img: &ImageBuffer,
    params: &[(String, ParamValue)],
) -> Result<Vec<u8>, WireError> {
    let params = wire_params(params);
    if params.len() > MAX_PARAMS as usize {
        return Err(WireError::Invalid("param count (v1 max 4)"));
    }
    let mut out = Vec::with_capacity(IMAGE_HEADER_LEN + img.data.len() + 4 + params.len() * 32);
    encode_image(&mut out, img);
    put_u32(&mut out, params.len() as u32);
    for (name, value) in params {
        let name_len = u32::try_from(name.len()).map_err(|_| WireError::Invalid("param name"))?;
        let mut payload = Vec::with_capacity(4 + name.len() + 12);
        put_u32(&mut payload, name_len);
        payload.extend_from_slice(name.as_bytes());
        let kind = encode_value(&mut payload, value)?;
        let len = u32::try_from(payload.len()).map_err(|_| WireError::Invalid("param value"))?;
        put_u32(&mut out, kind);
        put_u32(&mut out, len);
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

fn decode_value(kind: u32, v: &[u8]) -> Result<ParamValue, WireError> {
    let mut r = Reader::new(v);
    let value = match kind {
        KIND_FLOAT => ParamValue::Float(r.f32("float param")?),
        KIND_VEC2 => ParamValue::Vec2([r.f32("vec2 param")?, r.f32("vec2 param")?]),
        KIND_VEC3 => ParamValue::Vec3([
            r.f32("vec3 param")?,
            r.f32("vec3 param")?,
            r.f32("vec3 param")?,
        ]),
        KIND_COLOR => ParamValue::Color([
            r.f32("color param")?,
            r.f32("color param")?,
            r.f32("color param")?,
        ]),
        KIND_INT => ParamValue::Int(r.u32("int param")? as i32),
        KIND_BOOL => match r.u32("bool param")? {
            0 => ParamValue::Bool(false),
            1 => ParamValue::Bool(true),
            _ => return Err(WireError::Invalid("bool param")),
        },
        KIND_ASSET => {
            let s = r.take(v.len(), "asset param")?;
            ParamValue::Asset(
                std::str::from_utf8(s)
                    .map_err(|_| WireError::Invalid("asset param (UTF-8)"))?
                    .to_string(),
            )
        }
        _ => return Err(WireError::Invalid("param kind")),
    };
    r.finish("param value")?;
    Ok(value)
}

/// Decodes an input region back into its image + params (the guest's
/// view, host-side — used by tests and the SDK cross-check).
///
/// # Errors
///
/// Any [`WireError`]: bad magic, truncation, trailing bytes, unknown
/// kinds, bad UTF-8, more than [`MAX_PARAMS`] params.
pub fn decode_input(bytes: &[u8]) -> Result<(ImageBuffer, Vec<(String, ParamValue)>), WireError> {
    let mut r = Reader::new(bytes);
    let img = decode_image(&mut r, "input image")?;
    let count = r.u32("param count")?;
    if count > MAX_PARAMS {
        return Err(WireError::Invalid("param count (v1 max 4)"));
    }
    let mut params = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let kind = r.u32("param kind")?;
        let len = r.u32("param len")? as usize;
        let mut p = Reader::new(r.take(len, "param payload")?);
        let name_len = p.u32("param name len")? as usize;
        let name = std::str::from_utf8(p.take(name_len, "param name")?)
            .map_err(|_| WireError::Invalid("param name (UTF-8)"))?
            .to_string();
        let value = decode_value(kind, &p.bytes[p.pos..])?;
        params.push((name, value));
    }
    r.finish("input region")?;
    Ok((img, params))
}

/// Encodes an out region (the guest's job, host-side for tests).
#[must_use]
pub fn encode_output(img: &ImageBuffer) -> Vec<u8> {
    let mut out = Vec::with_capacity(IMAGE_HEADER_LEN + img.data.len());
    encode_image(&mut out, img);
    out
}

/// Decodes the out region the guest wrote (exactly `out_len` bytes).
///
/// # Errors
///
/// [`WireError::BadMagic`], [`WireError::Truncated`],
/// [`WireError::Trailing`] (length disagrees with `w*h*4`), or
/// [`WireError::Invalid`] for zero/oversized dimensions.
pub fn decode_output(bytes: &[u8]) -> Result<ImageBuffer, WireError> {
    let mut r = Reader::new(bytes);
    let img = decode_image(&mut r, "out region")?;
    r.finish("out region")?;
    Ok(img)
}

/// The size the host reserves for the out region: v1 filters write an
/// image the size of their input.
#[must_use]
pub fn out_capacity(img: &ImageBuffer) -> usize {
    IMAGE_HEADER_LEN + img.data.len()
}

/// Encodes a node-def record (the SDK builds these at compile time;
/// host-side for tests).
///
/// # Errors
///
/// [`WireError::Invalid`] when the def breaks a v1 limit.
pub fn encode_node_def(def: &NodeDef) -> Result<Vec<u8>, WireError> {
    validate_node_def(def)?;
    let mut out = Vec::with_capacity(NODE_DEF_HEADER_LEN + def.name.len());
    put_u32(&mut out, NODE_DEF_MAGIC);
    put_u32(&mut out, def.n_inputs);
    put_u32(&mut out, def.n_params);
    put_u32(&mut out, def.name.len() as u32);
    out.extend_from_slice(def.name.as_bytes());
    Ok(out)
}

/// Decodes the fixed header of a node-def record, returning the name
/// length still to read (the host reads the header from guest memory
/// first, then exactly `name_len` more bytes).
///
/// # Errors
///
/// [`WireError::BadMagic`] / [`WireError::Truncated`] /
/// [`WireError::Invalid`] (name longer than [`MAX_NAME_LEN`]).
pub fn decode_node_def_header(bytes: &[u8]) -> Result<(u32, u32, usize), WireError> {
    let mut r = Reader::new(bytes);
    if r.u32("node def")? != NODE_DEF_MAGIC {
        return Err(WireError::BadMagic("node def"));
    }
    let n_inputs = r.u32("node def")?;
    let n_params = r.u32("node def")?;
    let name_len = r.u32("node def")?;
    if name_len == 0 || name_len > MAX_NAME_LEN {
        return Err(WireError::Invalid("node def name length (1..=64)"));
    }
    Ok((n_inputs, n_params, name_len as usize))
}

/// Decodes a whole node-def record (header + name).
///
/// # Errors
///
/// Any [`WireError`]; also rejects defs outside the v1 limits.
pub fn decode_node_def(bytes: &[u8]) -> Result<NodeDef, WireError> {
    let header = bytes
        .get(..NODE_DEF_HEADER_LEN)
        .ok_or(WireError::Truncated("node def"))?;
    let (n_inputs, n_params, name_len) = decode_node_def_header(header)?;
    let mut r = Reader::new(&bytes[NODE_DEF_HEADER_LEN..]);
    let name = std::str::from_utf8(r.take(name_len, "node def name")?)
        .map_err(|_| WireError::Invalid("node def name (UTF-8)"))?
        .to_string();
    r.finish("node def")?;
    let def = NodeDef {
        name,
        n_inputs,
        n_params,
    };
    validate_node_def(&def)?;
    Ok(def)
}

fn validate_node_def(def: &NodeDef) -> Result<(), WireError> {
    if def.n_inputs != MAX_INPUTS {
        return Err(WireError::Invalid("node def input count (v1: exactly 1)"));
    }
    if def.n_params > MAX_PARAMS {
        return Err(WireError::Invalid("node def param count (v1 max 4)"));
    }
    if def.name.is_empty() || def.name.len() > MAX_NAME_LEN as usize {
        return Err(WireError::Invalid("node def name length (1..=64)"));
    }
    // The name becomes a registry key and an .mtlx nodedef name: keep it
    // to identifier characters so it can't smuggle XML or whitespace.
    if !def
        .name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(WireError::Invalid(
            "node def name characters ([A-Za-z0-9_-])",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire_consts::{
        ERR_BAD_MAGIC, ERR_GUEST_INTERNAL, ERR_OUT_OF_CAPACITY, ERR_TRUNCATED,
    };

    fn ramp(w: u32, h: u32) -> ImageBuffer {
        let data = (0..w * h * 4).map(|i| (i * 7 % 251) as u8).collect();
        ImageBuffer::new(w, h, data).unwrap()
    }

    fn all_kinds() -> Vec<(String, ParamValue)> {
        vec![
            ("strength".into(), ParamValue::Float(0.625)),
            ("center".into(), ParamValue::Vec2([0.25, -1.5])),
            ("tint".into(), ParamValue::Color([1.0, 0.5, 0.0])),
            ("radius".into(), ParamValue::Int(-3)),
        ]
    }

    #[test]
    fn input_roundtrip_is_exact_including_bytes() {
        let img = ramp(5, 3);
        let params = all_kinds();
        let bytes = encode_input(&img, &params).unwrap();
        let (img2, params2) = decode_input(&bytes).unwrap();
        assert_eq!(img2, img);
        assert_eq!(params2, params);
        // Byte-exact: re-encoding the decoded value reproduces the region.
        assert_eq!(encode_input(&img2, &params2).unwrap(), bytes);
    }

    #[test]
    fn input_layout_matches_the_documented_offsets() {
        let img = ImageBuffer::new(1, 1, vec![1, 2, 3, 4]).unwrap();
        let bytes = encode_input(&img, &[("k".into(), ParamValue::Int(7))]).unwrap();
        let expected: Vec<u8> = [
            &b"MBER"[..],
            &1u32.to_le_bytes(),
            &1u32.to_le_bytes(),
            &[1, 2, 3, 4],
            &1u32.to_le_bytes(), // param_count
            &KIND_INT.to_le_bytes(),
            &9u32.to_le_bytes(), // len = 4 (name_len) + 1 (name) + 4 (i32)
            &1u32.to_le_bytes(),
            b"k",
            &7i32.to_le_bytes(),
        ]
        .concat();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn remaining_kinds_roundtrip() {
        let img = ramp(2, 2);
        let params = vec![
            ("v3".into(), ParamValue::Vec3([1.0, 2.0, 3.0])),
            ("on".into(), ParamValue::Bool(true)),
            ("off".into(), ParamValue::Bool(false)),
            (
                "tex".into(),
                ParamValue::Asset("assets/ab/ünïcode.png".into()),
            ),
        ];
        let bytes = encode_input(&img, &params).unwrap();
        assert_eq!(decode_input(&bytes).unwrap(), (img, params));
    }

    #[test]
    fn nan_bits_survive_byte_exact() {
        // PartialEq can't compare NaN; the bytes can.
        let img = ramp(1, 1);
        let nan = f32::from_bits(0x7FC0_1234);
        let bytes = encode_input(&img, &[("x".into(), ParamValue::Float(nan))]).unwrap();
        let (_, back) = decode_input(&bytes).unwrap();
        match back[0].1 {
            ParamValue::Float(f) => assert_eq!(f.to_bits(), 0x7FC0_1234),
            ref other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn node_refs_never_cross_the_wire() {
        let img = ramp(1, 1);
        let params = vec![
            ("in".into(), ParamValue::NodeRef(9)),
            ("a".into(), ParamValue::Float(1.0)),
        ];
        let (_, back) = decode_input(&encode_input(&img, &params).unwrap()).unwrap();
        assert_eq!(back, vec![("a".into(), ParamValue::Float(1.0))]);
    }

    #[test]
    fn more_than_four_params_is_rejected_both_ways() {
        let img = ramp(1, 1);
        let five: Vec<(String, ParamValue)> = (0..5)
            .map(|i| (format!("p{i}"), ParamValue::Int(i)))
            .collect();
        assert!(matches!(
            encode_input(&img, &five),
            Err(WireError::Invalid(_))
        ));
        // Hand-built region claiming 5 params.
        let mut bytes = encode_input(&img, &[]).unwrap();
        let n = bytes.len();
        bytes[n - 4..].copy_from_slice(&5u32.to_le_bytes());
        assert!(matches!(decode_input(&bytes), Err(WireError::Invalid(_))));
    }

    #[test]
    fn output_roundtrip_is_exact() {
        let img = ramp(7, 2);
        let bytes = encode_output(&img);
        assert_eq!(bytes.len(), out_capacity(&img));
        assert_eq!(&bytes[..4], b"MBER");
        assert_eq!(decode_output(&bytes).unwrap(), img);
    }

    #[test]
    fn malformed_regions_name_their_failure() {
        let img = ramp(2, 2);
        let good = encode_output(&img);

        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(decode_output(&bad), Err(WireError::BadMagic("out region")));
        assert!(matches!(
            decode_output(&good[..good.len() - 1]),
            Err(WireError::Truncated(_))
        ));
        let mut long = good.clone();
        long.push(0);
        assert!(matches!(decode_output(&long), Err(WireError::Trailing(_))));
        // Zero width.
        let mut zero = good.clone();
        zero[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(decode_output(&zero), Err(WireError::Invalid(_))));
        // A hostile header can't size an allocation: u32::MAX² × 4.
        let mut huge = good;
        huge[4..12].copy_from_slice(&[0xFF; 8]);
        assert!(decode_output(&huge).is_err());

        // Input side: unknown kind, bad bool, bad UTF-8 name.
        let mut bytes = encode_input(&img, &[("b".into(), ParamValue::Bool(true))]).unwrap();
        let kind_at = IMAGE_HEADER_LEN + img.data.len() + 4;
        bytes[kind_at..kind_at + 4].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(decode_input(&bytes), Err(WireError::Invalid(_))));
        let mut bytes = encode_input(&img, &[("b".into(), ParamValue::Bool(true))]).unwrap();
        let n = bytes.len();
        bytes[n - 4..].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(decode_input(&bytes), Err(WireError::Invalid(_))));
        let mut bytes = encode_input(&img, &[("b".into(), ParamValue::Bool(true))]).unwrap();
        bytes[kind_at + 12] = 0xFF; // the 1-byte name
        assert!(matches!(decode_input(&bytes), Err(WireError::Invalid(_))));
    }

    #[test]
    fn node_def_roundtrip_and_limits() {
        let def = NodeDef {
            name: "blur5".into(),
            n_inputs: 1,
            n_params: 0,
        };
        let bytes = encode_node_def(&def).unwrap();
        assert_eq!(&bytes[..4], b"MDEF");
        assert_eq!(bytes.len(), NODE_DEF_HEADER_LEN + 5);
        assert_eq!(decode_node_def(&bytes).unwrap(), def);

        for bad in [
            NodeDef {
                n_inputs: 2,
                ..def.clone()
            },
            NodeDef {
                n_params: 5,
                ..def.clone()
            },
            NodeDef {
                name: String::new(),
                ..def.clone()
            },
            NodeDef {
                name: "x".repeat(65),
                ..def.clone()
            },
            NodeDef {
                name: "a b".into(),
                ..def.clone()
            },
            NodeDef {
                name: "<x/>".into(),
                ..def.clone()
            },
        ] {
            assert!(encode_node_def(&bad).is_err(), "{bad:?} must be rejected");
        }
        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert_eq!(
            decode_node_def(&wrong),
            Err(WireError::BadMagic("node def"))
        );
        assert!(matches!(
            decode_node_def(&bytes[..bytes.len() - 1]),
            Err(WireError::Truncated(_))
        ));
    }

    #[test]
    fn error_codes_are_distinct_negatives() {
        let codes = [
            ERR_BAD_MAGIC,
            ERR_TRUNCATED,
            ERR_OUT_OF_CAPACITY,
            ERR_GUEST_INTERNAL,
        ];
        for (i, a) in codes.iter().enumerate() {
            assert!(*a < 0);
            for b in &codes[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}

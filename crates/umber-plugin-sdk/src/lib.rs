//! umber-plugin-sdk — write Umber node plugins for `wasm32-unknown-unknown`.
//!
//! `no_std`, no allocator: a plugin is a pure function from one input
//! image (+ up to four named params) to an output image of the same
//! size, written straight into the host-provided out region. The wire
//! layout is crates/umber-wasm/WIRE.md; [`wire_consts`] is a
//! byte-identical copy of the host's constants file (umber-wasm's tests
//! assert the two copies match).
//!
//! ```ignore
//! use umber_plugin_sdk::{umber_node, Image, ImageMut, NodeError, Params};
//!
//! fn invert(input: &Image<'_>, _p: &Params<'_>, out: &mut ImageMut<'_>) -> Result<(), NodeError> {
//!     for (o, i) in out.data.chunks_exact_mut(4).zip(input.data.chunks_exact(4)) {
//!         o.copy_from_slice(&[255 - i[0], 255 - i[1], 255 - i[2], i[3]]);
//!     }
//!     Ok(())
//! }
//!
//! umber_node!(name: "invert", params: 0, eval: invert);
//! ```
//!
//! [`umber_node!`] emits the two C-ABI exports the host binds:
//! `umber_node_def() -> i32` (a pointer to a static node-def record) and
//! `umber_eval(in_ptr, in_len, out_ptr, out_cap) -> i32` (out_len, or a
//! negative `ERR_*` code). A declarative macro rather than an attribute
//! proc-macro: v1 needs no syntax rewriting, and it keeps the SDK a
//! single dependency-free crate.

#![cfg_attr(not(test), no_std)]

pub mod wire_consts;

use wire_consts::{
    ERR_BAD_MAGIC, ERR_GUEST_INTERNAL, ERR_OUT_OF_CAPACITY, ERR_TRUNCATED, IMAGE_HEADER_LEN,
    KIND_ASSET, KIND_BOOL, KIND_COLOR, KIND_FLOAT, KIND_INT, KIND_VEC2, KIND_VEC3, MAX_INPUTS,
    MAX_NAME_LEN, MAX_PARAMS, NODE_DEF_HEADER_LEN, NODE_DEF_MAGIC, WIRE_MAGIC,
};

/// Why a plugin eval failed; each maps to a wire error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeError {
    /// The input region's magic is wrong.
    BadMagic,
    /// The input region is truncated or structurally malformed.
    Truncated,
    /// The out region is too small for the output image.
    OutOfCapacity,
    /// Anything else (a mistyped param, an image beyond the plugin's
    /// limits, a plugin-specific failure).
    Internal,
}

impl NodeError {
    /// The negative `umber_eval` return code.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::BadMagic => ERR_BAD_MAGIC,
            Self::Truncated => ERR_TRUNCATED,
            Self::OutOfCapacity => ERR_OUT_OF_CAPACITY,
            Self::Internal => ERR_GUEST_INTERNAL,
        }
    }
}

/// The input image: RGBA8, row-major, top row first,
/// `data.len() == width*height*4` (same contract as umber-graph's
/// `ImageBuffer`).
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// RGBA8 bytes.
    pub data: &'a [u8],
}

/// The output image, pre-sized to the input's dimensions (v1 filters
/// transform in place of size). Every byte must be written.
#[derive(Debug)]
pub struct ImageMut<'a> {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// RGBA8 bytes (the out region, after its header).
    pub data: &'a mut [u8],
}

/// A decoded param value (umber-graph's `ParamValue` minus `NodeRef`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Param<'a> {
    /// `KIND_FLOAT`.
    Float(f32),
    /// `KIND_VEC2`.
    Vec2([f32; 2]),
    /// `KIND_VEC3`.
    Vec3([f32; 3]),
    /// `KIND_COLOR` (linear RGB).
    Color([f32; 3]),
    /// `KIND_INT`.
    Int(i32),
    /// `KIND_BOOL`.
    Bool(bool),
    /// `KIND_ASSET` (a path string).
    Asset(&'a str),
}

/// The named params of one eval call, borrowed from the input region.
/// Structure is validated once by [`decode_input`]; lookups can't fail
/// on malformed bytes.
#[derive(Debug, Clone, Copy)]
pub struct Params<'a> {
    tlv: &'a [u8],
    count: u32,
}

fn rd_u32(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_f32(b: &[u8], at: usize) -> Option<f32> {
    rd_u32(b, at).map(f32::from_bits)
}

/// Decodes one TLV entry at `at`: `(name, value, next_offset)`.
fn param_at(tlv: &[u8], at: usize) -> Option<(&str, Param<'_>, usize)> {
    let kind = rd_u32(tlv, at)?;
    let len = rd_u32(tlv, at + 4)? as usize;
    let start = at.checked_add(8)?;
    let end = start.checked_add(len)?;
    let payload = tlv.get(start..end)?;
    let name_len = rd_u32(payload, 0)? as usize;
    let name = core::str::from_utf8(payload.get(4..4usize.checked_add(name_len)?)?).ok()?;
    let v = &payload[4 + name_len..];
    let exact = |n: usize| (v.len() == n).then_some(());
    let value = match kind {
        KIND_FLOAT => {
            exact(4)?;
            Param::Float(rd_f32(v, 0)?)
        }
        KIND_VEC2 => {
            exact(8)?;
            Param::Vec2([rd_f32(v, 0)?, rd_f32(v, 4)?])
        }
        KIND_VEC3 => {
            exact(12)?;
            Param::Vec3([rd_f32(v, 0)?, rd_f32(v, 4)?, rd_f32(v, 8)?])
        }
        KIND_COLOR => {
            exact(12)?;
            Param::Color([rd_f32(v, 0)?, rd_f32(v, 4)?, rd_f32(v, 8)?])
        }
        KIND_INT => {
            exact(4)?;
            Param::Int(rd_u32(v, 0)? as i32)
        }
        KIND_BOOL => {
            exact(4)?;
            match rd_u32(v, 0)? {
                0 => Param::Bool(false),
                1 => Param::Bool(true),
                _ => return None,
            }
        }
        KIND_ASSET => Param::Asset(core::str::from_utf8(v).ok()?),
        _ => return None,
    };
    Some((name, value, end))
}

impl<'a> Params<'a> {
    /// Number of params present.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// True when no params were sent.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The `i`-th param in wire order.
    #[must_use]
    pub fn nth(&self, i: usize) -> Option<(&'a str, Param<'a>)> {
        let mut at = 0;
        for k in 0..self.count as usize {
            let (name, value, next) = param_at(self.tlv, at)?;
            if k == i {
                return Some((name, value));
            }
            at = next;
        }
        None
    }

    /// The param named `name`, if sent.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Param<'a>> {
        (0..self.len())
            .filter_map(|i| self.nth(i))
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
    }

    /// A `Float` param, `default` when absent.
    ///
    /// # Errors
    ///
    /// [`NodeError::Internal`] when present but not a finite `Float`.
    pub fn float(&self, name: &str, default: f32) -> Result<f32, NodeError> {
        match self.get(name) {
            None => Ok(default),
            Some(Param::Float(v)) if v.is_finite() => Ok(v),
            Some(_) => Err(NodeError::Internal),
        }
    }

    /// An `Int` param, `default` when absent.
    ///
    /// # Errors
    ///
    /// [`NodeError::Internal`] when present but not an `Int`.
    pub fn int(&self, name: &str, default: i32) -> Result<i32, NodeError> {
        match self.get(name) {
            None => Ok(default),
            Some(Param::Int(v)) => Ok(v),
            Some(_) => Err(NodeError::Internal),
        }
    }
}

/// Decodes and fully validates an input region.
///
/// # Errors
///
/// [`NodeError::BadMagic`] / [`NodeError::Truncated`] (which also
/// covers malformed params and trailing bytes).
pub fn decode_input(bytes: &[u8]) -> Result<(Image<'_>, Params<'_>), NodeError> {
    if rd_u32(bytes, 0).ok_or(NodeError::Truncated)? != WIRE_MAGIC {
        return Err(NodeError::BadMagic);
    }
    let width = rd_u32(bytes, 4).ok_or(NodeError::Truncated)?;
    let height = rd_u32(bytes, 8).ok_or(NodeError::Truncated)?;
    if width == 0 || height == 0 {
        return Err(NodeError::Truncated);
    }
    let n = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(NodeError::Truncated)?;
    let img_end = IMAGE_HEADER_LEN
        .checked_add(n)
        .ok_or(NodeError::Truncated)?;
    let data = bytes
        .get(IMAGE_HEADER_LEN..img_end)
        .ok_or(NodeError::Truncated)?;
    let count = rd_u32(bytes, img_end).ok_or(NodeError::Truncated)?;
    if count > MAX_PARAMS {
        return Err(NodeError::Truncated);
    }
    let tlv = &bytes[img_end + 4..];
    let mut at = 0;
    for _ in 0..count {
        let (_, _, next) = param_at(tlv, at).ok_or(NodeError::Truncated)?;
        at = next;
    }
    if at != tlv.len() {
        return Err(NodeError::Truncated);
    }
    Ok((
        Image {
            width,
            height,
            data,
        },
        Params { tlv, count },
    ))
}

/// A plugin's eval function, as [`umber_node!`] expects it.
pub type EvalFn = fn(&Image<'_>, &Params<'_>, &mut ImageMut<'_>) -> Result<(), NodeError>;

/// The whole guest side of one `umber_eval` call over safe slices:
/// decode the input, write the out header, run `f` on the out pixels.
/// Returns out_len or a negative `ERR_*` code. (Split from
/// [`eval_raw`] so the host can test it without wasm.)
pub fn eval_slices(input: &[u8], out: &mut [u8], f: EvalFn) -> i32 {
    let (img, params) = match decode_input(input) {
        Ok(v) => v,
        Err(e) => return e.code(),
    };
    let need = IMAGE_HEADER_LEN + img.data.len();
    let Ok(ret) = i32::try_from(need) else {
        return ERR_OUT_OF_CAPACITY;
    };
    let Some(region) = out.get_mut(..need) else {
        return ERR_OUT_OF_CAPACITY;
    };
    let (head, pixels) = region.split_at_mut(IMAGE_HEADER_LEN);
    head[0..4].copy_from_slice(&WIRE_MAGIC.to_le_bytes());
    head[4..8].copy_from_slice(&img.width.to_le_bytes());
    head[8..12].copy_from_slice(&img.height.to_le_bytes());
    let mut om = ImageMut {
        width: img.width,
        height: img.height,
        data: pixels,
    };
    match f(&img, &params, &mut om) {
        Ok(()) => ret,
        Err(e) => e.code(),
    }
}

/// The raw `umber_eval` body [`umber_node!`] emits.
///
/// # Safety
///
/// The host contract (WIRE.md): `[in_ptr, in_ptr+in_len)` and
/// `[out_ptr, out_ptr+out_cap)` are valid, non-overlapping ranges of
/// this module's linear memory, live for the call. Only meaningful on
/// wasm32, where pointers are 32-bit.
pub unsafe fn eval_raw(in_ptr: i32, in_len: i32, out_ptr: i32, out_cap: i32, f: EvalFn) -> i32 {
    if in_len < 0 || out_cap < 0 {
        return ERR_TRUNCATED;
    }
    // SAFETY: the caller upholds the host contract above.
    let (input, out) = unsafe {
        (
            core::slice::from_raw_parts(in_ptr as u32 as usize as *const u8, in_len as usize),
            core::slice::from_raw_parts_mut(out_ptr as u32 as usize as *mut u8, out_cap as usize),
        )
    };
    eval_slices(input, out, f)
}

/// Builds a node-def record at compile time:
/// `[MDEF][n_inputs=1][n_params][name_len][name]`. `N` must equal
/// `NODE_DEF_HEADER_LEN + name.len()`. v1 limits are enforced as
/// compile-time panics (a bad def never builds).
#[must_use]
pub const fn node_def_record<const N: usize>(name: &str, n_params: u32) -> [u8; N] {
    let name = name.as_bytes();
    assert!(
        N == NODE_DEF_HEADER_LEN + name.len(),
        "record size mismatch"
    );
    assert!(
        !name.is_empty() && name.len() <= MAX_NAME_LEN as usize,
        "node name must be 1..=64 bytes"
    );
    assert!(n_params <= MAX_PARAMS, "v1 allows at most 4 params");
    let mut out = [0u8; N];
    let fields = [NODE_DEF_MAGIC, MAX_INPUTS, n_params, name.len() as u32];
    let mut f = 0;
    while f < 4 {
        let b = fields[f].to_le_bytes();
        let mut j = 0;
        while j < 4 {
            out[f * 4 + j] = b[j];
            j += 1;
        }
        f += 1;
    }
    let mut i = 0;
    while i < name.len() {
        let c = name[i];
        assert!(
            c.is_ascii_alphanumeric() || c == b'_' || c == b'-',
            "node name characters must be [A-Za-z0-9_-]"
        );
        out[NODE_DEF_HEADER_LEN + i] = c;
        i += 1;
    }
    out
}

/// Declares a plugin node: emits the `umber_node_def` and `umber_eval`
/// C-ABI exports around an [`EvalFn`].
///
/// ```ignore
/// umber_node!(name: "vignette", params: 1, eval: vignette);
/// ```
///
/// `name` must be a string literal of `[A-Za-z0-9_-]`, 1..=64 bytes;
/// `params` the number of params the node declares (≤ 4). Both are
/// checked at compile time. One `umber_node!` per crate (the exports are
/// fixed symbol names).
#[macro_export]
macro_rules! umber_node {
    (name: $name:literal, params: $params:expr, eval: $eval:path $(,)?) => {
        const _: () = {
            const NAME: &str = $name;
            const LEN: usize = $crate::wire_consts::NODE_DEF_HEADER_LEN + NAME.len();
            static DEF: [u8; LEN] = $crate::node_def_record::<LEN>(NAME, $params);

            #[unsafe(no_mangle)]
            pub extern "C" fn umber_node_def() -> i32 {
                DEF.as_ptr() as usize as i32
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn umber_eval(
                in_ptr: i32,
                in_len: i32,
                out_ptr: i32,
                out_cap: i32,
            ) -> i32 {
                // SAFETY: the host calls this export only under the
                // WIRE.md contract (`eval_raw`'s safety section).
                unsafe { $crate::eval_raw(in_ptr, in_len, out_ptr, out_cap, $eval) }
            }
        };
    };
}

#[cfg(all(target_arch = "wasm32", feature = "panic-handler", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    // A panic becomes a trap; the host reports it as a plugin failure.
    core::arch::wasm32::unreachable()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built input region (independent of umber-wasm's encoder:
    /// the cross-crate check lives in umber-wasm's tests).
    fn region(w: u32, h: u32, params: &[(u32, &str, &[u8])]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"MBER");
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&h.to_le_bytes());
        b.extend((0..w * h * 4).map(|i| i as u8));
        b.extend_from_slice(&(params.len() as u32).to_le_bytes());
        for (kind, name, value) in params {
            b.extend_from_slice(&kind.to_le_bytes());
            b.extend_from_slice(&((4 + name.len() + value.len()) as u32).to_le_bytes());
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(value);
        }
        b
    }

    fn copy(i: &Image<'_>, _: &Params<'_>, o: &mut ImageMut<'_>) -> Result<(), NodeError> {
        o.data.copy_from_slice(i.data);
        Ok(())
    }

    #[test]
    fn decodes_image_and_params() {
        let bytes = region(
            2,
            1,
            &[
                (KIND_FLOAT, "strength", &0.75f32.to_le_bytes()),
                (KIND_ASSET, "tex", b"a/b.png"),
            ],
        );
        let (img, params) = decode_input(&bytes).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.data, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(params.len(), 2);
        assert_eq!(params.float("strength", 0.0), Ok(0.75));
        assert_eq!(params.float("absent", 0.5), Ok(0.5));
        assert_eq!(params.get("tex"), Some(Param::Asset("a/b.png")));
        assert_eq!(params.int("strength", 0), Err(NodeError::Internal));
    }

    #[test]
    fn rejects_malformed_regions_with_codes() {
        let good = region(1, 1, &[(KIND_INT, "r", &2i32.to_le_bytes())]);
        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(decode_input(&bad).unwrap_err(), NodeError::BadMagic);
        assert_eq!(
            decode_input(&good[..good.len() - 1]).unwrap_err(),
            NodeError::Truncated
        );
        let mut long = good.clone();
        long.push(0);
        assert_eq!(decode_input(&long).unwrap_err(), NodeError::Truncated);
        let mut kind = good;
        kind[20..24].copy_from_slice(&77u32.to_le_bytes());
        assert_eq!(decode_input(&kind).unwrap_err(), NodeError::Truncated);
    }

    #[test]
    fn eval_slices_writes_header_and_pixels() {
        let input = region(2, 2, &[]);
        let mut out = vec![0u8; 12 + 16 + 5]; // spare capacity is fine
        let n = eval_slices(&input, &mut out, copy);
        assert_eq!(n, 28);
        assert_eq!(&out[..4], b"MBER");
        assert_eq!(&out[4..12], &[2, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(&out[12..28], &input[12..28]);

        let mut small = vec![0u8; 27];
        assert_eq!(eval_slices(&input, &mut small, copy), ERR_OUT_OF_CAPACITY);
        assert_eq!(eval_slices(b"nope", &mut out, copy), ERR_BAD_MAGIC);
        let fails: EvalFn = |_, _, _| Err(NodeError::Internal);
        assert_eq!(eval_slices(&input, &mut out, fails), ERR_GUEST_INTERNAL);
    }

    /// The macro expands and type-checks on the host too (the exports
    /// are only CALLABLE on wasm32, where pointers are 32-bit — the
    /// runtime path is covered by umber-wasm's fixture tests).
    mod macro_expands {
        use crate::{Image, ImageMut, NodeError, Params};
        fn probe(_: &Image<'_>, _: &Params<'_>, _: &mut ImageMut<'_>) -> Result<(), NodeError> {
            Ok(())
        }
        crate::umber_node!(name: "probe_node", params: 2, eval: probe);
    }

    #[test]
    fn node_def_record_layout() {
        const REC: [u8; 21] = node_def_record::<21>("blur5", 0);
        assert_eq!(&REC[..4], b"MDEF");
        assert_eq!(&REC[4..8], &1u32.to_le_bytes());
        assert_eq!(&REC[8..12], &0u32.to_le_bytes());
        assert_eq!(&REC[12..16], &5u32.to_le_bytes());
        assert_eq!(&REC[16..], b"blur5");
    }
}

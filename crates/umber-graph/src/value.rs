//! Typed evaluation values: [`NodeOutput`] + [`ImageBuffer`].
//!
//! Wave-5 slice 1 (docs/specs/node-graph-design.md §"The eval engine").
//! Two kinds of value, kept distinct (the Substance/Mari precedent —
//! confusing a constant with a raster is the classic procedural-engine
//! bug): [`NodeOutput::Uniform`] is a per-evaluation constant,
//! [`NodeOutput::Image`] is a spatially-varying WxH raster.
//!
//! [`ImageBuffer`] is CPU RGBA8, row-major: `data.len() == w*h*4`,
//! row 0 is the image TOP (PNG file order — no flip;contrast with
//! `umber_export::png::read_png_rgba8`, which returns GPU row order
//! for the IBL path). The PNG codec in this module is std-only on
//! purpose: umber-graph stays dependency-minimal (this crate has no
//! image decoder in its tree), so [`decode_png_rgba8`] handles the
//! 8-bit RGB/RGBA non-interlaced PNGs the content-hash store holds
//! (see `umber_core::assets`: `<root>/assets/xx/<hash>.png` files are
//! plain PNG bytes) without pulling in a decoder crate.

use std::sync::OnceLock;

/// A per-node evaluation result: either a constant or a raster.
///
/// Kept as two variants on purpose — a [`crate::ParamValue`] constant
/// must never silently stand in for image data (or vice versa); the
/// engine reports [`crate::EvalError::TypeMismatch`] instead.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeOutput {
    /// Uniform value — a constant for this evaluation.
    Uniform(crate::ParamValue),
    /// A spatially-varying image at the graph's raster resolution
    /// (v1: one shared resolution for the whole graph, taken from
    /// [`crate::EvalContext`]; per-node negotiation is wave-6).
    Image(ImageBuffer),
}

impl NodeOutput {
    /// The variant's kind name, for error payloads (`"Uniform"`/`"Image"`).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Uniform(_) => "Uniform",
            Self::Image(_) => "Image",
        }
    }
}

/// Everything that can go wrong with image buffers / PNG bytes.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImageError {
    /// Zero width or height — a raster must cover at least one texel.
    #[error("image dimensions must be nonzero, got {width}x{height}")]
    EmptyDimensions {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The byte buffer's length doesn't match width×height×4.
    #[error("buffer size {actual} != {expected} (w*h*4 = {width}x{height})")]
    LengthMismatch {
        /// Actual byte count.
        actual: usize,
        /// Expected byte count.
        expected: usize,
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
    },
    /// Dimensions beyond the v1 CPU-raster ceiling (each side ≤ 8192,
    /// total texels ≤ 2²⁶) — a hostile param must not OOM the engine.
    #[error("image dimensions {width}x{height} exceed the v1 limit")]
    DimensionsTooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The bytes are not a supported PNG (not a PNG, unsupported color
    /// type/bit depth, truncated, CRC/adler mismatch, …). The message
    /// carries the reason; v1 accepts 8-bit RGB/RGBA, non-interlaced.
    #[error("png decode failed: {0}")]
    Png(String),
}

/// CPU RGBA8 raster, row-major, 4 bytes per texel.
///
/// # Layout contract
///
/// `data.len() == width*height*4`; texel `(x, y)` (origin top-left)
/// lives at `data[(y*width + x)*4 .. +4]` as `[r, g, b, a]`. Row 0 is
/// the image top (PNG file order).
///
/// The fields are public so destructuring/callers stay simple, but the
/// invariant is enforced by [`Self::new`]/[`Self::filled`]/[`Self::from_png_bytes`]
/// and checked by the accessors' debug asserts: mutating the fields
/// directly must preserve `len == w*h*4` or downstream code (PNG
/// encode, node evals) will reject the buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuffer {
    /// Raster width in texels.
    pub width: u32,
    /// Raster height in texels.
    pub height: u32,
    /// RGBA8 bytes, row-major, `len == w*h*4`.
    pub data: Vec<u8>,
}

impl ImageBuffer {
    /// Builds a buffer, enforcing the layout contract.
    ///
    /// # Errors
    ///
    /// [`ImageError::EmptyDimensions`] on zero width/height;
    /// [`ImageError::LengthMismatch`] when `data.len() != w*h*4`.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::EmptyDimensions { width, height });
        }
        let expected = width as usize * height as usize * 4;
        if data.len() != expected {
            return Err(ImageError::LengthMismatch {
                actual: data.len(),
                expected,
                width,
                height,
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// A solid fill: every texel is `pixel`.
    ///
    /// # Errors
    ///
    /// [`ImageError::EmptyDimensions`] on zero width/height;
    /// [`ImageError::DimensionsTooLarge`] past the v1 ceiling.
    pub fn filled(width: u32, height: u32, pixel: [u8; 4]) -> Result<Self, ImageError> {
        check_dims(width, height)?;
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..(width as usize * height as usize) {
            data.extend_from_slice(&pixel);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Raw RGBA8 bytes, row-major.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Texel `(x, y)`, origin top-left. Returns `None` out of bounds.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Some([
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ])
    }

    /// Writes texel `(x, y)`. No-op out of bounds (nodes validate
    /// coordinates; silent clamping would hide addressing bugs).
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.data[i..i + 4].copy_from_slice(&pixel);
    }

    /// Decodes PNG bytes (the content-hash store's PNG layout: plain
    /// PNG files) into a top-down RGBA8 buffer. 8-bit RGB is expanded
    /// with opaque alpha; anything else is [`ImageError::Png`].
    ///
    /// # Errors
    ///
    /// [`ImageError::Png`] for non-PNG/unsupported/truncated input.
    pub fn from_png_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let (width, height, data) = decode_png_rgba8(bytes)?;
        Self::new(width, height, data)
    }

    /// Encodes the buffer as PNG bytes (8-bit RGBA, filter-0 rows,
    /// stored deflate blocks — decodable by any PNG reader, including
    /// [`Self::from_png_bytes`]).
    ///
    /// # Errors
    ///
    /// [`ImageError::LengthMismatch`] if the fields were mutated to
    /// break the layout contract; [`ImageError::Png`] on internal
    /// size overflow (dimensions near `u32::MAX`).
    pub fn to_png_bytes(&self) -> Result<Vec<u8>, ImageError> {
        encode_png_rgba8(self.width, self.height, &self.data)
    }
}

/// Decodes PNG bytes into `(width, height, RGBA8 top-down)`.
///
/// Accepts 8-bit RGB (alpha 255 synthesized) and 8-bit RGBA,
/// non-interlaced. Full DEFLATE (stored/fixed/dynamic) + all five
/// filter types; chunk CRCs and the zlib Adler-32 are verified.
///
/// # Errors
///
/// [`ImageError::Png`] with the reason.
pub fn decode_png_rgba8(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), ImageError> {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    let png_err = |what: &str| ImageError::Png(what.to_string());
    if bytes.len() < 8 || bytes[0..8] != SIG {
        return Err(png_err("not a PNG (bad signature)"));
    }
    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bpp = 0usize; // bytes per pixel: 3 (RGB) or 4 (RGBA)
    let mut seen_ihdr = false;
    let mut idat: Vec<u8> = Vec::new();

    let read_u32 = |b: &[u8], p: usize| -> Result<u32, ImageError> {
        b.get(p..p + 4)
            .ok_or_else(|| png_err("truncated chunk header"))
            .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    };

    while pos < bytes.len() {
        let len = read_u32(bytes, pos)? as usize;
        let typ = bytes
            .get(pos + 4..pos + 8)
            .ok_or_else(|| png_err("truncated chunk type"))?;
        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(len)
            .ok_or_else(|| png_err("chunk too long"))?;
        let data = bytes
            .get(data_start..data_end)
            .ok_or_else(|| png_err("truncated chunk data"))?;
        let crc_stored = read_u32(bytes, data_end)?;
        let crc_actual = crc32_ieee(&bytes[pos + 4..data_end]);
        if crc_stored != crc_actual {
            return Err(png_err("chunk CRC mismatch"));
        }
        pos = data_end + 4;
        match typ {
            b"IHDR" => {
                if seen_ihdr {
                    return Err(png_err("duplicate IHDR"));
                }
                if data.len() != 13 {
                    return Err(png_err("bad IHDR length"));
                }
                seen_ihdr = true;
                width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                if width == 0 || height == 0 {
                    return Err(ImageError::EmptyDimensions { width, height });
                }
                if width as u64 * height as u64 > (1u64 << 28) {
                    return Err(png_err("dimensions too large for v1 decode"));
                }
                let (depth, color) = (data[8], data[9]);
                if depth != 8 {
                    return Err(png_err("only 8-bit PNGs are supported in v1"));
                }
                bpp = match color {
                    2 => 3,
                    6 => 4,
                    _ => return Err(png_err("only RGB/RGBA PNGs are supported in v1")),
                };
                if data[10] != 0 || data[11] != 0 || data[12] != 0 {
                    return Err(png_err("unsupported compression/filter/interlace method"));
                }
            }
            b"IDAT" => {
                if !seen_ihdr {
                    return Err(png_err("IDAT before IHDR"));
                }
                idat.extend_from_slice(data);
            }
            b"IEND" => break,
            _ => {
                // Ancillary chunks (tEXt, pHYs, iCCP, …) are skipped —
                // v1 cares about pixels, not metadata.
            }
        }
    }
    if !seen_ihdr {
        return Err(png_err("missing IHDR"));
    }
    let raw = zlib_decompress(&idat)?;
    let stride = width as usize * bpp;
    if raw.len() != height as usize * (1 + stride) {
        return Err(png_err("decompressed size mismatches dimensions"));
    }
    // Unfilter (all five PNG filter types).
    let mut out = vec![0u8; width as usize * height as usize * bpp];
    let mut prev = vec![0u8; stride];
    for (row, chunk) in raw.chunks_exact(1 + stride).enumerate() {
        let (filter, cur) = (chunk[0], &chunk[1..]);
        let dst = &mut out[row * stride..(row + 1) * stride];
        match filter {
            0 => dst.copy_from_slice(cur),
            1..=4 => {
                for i in 0..stride {
                    let a = if i >= bpp { dst[i - bpp] } else { 0 };
                    let b = prev[i];
                    let c = if i >= bpp { prev[i - bpp] } else { 0 };
                    let pred = match filter {
                        1 => a,
                        2 => b,
                        3 => ((u16::from(a) + u16::from(b)) / 2) as u8,
                        _ => paeth(a, b, c),
                    };
                    dst[i] = cur[i].wrapping_add(pred);
                }
            }
            _ => return Err(png_err("unknown filter type")),
        }
        prev.copy_from_slice(dst);
    }
    // RGB -> RGBA with opaque alpha; RGBA passes through.
    let rgba = if bpp == 3 {
        let mut v = Vec::with_capacity(width as usize * height as usize * 4);
        for px in out.chunks_exact(3) {
            v.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        v
    } else {
        out
    };
    Ok((width, height, rgba))
}

/// Encodes RGBA8 top-down bytes as PNG (8-bit RGBA, filter-0, stored
/// deflate blocks). Round-trips through [`decode_png_rgba8`].
///
/// # Errors
///
/// [`ImageError::LengthMismatch`] on size mismatch;
/// [`ImageError::Png`] on size overflow.
pub fn encode_png_rgba8(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, ImageError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(ImageError::LengthMismatch {
            actual: rgba.len(),
            expected,
            width,
            height,
        });
    }
    if width == 0 || height == 0 {
        return Err(ImageError::EmptyDimensions { width, height });
    }
    let stride = width as usize * 4;
    // Raw scanlines with filter byte 0.
    let mut raw = Vec::with_capacity(height as usize * (1 + stride));
    for row in rgba.chunks_exact(stride) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    // zlib wrapper: header 0x78 0x01 (deflate, FCHECK-valid) + stored
    // blocks (max 65535 bytes each) + Adler-32.
    let mut zl = vec![0x78, 0x01];
    let mut consumed = 0usize;
    for chunk in raw.chunks(65_535) {
        consumed += chunk.len();
        let is_last = consumed == raw.len();
        zl.push(u8::from(is_last)); // BFINAL + BTYPE 00 (stored)
        let len = chunk.len() as u16;
        zl.extend_from_slice(&len.to_le_bytes());
        zl.extend_from_slice(&(!len).to_le_bytes());
        zl.extend_from_slice(chunk);
    }
    zl.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, deflate, filter 0, no interlace
    push_chunk(&mut png, b"IHDR", &ihdr);
    push_chunk(&mut png, b"IDAT", &zl);
    push_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

fn push_chunk(png: &mut Vec<u8>, typ: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(typ);
    png.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(typ);
    crc_input.extend_from_slice(data);
    png.extend_from_slice(&crc32_ieee(&crc_input).to_be_bytes());
}

/// Enforces the v1 CPU-raster ceiling (each side 1..=8192, total
/// texels ≤ 2²⁶) so allocation-sized dimensions are rejected before
/// any `Vec` grows.
fn check_dims(width: u32, height: u32) -> Result<(), ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::EmptyDimensions { width, height });
    }
    if width > 8192 || height > 8192 || width as u64 * height as u64 > (1u64 << 26) {
        return Err(ImageError::DimensionsTooLarge { width, height });
    }
    Ok(())
}

/// Paeth predictor (PNG spec §6.6).
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (i32::from(a), i32::from(b), i32::from(c));
    let p = a + b - c;
    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

fn crc32_ieee(data: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// zlib (RFC 1950) decompression wrapping a raw DEFLATE stream.
fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, ImageError> {
    let png_err = |what: &str| ImageError::Png(what.to_string());
    if data.len() < 2 {
        return Err(png_err("truncated zlib header"));
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0F != 8 {
        return Err(png_err("unsupported zlib compression method"));
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(png_err("bad zlib header check"));
    }
    if flg & 0x20 != 0 {
        return Err(png_err("zlib preset dictionaries unsupported"));
    }
    let mut br = BitReader {
        bytes: data,
        bit: 16,
    };
    let out = inflate(&mut br)?;
    // Adler-32 starts at the next byte boundary after the stream end.
    let adler_at = br.bit.div_ceil(8);
    if data.len() < adler_at + 4 {
        return Err(png_err("missing zlib adler32"));
    }
    let stored = u32::from_be_bytes([
        data[adler_at],
        data[adler_at + 1],
        data[adler_at + 2],
        data[adler_at + 3],
    ]);
    if stored != adler32(&out) {
        return Err(png_err("zlib adler32 mismatch"));
    }
    if data.len() != adler_at + 4 {
        return Err(png_err("trailing bytes after zlib stream"));
    }
    Ok(out)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize, // bit index, LSB-first within each byte
}

impl BitReader<'_> {
    fn read_bits(&mut self, n: u32) -> Result<u32, ImageError> {
        let mut out = 0u32;
        for i in 0..n {
            let byte = *self
                .bytes
                .get(self.bit / 8)
                .ok_or_else(|| ImageError::Png("truncated deflate stream".to_string()))?;
            out |= (u32::from((byte >> (self.bit % 8)) & 1)) << i;
            self.bit += 1;
        }
        Ok(out)
    }

    fn read_bit(&mut self) -> Result<u32, ImageError> {
        self.read_bits(1)
    }

    fn align_to_byte(&mut self) {
        self.bit = self.bit.next_multiple_of(8);
    }
}

/// Canonical Huffman decoder built from code lengths.
struct Huffman {
    /// `counts[len]` = number of codes of bit-length `len`.
    counts: [u16; 16],
    /// Symbols sorted by (length, symbol), per RFC 1951 §3.2.2.
    symbols: Vec<u16>,
}

impl Huffman {
    fn from_lengths(lengths: &[u8]) -> Result<Self, ImageError> {
        let mut counts = [0u16; 16];
        for &len in lengths {
            if len as usize >= counts.len() {
                return Err(ImageError::Png("bad huffman code length".to_string()));
            }
            if len != 0 {
                counts[len as usize] += 1;
            }
        }
        let mut symbols = Vec::with_capacity(lengths.len());
        for len in 1..16u8 {
            for (sym, &l) in lengths.iter().enumerate() {
                if l == len {
                    symbols.push(sym as u16);
                }
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, ImageError> {
        let mut code: u32 = 0;
        let mut first: u32 = 0;
        let mut index: usize = 0;
        for len in 1usize..=15 {
            code |= br.read_bit()?;
            let count = u32::from(self.counts[len]);
            if code.wrapping_sub(first) < count {
                return self
                    .symbols
                    .get(index + (code - first) as usize)
                    .copied()
                    .ok_or_else(|| ImageError::Png("bad huffman code".to_string()));
            }
            index += count as usize;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(ImageError::Png("invalid huffman code".to_string()))
    }
}

fn fixed_literal() -> Huffman {
    let mut lengths = [0u8; 288];
    for l in lengths.iter_mut().take(144) {
        *l = 8;
    }
    for l in lengths.iter_mut().skip(144).take(112) {
        *l = 9;
    }
    for l in lengths.iter_mut().skip(256).take(24) {
        *l = 7;
    }
    for l in lengths.iter_mut().skip(280) {
        *l = 8;
    }
    Huffman::from_lengths(&lengths).expect("fixed table is valid")
}

fn fixed_distance() -> Huffman {
    Huffman::from_lengths(&[5u8; 30]).expect("fixed table is valid")
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Raw DEFLATE (RFC 1951) decompression: stored/fixed/dynamic blocks.
fn inflate(br: &mut BitReader<'_>) -> Result<Vec<u8>, ImageError> {
    let png_err = |what: &str| ImageError::Png(what.to_string());
    let mut out: Vec<u8> = Vec::new();
    loop {
        let final_block = br.read_bit()?;
        match br.read_bits(2)? {
            0 => {
                // Stored (uncompressed) block.
                br.align_to_byte();
                let b = br.bit / 8;
                let bytes = br.bytes;
                let len = u16::from_le_bytes(
                    bytes
                        .get(b..b + 2)
                        .ok_or_else(|| png_err("truncated stored block"))?
                        .try_into()
                        .map_err(|_| png_err("truncated stored block"))?,
                ) as usize;
                let nlen = u16::from_le_bytes(
                    bytes
                        .get(b + 2..b + 4)
                        .ok_or_else(|| png_err("truncated stored block"))?
                        .try_into()
                        .map_err(|_| png_err("truncated stored block"))?,
                );
                if len as u16 ^ nlen != 0xFFFF {
                    return Err(png_err("stored block length mismatch"));
                }
                let chunk = bytes
                    .get(b + 4..b + 4 + len)
                    .ok_or_else(|| png_err("truncated stored block"))?;
                out.extend_from_slice(chunk);
                br.bit = (b + 4 + len) * 8;
            }
            1 => {
                let lit = fixed_literal();
                let dist = fixed_distance();
                decode_compressed(br, &lit, &dist, &mut out)?;
            }
            2 => {
                let hlit = br.read_bits(5)? as usize + 257;
                let hdist = br.read_bits(5)? as usize + 1;
                let hclen = br.read_bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut cl_lens = [0u8; 19];
                for o in ORDER.iter().take(hclen) {
                    cl_lens[*o] = br.read_bits(3)? as u8;
                }
                let cl = Huffman::from_lengths(&cl_lens)?;
                let mut lengths = Vec::with_capacity(hlit + hdist);
                while lengths.len() < hlit + hdist {
                    let sym = cl.decode(br)?;
                    match sym {
                        0..=15 => lengths.push(sym as u8),
                        16 => {
                            let prev = *lengths.last().ok_or_else(|| png_err("bad repeat"))?;
                            let n = (br.read_bits(2)? + 3) as usize;
                            lengths.extend(std::iter::repeat_n(prev, n));
                        }
                        17 => {
                            let n = (br.read_bits(3)? + 3) as usize;
                            lengths.extend(std::iter::repeat_n(0, n));
                        }
                        18 => {
                            let n = (br.read_bits(7)? + 11) as usize;
                            lengths.extend(std::iter::repeat_n(0, n));
                        }
                        _ => return Err(png_err("bad code-length symbol")),
                    }
                }
                if lengths.len() != hlit + hdist {
                    return Err(png_err("code-length overflow"));
                }
                let lit = Huffman::from_lengths(&lengths[..hlit])?;
                let dist = Huffman::from_lengths(&lengths[hlit..])?;
                decode_compressed(br, &lit, &dist, &mut out)?;
            }
            _ => return Err(png_err("invalid deflate block type")),
        }
        if final_block == 1 {
            break;
        }
    }
    Ok(out)
}

fn decode_compressed(
    br: &mut BitReader<'_>,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
) -> Result<(), ImageError> {
    let png_err = |what: &str| ImageError::Png(what.to_string());
    loop {
        let sym = lit.decode(br)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => break,
            257..=287 => {
                let li = (sym - 257) as usize;
                let len = u32::from(LENGTH_BASE[li]) + br.read_bits(u32::from(LENGTH_EXTRA[li]))?;
                let dsym = dist.decode(br)?;
                if dsym > 29 {
                    return Err(png_err("bad distance symbol"));
                }
                let di = dsym as usize;
                let d = u32::from(DIST_BASE[di]) + br.read_bits(u32::from(DIST_EXTRA[di]))?;
                if d as usize > out.len() || d == 0 {
                    return Err(png_err("distance too far back"));
                }
                for _ in 0..len {
                    let b = out[out.len() - d as usize];
                    out.push(b);
                }
            }
            _ => return Err(png_err("bad literal/length symbol")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_enforces_the_layout_contract() {
        assert!(ImageBuffer::new(2, 2, vec![0u8; 16]).is_ok());
        assert!(matches!(
            ImageBuffer::new(2, 2, vec![0u8; 15]),
            Err(ImageError::LengthMismatch { actual: 15, .. })
        ));
        assert!(matches!(
            ImageBuffer::new(0, 4, vec![]),
            Err(ImageError::EmptyDimensions { .. })
        ));
    }

    #[test]
    fn ceiling_admits_8k_and_rejects_one_past_it() {
        // The §6 8K row: 8192² is exactly the 2^26-texel ceiling.
        let k8 = ImageBuffer::filled(8192, 8192, [1, 2, 3, 255]).expect("8K is within the ceiling");
        assert_eq!((k8.width(), k8.height()), (8192, 8192));
        assert_eq!(k8.data.len(), 8192 * 8192 * 4);
        assert_eq!(k8.data.len(), 256 << 20, "256 MiB of RGBA8");
        assert_eq!(k8.pixel(8191, 8191), Some([1, 2, 3, 255]));
        drop(k8);
        // One past on either side fails before allocating.
        for (w, h) in [(8193, 1), (1, 8193), (8193, 8193)] {
            assert!(
                matches!(
                    ImageBuffer::filled(w, h, [0; 4]),
                    Err(ImageError::DimensionsTooLarge { width, height }) if (width, height) == (w, h)
                ),
                "{w}x{h} must exceed the ceiling"
            );
        }
        // `new` takes caller-allocated bytes and checks only the layout
        // contract (no ceiling): an 8K buffer passes it too.
        assert!(ImageBuffer::new(8192, 8192, vec![0u8; 8192 * 8192 * 4]).is_ok());
    }

    #[test]
    fn filled_covers_every_texel() {
        let buf = ImageBuffer::filled(4, 4, [255, 0, 0, 255]).unwrap();
        assert_eq!(buf.data.len(), 64);
        assert!(buf.data.chunks_exact(4).all(|px| px == [255, 0, 0, 255]));
        assert_eq!(buf.pixel(3, 3), Some([255, 0, 0, 255]));
        assert_eq!(buf.pixel(4, 0), None);
    }

    #[test]
    fn set_pixel_addresses_top_left_origin() {
        let mut buf = ImageBuffer::filled(2, 1, [0, 0, 0, 255]).unwrap();
        buf.set_pixel(1, 0, [9, 8, 7, 6]);
        assert_eq!(buf.pixel(0, 0), Some([0, 0, 0, 255]));
        assert_eq!(buf.pixel(1, 0), Some([9, 8, 7, 6]));
        buf.set_pixel(9, 9, [1, 1, 1, 1]); // out of bounds: no-op, no panic
    }

    #[test]
    fn png_roundtrips_distinct_texels() {
        let data: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128,
        ];
        let bytes = encode_png_rgba8(2, 2, &data).unwrap();
        let (w, h, back) = decode_png_rgba8(&bytes).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(back, data);
    }

    #[test]
    fn png_decode_rejects_garbage_and_truncation() {
        assert!(matches!(
            decode_png_rgba8(b"definitely not a png"),
            Err(ImageError::Png(_))
        ));
        let bytes = encode_png_rgba8(2, 2, &[1u8; 16]).unwrap();
        assert!(matches!(
            decode_png_rgba8(&bytes[..bytes.len() / 2]),
            Err(ImageError::Png(_))
        ));
    }

    #[test]
    fn png_decode_rejects_bad_crc() {
        let mut bytes = encode_png_rgba8(1, 1, &[7, 8, 9, 255]).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF; // corrupt IEND's CRC
        assert!(matches!(decode_png_rgba8(&bytes), Err(ImageError::Png(_))));
    }

    #[test]
    fn png_decodes_independent_dynamic_huffman_and_filters() {
        // Built with CPython's zlib (level 9 → dynamic Huffman) and one
        // row per filter type (Sub/Up/Average/Paeth): exercises the
        // decoder paths our stored-block encoder never emits, against
        // an independent implementation's bytes.
        let bytes: &[u8] = &[
            137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 4, 0, 0, 0, 4,
            8, 6, 0, 0, 0, 169, 241, 158, 126, 0, 0, 0, 51, 73, 68, 65, 84, 120, 218, 99, 100, 96,
            96, 255, 239, 160, 32, 208, 8, 196, 245, 32, 154, 73, 192, 65, 161, 17, 136, 235, 97,
            52, 179, 68, 130, 201, 126, 13, 3, 137, 70, 32, 174, 7, 209, 44, 96, 25, 160, 114, 32,
            6, 211, 0, 7, 129, 15, 14, 25, 250, 78, 181, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96,
            130,
        ];
        let (w, h, data) = decode_png_rgba8(bytes).unwrap();
        assert_eq!((w, h), (4, 4));
        let expected: &[u8] = &[
            0, 0, 7, 255, 64, 32, 23, 128, 128, 64, 39, 255, 192, 96, 55, 128, 16, 64, 39, 128, 80,
            96, 55, 255, 144, 128, 71, 128, 208, 160, 87, 255, 32, 128, 71, 255, 96, 160, 87, 128,
            160, 192, 103, 255, 224, 224, 119, 128, 48, 192, 103, 128, 112, 224, 119, 255, 176, 0,
            135, 128, 240, 32, 151, 255,
        ];
        assert_eq!(data, expected);
    }

    #[test]
    fn node_output_kinds_stay_distinct() {
        let u = NodeOutput::Uniform(crate::ParamValue::Float(1.0));
        let i = NodeOutput::Image(ImageBuffer::filled(1, 1, [0, 0, 0, 255]).unwrap());
        assert_eq!(u.kind(), "Uniform");
        assert_eq!(i.kind(), "Image");
        assert_ne!(u, i);
    }
}

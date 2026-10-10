//! PNG image encoding for exported paint maps.
//!
//! Encodes RGBA8 paint-target data (the `PaintTarget` readback format:
//! width×height×4 bytes, row-major, no padding) into a PNG file. Pure
//! CPU — no GPU types — so it is testable on any platform and usable
//! from any caller with raw bytes.
//!
//! Format decisions (docs/specs/requirements.md §6):
//! - 8-bit RGBA output via the `png` crate (Wave 2 scope; 16-bit +
//!   EXR float arrive in Wave 3).
//! - SRGB transfer applied at export time when `Srgb` is requested:
//!   the paint buffer is linear rgba8unorm; PNG consumers expect sRGB
//!   encoding for basecolor-class maps.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Errors from encoding or writing a PNG.
#[derive(Debug, thiserror::Error)]
pub enum PngError {
    /// The byte buffer's length doesn't match width×height×4.
    #[error("buffer size {actual} != {expected} (w*h*4 = {width}x{height})")]
    SizeMismatch {
        /// Actual byte count.
        actual: usize,
        /// Expected byte count.
        expected: usize,
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
    },
    /// The encoder rejected the stream.
    #[error("png encode failed: {0}")]
    Encode(String),
    /// The decoder rejected the stream (not a PNG, unsupported color
    /// type/bit depth, truncated file).
    #[error("png decode failed: {0}")]
    Decode(String),
    /// The OS refused the write (missing dir, permissions, …).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Output color transfer for the encoded PNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Transfer {
    /// Write the bytes as-is (linear data in a linear-tagged PNG).
    #[default]
    Linear,
    /// Apply the sRGB transfer curve to RGB channels (alpha untouched).
    Srgb,
}

/// Encodes `rgba` (width×height×4, row-major) into a PNG at `path`,
/// creating parent directories as needed.
///
/// # Errors
///
/// [`PngError::SizeMismatch`] when the buffer doesn't match the
/// dimensions; see the other variants for encode/io failures.
pub fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
    transfer: Transfer,
) -> Result<(), PngError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(PngError::SizeMismatch {
            actual: rgba.len(),
            expected,
            width,
            height,
        });
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    if transfer == Transfer::Srgb {
        // Tag the chunk so viewers decode as sRGB; the pixel values
        // themselves are encoded below.
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    }
    let mut writer = encoder
        .write_header()
        .map_err(|e| PngError::Encode(e.to_string()))?;

    let data: Vec<u8> = match transfer {
        Transfer::Linear => rgba.to_vec(),
        Transfer::Srgb => {
            let mut out = rgba.to_vec();
            for px in out.chunks_exact_mut(4) {
                px[0] = linear_to_srgb_u8(px[0]);
                px[1] = linear_to_srgb_u8(px[1]);
                px[2] = linear_to_srgb_u8(px[2]);
                // alpha untouched
            }
            out
        }
    };
    writer
        .write_image_data(&data)
        .map_err(|e| PngError::Encode(e.to_string()))?;
    Ok(())
}

/// Reads a PNG file into RGBA8 bytes (width×height×4, row-major).
///
/// Accepted inputs: 8-bit RGBA (returned as-is) and 8-bit RGB (opaque
/// alpha is synthesized). Anything else (16-bit, gray, palette,
/// non-PNG bytes) is a [`PngError::Decode`] — Wave-4 IBL scope is LDR
/// equirects; 16-bit PNG support rides the bit-depth follow-up.
///
/// # Row order (the environment-map contract)
///
/// The returned rows are in *GPU row order*: output row 0 is the PNG's
/// LAST stored row (the displayed bottom), output row H-1 is the PNG's
/// first stored row (the displayed top). Rationale: umber-gpu's
/// equirect convention (`equirect_uv(n) = (atan2/2π+0.5,
/// asin(n.y)/π+0.5)`, sampled with a v=0-first-row sampler) maps +Y to
/// the last data row, so a viewer-conventional PNG (zenith/sky at the
/// displayed top) lands on +Y with no further flipping. See
/// `umber_gpu::ibl`'s convention block. A plain image viewer and the
/// renderer therefore agree about which end is up.
///
/// # Errors
///
/// [`PngError::Decode`] for unreadable/unsupported files;
/// [`PngError::Io`] from the filesystem.
pub fn read_png_rgba8(path: &Path) -> Result<(u32, u32, Vec<u8>), PngError> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .map_err(|e| PngError::Decode(e.to_string()))?;
    let (width, height) = {
        let info = reader.info();
        (info.width, info.height)
    };
    let color = reader.info().color_type;
    let depth = reader.info().bit_depth;
    let rgb_source = match (color, depth) {
        (png::ColorType::Rgba, png::BitDepth::Eight) => false,
        (png::ColorType::Rgb, png::BitDepth::Eight) => true,
        _ => {
            return Err(PngError::Decode(format!(
                "unsupported PNG: {color:?} at {depth:?} (want 8-bit RGB or RGBA)"
            )));
        }
    };

    let mut raw = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut raw)
        .map_err(|e| PngError::Decode(e.to_string()))?;

    let rgba: Vec<u8> = if rgb_source {
        let mut out = Vec::with_capacity(width as usize * height as usize * 4);
        for px in raw.chunks_exact(3) {
            out.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        out
    } else {
        raw
    };

    // Flip to GPU row order (see this function's doc comment): output
    // row 0 = the file's last stored row.
    let stride = width as usize * 4;
    let mut flipped = vec![0u8; rgba.len()];
    for (dst_row, src_row) in (0..height as usize).zip((0..height as usize).rev()) {
        flipped[dst_row * stride..(dst_row + 1) * stride]
            .copy_from_slice(&rgba[src_row * stride..(src_row + 1) * stride]);
    }
    Ok((width, height, flipped))
}

/// Linear (0..255) → sRGB (0..255), the canonical 2.2-gamma-style
/// transfer from IEC 61966-2-1 rounded to nearest.
fn linear_to_srgb_u8(linear: u8) -> u8 {
    let l = linear as f32 / 255.0;
    let s = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Linear (0..65535) → sRGB (0..65535): the same IEC 61966-2-1 curve
/// at 16-bit precision (the 8-bit version's exact sibling).
fn linear_to_srgb_u16(linear: u16) -> u16 {
    let l = linear as f32 / 65_535.0;
    let s = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 65_535.0 + 0.5).clamp(0.0, 65_535.0) as u16
}

/// Encodes `rgba` (width×height×4 **u16**, row-major) into a 16-bit
/// PNG at `path` — the `Png16` half of the bit-depth requirement
/// (§6: "Bit depths 8/16/32F").
///
/// The u16 values are written as-is under [`Transfer::Linear`]; under
/// [`Transfer::Srgb`] the RGB channels take the 16-bit sRGB curve
/// (alpha untouched), mirroring [`write_png`]'s contract.
///
/// # Errors
///
/// [`PngError::SizeMismatch`] when the slice doesn't match the
/// dimensions; see the other variants for encode/io failures.
pub fn write_png16(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u16],
    transfer: Transfer,
) -> Result<(), PngError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(PngError::SizeMismatch {
            actual: rgba.len(),
            expected,
            width,
            height,
        });
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Sixteen);
    if transfer == Transfer::Srgb {
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    }
    let mut writer = encoder
        .write_header()
        .map_err(|e| PngError::Encode(e.to_string()))?;

    let mut data = rgba.to_vec();
    if transfer == Transfer::Srgb {
        for px in data.chunks_exact_mut(4) {
            px[0] = linear_to_srgb_u16(px[0]);
            px[1] = linear_to_srgb_u16(px[1]);
            px[2] = linear_to_srgb_u16(px[2]);
        }
    }
    // png crate takes 16-bit data as native-endian u8 pairs.
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    writer
        .write_image_data(&bytes)
        .map_err(|e| PngError::Encode(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_png(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("umber-export-test-{name}.png"))
    }

    #[test]
    fn srgb_transfer_matches_spec_corners() {
        // IEC 61966-2-1 anchor points.
        assert_eq!(linear_to_srgb_u8(0), 0);
        assert_eq!(linear_to_srgb_u8(255), 255);
        // 0.5 linear ≈ 0.7354 sRGB ≈ 188.
        assert_eq!(linear_to_srgb_u8(128), 188);
        // Below the linear-segment threshold stays linear: 1/255*12.92*255
        assert_eq!(linear_to_srgb_u8(1), 13);
    }

    #[test]
    fn u16_srgb_matches_the_u8_curve_at_scale() {
        // Same anchor logic at 16-bit: ends fixed, mid boosted.
        assert_eq!(linear_to_srgb_u16(0), 0);
        assert_eq!(linear_to_srgb_u16(65_535), 65_535);
        // 0.5 linear ≈ 0.7354 sRGB ≈ 48_267.
        let mid = linear_to_srgb_u16(32_768);
        assert!((47_000..=49_500).contains(&mid), "mid: {mid}");
    }

    #[test]
    fn png16_roundtrips_a_tiny_map() {
        // Write 2x2 distinct u16 texels, read the header back: the
        // png crate's decoder reports the right depth + dimensions.
        let path = temp_png("roundtrip16");
        let rgba: Vec<u16> = vec![
            0, 10_000, 20_000, 65_535, 30_000, 40_000, 50_000, 60_000, 1, 2, 3, 4, 65_535, 0,
            32_768, 16_384,
        ];
        write_png16(&path, 2, 2, &rgba, Transfer::Linear).unwrap();
        let decoder = png::Decoder::new(File::open(&path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!(info.width, 2);
        assert_eq!(info.height, 2);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        // Byte length: 2x2 texels * 4 channels * 2 bytes.
        let mut bytes = vec![0u8; reader.output_buffer_size()];
        let _ = reader.next_frame(&mut bytes).unwrap();
        // First texel's red reads back as the same u16 (little-endian
        // native order on this platform).
        let r0 = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(r0, 0);
        let g0 = u16::from_le_bytes([bytes[2], bytes[3]]);
        assert_eq!(g0, 10_000);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn png16_rejects_mismatched_size() {
        let path = temp_png("mismatch16");
        let short = vec![0u16; 15]; // 2x2x4 = 16 needed
        let err = write_png16(&path, 2, 2, &short, Transfer::Linear).unwrap_err();
        assert!(matches!(err, PngError::SizeMismatch { .. }));
    }

    #[test]
    fn write_then_size_mismatch_reported() {
        let path = temp_png("mismatch");
        let err = write_png(&path, 4, 4, &[0u8; 7], Transfer::Linear).unwrap_err();
        assert!(matches!(err, PngError::SizeMismatch { actual: 7, .. }));
    }

    #[test]
    fn roundtrip_png_file_is_decodable() {
        // Write a 2x2 image with distinct pixels, then re-open and check
        // the header round-trips through the png crate's decoder.
        let path = temp_png("roundtrip");
        let data: Vec<u8> = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 255, 128, // white, half-alpha
        ];
        write_png(&path, 2, 2, &data, Transfer::Srgb).expect("write succeeds");
        let file = File::open(&path).expect("file exists");
        let decoder = png::Decoder::new(BufReaderWrapper(file));
        let reader = decoder.read_info().expect("png header decodes after write");
        assert_eq!(reader.info().width, 2);
        assert_eq!(reader.info().height, 2);
        let _ = std::fs::remove_file(&path);
    }

    /// Thin wrapper so the test doesn't need a BufReader import dance.
    struct BufReaderWrapper(File);
    impl std::io::Read for BufReaderWrapper {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            std::io::Read::read(&mut self.0, buf)
        }
    }

    #[test]
    fn read_returns_gpu_row_order_with_a_vertical_flip() {
        // 1x2 map, stored top-down: row 0 = red, row 1 = green.
        // The reader flips to GPU row order: output row 0 = the
        // file's last stored row (green), output row 1 = red.
        let path = temp_png("read-flip");
        let stored: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 255];
        write_png(&path, 1, 2, &stored, Transfer::Linear).expect("write succeeds");
        let (w, h, bytes) = read_png_rgba8(&path).expect("read succeeds");
        assert_eq!((w, h), (1, 2));
        assert_eq!(&bytes[0..4], &[0, 255, 0, 255], "output row 0 = green");
        assert_eq!(&bytes[4..8], &[255, 0, 0, 255], "output row 1 = red");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_expands_rgb_with_opaque_alpha() {
        // 2x1 RGB file: the reader must synthesize alpha 255 per texel
        // (then flip the single row, a no-op).
        let path = temp_png("read-rgb");
        {
            let file = File::create(&path).unwrap();
            let mut encoder = png::Encoder::new(file, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[10, 20, 30, 40, 50, 60]).unwrap();
        }
        let (w, h, bytes) = read_png_rgba8(&path).expect("read succeeds");
        assert_eq!((w, h), (2, 1));
        assert_eq!(bytes, vec![10, 20, 30, 255, 40, 50, 60, 255]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_rejects_missing_and_non_png_files() {
        let missing = std::env::temp_dir().join("umber-export-test-no-such-file.png");
        let err = read_png_rgba8(&missing).unwrap_err();
        // Missing files surface as Decode (the png crate wraps the OS
        // error) or Io — either is a rejection, never a panic.
        assert!(matches!(err, PngError::Decode(_) | PngError::Io(_)));

        let path = temp_png("read-garbage");
        std::fs::write(&path, b"definitely not a png file").unwrap();
        let err = read_png_rgba8(&path).unwrap_err();
        assert!(matches!(err, PngError::Decode(_)), "got {err:?}");
        let _ = std::fs::remove_file(&path);
    }
}

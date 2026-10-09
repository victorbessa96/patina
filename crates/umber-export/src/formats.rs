//! TIFF + JPEG image encoding for exported paint maps (requirements.md
//! §6: "formats: PNG, EXR, TIFF, JPEG").
//!
//! Encodes RGBA8 paint-target data (the `PaintTarget` readback format:
//! width×height×4 bytes, row-major, no padding — the same shape
//! [`crate::png::write_png`] takes) via the `image` crate's TIFF and
//! JPEG encoders.
//!
//! Format tradeoffs (see LANDING_NOTES_FORMATS.md for the full writeup):
//! - TIFF is lossless and keeps the alpha channel — the lossless
//!   archival sibling of PNG for pipelines that want TIFF specifically
//!   (e.g. some DCC/print tooling).
//! - JPEG's color model (YCbCr with chroma subsampling) has no alpha
//!   plane at all, so the alpha channel is dropped before encoding —
//!   not an oversight, a hard format constraint. JPEG is lossy and
//!   meant for quick previews/thumbnails, never for maps that need
//!   exact values (normal maps, masks) or coverage/alpha data.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::tiff::TiffEncoder;
use image::{ExtendedColorType, ImageEncoder};

use crate::png::Transfer;

/// Errors from encoding or writing a TIFF.
#[derive(Debug, thiserror::Error)]
pub enum TiffError {
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
    /// The encoder rejected the stream (wrapped `image` error text).
    #[error("tiff encode failed: {0}")]
    Encode(String),
    /// The OS refused the write (missing dir, permissions, …).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<image::ImageError> for TiffError {
    fn from(err: image::ImageError) -> Self {
        Self::Encode(err.to_string())
    }
}

/// Errors from encoding or writing a JPEG.
#[derive(Debug, thiserror::Error)]
pub enum JpegError {
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
    /// The encoder rejected the stream (wrapped `image` error text).
    #[error("jpeg encode failed: {0}")]
    Encode(String),
    /// The OS refused the write (missing dir, permissions, …).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<image::ImageError> for JpegError {
    fn from(err: image::ImageError) -> Self {
        Self::Encode(err.to_string())
    }
}

/// Creates `path`'s parent directory tree if it doesn't exist yet
/// (mirrors [`crate::png::write_png`]'s behavior).
fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Linear (0..255) → sRGB (0..255); duplicated from
/// [`crate::png`] rather than shared, since that module is owned by a
/// parallel edit and this crate keeps writers self-contained.
fn linear_to_srgb_u8(linear: u8) -> u8 {
    let l = linear as f32 / 255.0;
    let s = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Applies `transfer` to an RGBA8 buffer, leaving alpha untouched.
fn apply_transfer(rgba: &[u8], transfer: Transfer) -> Vec<u8> {
    match transfer {
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
    }
}

/// Encodes `rgba` (width×height×4, row-major) into a lossless RGBA
/// TIFF at `path`, creating parent directories as needed.
///
/// # Errors
///
/// [`TiffError::SizeMismatch`] when the buffer doesn't match the
/// dimensions; [`TiffError::Encode`] from the encoder;
/// [`TiffError::Io`] from the filesystem.
pub fn write_tiff_rgba8(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
    transfer: Transfer,
) -> Result<(), TiffError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(TiffError::SizeMismatch {
            actual: rgba.len(),
            expected,
            width,
            height,
        });
    }

    ensure_parent_dir(path)?;

    let data = apply_transfer(rgba, transfer);
    let file = File::create(path)?;
    TiffEncoder::new(file).write_image(&data, width, height, ExtendedColorType::Rgba8)?;
    Ok(())
}

/// Encodes `rgba` (width×height×4, row-major) into a lossy JPEG at
/// `path`, creating parent directories as needed.
///
/// JPEG has no alpha channel, so the source's alpha byte is dropped
/// before encoding — only R/G/B survive. `quality` is the `image`
/// crate's 1-100 JPEG quality scale (higher = larger file, fewer
/// artifacts).
///
/// # Errors
///
/// [`JpegError::SizeMismatch`] when the buffer doesn't match the
/// dimensions; [`JpegError::Encode`] from the encoder;
/// [`JpegError::Io`] from the filesystem.
pub fn write_jpeg_rgba8(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
    quality: u8,
) -> Result<(), JpegError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(JpegError::SizeMismatch {
            actual: rgba.len(),
            expected,
            width,
            height,
        });
    }

    ensure_parent_dir(path)?;

    // Drop alpha: JPEG's YCbCr color model carries no alpha plane, so
    // encoding RGBA data isn't an option — only R/G/B are kept.
    let rgb: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();

    let file = File::create(path)?;
    let w = BufWriter::new(file);
    JpegEncoder::new_with_quality(w, quality).write_image(
        &rgb,
        width,
        height,
        ExtendedColorType::Rgb8,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(tag: &str, ext: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "umber-formats-test-{tag}-{}.{ext}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn tiff_roundtrips_a_tiny_map_exactly() {
        let path = tmp_path("tiff-roundtrip", "tiff");
        let data: Vec<u8> = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 200, // green, partial alpha
            0, 0, 255, 128, // blue, half alpha
            255, 255, 0, 0, // yellow, transparent
        ];
        write_tiff_rgba8(&path, 2, 2, &data, Transfer::Linear).expect("write succeeds");

        let img = image::open(&path).expect("tiff decodes after write");
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        let rgba = img.into_rgba8();
        assert_eq!(rgba.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(rgba.get_pixel(1, 0).0, [0, 255, 0, 200]);
        assert_eq!(rgba.get_pixel(0, 1).0, [0, 0, 255, 128]);
        assert_eq!(rgba.get_pixel(1, 1).0, [255, 255, 0, 0]);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn tiff_applies_srgb_transfer_like_png() {
        let path = tmp_path("tiff-srgb", "tiff");
        let data: Vec<u8> = vec![
            128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255,
        ];
        write_tiff_rgba8(&path, 2, 2, &data, Transfer::Srgb).expect("write succeeds");
        let img = image::open(&path).expect("decodes");
        let rgba = img.into_rgba8();
        // 128 linear -> 188 sRGB (same anchor the PNG writer tests).
        assert_eq!(rgba.get_pixel(0, 0).0, [188, 188, 188, 255]);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn tiff_rejects_mismatched_buffer_size() {
        let path = tmp_path("tiff-mismatch", "tiff");
        let err = write_tiff_rgba8(&path, 4, 4, &[0u8; 7], Transfer::Linear).unwrap_err();
        assert!(matches!(err, TiffError::SizeMismatch { actual: 7, .. }));
        assert!(!path.exists(), "no file may be written on error");
    }

    #[test]
    fn tiff_creates_parent_directories() {
        let base =
            std::env::temp_dir().join(format!("umber-tiff-test-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let path = base.join("nested/deep/map.tiff");
        let rgba = vec![200u8; 16]; // 2x2
        write_tiff_rgba8(&path, 2, 2, &rgba, Transfer::Linear).unwrap();
        assert!(path.exists());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn jpeg_roundtrips_dimensions_and_drops_alpha() {
        let path = tmp_path("jpeg-roundtrip", "jpg");
        let data: Vec<u8> = vec![
            220, 20, 20, 255, // red-ish
            20, 220, 20, 200, // green-ish, alpha ignored
            20, 20, 220, 128, // blue-ish, alpha ignored
            220, 220, 20, 0, // yellow-ish, alpha ignored
        ];
        write_jpeg_rgba8(&path, 2, 2, &data, 90).expect("write succeeds");

        let img = image::open(&path).expect("jpeg decodes after write");
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.color(), image::ColorType::Rgb8, "jpeg carries no alpha");

        // Lossy: allow generous tolerance, but the dominant channel of
        // each quadrant must survive recognizably.
        let rgb = img.into_rgb8();
        let red = rgb.get_pixel(0, 0).0;
        assert!(
            red[0] > red[1] && red[0] > red[2],
            "top-left stays red-dominant: {red:?}"
        );
        let green = rgb.get_pixel(1, 0).0;
        assert!(
            green[1] > green[0] && green[1] > green[2],
            "top-right stays green-dominant: {green:?}"
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn jpeg_rejects_mismatched_buffer_size() {
        let path = tmp_path("jpeg-mismatch", "jpg");
        let err = write_jpeg_rgba8(&path, 4, 4, &[0u8; 7], 80).unwrap_err();
        assert!(matches!(err, JpegError::SizeMismatch { actual: 7, .. }));
        assert!(!path.exists(), "no file may be written on error");
    }

    #[test]
    fn jpeg_creates_parent_directories() {
        let base =
            std::env::temp_dir().join(format!("umber-jpeg-test-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let path = base.join("nested/deep/map.jpg");
        let rgba = vec![100u8; 16]; // 2x2
        write_jpeg_rgba8(&path, 2, 2, &rgba, 80).unwrap();
        assert!(path.exists());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn jpeg_lower_quality_yields_smaller_file() {
        // Same input, two quality settings: low quality must compress
        // smaller than high quality (the whole point of the knob).
        let width = 64u32;
        let height = 64u32;
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
            // A gradient-ish pattern so the encoder has real work to do
            // (a flat color compresses to ~same size at any quality).
            let v = (i % 256) as u8;
            px[0] = v;
            px[1] = v.wrapping_mul(3);
            px[2] = v.wrapping_mul(7);
            px[3] = 255;
        }

        let low_path = tmp_path("jpeg-quality-low", "jpg");
        let high_path = tmp_path("jpeg-quality-high", "jpg");
        write_jpeg_rgba8(&low_path, width, height, &rgba, 5).unwrap();
        write_jpeg_rgba8(&high_path, width, height, &rgba, 95).unwrap();

        let low_size = std::fs::metadata(&low_path).unwrap().len();
        let high_size = std::fs::metadata(&high_path).unwrap().len();
        assert!(
            low_size < high_size,
            "low quality ({low_size}B) should be smaller than high quality ({high_size}B)"
        );

        std::fs::remove_file(&low_path).unwrap();
        std::fs::remove_file(&high_path).unwrap();
    }
}

//! EXR float export (requirements.md §6: "formats: PNG, EXR, TIFF,
//! JPEG" — 32-bit float half of the bit-depth requirement).
//!
//! Encodes RGBA f32 maps (the bake engine's `rgba32float` readback
//! shape: width×height×4 floats, row-major) into OpenEXR files via the
//! `exr` crate's prelude. Pure CPU — testable anywhere, callable from
//! any context holding raw floats (the position map's natural writer).
//!
//! The f32 buffer is written as-is: EXR is the linear-float interchange
//! format, so no transfer conversion applies (contrast
//! [`crate::png::Transfer`]); color management is the caller's concern.

use std::path::Path;

use exr::prelude as exr_prelude;

/// Errors from encoding or writing an EXR file.
#[derive(Debug, thiserror::Error)]
pub enum ExrError {
    /// The byte buffer's length doesn't match width×height×4.
    #[error("buffer size {actual} floats != {expected} (w*h*4 = {width}x{height})")]
    SizeMismatch {
        /// Actual float count.
        actual: usize,
        /// Expected float count.
        expected: usize,
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
    },
    /// The encoder rejected the stream (wrapped `exr` error text).
    #[error("exr encode failed: {0}")]
    Encode(String),
    /// The OS refused the write (missing dir, permissions, …).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<exr::error::Error> for ExrError {
    fn from(err: exr::error::Error) -> Self {
        Self::Encode(err.to_string())
    }
}

/// Encodes `rgba` (width×height×4 f32, row-major) into an EXR file at
/// `path`, creating parent directories as needed.
///
/// The closure-based `exr` prelude writer is fed per-texel reads out of
/// the flat slice — no intermediate image allocation.
///
/// # Errors
///
/// [`ExrError::SizeMismatch`] when the slice doesn't match the
/// dimensions; [`ExrError::Encode`] from the encoder;
/// [`ExrError::Io`] from the filesystem.
pub fn write_exr_f32(path: &Path, width: u32, height: u32, rgba: &[f32]) -> Result<(), ExrError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(ExrError::SizeMismatch {
            actual: rgba.len(),
            expected,
            width,
            height,
        });
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let w = width as usize;
    let h = height as usize;
    exr_prelude::write_rgba_file(
        path,
        w,
        h,
        // The prelude's y axis runs bottom-up; the bake readback is
        // top-down row-major, so flip y when reading the slice (the
        // same orientation note the PNG writer carries in its tests).
        |x: usize, y: usize| {
            let row = h - 1 - y;
            let i = (row * w + x) * 4;
            (rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3])
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("umber-exr-test-{tag}-{}.exr", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn roundtrips_a_tiny_map_through_file() {
        // 2x2 RGBA, distinct values per texel; read back with the
        // exr crate and verify the floats survive exactly.
        let path = tmp_path("roundtrip");
        let rgba: Vec<f32> = vec![
            0.0, 0.1, 0.2, 1.0, // (0,0)
            0.3, 0.4, 0.5, 0.9, // (1,0)
            0.6, 0.7, 0.8, 0.5, // (0,1)
            0.25, 0.5, 0.75, 0.125, // (1,1)
        ];
        write_exr_f32(&path, 2, 2, &rgba).unwrap();

        // Read back pixel-by-pixel and compare with what we wrote,
        // accounting for the bottom-up row orientation.
        let image = exr::prelude::read_first_rgba_layer_from_file(
            &path,
            |resolution, _| vec![vec![[0.0f32; 4]; resolution.width()]; resolution.height()],
            |pixels, position, (r, g, b, a): (f32, f32, f32, f32)| {
                pixels[position.y()][position.x()] = [r, g, b, a]
            },
        )
        .unwrap();
        let layer = image.layer_data;
        // EXR's y=0 is the bottom = our slice's last row: (0,1) then (1,1).
        let bottom_left = layer.channel_data.pixels[0][0];
        assert!((bottom_left[0] - 0.6).abs() < 1e-6);
        assert!((bottom_left[1] - 0.7).abs() < 1e-6);
        let bottom_right = layer.channel_data.pixels[0][1];
        assert!((bottom_right[0] - 0.25).abs() < 1e-6);
        let top_left = layer.channel_data.pixels[1][0];
        assert!(top_left[0].abs() < 1e-6);
        assert!((top_left[3] - 1.0).abs() < 1e-6);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_mismatched_buffer_size() {
        let path = tmp_path("mismatch");
        let short = vec![0.0f32; 7]; // 2x2x4 = 16 needed
        let err = write_exr_f32(&path, 2, 2, &short).unwrap_err();
        assert!(matches!(err, ExrError::SizeMismatch { .. }));
        assert!(!path.exists(), "no file may be written on error");
    }

    #[test]
    fn creates_parent_directories() {
        let base = std::env::temp_dir().join(format!("umber-exr-test-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let path = base.join("nested/deep/map.exr");
        let rgba = vec![0.5f32; 16]; // 2x2
        write_exr_f32(&path, 2, 2, &rgba).unwrap();
        assert!(path.exists());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn orientation_preserves_display_topology() {
        // Slice row 0 is the texture's TOP row (top-down row-major);
        // the writer flips y so the EXR displays identically: file
        // y=0 (bottom) = slice's LAST row, y=h-1 (top) = slice's row 0.
        let path = tmp_path("orient");
        // 1x2 map: texel (0,0)=red, texel (0,1)=green (row-major).
        let rgba: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        write_exr_f32(&path, 1, 2, &rgba).unwrap();

        let image = exr::prelude::read_first_rgba_layer_from_file(
            &path,
            |resolution, _| vec![vec![[0.0f32; 4]; resolution.width()]; resolution.height()],
            |pixels, position, (r, g, b, a): (f32, f32, f32, f32)| {
                pixels[position.y()][position.x()] = [r, g, b, a]
            },
        )
        .unwrap();
        let layer = image.layer_data;
        // File top (y=1) = slice row 0 = red.
        let top = layer.channel_data.pixels[1][0];
        assert!((top[0] - 1.0).abs() < 1e-6, "top must be red (slice row 0)");
        // File bottom (y=0) = slice's last row = green.
        let bottom = layer.channel_data.pixels[0][0];
        assert!((bottom[1] - 1.0).abs() < 1e-6, "bottom must be green");
        std::fs::remove_file(&path).unwrap();
    }
}

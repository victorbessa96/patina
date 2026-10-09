//! Dithering (requirements.md §6: "dithering option") — Floyd-
//! Steinberg error diffusion applied at the f32→u8 quantization
//! boundary.
//!
//! Precision is lost when scene-linear f32 texels quantize to 8 bits;
//! error diffusion spreads each channel's rounding error to the
//! right/below neighbors, turning banding into perceptually-uniform
//! noise. On integral input (already-u8 buffers) the quantization
//! error is zero and this is a no-op — the option exists for the
//! f32 sources the driver gains when painted maps arrive.
//!
//! Algorithm: Floyd–Steinberg (7/16 right, 3/16 below-left, 5/16
//! below, 1/16 below-right), per-channel on RGB, alpha carried
//! (not diffused — coverage/opacity is data), deterministic
//! left-to-right rows (golden-test friendly, no RNG).

/// Quantizes an f32 RGBA map (0.0..=1.0 per channel, w*h*4 floats)
/// to RGBA8 with Floyd–Steinberg error diffusion on the RGB channels.
/// Alpha quantizes independently (round-to-nearest, no diffusion).
pub fn dither_quantize_rgba8(rgba_f32: &[f32], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(rgba_f32.len(), w * h * 4, "input must be w*h*4 floats");

    let mut out = vec![0u8; w * h * 4];
    // Per-channel error accumulators (RGB only).
    let mut err = vec![[0.0f32; 3]; w * h];

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let src = i * 4;
            // Alpha: plain round, never diffused.
            out[src + 3] = (rgba_f32[src + 3] * 255.0).round().clamp(0.0, 255.0) as u8;
            for c in 0..3 {
                let old_v = rgba_f32[src + c] * 255.0 + err[i][c];
                let new_v = old_v.round().clamp(0.0, 255.0);
                out[src + c] = new_v as u8;
                let e = old_v - new_v;

                if x + 1 < w {
                    err[i + 1][c] += e * 7.0 / 16.0;
                }
                if y + 1 < h {
                    if x > 0 {
                        err[i + w - 1][c] += e * 3.0 / 16.0;
                    }
                    err[i + w][c] += e * 5.0 / 16.0;
                    if x + 1 < w {
                        err[i + w + 1][c] += e * 1.0 / 16.0;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_inputs_quantize_exactly() {
        // 0.5 = 127.5/255 rounds… use exact eighth-values: 0.0 and
        // 1.0 quantize exactly; error is zero, output deterministic.
        let input: Vec<f32> = vec![0.0, 1.0, 0.5, 1.0];
        let out = dither_quantize_rgba8(&input, 1, 1);
        assert_eq!(&out[..3], &[0, 255, 128]); // 0.5*255=127.5 rounds to 128
        assert_eq!(out[3], 255);
    }

    #[test]
    fn fractional_input_diffuses_to_the_right() {
        // 2x1 image, texel0 r = 0.502 (≈128.01): rounds to 128 with
        // error +0.01… too small to move the neighbor. Use a half
        // step: r = 128.5/255 → error 0.5 pushes 7/16*0.5 = 0.219
        // right; texel1 r = 128/255 (128.0) + 0.219 = 128.219 still
        // rounds to 128. Force the visible effect: texel1 sits at
        // x.5+ threshold: 128.5/255 + 0.219 crosses to 129.
        let input: Vec<f32> = vec![
            128.5 / 255.0,
            0.0,
            0.0,
            1.0, //
            128.5 / 255.0,
            0.0,
            0.0,
            1.0,
        ];
        let out = dither_quantize_rgba8(&input, 2, 1);
        // f32::round is half-away-from-zero: 128.5 → 129, error -0.5
        // diffuses right.
        assert_eq!(out[0], 129, "128.5 rounds half-up (f32::round)");
        // Whether texel0 lands 128 (err +0.5) or 129 (err -0.5),
        // the RIGHT neighbor must feel it: its own 128.5 shifts off
        // the .5 boundary.
        //
        // Deterministic pin (probe-verified 2026-10-09): texel0's
        // 128.5 rounds half-away-from-zero to 129 (error -0.5), 7/16
        // of that (-0.21875) diffuses right, so texel1 sits at
        // 128.5 - 0.21875 = 128.28 -> 128. Without diffusion both
        // texels would be 129; the pinned pair (129, 128) is reachable
        // ONLY through error diffusion, so deleting the diffusion
        // lines fails this test.
        assert_eq!(
            out[0..2],
            [129, 0],
            "texel0 red quantizes 128.5 -> 129 (half-away-from-zero)"
        );
        assert_eq!(out[4], 128, "right neighbor absorbs 7/16 of the -0.5 error");
        assert_eq!(
            out[0] as u32 + out[4] as u32,
            257,
            "twin .5-boundary texels preserve total energy: 129 + 128"
        );
    }

    #[test]
    fn alpha_is_never_diffused() {
        // A field of alpha 0.5 with RGB extremes beside it: every
        // alpha byte must be exactly 128 regardless of color error.
        let mut input = vec![0.0f32; 4 * 4 * 4];
        for (i, px) in input.chunks_exact_mut(4).enumerate() {
            px[3] = 0.5;
            if i % 2 == 0 {
                px[0..3].fill(1.0);
            }
        }
        let out = dither_quantize_rgba8(&input, 4, 4);
        for px in out.chunks_exact(4) {
            assert_eq!(px[3], 128, "alpha = 0.5 quantizes to 128 always");
        }
    }

    #[test]
    fn deterministic_output() {
        let input: Vec<f32> = (0..8 * 8 * 4).map(|i| (i as f32 * 0.017) % 1.0).collect();
        let a = dither_quantize_rgba8(&input, 8, 8);
        let b = dither_quantize_rgba8(&input, 8, 8);
        assert_eq!(a, b, "same input → identical dither (no RNG)");
    }

    #[test]
    fn values_stay_in_range_and_close() {
        // Random-ish ramp: every output byte in [0,255] and within
        // one quantization step of its input.
        let input: Vec<f32> = (0..16 * 16 * 4)
            .map(|i| ((i * 37) as f32 % 255.0) / 255.0)
            .collect();
        let out = dither_quantize_rgba8(&input, 16, 16);
        for (f, b) in input.chunks_exact(4).zip(out.chunks_exact(4)) {
            for (fv, bv) in f.iter().zip(b.iter()) {
                let diff = (fv * 255.0 - *bv as f32).abs();
                assert!(diff <= 1.0, "quantization error must stay within 1 step");
            }
        }
    }
}

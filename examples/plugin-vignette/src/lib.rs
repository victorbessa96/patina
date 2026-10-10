//! `vignette`: radial darkening — proves the param path.
//!
//! Param `strength: Float` (default 0.5). For texel centre `p` and image
//! centre `c`, `t = |p - c|² / |corner - c|²` (0 at the centre, ~1 at the
//! corners) and `f = clamp(1 - strength * t, 0, 1)`. RGB scale by `f`
//! (round-half-up); alpha is untouched. `strength = 0` is the identity.
//! Squared distance on purpose: no `sqrt` in `core`, and the quadratic
//! falloff is the classic lens-vignette shape anyway.

#![no_std]

use umber_plugin_sdk::{umber_node, Image, ImageMut, NodeError, Params};

fn vignette(
    input: &Image<'_>,
    params: &Params<'_>,
    out: &mut ImageMut<'_>,
) -> Result<(), NodeError> {
    let strength = params.float("strength", 0.5)?;
    let (w, h) = (input.width as usize, input.height as usize);
    let (cx, cy) = (input.width as f32 * 0.5, input.height as f32 * 0.5);
    let d2_max = cx * cx + cy * cy;
    for y in 0..h {
        let dy = y as f32 + 0.5 - cy;
        for x in 0..w {
            let dx = x as f32 + 0.5 - cx;
            let t = (dx * dx + dy * dy) / d2_max;
            let f = (1.0 - strength * t).clamp(0.0, 1.0);
            let i = (y * w + x) * 4;
            for c in 0..3 {
                out.data[i + c] = (f32::from(input.data[i + c]) * f + 0.5) as u8;
            }
            out.data[i + 3] = input.data[i + 3];
        }
    }
    Ok(())
}

umber_node!(name: "vignette", params: 1, eval: vignette);

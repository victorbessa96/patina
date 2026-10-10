//! `blur5`: a 5-px separable box blur — the reference plugin.
//!
//! Mirrors umber-graph's native `blur` node at `radius: 2` EXACTLY
//! (crates/umber-graph/src/nodes/filter.rs `box_pass_1d`/`box_blur`):
//! window `k = 5`, edge clamp, per-pass round-to-nearest
//! `(sum + k/2) / k`, horizontal pass first, then vertical over the
//! rounded horizontal result, all four channels (alpha blurs too).
//! umber-wasm's equivalence test asserts byte-for-byte agreement.
//!
//! No allocator: the horizontal pass writes straight into the out
//! region, the vertical pass works column by column in place, with two
//! stack line buffers sized to the engine's 8192-px side limit.

#![no_std]

use umber_plugin_sdk::{umber_node, Image, ImageMut, NodeError, Params};

const RADIUS: i64 = 2;
/// umber-graph's per-side raster ceiling.
const MAX_SIDE: usize = 8192;

/// One 1-D box pass, `src` → `dst` (same length) — a line-for-line
/// mirror of the native `box_pass_1d` at `radius = RADIUS`.
fn box_pass(src: &[u8], dst: &mut [u8]) {
    let n = src.len();
    let r = RADIUS as u64;
    let k = 2 * r + 1;
    let last = (n - 1) as i64;
    let mut sum: u64 = r * u64::from(src[0]);
    let upto = (RADIUS.min(last) as usize).min(n - 1);
    for s in src.iter().take(upto + 1) {
        sum += u64::from(*s);
    }
    sum += (r - upto as u64) * u64::from(src[n - 1]);
    dst[0] = ((sum + k / 2) / k) as u8;
    for (x, slot) in dst.iter_mut().enumerate().skip(1) {
        let xi = x as i64;
        let sub = src[(xi - RADIUS - 1).clamp(0, last) as usize];
        let add = src[(xi + RADIUS).clamp(0, last) as usize];
        sum = sum - u64::from(sub) + u64::from(add);
        *slot = ((sum + k / 2) / k) as u8;
    }
}

fn blur5(input: &Image<'_>, _params: &Params<'_>, out: &mut ImageMut<'_>) -> Result<(), NodeError> {
    let (w, h) = (input.width as usize, input.height as usize);
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err(NodeError::Internal);
    }
    let mut line = [0u8; MAX_SIDE];
    let mut res = [0u8; MAX_SIDE];
    // Horizontal: input rows → out.
    for y in 0..h {
        for c in 0..4 {
            for x in 0..w {
                line[x] = input.data[(y * w + x) * 4 + c];
            }
            box_pass(&line[..w], &mut res[..w]);
            for x in 0..w {
                out.data[(y * w + x) * 4 + c] = res[x];
            }
        }
    }
    // Vertical: out columns, in place (each column is read whole into
    // `line` before any of it is overwritten).
    for x in 0..w {
        for c in 0..4 {
            for y in 0..h {
                line[y] = out.data[(y * w + x) * 4 + c];
            }
            box_pass(&line[..h], &mut res[..h]);
            for y in 0..h {
                out.data[(y * w + x) * 4 + c] = res[y];
            }
        }
    }
    Ok(())
}

umber_node!(name: "blur5", params: 0, eval: blur5);

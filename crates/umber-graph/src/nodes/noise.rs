//! Noise generators: `noise_perlin`, `noise_value`, `noise_worley`.
//!
//! Wave-5 slice 2a (`docs/specs/node-graph-design.md`, "The node set").
//! Three tiling functions sharing one param block (scale/offset/seed).
//!
//! # Shared params
//!
//! * `scale: Float` (default 8.0) — noise-space frequency multiplier.
//! * `offset: Vec2` (default `[0, 0]`) — noise-space translation.
//! * `seed: Int` (default 0) — selects the permutation table.
//!
//! # UV convention (V-flip)
//!
//! Pixel `(x, y)` samples noise space at
//! `u = (x / w) * scale + offset[0]`, `v = (1 - y / h) * scale + offset[1]`.
//! Row 0 is the image top (see [`crate::value::ImageBuffer`]) but samples
//! `v = scale` (UV origin bottom-left, the MaterialX convention), so noise
//! orientation agrees with the mtlx/app UV layers. Documented here so all
//! layers sample the same field.
//!
//! # Permutation table
//!
//! [`build_perm`] shuffles `0..256` with Fisher-Yates driven by a simple
//! LCG (`state = state * 1664525 + 1013904223`, `j = state % (i + 1)` from
//! `i = 255` down to 1, starting `state = seed as u32`). Same seed gives
//! the same table (deterministic); the table is doubled to 512 entries so
//! index wrapping is a bitmask, never a branch.

use crate::value::ImageBuffer;
use crate::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use std::sync::Arc;

/// Builds the doubled permutation table for `seed` (see module docs).
#[must_use]
pub fn build_perm(seed: i32) -> [u8; 512] {
    let mut p = [0u8; 256];
    for (i, slot) in p.iter_mut().enumerate() {
        *slot = i as u8;
    }
    let mut state = seed as u32;
    for i in (1..256).rev() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let j = (state as usize) % (i + 1);
        p.swap(i, j);
    }
    let mut doubled = [0u8; 512];
    for (i, slot) in doubled.iter_mut().enumerate() {
        *slot = p[i & 255];
    }
    doubled
}

/// Quintic fade (`6t^5 - 15t^4 + 10t^3`) for Perlin interpolation.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Smoothstep (`3t^2 - 2t^3`) on `[0, 1]` for value-noise interpolation.
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// 2D gradient dot product: `h & 7` selects one of 8 unit/diagonal
/// gradients (diagonals pre-normalized by `1/sqrt(2)`), so the Perlin
/// field stays in `[-1, 1]`.
fn grad2(h: u8, x: f32, y: f32) -> f32 {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    match h & 7 {
        0 => (x + y) * INV_SQRT2,
        1 => (-x + y) * INV_SQRT2,
        2 => (x - y) * INV_SQRT2,
        3 => (-x - y) * INV_SQRT2,
        4 => x,
        5 => -x,
        6 => y,
        _ => -y,
    }
}

/// Classic 2D Perlin noise in `[-1, 1]`: hashed gradients at the four
/// lattice corners, quintic-fade trilinear (bilinear, 2D) interpolation.
fn perlin2(x: f32, y: f32, perm: &[u8; 512]) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;
    let x0 = (xi & 255) as usize;
    let y0 = (yi & 255) as usize;
    let x1 = ((xi + 1) & 255) as usize;
    let y1 = ((yi + 1) & 255) as usize;
    let h00 = perm[x0 + perm[y0] as usize];
    let h10 = perm[x1 + perm[y0] as usize];
    let h01 = perm[x0 + perm[y1] as usize];
    let h11 = perm[x1 + perm[y1] as usize];
    let n00 = grad2(h00, xf, yf);
    let n10 = grad2(h10, xf - 1.0, yf);
    let n01 = grad2(h01, xf, yf - 1.0);
    let n11 = grad2(h11, xf - 1.0, yf - 1.0);
    let u = fade(xf);
    let v = fade(yf);
    (n00 * (1.0 - u) + n10 * u) * (1.0 - v) + (n01 * (1.0 - u) + n11 * u) * v
}

/// Lattice-corner hash in `[0, 1]`: `perm[(xi + perm[yi & 255]) & 255] / 255`.
fn lattice_hash(xi: i32, yi: i32, perm: &[u8; 512]) -> f32 {
    let x = (xi & 255) as usize;
    let y = (yi & 255) as usize;
    f32::from(perm[(x + perm[y] as usize) & 255]) / 255.0
}

/// Value noise in `[0, 1]`: hashed lattice corners, smoothstep
/// interpolation.
fn value2(x: f32, y: f32, perm: &[u8; 512]) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;
    let v00 = lattice_hash(xi, yi, perm);
    let v10 = lattice_hash(xi + 1, yi, perm);
    let v01 = lattice_hash(xi, yi + 1, perm);
    let v11 = lattice_hash(xi + 1, yi + 1, perm);
    let u = smooth(xf);
    let v = smooth(yf);
    (v00 * (1.0 - u) + v10 * u) * (1.0 - v) + (v01 * (1.0 - u) + v11 * u) * v
}

/// Second hash axis for Worley feature points (offset selector so the
/// y-coordinate differs from the x-coordinate's hash stream).
fn lattice_hash_b(xi: i32, yi: i32, perm: &[u8; 512]) -> f32 {
    let x = ((xi + 57) & 255) as usize;
    let y = ((yi + 131) & 255) as usize;
    f32::from(perm[(x + perm[y] as usize) & 255]) / 255.0
}

/// Worley F1 in `[0, 1]`: distance (cell units, cell size 1) to the
/// nearest feature point over the 3x3 cell neighbourhood, clamped to
/// `[0, 1]`. One feature point per cell (v1), hashed from the table:
/// `fx = hash_a(cell)`, `fy = hash_b(cell)`.
fn worley_f1(x: f32, y: f32, perm: &[u8; 512]) -> f32 {
    let cx = x.floor() as i32;
    let cy = y.floor() as i32;
    let mut best = f32::MAX;
    for jy in -1..=1 {
        for jx in -1..=1 {
            let nx = cx + jx;
            let ny = cy + jy;
            let fx = nx as f32 + lattice_hash(nx, ny, perm);
            let fy = ny as f32 + lattice_hash_b(nx, ny, perm);
            let dx = x - fx;
            let dy = y - fy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
            }
        }
    }
    best.clamp(0.0, 1.0)
}

fn find_param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

/// Shared param block: `(scale, offset, seed)` with documented defaults.
fn noise_params(params: &[(String, ParamValue)]) -> Result<(f32, [f32; 2], i32), EvalError> {
    let scale = match find_param(params, "scale") {
        None => 8.0,
        Some(ParamValue::Float(s)) => *s,
        Some(_) => {
            return Err(EvalError::BadParam {
                node: 0,
                param: "scale".into(),
            });
        }
    };
    if !scale.is_finite() {
        return Err(EvalError::BadParam {
            node: 0,
            param: "scale".into(),
        });
    }
    let offset = match find_param(params, "offset") {
        None => [0.0, 0.0],
        Some(ParamValue::Vec2(o)) => *o,
        Some(_) => {
            return Err(EvalError::BadParam {
                node: 0,
                param: "offset".into(),
            });
        }
    };
    if !offset[0].is_finite() || !offset[1].is_finite() {
        return Err(EvalError::BadParam {
            node: 0,
            param: "offset".into(),
        });
    }
    let seed = match find_param(params, "seed") {
        None => 0,
        Some(ParamValue::Int(s)) => *s,
        Some(_) => {
            return Err(EvalError::BadParam {
                node: 0,
                param: "seed".into(),
            });
        }
    };
    Ok((scale, offset, seed))
}

/// Rejects hostile raster sizes before allocation (each side `<= 8192`,
/// total texels `<= 2^26` — the [`ImageBuffer`] v1 ceiling).
fn check_resolution(width: u32, height: u32) -> Result<(), EvalError> {
    if width == 0 || height == 0 {
        return Err(EvalError::Image(crate::ImageError::EmptyDimensions {
            width,
            height,
        }));
    }
    if width > 8192 || height > 8192 || width as u64 * height as u64 > (1u64 << 26) {
        return Err(EvalError::Image(crate::ImageError::DimensionsTooLarge {
            width,
            height,
        }));
    }
    Ok(())
}

fn gray_byte(t: f32) -> u8 {
    (t.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Renders one grayscale image by sampling `f` (noise space → `[0, 1]`)
/// at the V-flipped UV of every texel.
fn render_field(
    ctx: &EvalContext,
    scale: f32,
    offset: [f32; 2],
    f: impl Fn(f32, f32) -> f32,
) -> Result<NodeOutput, EvalError> {
    let (w, h) = ctx.resolution;
    check_resolution(w, h)?;
    let mut data = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 / w as f32) * scale + offset[0];
            let v = (1.0 - y as f32 / h as f32) * scale + offset[1];
            let b = gray_byte(f(u, v));
            data.extend_from_slice(&[b, b, b, 255]);
        }
    }
    Ok(NodeOutput::Image(ImageBuffer::new(w, h, data)?))
}

/// `noise_perlin`: classic Perlin ([`perlin2`], `[-1, 1]` mapped by
/// `(v * 0.5 + 0.5) * 255` per channel, alpha 255; grayscale `r = g = b`).
/// Params: `scale: Float` (default 8.0), `offset: Vec2` (default
/// `[0, 0]`), `seed: Int` (default 0). Output: `Image` at
/// `ctx.resolution`.
pub struct PerlinNode;

impl NodeImpl for PerlinNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let (scale, offset, seed) = noise_params(params)?;
        let perm = build_perm(seed);
        render_field(ctx, scale, offset, |u, v| perlin2(u, v, &perm) * 0.5 + 0.5)
    }
}

/// `noise_value`: value noise ([`value2`], `[0, 1]` → RGBA8 grayscale).
/// Params: `scale: Float` (default 8.0), `offset: Vec2` (default
/// `[0, 0]`), `seed: Int` (default 0). Output: `Image` at
/// `ctx.resolution`.
pub struct ValueNode;

impl NodeImpl for ValueNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let (scale, offset, seed) = noise_params(params)?;
        let perm = build_perm(seed);
        render_field(ctx, scale, offset, |u, v| value2(u, v, &perm))
    }
}

/// `noise_worley`: Worley/cellular F1 ([`worley_f1`], `[0, 1]` → RGBA8
/// grayscale). Params: `scale: Float` (default 8.0), `offset: Vec2`
/// (default `[0, 0]`), `seed: Int` (default 0). Output: `Image` at
/// `ctx.resolution`.
pub struct WorleyNode;

impl NodeImpl for WorleyNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let (scale, offset, seed) = noise_params(params)?;
        let perm = build_perm(seed);
        render_field(ctx, scale, offset, |u, v| worley_f1(u, v, &perm))
    }
}

/// Registers `noise_perlin`, `noise_value`, `noise_worley`.
pub fn register_noise_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("noise_perlin", Arc::new(PerlinNode));
    registry.register("noise_value", Arc::new(ValueNode));
    registry.register("noise_worley", Arc::new(WorleyNode));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeRegistry;

    fn eval_def(
        registry: &NodeRegistry,
        def: &str,
        params: &[(String, ParamValue)],
        w: u32,
        h: u32,
    ) -> ImageBuffer {
        let imp = registry.get(def).expect("registered");
        match imp
            .eval(vec![], params, &EvalContext::new(w, h))
            .expect("noise evaluates")
        {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    fn seeded_registry() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        register_noise_nodes(&mut r);
        r
    }

    #[test]
    fn perlin_is_deterministic_per_seed_and_varies_across_seeds() {
        // Same seed twice → byte-identical; different seed → differs.
        // (A nondeterministic table — e.g. time-seeded — fails the first;
        // a seed-ignoring table fails the second.)
        let r = seeded_registry();
        let params = |seed: i32| {
            vec![
                ("scale".into(), ParamValue::Float(4.0)),
                ("seed".into(), ParamValue::Int(seed)),
            ]
        };
        let a = eval_def(&r, "noise_perlin", &params(7), 8, 8);
        let b = eval_def(&r, "noise_perlin", &params(7), 8, 8);
        assert_eq!(a.data, b.data, "same seed must be byte-identical");
        let c = eval_def(&r, "noise_perlin", &params(8), 8, 8);
        assert_ne!(a.data, c.data, "different seeds must differ");
    }

    #[test]
    fn perlin_mapping_covers_range_and_is_not_constant() {
        // u8 channels are trivially in [0, 255]; the failable asserts are
        // min < max (a broken [-1,1]→byte map — e.g. truncation instead of
        // the documented (v*0.5+0.5)*255 — collapses to constant) and that
        // every texel is grayscale with opaque alpha.
        let r = seeded_registry();
        let buf = eval_def(
            &r,
            "noise_perlin",
            &[("scale".into(), ParamValue::Float(8.0))],
            16,
            16,
        );
        let (mut min, mut max) = (255u8, 0u8);
        for px in buf.data.chunks_exact(4) {
            assert_eq!(px[0], px[1], "grayscale r == g");
            assert_eq!(px[0], px[2], "grayscale r == b");
            assert_eq!(px[3], 255, "opaque alpha");
            min = min.min(px[0]);
            max = max.max(px[0]);
        }
        assert!(min < max, "field must vary (min {min} < max {max})");
    }

    #[test]
    fn value_noise_corners_equal_the_hashed_lattice() {
        // 2x2 buffer, scale 1.0, offset [0,0]: samples are
        // u = x/2 ∈ {0, 0.5}, v = 1 - y/2 ∈ {1, 0.5}. Only pixel (0,0)
        // lands exactly on a lattice corner — noise space (0, 1) — so it
        // must equal lattice_hash(0, 1) exactly; the other three pixels
        // are smoothstep interpolations, mirrored here in f32 (the impl's
        // precision) for exact-byte asserts.
        let r = seeded_registry();
        let seed = 3;
        let buf = eval_def(
            &r,
            "noise_value",
            &[
                ("scale".into(), ParamValue::Float(1.0)),
                ("seed".into(), ParamValue::Int(seed)),
            ],
            2,
            2,
        );
        let perm = build_perm(seed);
        // Ported corner hash: perm[(0 + perm[1]) & 255] / 255.
        let corner = f32::from(perm[perm[1] as usize]) / 255.0;
        let corner_byte = (corner.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        assert_eq!(
            buf.pixel(0, 0),
            Some([corner_byte, corner_byte, corner_byte, 255]),
            "pixel (0,0) samples lattice corner (0,1) exactly"
        );
        // Full mirrored field: smoothstep bilinear over the four corners
        // (0,1),(1,1),(0,2),(1,2) — derivation: v=1-y/2 spans lattice rows
        // y=1..2, u=x/2 spans columns 0..1.
        let h = |xi: i32, yi: i32| lattice_hash(xi, yi, &perm);
        let at = |u: f32, v: f32| {
            let xi = u.floor() as i32;
            let yi = v.floor() as i32;
            let xf = u - xi as f32;
            let yf = v - yi as f32;
            let s = |t: f32| t * t * (3.0 - 2.0 * t);
            let (su, sv) = (s(xf), s(yf));
            let val = (h(xi, yi) * (1.0 - su) + h(xi + 1, yi) * su) * (1.0 - sv)
                + (h(xi, yi + 1) * (1.0 - su) + h(xi + 1, yi + 1) * su) * sv;
            let b = (val.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [b, b, b, 255]
        };
        assert_eq!(buf.pixel(1, 0), Some(at(0.5, 1.0)), "top edge midpoint");
        assert_eq!(buf.pixel(0, 1), Some(at(0.0, 0.5)), "left edge midpoint");
        assert_eq!(buf.pixel(1, 1), Some(at(0.5, 0.5)), "cell center");
    }

    #[test]
    fn worley_feature_point_is_zero_and_corner_is_max() {
        // Part 1 — nearest distance ~0: 64x64, scale 1.0, seed 11. The
        // sample nearest cell (0,0)'s feature point is at most half a
        // pixel diagonal away — sqrt(2)/2/64 ≈ 0.011 → byte ≤ 3 < 4
        // regardless of the hashed position (geometric bound, not seed
        // luck). Part 2 — corner max: 8x8, scale 1.0, seed 8. A probe over
        // seeds 0..50 showed the mirrored field attains its UNIQUE max
        // 0.95496434 at corner (0,0) → byte 244; the test re-derives this
        // from mirrored math and asserts uniqueness, so a wrong
        // hash/interp (a different field) fails.
        let r = seeded_registry();
        let seed = 11;
        let (w, h) = (64u32, 64u32);
        let buf = eval_def(
            &r,
            "noise_worley",
            &[
                ("scale".into(), ParamValue::Float(1.0)),
                ("seed".into(), ParamValue::Int(seed)),
            ],
            w,
            h,
        );
        let perm = build_perm(seed);
        let mirrored = |x: u32, y: u32| {
            let u = x as f32 / w as f32;
            let v = 1.0 - y as f32 / h as f32;
            let b = (worley_f1(u, v, &perm).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [b, b, b, 255]
        };
        // Feature point of cell (0,0) in noise space.
        let (fx, fy) = (lattice_hash(0, 0, &perm), lattice_hash_b(0, 0, &perm));
        // Nearest sampled texel to (fx, fy): u = x/64 → x = fx*64 rounded,
        // v = 1 - y/64 → y = (1 - fy)*64 rounded, clamped into the buffer.
        let nx = ((fx * 64.0).round() as i64).clamp(0, 63) as u32;
        let ny = (((1.0 - fy) * 64.0).round() as i64).clamp(0, 63) as u32;
        let near = buf.pixel(nx, ny).expect("in bounds");
        assert_eq!(near, mirrored(nx, ny), "mirrored exact byte");
        assert!(
            near[0] < 4,
            "nearest sample to the feature point ({fx:.4}, {fy:.4}) at ({nx}, {ny}) must be ~0, got {}",
            near[0]
        );

        // Part 2: 8x8 seed 8 — unique max at corner (0,0), byte 244.
        let buf8 = eval_def(
            &r,
            "noise_worley",
            &[
                ("scale".into(), ParamValue::Float(1.0)),
                ("seed".into(), ParamValue::Int(8)),
            ],
            8,
            8,
        );
        let perm8 = build_perm(8);
        let mirrored8 = |x: u32, y: u32| {
            let u = x as f32 / 8.0;
            let v = 1.0 - y as f32 / 8.0;
            let b = (worley_f1(u, v, &perm8).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [b, b, b, 255]
        };
        assert_eq!(
            buf8.pixel(0, 0),
            Some([244, 244, 244, 255]),
            "corner (0,0) holds the derived max 244"
        );
        assert_eq!(
            buf8.pixel(0, 0),
            Some(mirrored8(0, 0)),
            "corner (0,0) mirrored exact"
        );
        let max = buf8
            .data
            .chunks_exact(4)
            .map(|px| px[0])
            .max()
            .expect("nonempty");
        assert_eq!(max, 244, "buffer max is the corner's 244");
        let at_max = buf8.data.chunks_exact(4).filter(|px| px[0] == max).count();
        assert_eq!(at_max, 1, "the max must be unique to the corner");
    }

    #[test]
    fn noise_rejects_mistyped_params() {
        let r = seeded_registry();
        let imp = r.get("noise_perlin").expect("registered");
        let bad = imp.eval(
            vec![],
            &[("scale".into(), ParamValue::Int(4))],
            &EvalContext::new(4, 4),
        );
        assert!(
            matches!(bad, Err(EvalError::BadParam { .. })),
            "mistyped scale must fail, got {bad:?}"
        );
    }
}

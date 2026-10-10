//! Filter nodes: `blur`, `sharpen`, `levels`, `curves`, `invert`.
//!
//! Wave-5 slice 2b (`docs/specs/node-graph-design.md`, "The node set",
//! Filter row). Every node consumes Image inputs (a `Uniform` input is
//! [`EvalError::TypeMismatch`] — the engine enforces Image-only via the
//! impls below) and produces an `Image` the same size as its input (NOT
//! `ctx.resolution`: filters transform, they don't generate).

use crate::value::ImageBuffer;
use crate::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use std::sync::Arc;

fn find_param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

/// The node's single image input (any name — filters take exactly one
/// wiring, like `passthrough`).
fn single_image(inputs: &[(String, NodeOutput)]) -> Result<&ImageBuffer, EvalError> {
    match inputs.len() {
        1 => match &inputs[0].1 {
            NodeOutput::Image(buf) => Ok(buf),
            other => Err(EvalError::TypeMismatch {
                node: 0,
                expected: "Image".into(),
                got: other.kind().into(),
            }),
        },
        0 => Err(EvalError::MissingInput {
            node: 0,
            input: "in".into(),
        }),
        n => Err(EvalError::BadParam {
            node: 0,
            param: format!("expected exactly 1 input, got {n}"),
        }),
    }
}

fn float_param(
    params: &[(String, ParamValue)],
    name: &str,
    default: f32,
) -> Result<f32, EvalError> {
    let v = match find_param(params, name) {
        None => default,
        Some(ParamValue::Float(v)) => *v,
        Some(_) => {
            return Err(EvalError::BadParam {
                node: 0,
                param: name.into(),
            });
        }
    };
    if !v.is_finite() {
        return Err(EvalError::BadParam {
            node: 0,
            param: name.into(),
        });
    }
    Ok(v)
}

fn int_param(params: &[(String, ParamValue)], name: &str, default: i32) -> Result<i32, EvalError> {
    match find_param(params, name) {
        None => Ok(default),
        Some(ParamValue::Int(v)) => Ok(*v),
        Some(_) => Err(EvalError::BadParam {
            node: 0,
            param: name.into(),
        }),
    }
}

/// Round-half-up float→byte, matching the merged convention
/// (`eval.rs`'s `f32_to_u8`, noise's `gray_byte`). NaN normalizes to 0
/// through the saturating `as u8` cast.
fn byte_round(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// One 1-D box pass over `src` (length `n`) with edge clamping.
///
/// Window size `k = 2*radius + 1`; each output is
/// `(sum + k/2) / k` (integer round-to-nearest). The running sum makes
/// this O(n) for ANY radius (a hostile `radius: Int` must not hang the
/// engine): `sum(0) = r*s[0] + Σ_{dx=0}^{r} s[min(dx, n-1)]`, then
/// `sum(x) = sum(x-1) - s[clamp(x-r-1)] + s[clamp(x+r)]`.
fn box_pass_1d(src: &[u8], n: usize, radius: i64) -> Vec<u8> {
    if radius <= 0 {
        return src.to_vec();
    }
    let r = radius as u64;
    let k = 2 * r + 1;
    let last = (n - 1) as i64;
    let mut dst = vec![0u8; n];
    // Seed: dx in -r..=-1 clamp to s[0] (r copies) + dx in 0..=r.
    let mut sum: u64 = r * u64::from(src[0]);
    let upto = (radius.min(last) as usize).min(n - 1);
    for s in src.iter().take(upto + 1) {
        sum += u64::from(*s);
    }
    sum += (r - upto as u64) * u64::from(src[n - 1]);
    dst[0] = ((sum + k / 2) / k) as u8;
    for (x, slot) in dst.iter_mut().enumerate().skip(1) {
        let xi = x as i64;
        let sub = src[(xi - radius - 1).clamp(0, last) as usize];
        let add = src[(xi + radius).clamp(0, last) as usize];
        sum = sum - u64::from(sub) + u64::from(add);
        *slot = ((sum + k / 2) / k) as u8;
    }
    dst
}

/// Shared separable box-blur kernel (reused by `blur` and `sharpen`):
/// horizontal [`box_pass_1d`] per row per channel, then vertical per
/// column per channel. All four RGBA channels blur identically
/// (documented: alpha blurs too — a blurred mask's coverage fades).
/// `radius <= 0` returns a clone (the clamp target for negative radii).
fn box_blur(src: &ImageBuffer, radius: i32) -> Result<ImageBuffer, EvalError> {
    if radius <= 0 {
        return Ok(src.clone());
    }
    let (w, h) = (src.width as usize, src.height as usize);
    let r = i64::from(radius);
    // Horizontal pass, channel-interleaved source → planar temp.
    let mut tmp = vec![0u8; w * h * 4];
    let mut row = vec![0u8; w.max(1)];
    for y in 0..h {
        for c in 0..4 {
            for (x, slot) in row.iter_mut().enumerate().take(w) {
                *slot = src.data[(y * w + x) * 4 + c];
            }
            let out = box_pass_1d(&row[..w], w, r);
            for (x, b) in out.iter().enumerate().take(w) {
                tmp[(y * w + x) * 4 + c] = *b;
            }
        }
    }
    // Vertical pass over columns into the final buffer.
    let mut data = vec![0u8; w * h * 4];
    let mut col = vec![0u8; h.max(1)];
    let mut col_out = vec![0u8; h.max(1)];
    for x in 0..w {
        for c in 0..4 {
            for (y, slot) in col.iter_mut().enumerate().take(h) {
                *slot = tmp[(y * w + x) * 4 + c];
            }
            let out = box_pass_1d(&col[..h], h, r);
            col_out[..h].copy_from_slice(&out);
            for (y, b) in col_out.iter().enumerate().take(h) {
                data[(y * w + x) * 4 + c] = *b;
            }
        }
    }
    Ok(ImageBuffer::new(src.width, src.height, data)?)
}

/// `blur`: separable box blur v1 (gaussian is wave-6, per the design).
/// Params: `radius: Int` (default 1, clamped `>= 0` — negative means
/// no blur, NOT an error). Edge-clamp, horizontal pass then vertical,
/// per-pass round-to-nearest. Output: `Image`, input size.
pub struct BlurNode;

impl NodeImpl for BlurNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = single_image(&inputs)?;
        let radius = int_param(params, "radius", 1)?.max(0);
        Ok(NodeOutput::Image(box_blur(src, radius)?))
    }
}

/// `sharpen`: unsharp mask `out = input + amount * (input - blur(input))`
/// (the blur is the shared [`box_blur`] kernel). Params: `amount: Float`
/// (default 0.5, any finite value — negative amounts soften further),
/// `radius: Int` (default 1, clamped `>= 0`). RGB is sharpened; alpha
/// passes through untouched (sharpening coverage would erode mask
/// edges — v1 keeps alpha). Output: `Image`, input size.
pub struct SharpenNode;

impl NodeImpl for SharpenNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = single_image(&inputs)?;
        let amount = float_param(params, "amount", 0.5)?;
        let radius = int_param(params, "radius", 1)?.max(0);
        let blurred = box_blur(src, radius)?;
        let mut data = Vec::with_capacity(src.data.len());
        for (px, bp) in src.data.chunks_exact(4).zip(blurred.data.chunks_exact(4)) {
            for (p, b) in px.iter().take(3).zip(bp.iter().take(3)) {
                let v = f32::from(*p) + amount * (f32::from(*p) - f32::from(*b));
                data.push(v.clamp(0.0, 255.0).round() as u8);
            }
            data.push(px[3]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// `levels`: per-channel remap. Params (all `Float`):
/// `in_low` (default 0.0), `in_high` (default 1.0), `gamma` (default
/// 1.0, must be finite and `>= 0`), `out_low` (default 0.0), `out_high`
/// (default 1.0). Per channel: `t = clamp((v - in_low) / (in_high -
/// in_low))`; `t = t^gamma` (skipped when `gamma == 1.0`, keeping the
/// value bit-exact); `out = out_low + t * (out_high - out_low)`.
/// Degenerate `in_high == in_low` is [`EvalError::BadParam`] (division
/// by zero made loud). RGB is remapped; alpha passes through. Exact
/// identity params (`0/1/1/0/1`) return the input clone byte-identical
/// (fast path — no float round-trip). Output: `Image`, input size.
pub struct LevelsNode;

impl NodeImpl for LevelsNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = single_image(&inputs)?;
        let in_low = float_param(params, "in_low", 0.0)?;
        let in_high = float_param(params, "in_high", 1.0)?;
        let gamma = float_param(params, "gamma", 1.0)?;
        let out_low = float_param(params, "out_low", 0.0)?;
        let out_high = float_param(params, "out_high", 1.0)?;
        if gamma < 0.0 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "gamma".into(),
            });
        }
        if in_high == in_low {
            return Err(EvalError::BadParam {
                node: 0,
                param: "in_low == in_high (degenerate range)".into(),
            });
        }
        if in_low == 0.0 && in_high == 1.0 && gamma == 1.0 && out_low == 0.0 && out_high == 1.0 {
            return Ok(NodeOutput::Image(src.clone()));
        }
        let span = in_high - in_low;
        let out_span = out_high - out_low;
        let mut data = Vec::with_capacity(src.data.len());
        for px in src.data.chunks_exact(4) {
            for p in px.iter().take(3) {
                let v = f32::from(*p) / 255.0;
                let t = ((v - in_low) / span).clamp(0.0, 1.0);
                let g = if gamma == 1.0 { t } else { t.powf(gamma) };
                data.push(byte_round(out_low + g * out_span));
            }
            data.push(px[3]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// The 4-point control curve, mirroring `umber_brush::ControlCurve`
/// WITHOUT depending on umber-brush (umber-graph stays
/// dependency-minimal — see crate `Cargo.toml`: only `thiserror`).
/// Identical semantics: four points, finite coordinates, strictly
/// increasing x (`x[0] < x[1] < x[2] < x[3]`, enforced at eval →
/// [`EvalError::BadParam`); clamped piecewise-linear lerp: below `x[0]`
/// yields `y[0]`, above `x[3]` yields `y[3]`, else the bracketing lerp.
/// A conversion to the real `ControlCurve` belongs at the app-bridge
/// layer (wave-5 slice 5), not here.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Curve4 {
    points: [[f32; 2]; 4],
}

impl Curve4 {
    fn try_new(points: [[f32; 2]; 4]) -> Result<Self, EvalError> {
        if points
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err(EvalError::BadParam {
                node: 0,
                param: "curve_p* (non-finite coordinate)".into(),
            });
        }
        if !(points[0][0] < points[1][0]
            && points[1][0] < points[2][0]
            && points[2][0] < points[3][0])
        {
            return Err(EvalError::BadParam {
                node: 0,
                param: "curve_p* (x must be strictly increasing)".into(),
            });
        }
        Ok(Self { points })
    }

    fn evaluate(&self, x: f32) -> f32 {
        let p = self.points;
        if x <= p[0][0] {
            return p[0][1];
        }
        if x >= p[3][0] {
            return p[3][1];
        }
        for i in 0..3 {
            if x <= p[i + 1][0] {
                let (x0, y0) = (p[i][0], p[i][1]);
                let (x1, y1) = (p[i + 1][0], p[i + 1][1]);
                let t = (x - x0) / (x1 - x0);
                return y0 + (y1 - y0) * t;
            }
        }
        p[3][1]
    }
}

fn curve_param(params: &[(String, ParamValue)], name: &str) -> Result<[f32; 2], EvalError> {
    match find_param(params, name) {
        Some(ParamValue::Vec2(v)) => Ok(*v),
        _ => Err(EvalError::BadParam {
            node: 0,
            param: name.into(),
        }),
    }
}

/// `curves`: the 4-point curve applied per RGB channel identically;
/// alpha passes through. `ParamValue` has no curve variant, so the
/// curve travels as four `Vec2` params `curve_p0..curve_p3` (x/y pairs;
/// monotonic-x enforced at eval → [`EvalError::BadParam`]). Missing
/// params are also `BadParam` (no silent default curve — an unwired
/// curve is a graph bug). The identity curve
/// (`(0,0),(1/3,1/3),(2/3,2/3),(1,1)`) reproduces the input
/// byte-exact (the lerp error is ~ulps, far from any rounding
/// boundary — pinned by test). Output: `Image`, input size.
pub struct CurvesNode;

impl NodeImpl for CurvesNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = single_image(&inputs)?;
        let curve = Curve4::try_new([
            curve_param(params, "curve_p0")?,
            curve_param(params, "curve_p1")?,
            curve_param(params, "curve_p2")?,
            curve_param(params, "curve_p3")?,
        ])?;
        let mut data = Vec::with_capacity(src.data.len());
        for px in src.data.chunks_exact(4) {
            for p in px.iter().take(3) {
                data.push(byte_round(curve.evaluate(f32::from(*p) / 255.0)));
            }
            data.push(px[3]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// `invert`: `255 - v` per RGB channel; alpha passes through. No params.
/// Output: `Image`, input size.
pub struct InvertNode;

impl NodeImpl for InvertNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        _params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = single_image(&inputs)?;
        let mut data = Vec::with_capacity(src.data.len());
        for px in src.data.chunks_exact(4) {
            data.extend_from_slice(&[255 - px[0], 255 - px[1], 255 - px[2], px[3]]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// Registers `blur`, `sharpen`, `levels`, `curves`, `invert`.
pub fn register_filter_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("blur", Arc::new(BlurNode));
    registry.register("sharpen", Arc::new(SharpenNode));
    registry.register("levels", Arc::new(LevelsNode));
    registry.register("curves", Arc::new(CurvesNode));
    registry.register("invert", Arc::new(InvertNode));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeRegistry;

    fn registry() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        register_filter_nodes(&mut r);
        r
    }

    fn eval_img(
        r: &NodeRegistry,
        def: &str,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
    ) -> ImageBuffer {
        let imp = r.get(def).expect("registered");
        match imp
            .eval(inputs, params, &EvalContext::new(64, 64))
            .expect("filter evaluates")
        {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    fn img_input(buf: ImageBuffer) -> Vec<(String, NodeOutput)> {
        vec![("in".into(), NodeOutput::Image(buf))]
    }

    /// Single white pixel on black 5x5 (alpha 255 everywhere).
    fn impulse_5x5() -> ImageBuffer {
        let mut buf = ImageBuffer::filled(5, 5, [0, 0, 0, 255]).unwrap();
        buf.set_pixel(2, 2, [255, 255, 255, 255]);
        buf
    }

    #[test]
    fn blur_impulse_is_28_block_with_derivation() {
        // Radius 1 (k=3), horizontal pass first: only row 2 is nonzero;
        // its window over [0,0,255,0,0] gives x=1,2,3: (0+0+255)/3,
        // (0+255+0)/3, (255+0+0)/3 = 85 each — (255+1)/3 = 85 with the
        // round-to-nearest (sum + k/2)/k. So row 2 = [0,85,85,85,0].
        // Vertical pass: columns 1..3 each hold a single 85 at row 2, so
        // rows 1..3 read windows {0,0,85}/{0,85,0}/{85,0,0} = 85/3 →
        // (85+1)/3 = 28. The 3x3 neighborhood is uniformly 28 (NOT 85:
        // the second pass divides again — a single-pass box/9 gives the
        // same 28 here, 255/9 = 28.33 → 28). Everything outside is 0;
        // alpha stays 255. A wrong pass order or truncation-vs-rounding
        // mix fails the exact 28s.
        let r = registry();
        let buf = eval_img(
            &r,
            "blur",
            img_input(impulse_5x5()),
            &[("radius".into(), ParamValue::Int(1))],
        );
        assert_eq!((buf.width, buf.height), (5, 5));
        for y in 0..5 {
            for x in 0..5 {
                let in_block = (1..=3).contains(&x) && (1..=3).contains(&y);
                let expected = if in_block { 28 } else { 0 };
                assert_eq!(
                    buf.pixel(x, y),
                    Some([expected, expected, expected, 255]),
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn blur_radius_zero_and_negative_are_identity() {
        // radius 0 skips the kernel (clone); negative clamps to 0 (the
        // documented clamp, NOT BadParam).
        let r = registry();
        let src = impulse_5x5();
        for radius in [0, -3] {
            let buf = eval_img(
                &r,
                "blur",
                img_input(src.clone()),
                &[("radius".into(), ParamValue::Int(radius))],
            );
            assert_eq!(buf.data, src.data, "radius {radius} must be identity");
        }
    }

    #[test]
    fn blur_rejects_images_needing_no_input_or_uniform() {
        let r = registry();
        let imp = r.get("blur").expect("registered");
        let ctx = EvalContext::new(2, 2);
        assert!(
            matches!(
                imp.eval(vec![], &[], &ctx),
                Err(EvalError::MissingInput { .. })
            ),
            "unwired blur must be MissingInput"
        );
        let uniform = vec![("in".into(), NodeOutput::Uniform(ParamValue::Float(1.0)))];
        match imp.eval(uniform, &[], &ctx) {
            Err(EvalError::TypeMismatch { expected, got, .. }) => {
                assert_eq!(expected, "Image");
                assert_eq!(got, "Uniform");
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
        let bad = imp.eval(
            img_input(impulse_5x5()),
            &[("radius".into(), ParamValue::Float(1.0))],
            &ctx,
        );
        assert!(
            matches!(bad, Err(EvalError::BadParam { .. })),
            "mistyped radius must fail, got {bad:?}"
        );
    }

    #[test]
    fn sharpen_amount_zero_is_exact_identity() {
        // amount 0 → out = input + 0*(input - blur) = input, float-exact.
        let r = registry();
        let src = impulse_5x5();
        let buf = eval_img(
            &r,
            "sharpen",
            img_input(src.clone()),
            &[
                ("amount".into(), ParamValue::Float(0.0)),
                ("radius".into(), ParamValue::Int(1)),
            ],
        );
        assert_eq!(buf.data, src.data);
    }

    #[test]
    fn sharpen_unsharp_clamps_with_derivation() {
        // Same impulse, amount 1.0, radius 1: blur center = 28 (above).
        // Center: 255 + 1*(255-28) = 482 → clamp 255. Dark neighbor:
        // 0 + 1*(0-28) = -28 → clamp 0. Alpha passes through (set 128
        // on the impulse to prove it is NOT sharpened).
        let r = registry();
        let mut src = impulse_5x5();
        for px in src.data.chunks_exact_mut(4) {
            px[3] = 128;
        }
        let buf = eval_img(
            &r,
            "sharpen",
            img_input(src),
            &[
                ("amount".into(), ParamValue::Float(1.0)),
                ("radius".into(), ParamValue::Int(1)),
            ],
        );
        assert_eq!(buf.pixel(2, 2), Some([255, 255, 255, 128]), "hot center");
        assert_eq!(buf.pixel(0, 0), Some([0, 0, 0, 128]), "dark corner");
        assert_eq!(buf.pixel(2, 1), Some([0, 0, 0, 128]), "dark neighbor");
    }

    #[test]
    fn levels_identity_is_byte_exact_and_clamp_derived() {
        // Identity params take the clone fast path (no float trip).
        let r = registry();
        let src = ImageBuffer::new(
            4,
            2,
            vec![
                0, 10, 63, 255, 64, 128, 191, 255, 200, 223, 254, 255, 255, 1, 2, 128, 17, 34, 51,
                200, 99, 100, 101, 255, 5, 6, 7, 255, 250, 251, 252, 255,
            ],
        )
        .unwrap();
        let buf = eval_img(&r, "levels", img_input(src.clone()), &[]);
        assert_eq!(buf.data, src.data, "identity levels must pass through");

        // in_low 0.25: v < 0.25 clamps to t = 0 → byte 0. 63/255 ≈
        // 0.2471 < 0.25 → 0 (exact, no float luck: the comparison is
        // against the clamp, and any negative numerator clamps). 0 → 0,
        // 255 → t = 1 → 255. 128/255 ≈ 0.5020 → t ≈ 0.3359 → byte 86
        // (mirrored f32 math pins it exactly).
        let params = vec![("in_low".into(), ParamValue::Float(0.25))];
        let one = |b: u8| {
            eval_img(
                &r,
                "levels",
                img_input(ImageBuffer::filled(1, 1, [b, b, b, 255]).unwrap()),
                &params,
            )
            .pixel(0, 0)
            .expect("in bounds")[0]
        };
        assert_eq!(one(0), 0);
        assert_eq!(one(63), 0, "63/255 < 0.25 clamps to 0");
        assert_eq!(one(255), 255);
        let v = 128f32 / 255.0;
        let t = ((v - 0.25) / 0.75).clamp(0.0, 1.0);
        let expected = (t * 255.0 + 0.5) as u8;
        assert_eq!(expected, 86, "hand derivation pins 86");
        assert_eq!(one(128), expected, "mirrored f32 math");
    }

    #[test]
    fn levels_degenerate_range_is_bad_param() {
        let r = registry();
        let imp = r.get("levels").expect("registered");
        let ctx = EvalContext::new(2, 2);
        let err = imp.eval(
            img_input(ImageBuffer::filled(2, 2, [9, 9, 9, 255]).unwrap()),
            &[
                ("in_low".into(), ParamValue::Float(0.5)),
                ("in_high".into(), ParamValue::Float(0.5)),
            ],
            &ctx,
        );
        assert!(
            matches!(err, Err(EvalError::BadParam { .. })),
            "in_low == in_high must fail, got {err:?}"
        );
        let neg_gamma = imp.eval(
            img_input(ImageBuffer::filled(2, 2, [9, 9, 9, 255]).unwrap()),
            &[("gamma".into(), ParamValue::Float(-1.0))],
            &ctx,
        );
        assert!(
            matches!(neg_gamma, Err(EvalError::BadParam { .. })),
            "negative gamma must fail, got {neg_gamma:?}"
        );
    }

    fn identity_curve_params() -> Vec<(String, ParamValue)> {
        vec![
            ("curve_p0".into(), ParamValue::Vec2([0.0, 0.0])),
            ("curve_p1".into(), ParamValue::Vec2([1.0 / 3.0, 1.0 / 3.0])),
            ("curve_p2".into(), ParamValue::Vec2([2.0 / 3.0, 2.0 / 3.0])),
            ("curve_p3".into(), ParamValue::Vec2([1.0, 1.0])),
        ]
    }

    #[test]
    fn curves_identity_is_exact_over_all_256_values() {
        // 256x1 ramp covering every byte; slope-1 segments keep the
        // f32 lerp error at ~ulps, far from any .5 rounding boundary,
        // so every byte round-trips exactly. Any nonlinear-segment bug
        // fails somewhere in the 256.
        let r = registry();
        let data: Vec<u8> = (0..=255u8).flat_map(|b| [b, b, b, 255]).collect();
        let src = ImageBuffer::new(256, 1, data).unwrap();
        let buf = eval_img(
            &r,
            "curves",
            img_input(src.clone()),
            &identity_curve_params(),
        );
        assert_eq!(buf.data, src.data);
    }

    #[test]
    fn curves_steep_midpoint_derived_and_monotonicity_guarded() {
        // Points (0,0),(0.25,0.75),(0.75,0.75),(1,1): input 128 lands in
        // the flat middle segment, y1 - y0 = 0.0 so y = 0.75 EXACTLY
        // regardless of t → byte (0.75*255+0.5) = 191.75 → 191. A
        // segment-indexing bug (wrong bracketing) fails this.
        let r = registry();
        let params = vec![
            ("curve_p0".into(), ParamValue::Vec2([0.0, 0.0])),
            ("curve_p1".into(), ParamValue::Vec2([0.25, 0.75])),
            ("curve_p2".into(), ParamValue::Vec2([0.75, 0.75])),
            ("curve_p3".into(), ParamValue::Vec2([1.0, 1.0])),
        ];
        let buf = eval_img(
            &r,
            "curves",
            img_input(ImageBuffer::filled(1, 1, [128, 128, 128, 200]).unwrap()),
            &params,
        );
        assert_eq!(
            buf.pixel(0, 0),
            Some([191, 191, 191, 200]),
            "flat-segment midpoint with alpha passthrough"
        );

        // Non-monotonic x (p1.x == p2.x — the ControlCurve rejection)
        // and mistyped params are BadParam.
        let imp = r.get("curves").expect("registered");
        let ctx = EvalContext::new(1, 1);
        let flat = img_input(ImageBuffer::filled(1, 1, [9, 9, 9, 255]).unwrap());
        let non_mono = vec![
            ("curve_p0".into(), ParamValue::Vec2([0.0, 0.0])),
            ("curve_p1".into(), ParamValue::Vec2([0.5, 0.5])),
            ("curve_p2".into(), ParamValue::Vec2([0.5, 0.8])),
            ("curve_p3".into(), ParamValue::Vec2([1.0, 1.0])),
        ];
        assert!(
            matches!(
                imp.eval(flat.clone(), &non_mono, &ctx),
                Err(EvalError::BadParam { .. })
            ),
            "equal x must fail"
        );
        let mistyped = vec![
            ("curve_p0".into(), ParamValue::Float(0.0)),
            ("curve_p1".into(), ParamValue::Vec2([0.3, 0.3])),
            ("curve_p2".into(), ParamValue::Vec2([0.6, 0.6])),
            ("curve_p3".into(), ParamValue::Vec2([1.0, 1.0])),
        ];
        assert!(
            matches!(
                imp.eval(flat, &mistyped, &ctx),
                Err(EvalError::BadParam { .. })
            ),
            "non-Vec2 curve point must fail"
        );
    }

    #[test]
    fn invert_is_exact_with_mid_gray_derivation() {
        // 255 - v per RGB; 128 → 127 (255-128); alpha untouched.
        let r = registry();
        let src = ImageBuffer::new(2, 1, vec![0, 0, 0, 200, 128, 128, 128, 200]).unwrap();
        let buf = eval_img(&r, "invert", img_input(src), &[]);
        assert_eq!(
            buf.pixel(0, 0),
            Some([255, 255, 255, 200]),
            "black inverts to white"
        );
        assert_eq!(
            buf.pixel(1, 0),
            Some([127, 127, 127, 200]),
            "mid-gray 128 → 127"
        );
    }
}

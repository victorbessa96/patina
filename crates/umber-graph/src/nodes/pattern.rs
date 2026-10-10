//! Pattern generators: `checkerboard`, `dots`, `brick_pattern`.
//!
//! Wave-5 slice 2a (`docs/specs/node-graph-design.md`, "The node set").
//!
//! * `checkerboard`: MaterialX names it `checkerboard` (not `checker`) —
//!   the node_def is `checkerboard` so the mtlx layer maps it standard.
//!   "50% gray/white" in the design means equal coverage: v1 renders
//!   white `[255, 255, 255, 255]` and black `[0, 0, 0, 255]` cells.
//! * `dots`: white dots on black, one per cell center.
//! * `brick_pattern`: the DECLARED CUSTOM nodedef from the mtlx slice —
//!   its input list is taken from the merged `mtlx.rs`
//!   ([`crate::mtlx::painter_nodedefs`]): `scale: vector2`,
//!   `mortar: float`. The eval accepts the same params, so mtlx
//!   round-tripped graphs evaluate without renaming.

use crate::value::ImageBuffer;
use crate::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use std::sync::Arc;

fn find_param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

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

fn finite_float(
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

/// `checkerboard`: `scale: Float` (cells across, default 8.0, must be
/// finite and `> 0`). Cell `(cx, cy) = (floor(x * scale / w),
/// floor(y * scale / h))`; even parity `(cx + cy) % 2 == 0` is white,
/// odd is black. Output: `Image` at `ctx.resolution`.
pub struct CheckerNode;

impl NodeImpl for CheckerNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let scale = finite_float(params, "scale", 8.0)?;
        if scale <= 0.0 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "scale".into(),
            });
        }
        let (w, h) = ctx.resolution;
        check_resolution(w, h)?;
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let cx = (x as f32 * scale / w as f32).floor() as i64;
                let cy = (y as f32 * scale / h as f32).floor() as i64;
                let px = if (cx + cy) % 2 == 0 {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                data.extend_from_slice(&px);
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(w, h, data)?))
    }
}

/// Smoothstep on `[edge0, edge1]` (edges may straddle the sample).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `dots`: `scale: Float` (cells across, default 8.0, must be finite and
/// `> 0`), `radius: Float` (dot radius in cell units, default 0.25, must
/// be finite and `>= 0`). One white dot per cell center on black.
/// Antialiasing: a 1-pixel-wide smoothstep (half a pixel each side of
/// the radius, converted to cell units via the cell's pixel size), i.e.
/// `coverage = 1 - smoothstep(r - half_px, r + half_px, dist)` where
/// `dist` is the distance from the cell center in cell units. Output:
/// `Image` at `ctx.resolution`.
pub struct DotsNode;

impl NodeImpl for DotsNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let scale = finite_float(params, "scale", 8.0)?;
        if scale <= 0.0 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "scale".into(),
            });
        }
        let radius = finite_float(params, "radius", 0.25)?;
        if radius < 0.0 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "radius".into(),
            });
        }
        let (w, h) = ctx.resolution;
        check_resolution(w, h)?;
        // Pixels per cell (x/y may differ on non-square buffers); the AA
        // half-width uses the smaller so the edge never exceeds 1 pixel.
        let cell_px = (w as f32 / scale).min(h as f32 / scale).max(1e-6);
        let half = 0.5 / cell_px;
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let gx = (x as f32 + 0.5) * scale / w as f32;
                let gy = (y as f32 + 0.5) * scale / h as f32;
                let lx = gx - gx.floor() - 0.5;
                let ly = gy - gy.floor() - 0.5;
                let d = (lx * lx + ly * ly).sqrt();
                let cov = 1.0 - smoothstep(radius - half, radius + half, d);
                let b = (cov.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                data.extend_from_slice(&[b, b, b, 255]);
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(w, h, data)?))
    }
}

/// Brick-face and mortar colors (exact, documented for the mtlx layer).
pub const BRICK_FACE: [u8; 4] = [180, 90, 70, 255];
/// Brick-face and mortar colors (exact, documented for the mtlx layer).
pub const BRICK_MORTAR: [u8; 4] = [64, 64, 64, 255];

/// `brick_pattern`: Substance-style running-bond brick. Params (the
/// mtlx-declared input list — `scale: vector2`, `mortar: float`):
///
/// * `scale: Vec2` (default `[8.0, 8.0]`; a `Float` is also accepted as a
///   uniform `sx = sy` convenience) — bricks across × courses tall.
/// * `mortar: Float` (default 0.1, must satisfy `0 <= mortar < 1`) —
///   mortar width as a fraction of a brick cell.
///
/// Odd courses are offset by half a brick (`+0.5` cell in x). A texel is
/// mortar when `|fx| >= 0.5 - m/2` or `|fy| >= 0.5 - m/2` (fractional
/// cell coords centered on the brick); otherwise it is brick face.
/// Output: `Image` at `ctx.resolution`.
pub struct BrickNode;

impl NodeImpl for BrickNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let (sx, sy) = match find_param(params, "scale") {
            None => (8.0, 8.0),
            Some(ParamValue::Vec2([a, b])) => (*a, *b),
            Some(ParamValue::Float(a)) => (*a, *a),
            Some(_) => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "scale".into(),
                });
            }
        };
        if !sx.is_finite() || !sy.is_finite() || sx <= 0.0 || sy <= 0.0 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "scale".into(),
            });
        }
        let mortar = finite_float(params, "mortar", 0.1)?;
        if !(0.0..1.0).contains(&mortar) {
            return Err(EvalError::BadParam {
                node: 0,
                param: "mortar".into(),
            });
        }
        let (w, h) = ctx.resolution;
        check_resolution(w, h)?;
        let inset = 0.5 - mortar / 2.0;
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let gy = y as f32 * sy / h as f32;
                let iy = gy.floor() as i64;
                // Running bond: odd courses shift half a brick.
                let shift = if iy % 2 == 0 { 0.0 } else { 0.5 };
                let gx = x as f32 * sx / w as f32 + shift;
                let fx = (gx - gx.floor() - 0.5).abs();
                let fy = (gy - gy.floor() - 0.5).abs();
                let px = if fx >= inset || fy >= inset {
                    BRICK_MORTAR
                } else {
                    BRICK_FACE
                };
                data.extend_from_slice(&px);
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(w, h, data)?))
    }
}

/// Registers `checkerboard`, `dots`, `brick_pattern`.
pub fn register_pattern_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("checkerboard", Arc::new(CheckerNode));
    registry.register("dots", Arc::new(DotsNode));
    registry.register("brick_pattern", Arc::new(BrickNode));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_def(params: &[(String, ParamValue)], def: &str, w: u32, h: u32) -> ImageBuffer {
        let mut r = crate::NodeRegistry::new();
        register_pattern_nodes(&mut r);
        let imp = r.get(def).expect("registered");
        match imp
            .eval(vec![], params, &EvalContext::new(w, h))
            .expect("pattern evaluates")
        {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    #[test]
    fn checkerboard_parity_is_exact() {
        // 8x8, scale 8: pixel == cell, cx = x, cy = y. (0,0): parity 0 →
        // white; (1,0): parity 1 → black. (A flipped-parity impl fails the
        // first assert; a constant impl fails the second.)
        let buf = eval_def(
            &[("scale".into(), ParamValue::Float(8.0))],
            "checkerboard",
            8,
            8,
        );
        assert_eq!(
            buf.pixel(0, 0),
            Some([255, 255, 255, 255]),
            "even cell (0,0) is white"
        );
        assert_eq!(
            buf.pixel(1, 0),
            Some([0, 0, 0, 255]),
            "odd cell (1,0) is black"
        );
        assert_eq!(
            buf.pixel(0, 1),
            Some([0, 0, 0, 255]),
            "odd cell (0,1) is black"
        );
        assert_eq!(
            buf.pixel(7, 7),
            Some([255, 255, 255, 255]),
            "(7+7) % 2 == 0 → white"
        );
    }

    #[test]
    fn dots_center_is_white_and_corner_is_black() {
        // 16x16, scale 4 (4px cells), radius 0.25: cell (0,0) spans x,y in
        // 0..4; pixel (1,1) has local (−0.125, −0.125), d ≈ 0.177, inside
        // the AA band's bright side → > 200. Pixel (0,0) has local
        // (−0.375, −0.375), d ≈ 0.53, far outside → < 10. Mirrored exact
        // bytes pin both (a swapped-colors impl fails the thresholds; a
        // wrong-radius impl fails the exact bytes).
        let buf = eval_def(
            &[
                ("scale".into(), ParamValue::Float(4.0)),
                ("radius".into(), ParamValue::Float(0.25)),
            ],
            "dots",
            16,
            16,
        );
        let mirrored = |x: u32, y: u32| {
            let gx = (x as f32 + 0.5) * 4.0 / 16.0;
            let gy = (y as f32 + 0.5) * 4.0 / 16.0;
            let lx = gx - gx.floor() - 0.5;
            let ly = gy - gy.floor() - 0.5;
            let d = (lx * lx + ly * ly).sqrt();
            let half = 0.5 / 4.0;
            let t = ((d - (0.25 - half)) / (2.0 * half)).clamp(0.0, 1.0);
            let cov = 1.0 - (t * t * (3.0 - 2.0 * t));
            let b = (cov.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [b, b, b, 255]
        };
        let center = buf.pixel(1, 1).expect("in bounds");
        assert_eq!(center, mirrored(1, 1), "near-center mirrored exact");
        assert!(
            center[0] > 200,
            "cell-center pixel must be white, got {}",
            center[0]
        );
        let corner = buf.pixel(0, 0).expect("in bounds");
        assert_eq!(corner, mirrored(0, 0), "corner mirrored exact");
        assert!(
            corner[0] < 10,
            "cell corner must be black, got {}",
            corner[0]
        );
    }

    #[test]
    fn brick_mortar_and_half_bond_offset_are_exact() {
        // 16x16, scale [4,2] (4px bricks, 8px courses), mortar 0.2
        // (inset = 0.4 cell). Course boundary at y = 8 (gy = 1.0 →
        // |fy| = 0.5 ≥ 0.4): pixel (0,8) is mortar [64,64,64,255].
        // Pixel (2,2): gx = 0.5 → fx = 0, gy = 0.25 → |fy| = 0.25 < 0.4:
        // face [180,90,70,255]. Vertical joint in course 0 at x = 0
        // (gx = 0 → |fx| = 0.5): pixel (0,2) is mortar; course 1 shifts
        // +0.5, so pixel (0,10) has gx = 0.5 → face. A bond-ignoring impl
        // fails the last pair.
        let params = vec![
            ("scale".into(), ParamValue::Vec2([4.0, 2.0])),
            ("mortar".into(), ParamValue::Float(0.2)),
        ];
        let buf = eval_def(&params, "brick_pattern", 16, 16);
        assert_eq!(
            buf.pixel(0, 8),
            Some(BRICK_MORTAR),
            "course boundary is mortar"
        );
        assert_eq!(buf.pixel(2, 2), Some(BRICK_FACE), "brick face");
        assert_eq!(
            buf.pixel(0, 2),
            Some(BRICK_MORTAR),
            "course-0 vertical joint at x = 0"
        );
        assert_eq!(
            buf.pixel(0, 10),
            Some(BRICK_FACE),
            "course-1 is shifted half a brick: x = 0 is face"
        );
        assert_eq!(
            buf.pixel(2, 10),
            Some(BRICK_MORTAR),
            "course-1 joint lands at x = 2"
        );
        // Mortar darkness: exact declared colors differ as documented.
        assert_ne!(BRICK_FACE, BRICK_MORTAR);
    }

    #[test]
    fn brick_accepts_float_scale_and_rejects_bad_mortar() {
        let mut r = crate::NodeRegistry::new();
        register_pattern_nodes(&mut r);
        let imp = r.get("brick_pattern").expect("registered");
        let ctx = EvalContext::new(8, 8);
        let float_scale = imp.eval(
            vec![],
            &[
                ("scale".into(), ParamValue::Float(4.0)),
                ("mortar".into(), ParamValue::Float(0.1)),
            ],
            &ctx,
        );
        assert!(
            float_scale.is_ok(),
            "Float scale is accepted, got {float_scale:?}"
        );
        let bad_mortar = imp.eval(vec![], &[("mortar".into(), ParamValue::Float(1.5))], &ctx);
        assert!(
            matches!(bad_mortar, Err(EvalError::BadParam { .. })),
            "mortar ≥ 1 must fail, got {bad_mortar:?}"
        );
    }
}

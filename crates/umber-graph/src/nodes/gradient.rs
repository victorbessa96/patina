//! Gradient generator: `gradient` (linear / radial / angular).
//!
//! Wave-5 slice 2a (`docs/specs/node-graph-design.md`, "The node set").
//!
//! # Params
//!
//! * `type: Int` (default 0) — 0 = linear, 1 = radial, 2 = angular.
//! * `repeat: Int` (default 1, must be `>= 1`) — ramp cycle count.
//! * `center: Vec2` (default `[0.5, 0.5]`) — rotation pivot (linear),
//!   ring origin (radial), angle origin (angular).
//! * `angle: Float` (default 0.0, degrees) — see rotation math below.
//!
//! # UV convention
//!
//! Screen space: `u = x / w`, `v = y / h` (y down). This differs from the
//! noise nodes' V-flipped UV on purpose: a gradient's ramp direction is
//! defined top→bottom for `angle = 0`, which is the screen-space reading.
//! Each node's convention is documented so the mtlx/app layers agree.
//!
//! # Rotation math
//!
//! `p = uv - center`; `rad = angle * π / 180`; with `c = cos(rad)` and
//! `s = sin(rad)`, `ruv = center + (p.x * c - p.y * s, p.x * s + p.y * c)`
//! (CCW in screen space, y-down). The linear ramp reads `ruv.y`, so
//! `angle = 0` ramps top→bottom and `angle = 90` ramps left→right
//! (`ruv.y` reduces to `u` for the default center). Radial distance is
//! rotation-invariant and uses unrotated `uv`; angular uses unrotated
//! `uv - center` (a rotation would only phase-shift the cycle).
//!
//! # Ramp mapping
//!
//! * linear: `t = fract(ruv.y * repeat)`.
//! * radial: `t = fract(dist(uv, center) * repeat)`.
//! * angular: `t = fract((atan2(d.y, d.x) / 2π + 0.5) * repeat)` where
//!   `d = uv - center` (so the −x axis maps to 0 and the +x axis to 0.5
//!   at `repeat = 1`).
//!
//! `fract(x) = x - floor(x)` (always in `[0, 1)`); `t` → RGBA8 grayscale
//! ramp `(t * 255 + 0.5) as u8`, alpha 255. Note the exclusive upper
//! bound: on a height-`h` buffer the last row reads `(h-1)/h`, i.e. 223
//! for `h = 8` — the ramp never touches 255 because `t = 1` would wrap
//! to 0 under `fract`.

use crate::value::ImageBuffer;
use crate::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use std::sync::Arc;

fn find_param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// `gradient`: linear/radial/angular grayscale ramp (see module docs).
/// Output: `Image` at `ctx.resolution`.
pub struct GradientNode;

impl NodeImpl for GradientNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let kind = match find_param(params, "type") {
            None => 0,
            Some(ParamValue::Int(t)) => *t,
            Some(_) => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "type".into(),
                });
            }
        };
        if !(0..=2).contains(&kind) {
            return Err(EvalError::BadParam {
                node: 0,
                param: "type".into(),
            });
        }
        let repeat = match find_param(params, "repeat") {
            None => 1,
            Some(ParamValue::Int(r)) => *r,
            Some(_) => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "repeat".into(),
                });
            }
        };
        if repeat < 1 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "repeat".into(),
            });
        }
        let center = match find_param(params, "center") {
            None => [0.5, 0.5],
            Some(ParamValue::Vec2(c)) => *c,
            Some(_) => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "center".into(),
                });
            }
        };
        if !center[0].is_finite() || !center[1].is_finite() {
            return Err(EvalError::BadParam {
                node: 0,
                param: "center".into(),
            });
        }
        let angle = match find_param(params, "angle") {
            None => 0.0,
            Some(ParamValue::Float(a)) => *a,
            Some(_) => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "angle".into(),
                });
            }
        };
        if !angle.is_finite() {
            return Err(EvalError::BadParam {
                node: 0,
                param: "angle".into(),
            });
        }

        let (w, h) = ctx.resolution;
        if w == 0 || h == 0 {
            return Err(EvalError::Image(crate::ImageError::EmptyDimensions {
                width: w,
                height: h,
            }));
        }
        if w > 8192 || h > 8192 || w as u64 * h as u64 > (1u64 << 26) {
            return Err(EvalError::Image(crate::ImageError::DimensionsTooLarge {
                width: w,
                height: h,
            }));
        }

        let rad = angle * std::f32::consts::PI / 180.0;
        let (c, s) = (rad.cos(), rad.sin());
        let rep = repeat as f32;
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let u = x as f32 / w as f32;
                let v = y as f32 / h as f32;
                let px = u - center[0];
                let py = v - center[1];
                let ruy = center[1] + (px * s + py * c);
                let t = match kind {
                    0 => fract(ruy * rep),
                    1 => {
                        let d = (px * px + py * py).sqrt();
                        fract(d * rep)
                    }
                    _ => {
                        let a = py.atan2(px);
                        fract((a / (2.0 * std::f32::consts::PI) + 0.5) * rep)
                    }
                };
                let b = (t.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                data.extend_from_slice(&[b, b, b, 255]);
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(w, h, data)?))
    }
}

/// Registers `gradient`.
pub fn register_gradient_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("gradient", Arc::new(GradientNode));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_gradient(params: &[(String, ParamValue)], w: u32, h: u32) -> ImageBuffer {
        let mut r = crate::NodeRegistry::new();
        register_gradient_nodes(&mut r);
        let imp = r.get("gradient").expect("registered");
        match imp
            .eval(vec![], params, &EvalContext::new(w, h))
            .expect("gradient evaluates")
        {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    #[test]
    fn linear_ramps_top_to_bottom_with_exact_endpoints() {
        // 8x8, repeat 1, angle 0: t = y/8, so row y reads
        // (y/8*255+0.5) as u8. Row 0 → 0; row 7 → (7/8*255+0.5)=223
        // (NOT ~254: the fract ramp's exclusive upper bound keeps t < 1,
        // since t = 1 would wrap to 0 — see module docs).
        let buf = eval_gradient(
            &[
                ("type".into(), ParamValue::Int(0)),
                ("repeat".into(), ParamValue::Int(1)),
                ("angle".into(), ParamValue::Float(0.0)),
            ],
            8,
            8,
        );
        for y in 0..8 {
            let expected = ((y as f32 / 8.0) * 255.0 + 0.5) as u8;
            for x in 0..8 {
                assert_eq!(
                    buf.pixel(x, y),
                    Some([expected, expected, expected, 255]),
                    "row {y} must read the mirrored ramp value {expected}"
                );
            }
        }
        assert_eq!(buf.pixel(0, 0), Some([0, 0, 0, 255]), "top row ~0");
        assert_eq!(
            buf.pixel(0, 7),
            Some([223, 223, 223, 255]),
            "bottom row is the derived max 223"
        );
        assert!(
            buf.pixel(0, 0).unwrap()[0] < buf.pixel(0, 7).unwrap()[0],
            "ramp direction: top < bottom"
        );
    }

    #[test]
    fn radial_is_zero_at_center_and_max_at_corner() {
        // 8x8, center [0.5,0.5]: pixel (4,4) samples (0.5,0.5) exactly →
        // dist 0 → byte 0. Corner (0,0) samples (0,0): dist sqrt(0.5) ≈
        // 0.7071 → (0.7071*255+0.5) = 180, the farthest sample, hence the
        // buffer max (repeat 1 keeps every dist < 1, so no fract wrap).
        let buf = eval_gradient(
            &[
                ("type".into(), ParamValue::Int(1)),
                ("repeat".into(), ParamValue::Int(1)),
            ],
            8,
            8,
        );
        assert_eq!(
            buf.pixel(4, 4),
            Some([0, 0, 0, 255]),
            "center pixel samples dist 0"
        );
        let expected_corner = ((0.5f32.sqrt()) * 255.0 + 0.5) as u8;
        assert_eq!(
            buf.pixel(0, 0),
            Some([expected_corner, expected_corner, expected_corner, 255]),
            "corner (0,0) is the mirrored max"
        );
        assert_eq!(expected_corner, 180);
        let max = buf
            .data
            .chunks_exact(4)
            .map(|px| px[0])
            .max()
            .expect("nonempty");
        assert_eq!(max, expected_corner, "corner holds the buffer max");
    }

    #[test]
    fn angular_spans_a_full_cycle_and_is_not_constant() {
        // Smoke + derivation: pixel (7,4) samples d = (0.375, 0) → a = 0 →
        // t = 0.5 → 128; pixel (0,4) samples d = (-0.5, 0) → a = π → t = 0
        // (fract(1.0) = 0). A broken atan2 mapping fails these; a constant
        // impl fails the min < max assert.
        let buf = eval_gradient(&[("type".into(), ParamValue::Int(2))], 8, 8);
        assert_eq!(
            buf.pixel(7, 4),
            Some([128, 128, 128, 255]),
            "+x axis maps to half cycle"
        );
        assert_eq!(
            buf.pixel(0, 4),
            Some([0, 0, 0, 255]),
            "−x axis maps to cycle start"
        );
        let (mut min, mut max) = (255u8, 0u8);
        for px in buf.data.chunks_exact(4) {
            min = min.min(px[0]);
            max = max.max(px[0]);
        }
        assert!(min < max, "angular ramp must vary");
    }

    #[test]
    fn gradient_rejects_bad_type_and_repeat() {
        let mut r = crate::NodeRegistry::new();
        register_gradient_nodes(&mut r);
        let imp = r.get("gradient").expect("registered");
        let ctx = EvalContext::new(4, 4);
        let bad_type = imp.eval(vec![], &[("type".into(), ParamValue::Int(9))], &ctx);
        assert!(
            matches!(bad_type, Err(EvalError::BadParam { .. })),
            "type 9 must fail, got {bad_type:?}"
        );
        let bad_repeat = imp.eval(vec![], &[("repeat".into(), ParamValue::Int(0))], &ctx);
        assert!(
            matches!(bad_repeat, Err(EvalError::BadParam { .. })),
            "repeat 0 must fail, got {bad_repeat:?}"
        );
    }
}

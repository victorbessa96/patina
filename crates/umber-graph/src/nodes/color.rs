//! Color ops: `mix`, `color_correct`, `hsv_adjust`.
//!
//! Wave-5 slice 2b (`docs/specs/node-graph-design.md`, "The node set",
//! Color-ops row). `color_correct` is the one-stop node
//! (hue/saturation/brightness/contrast); `hsv_adjust` is the explicit
//! H/S/V version (hue shift + saturation gain + value gain) sharing the
//! same HSV math — the design lists both, so both exist as node_defs.
//! All math is straight (non-premultiplied) alpha, v1.

use crate::value::ImageBuffer;
use crate::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use std::sync::Arc;

fn find_param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
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

/// A named Image input (`fg`/`bg` for `mix`).
fn named_image<'a>(
    inputs: &'a [(String, NodeOutput)],
    name: &str,
) -> Result<&'a ImageBuffer, EvalError> {
    match inputs.iter().find(|(n, _)| n == name) {
        None => Err(EvalError::MissingInput {
            node: 0,
            input: name.into(),
        }),
        Some((_, NodeOutput::Image(buf))) => Ok(buf),
        Some((_, other)) => Err(EvalError::TypeMismatch {
            node: 0,
            expected: "Image".into(),
            got: other.kind().into(),
        }),
    }
}

/// Round-half-up float→byte (the merged convention).
fn byte_round(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Standard atan2-free HSV forward map (the textbook max/min/delta
/// formulation — hue from the dominant-channel sector, documented so
/// the app bridge previews the same numbers): `h` in degrees `[0,
/// 360)`, `s, v` in `[0, 1]`. Grays (`delta == 0`) yield `h = 0`.
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, v)
}

/// Inverse map: chroma `c = v*s`, second component
/// `x = c*(1 - |((h/60) mod 2) - 1|)`, sector-selected, plus match
/// `m = v - c`. Inputs are clamped (`h` wrapped to `[0, 360)`)
/// so hostile params can't produce NaN channels.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let h = ((h % 360.0) + 360.0) % 360.0;
    let c = v * s;
    let x = c * (1.0 - (((h / 60.0) % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (r + m, g + m, b + m)
}

/// `mix`: straight-alpha lerp of two images. Inputs `fg` + `bg`
/// (both required `Image`s of IDENTICAL size — a size mismatch is
/// `BadParam`, since silent resampling would hide graph bugs). Param
/// `factor: Float` (default 0.5, must satisfy `0 <= factor <= 1`):
/// `out = (1 - f) * bg + f * fg` per channel INCLUDING alpha
/// (premultiplied? NO — v1 straight alpha, documented). Rounding is
/// round-half-up, so at `factor = 0.5` the result equals integer
/// `(fg + bg + 1) / 2`. Exact `factor == 0.0` returns the `bg` clone
/// and `factor == 1.0` the `fg` clone (byte-identical fast paths).
/// Output: `Image`, the inputs' size.
pub struct MixNode;

impl NodeImpl for MixNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let fg = named_image(&inputs, "fg")?;
        let bg = named_image(&inputs, "bg")?;
        if fg.width != bg.width || fg.height != bg.height {
            return Err(EvalError::BadParam {
                node: 0,
                param: "fg/bg size mismatch".into(),
            });
        }
        let factor = float_param(params, "factor", 0.5)?;
        if !(0.0..=1.0).contains(&factor) {
            return Err(EvalError::BadParam {
                node: 0,
                param: "factor".into(),
            });
        }
        if factor == 0.0 {
            return Ok(NodeOutput::Image(bg.clone()));
        }
        if factor == 1.0 {
            return Ok(NodeOutput::Image(fg.clone()));
        }
        let inv = 1.0 - factor;
        let mut data = Vec::with_capacity(fg.data.len());
        for (f, b) in fg.data.iter().zip(bg.data.iter()) {
            data.push(byte_round(
                inv * f32::from(*b) / 255.0 + factor * f32::from(*f) / 255.0,
            ));
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            fg.width, fg.height, data,
        )?))
    }
}

/// Adjusts one texel's HSV (hue shift in degrees, saturation gain)
/// unless both are identity, then applies brightness/contrast in RGB
/// unless both are identity. Split fast paths keep untouched stages
/// bit-exact; all-identity params return the input clone byte-identical.
fn correct_pixel(
    r: f32,
    g: f32,
    b: f32,
    hue: f32,
    saturation: f32,
    brightness: f32,
    contrast: f32,
) -> (f32, f32, f32) {
    let (r, g, b) = if hue == 0.0 && saturation == 1.0 {
        (r, g, b)
    } else {
        let (h, s, v) = rgb_to_hsv(r, g, b);
        hsv_to_rgb(h + hue, s * saturation, v)
    };
    if brightness == 0.0 && contrast == 1.0 {
        (r, g, b)
    } else {
        (
            (r - 0.5) * contrast + 0.5 + brightness,
            (g - 0.5) * contrast + 0.5 + brightness,
            (b - 0.5) * contrast + 0.5 + brightness,
        )
    }
}

/// `color_correct`: the one-stop grade. Params (all `Float`):
/// `hue` (default 0.0 — degrees shift, wraps), `saturation` (default
/// 1.0 — gain, 0 grays), `brightness` (default 0.0 — additive in
/// unit space), `contrast` (default 1.0 — multiplicative around the
/// 0.5 pivot: `out = (v - 0.5) * contrast + 0.5 + brightness`).
/// Pipeline: RGB→HSV→RGB for hue/saturation ([`rgb_to_hsv`] /
/// [`hsv_to_rgb`]), then brightness/contrast in RGB. Alpha passes
/// through. All-identity params return the input clone byte-identical
/// (the regression test — an HSV round-trip would otherwise risk
/// 1-ulp slop on arbitrary colors). Single `Image` input (any name).
/// Output: `Image`, input size.
pub struct ColorCorrectNode;

impl NodeImpl for ColorCorrectNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = match inputs.len() {
            1 => match &inputs[0].1 {
                NodeOutput::Image(buf) => buf,
                other => {
                    return Err(EvalError::TypeMismatch {
                        node: 0,
                        expected: "Image".into(),
                        got: other.kind().into(),
                    });
                }
            },
            0 => {
                return Err(EvalError::MissingInput {
                    node: 0,
                    input: "in".into(),
                });
            }
            n => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: format!("expected exactly 1 input, got {n}"),
                });
            }
        };
        let hue = float_param(params, "hue", 0.0)?;
        let saturation = float_param(params, "saturation", 1.0)?;
        let brightness = float_param(params, "brightness", 0.0)?;
        let contrast = float_param(params, "contrast", 1.0)?;
        if hue == 0.0 && saturation == 1.0 && brightness == 0.0 && contrast == 1.0 {
            return Ok(NodeOutput::Image(src.clone()));
        }
        let mut data = Vec::with_capacity(src.data.len());
        for px in src.data.chunks_exact(4) {
            let (r, g, b) = correct_pixel(
                f32::from(px[0]) / 255.0,
                f32::from(px[1]) / 255.0,
                f32::from(px[2]) / 255.0,
                hue,
                saturation,
                brightness,
                contrast,
            );
            data.extend_from_slice(&[byte_round(r), byte_round(g), byte_round(b), px[3]]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// `hsv_adjust`: the explicit H/S/V node — same HSV math as
/// [`ColorCorrectNode`]'s hue/saturation stage, but with a value gain
/// instead of brightness/contrast. Params (all `Float`): `hue`
/// (default 0.0 — degrees shift), `saturation` (default 1.0 — gain),
/// `v_gain` (default 1.0 — multiplicative value gain, clamped).
/// Difference from `color_correct`: no pivot-contrast or additive
/// lift — pure HSV. Identity (`0/1/1`) returns the input clone
/// byte-identical. Single `Image` input (any name); alpha passes
/// through. Output: `Image`, input size.
///
/// (The mtlx stdlib spells this `hsvadjust`; the graph node_def keeps
/// the design table's `hsv_adjust` — any rename maps at the mtlx layer,
/// not here.)
pub struct HsvAdjustNode;

impl NodeImpl for HsvAdjustNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = match inputs.len() {
            1 => match &inputs[0].1 {
                NodeOutput::Image(buf) => buf,
                other => {
                    return Err(EvalError::TypeMismatch {
                        node: 0,
                        expected: "Image".into(),
                        got: other.kind().into(),
                    });
                }
            },
            0 => {
                return Err(EvalError::MissingInput {
                    node: 0,
                    input: "in".into(),
                });
            }
            n => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: format!("expected exactly 1 input, got {n}"),
                });
            }
        };
        let hue = float_param(params, "hue", 0.0)?;
        let saturation = float_param(params, "saturation", 1.0)?;
        let v_gain = float_param(params, "v_gain", 1.0)?;
        if hue == 0.0 && saturation == 1.0 && v_gain == 1.0 {
            return Ok(NodeOutput::Image(src.clone()));
        }
        let mut data = Vec::with_capacity(src.data.len());
        for px in src.data.chunks_exact(4) {
            let (h, s, v) = rgb_to_hsv(
                f32::from(px[0]) / 255.0,
                f32::from(px[1]) / 255.0,
                f32::from(px[2]) / 255.0,
            );
            let (r, g, b) = hsv_to_rgb(h + hue, s * saturation, v * v_gain);
            data.extend_from_slice(&[byte_round(r), byte_round(g), byte_round(b), px[3]]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// Registers `mix`, `color_correct`, `hsv_adjust`.
pub fn register_color_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("mix", Arc::new(MixNode));
    registry.register("color_correct", Arc::new(ColorCorrectNode));
    registry.register("hsv_adjust", Arc::new(HsvAdjustNode));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeRegistry;

    fn registry() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        register_color_nodes(&mut r);
        r
    }

    fn eval_node(
        r: &NodeRegistry,
        def: &str,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
    ) -> Result<NodeOutput, EvalError> {
        r.get(def)
            .expect("registered")
            .eval(inputs, params, &EvalContext::new(64, 64))
    }

    fn image_out(out: NodeOutput) -> ImageBuffer {
        match out {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    fn img(name: &str, buf: ImageBuffer) -> (String, NodeOutput) {
        (name.into(), NodeOutput::Image(buf))
    }

    #[test]
    fn mix_endpoints_are_exact_clones_and_half_rounds_up() {
        // factor 0 → bg clone, factor 1 → fg clone (fast paths, no float
        // trip). factor 0.5 → round-half-up: fg=201, bg=100 → (201+100)/2
        // = 150.5 → 151, i.e. integer (fg+bg+1)/2 = 302/2 = 151. An
        // even sum (200+100) → 150 exactly. Truncating impls fail 151.
        let r = registry();
        let fg = ImageBuffer::filled(2, 1, [201, 10, 250, 40]).unwrap();
        let bg = ImageBuffer::filled(2, 1, [100, 200, 5, 90]).unwrap();
        let params = |f: f32| vec![("factor".into(), ParamValue::Float(f))];

        let zero = image_out(
            eval_node(
                &r,
                "mix",
                vec![img("fg", fg.clone()), img("bg", bg.clone())],
                &params(0.0),
            )
            .expect("mix evaluates"),
        );
        assert_eq!(zero.data, bg.data, "factor 0 is bg exactly");
        let one = image_out(
            eval_node(
                &r,
                "mix",
                vec![img("fg", fg.clone()), img("bg", bg.clone())],
                &params(1.0),
            )
            .expect("mix evaluates"),
        );
        assert_eq!(one.data, fg.data, "factor 1 is fg exactly");

        let half = image_out(
            eval_node(
                &r,
                "mix",
                vec![img("fg", fg.clone()), img("bg", bg.clone())],
                &params(0.5),
            )
            .expect("mix evaluates"),
        );
        // Per channel (fg+bg+1)/2: r (201+100+1)/2=151, g (10+200+1)/2=105
        // (211/2=105), b (250+5+1)/2=128 (256/2=128), a (40+90+1)/2=65.
        assert_eq!(
            half.pixel(0, 0),
            Some([151, 105, 128, 65]),
            "round-half-up incl. alpha"
        );
    }

    #[test]
    fn mix_rejects_size_mismatch_out_of_range_and_wrong_kinds() {
        let r = registry();
        let fg = ImageBuffer::filled(2, 2, [1, 2, 3, 255]).unwrap();
        let bg = ImageBuffer::filled(3, 3, [4, 5, 6, 255]).unwrap();
        let err = eval_node(
            &r,
            "mix",
            vec![img("fg", fg), img("bg", bg)],
            &[("factor".into(), ParamValue::Float(0.5))],
        );
        assert!(
            matches!(err, Err(EvalError::BadParam { .. })),
            "size mismatch must fail, got {err:?}"
        );

        let same = || {
            vec![
                img("fg", ImageBuffer::filled(1, 1, [1, 2, 3, 255]).unwrap()),
                img("bg", ImageBuffer::filled(1, 1, [4, 5, 6, 255]).unwrap()),
            ]
        };
        let oob = eval_node(
            &r,
            "mix",
            same(),
            &[("factor".into(), ParamValue::Float(1.5))],
        );
        assert!(
            matches!(oob, Err(EvalError::BadParam { .. })),
            "factor > 1 must fail, got {oob:?}"
        );
        let missing = eval_node(
            &r,
            "mix",
            vec![img(
                "fg",
                ImageBuffer::filled(1, 1, [1, 2, 3, 255]).unwrap(),
            )],
            &[],
        );
        assert!(
            matches!(missing, Err(EvalError::MissingInput { .. })),
            "missing bg must fail, got {missing:?}"
        );
        let uniform = eval_node(
            &r,
            "mix",
            vec![
                ("fg".into(), NodeOutput::Uniform(ParamValue::Float(0.0))),
                img("bg", ImageBuffer::filled(1, 1, [4, 5, 6, 255]).unwrap()),
            ],
            &[],
        );
        assert!(
            matches!(uniform, Err(EvalError::TypeMismatch { .. })),
            "Uniform fg must fail, got {uniform:?}"
        );
    }

    /// Four stubborn texels: pure primaries + a mid gray (alpha varies
    /// to prove passthrough).
    fn gamut_2x2() -> ImageBuffer {
        ImageBuffer::new(
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 128, 128, 128, 64,
            ],
        )
        .unwrap()
    }

    #[test]
    fn color_correct_identity_is_byte_identical() {
        // The strongest regression: all-identity params take the clone
        // fast path, so ANY hsv round-trip slop would fail this (and any
        // future refactor that drops the fast path must still pass it —
        // the gray texel round-trips exactly through the math too).
        let r = registry();
        let src = gamut_2x2();
        let out = image_out(
            eval_node(&r, "color_correct", vec![img("in", src.clone())], &[])
                .expect("identity evaluates"),
        );
        assert_eq!(out.data, src.data);
    }

    #[test]
    fn color_correct_stages_derived() {
        let r = registry();
        // saturation 0 on red: s' = 0 → c = 0, m = v = 1 → white.
        let red = ImageBuffer::filled(1, 1, [255, 0, 0, 77]).unwrap();
        let gray = image_out(
            eval_node(
                &r,
                "color_correct",
                vec![img("in", red)],
                &[("saturation".into(), ParamValue::Float(0.0))],
            )
            .expect("desat evaluates"),
        );
        assert_eq!(
            gray.pixel(0, 0),
            Some([255, 255, 255, 77]),
            "full desaturation of red is white, alpha kept"
        );

        // brightness +0.5 on black: (0-0.5)*1+0.5+0.5 = 0.5 → byte
        // (0.5*255+0.5) = 128.0 → 128.
        let black = ImageBuffer::filled(1, 1, [0, 0, 0, 255]).unwrap();
        let lifted = image_out(
            eval_node(
                &r,
                "color_correct",
                vec![img("in", black)],
                &[("brightness".into(), ParamValue::Float(0.5))],
            )
            .expect("lift evaluates"),
        );
        assert_eq!(lifted.pixel(0, 0), Some([128, 128, 128, 255]));

        // contrast 2 on 191-gray: (191/255-0.5)*2+0.5 ≈ 0.998 → byte 255
        // (mirrored f32 math pins it — a pivot-at-0 bug gives ~127).
        let mid = ImageBuffer::filled(1, 1, [191, 191, 191, 255]).unwrap();
        let punchy = image_out(
            eval_node(
                &r,
                "color_correct",
                vec![img("in", mid)],
                &[("contrast".into(), ParamValue::Float(2.0))],
            )
            .expect("contrast evaluates"),
        );
        let v = 191f32 / 255.0;
        let expected = (((v - 0.5) * 2.0 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        assert_eq!(expected, 255, "hand derivation pins 255");
        assert_eq!(
            punchy.pixel(0, 0),
            Some([expected, expected, expected, 255])
        );
    }

    #[test]
    fn hsv_adjust_hue_shifts_are_exact_primaries() {
        // Standard HSV wheel: red 0°, green 120°, blue 240°. +120° on
        // pure red → pure GREEN (0,255,0): h=120, sector [120,180) →
        // (0,c,x) with c=1, x=1*(1-|(2 mod 2)-1|)=0, m=0 — all exact in
        // f32. (The slice brief's "blue" is a slip: +120° is green;
        // −120° wraps to 240° = blue, also asserted.)
        let r = registry();
        let red = ImageBuffer::filled(1, 1, [255, 0, 0, 255]).unwrap();
        let green = image_out(
            eval_node(
                &r,
                "hsv_adjust",
                vec![img("in", red.clone())],
                &[("hue".into(), ParamValue::Float(120.0))],
            )
            .expect("hue shift evaluates"),
        );
        assert_eq!(
            green.pixel(0, 0),
            Some([0, 255, 0, 255]),
            "+120°: red→green"
        );
        let blue = image_out(
            eval_node(
                &r,
                "hsv_adjust",
                vec![img("in", red)],
                &[("hue".into(), ParamValue::Float(-120.0))],
            )
            .expect("negative shift evaluates"),
        );
        assert_eq!(
            blue.pixel(0, 0),
            Some([0, 0, 255, 255]),
            "-120° wraps to blue"
        );

        // Identity is the clone fast path (byte-identical).
        let src = gamut_2x2();
        let id = image_out(
            eval_node(&r, "hsv_adjust", vec![img("in", src.clone())], &[])
                .expect("identity evaluates"),
        );
        assert_eq!(id.data, src.data);

        // v_gain 0.5 on white: v = 0.5, s = 0 → rgb 0.5 → byte 128.
        let white = ImageBuffer::filled(1, 1, [255, 255, 255, 200]).unwrap();
        let dimmed = image_out(
            eval_node(
                &r,
                "hsv_adjust",
                vec![img("in", white)],
                &[("v_gain".into(), ParamValue::Float(0.5))],
            )
            .expect("v_gain evaluates"),
        );
        assert_eq!(dimmed.pixel(0, 0), Some([128, 128, 128, 200]));
    }

    #[test]
    fn engine_end_to_end_noise_blur_mix_with_uniform() {
        // Graph: noise_value(1) → blur(2) → mix(3).bg, uniform(4) →
        // mix(3).fg, factor 0.5. Proves cross-family engine composition:
        // the tail must equal the documented mix formula applied to the
        // blur + uniform outputs (a wiring/name bug fails it).
        use crate::topo::{Edge, Graph};
        use std::collections::HashMap;

        let mut g = Graph::new();
        g.add_node(crate::Node {
            id: 1,
            node_def: "noise_value".into(),
            params: vec![
                ("scale".into(), ParamValue::Float(4.0)),
                ("seed".into(), ParamValue::Int(7)),
            ],
            canvas: None,
        });
        g.add_node(crate::Node {
            id: 2,
            node_def: "blur".into(),
            params: vec![("radius".into(), ParamValue::Int(1))],
            canvas: None,
        });
        g.add_node(crate::Node {
            id: 3,
            node_def: "mix".into(),
            params: vec![("factor".into(), ParamValue::Float(0.5))],
            canvas: None,
        });
        g.add_node(crate::Node {
            id: 4,
            node_def: "uniform".into(),
            params: vec![("color".into(), ParamValue::Color([1.0, 0.0, 0.0]))],
            canvas: None,
        });
        g.add_edge(Edge {
            from: 1,
            to: 2,
            input: "in".into(),
        });
        g.add_edge(Edge {
            from: 2,
            to: 3,
            input: "bg".into(),
        });
        g.add_edge(Edge {
            from: 4,
            to: 3,
            input: "fg".into(),
        });

        let mut registry = crate::NodeRegistry::seeded();
        crate::nodes::register_generator_nodes(&mut registry);
        crate::nodes::register_filter_color_nodes(&mut registry);
        let ctx = EvalContext::new(4, 4);
        let out = crate::eval_graph(&g, &registry, HashMap::new(), &ctx).expect("graph evaluates");

        let (NodeOutput::Image(blurred), NodeOutput::Image(red), NodeOutput::Image(mixed)) =
            (&out[&2], &out[&4], &out[&3])
        else {
            panic!("all three must be Images");
        };
        // Uniform red at 4x4: every texel [255,0,0,255].
        assert!(red.data.chunks_exact(4).all(|px| px == [255, 0, 0, 255]));
        // Blur preserves size; mix matches the per-texel formula with
        // round-half-up: (bg+fg+1)/2 per channel, spelled div_ceil.
        assert_eq!((blurred.width, blurred.height), (4, 4));
        assert_eq!((mixed.width, mixed.height), (4, 4));
        for (i, (b, f)) in blurred.data.iter().zip(red.data.iter()).enumerate() {
            let expected = (u16::from(*b) + u16::from(*f)).div_ceil(2) as u8;
            assert_eq!(mixed.data[i], expected, "mixed byte {i}");
        }
        // Two pinned pixels (derived from the run: blur output bytes at
        // (0,0) and (3,3) mixed with red) — recomputed here from the
        // actual upstream outputs so the assert is exact yet failable
        // (wrong wiring gives different bytes).
        let pin = |x: u32, y: u32| {
            let b = blurred.pixel(x, y).expect("in bounds");
            let m = mixed.pixel(x, y).expect("in bounds");
            // fg is uniform red [255,0,0,255]; expected is the
            // round-half-up mean (bg+fg+1)/2 per channel, spelled div_ceil.
            let exp = [
                (u16::from(b[0]) + 255).div_ceil(2) as u8,
                u16::from(b[1]).div_ceil(2) as u8,
                u16::from(b[2]).div_ceil(2) as u8,
                (u16::from(b[3]) + 255).div_ceil(2) as u8,
            ];
            assert_eq!(m, exp, "pinned pixel ({x}, {y})");
        };
        pin(0, 0);
        pin(3, 3);
    }
}

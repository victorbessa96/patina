//! Mask ops: `flood_fill`, `edge_detect`, `histogram_match`.
//!
//! Wave-5 slice 2c (`docs/specs/node-graph-design.md`, "The node set",
//! Mask-ops row). All three consume `Image` inputs (a `Uniform` input is
//! [`EvalError::TypeMismatch`]) and produce an `Image` the same size as
//! their input (NOT `ctx.resolution`: mask ops transform, they don't
//! generate).
//!
//! * `flood_fill`: connected-component labeling of a mask on luminance
//!   (`(r+g+b)/3 > 127`, integer-exact as `r+g+b > 381`), 4-neighborhood,
//!   row-major discovery order (smallest unvisited coordinate first, so
//!   label ids follow discovery order). Label `i` (1-based) renders as
//!   the deterministic gray `(i*37+11) % 255 + 1` (never 0: label 1 is
//!   49, label 2 is 86, label 3 is 123, …); background (outside the mask)
//!   renders opaque black `[0,0,0,255]`. v1 labels only — per-component
//!   attribute extraction (area, centroid, bbox à la Substance) is wave-6.
//! * `edge_detect`: RAW Sobel magnitude on luminance, normalized to
//!   `[0,255]` — deliberately NO threshold param (thresholding composes
//!   via `levels`, the node-graph way). `radius: Int` (default 1, clamped
//!   `1..=3`) dilates the kernel: same weights, samples taken at ±radius
//!   with edge clamping.
//! * `histogram_match`: per-channel CDF transfer (`input` + `target`
//!   images, sizes MAY differ — CDFs are normalized by pixel count, so a
//!   small target grades a large input). 256-bin CDFs, nearest-bin lookup
//!   with NO lerp: for input value `v` with `t = cdf_in[v]`, the output is
//!   the smallest `w` with `cdf_tgt[w] >= t`. RGB is matched; alpha passes
//!   through untouched (alpha is coverage, not color distribution).

use crate::value::ImageBuffer;
use crate::{EvalContext, EvalError, NodeImpl, NodeOutput, ParamValue};
use std::sync::Arc;

fn find_param<'a>(params: &'a [(String, ParamValue)], name: &str) -> Option<&'a ParamValue> {
    params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
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

/// A named `Image` input (`mask` for `flood_fill`/`edge_detect`,
/// `input`/`target` for `histogram_match` — the exact slot names from the
/// declared mtlx nodedefs in [`crate::mtlx::painter_nodedefs`]).
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

/// `flood_fill`: label the 4-connected components of the mask input.
/// Input `mask` (`Image`). No params. Output: `Image`, input size —
/// component pixels opaque gray by discovery order, background opaque
/// black. See the module docs for the inside test and gray mapping.
pub struct FloodFillNode;

impl NodeImpl for FloodFillNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        _params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = named_image(&inputs, "mask")?;
        let (w, h) = (src.width as usize, src.height as usize);
        // Inside test, integer-exact: mean(r,g,b) > 127 ⟺ r+g+b > 381.
        let inside: Vec<bool> = src
            .data
            .chunks_exact(4)
            .map(|px| u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2]) > 381)
            .collect();
        // Row-major discovery: the first unvisited inside texel in scan
        // order seeds the next label, so ids follow discovery order.
        let mut labels = vec![0u32; w * h];
        let mut next: u32 = 0;
        let mut stack: Vec<usize> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let seed = y * w + x;
                if !inside[seed] || labels[seed] != 0 {
                    continue;
                }
                next += 1;
                labels[seed] = next;
                stack.push(seed);
                while let Some(j) = stack.pop() {
                    let (jx, jy) = (j % w, j / w);
                    // 4-neighborhood ONLY: diagonals do NOT connect (an
                    // 8-neighborhood implementation fails the diagonal test).
                    if jx > 0 {
                        visit(&inside, &mut labels, &mut stack, next, j - 1);
                    }
                    if jx + 1 < w {
                        visit(&inside, &mut labels, &mut stack, next, j + 1);
                    }
                    if jy > 0 {
                        visit(&inside, &mut labels, &mut stack, next, j - w);
                    }
                    if jy + 1 < h {
                        visit(&inside, &mut labels, &mut stack, next, j + w);
                    }
                }
            }
        }
        let mut data = vec![0u8; w * h * 4];
        for (i, lab) in labels.iter().enumerate() {
            // Hash-free deterministic spread, never 0 (label 1 → 49,
            // label 2 → 86): adjacent components stay visibly distinct.
            // `lab <= w*h <= 2^26`, so `lab*37` cannot overflow u32.
            let g = if *lab == 0 {
                0
            } else {
                ((*lab * 37 + 11) % 255 + 1) as u8
            };
            data[i * 4..i * 4 + 4].copy_from_slice(&[g, g, g, 255]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

fn visit(inside: &[bool], labels: &mut [u32], stack: &mut Vec<usize>, label: u32, k: usize) {
    if inside[k] && labels[k] == 0 {
        labels[k] = label;
        stack.push(k);
    }
}

/// `edge_detect`: Sobel magnitude on luminance. Input `mask` (`Image`).
/// Param `radius: Int` (default 1, clamped `1..=3` — a dilated Sobel: the
/// classic weights sample neighbors at ±radius, edge-clamped). Output:
/// the RAW magnitude normalized by the kernel maximum `1020√2`
/// (`|gx|,|gy| <= (1+2+1)*255 = 1020` per axis, so the max magnitude is
/// `1020√2`), as opaque gray `[m,m,m,255]`. No threshold — feed `levels`.
pub struct EdgeDetectNode;

/// The kernel maximum: per-axis `|g| <= (1+2+1)*255 = 1020` (weights
/// unchanged by dilation), so `max |g| = 1020√2`.
const SOBEL_MAX_MAG: f32 = 1020.0 * std::f32::consts::SQRT_2;

impl NodeImpl for EdgeDetectNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = named_image(&inputs, "mask")?;
        let radius = int_param(params, "radius", 1)?.clamp(1, 3);
        let (w, h) = (src.width as usize, src.height as usize);
        let lum: Vec<f32> = src
            .data
            .chunks_exact(4)
            .map(|px| (f32::from(px[0]) + f32::from(px[1]) + f32::from(px[2])) / 3.0)
            .collect();
        // Luminance at clamped coordinates (edge-clamp: borders reuse the
        // edge texel, so a flat field has exactly zero gradient).
        let at = |x: i64, y: i64| -> f32 {
            let cx = x.clamp(0, w as i64 - 1) as usize;
            let cy = y.clamp(0, h as i64 - 1) as usize;
            lum[cy * w + cx]
        };
        let r = i64::from(radius);
        let mut data = Vec::with_capacity(w * h * 4);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let gx = (at(x + r, y - r) + 2.0 * at(x + r, y) + at(x + r, y + r))
                    - (at(x - r, y - r) + 2.0 * at(x - r, y) + at(x - r, y + r));
                let gy = (at(x - r, y + r) + 2.0 * at(x, y + r) + at(x + r, y + r))
                    - (at(x - r, y - r) + 2.0 * at(x, y - r) + at(x + r, y - r));
                let mag = (gx * gx + gy * gy).sqrt();
                let m = byte_round(mag / SOBEL_MAX_MAG);
                data.extend_from_slice(&[m, m, m, 255]);
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// `histogram_match`: grade `input` onto `target`'s distribution.
/// Inputs `input` + `target` (both `Image`; sizes MAY differ — CDFs are
/// normalized by pixel count). No params. Per RGB channel: 256-bin CDFs,
/// nearest-bin transfer (NO lerp — byte values assert exact): input value
/// `v` with `t = cdf_in[v]` maps to the smallest `w` with
/// `cdf_tgt[w] >= t`. Alpha passes through. Output: `Image`, input size.
pub struct HistogramMatchNode;

fn channel_cdf(hist: &[u32; 256], total: u32) -> [f64; 256] {
    let mut cdf = [0f64; 256];
    let mut acc: u64 = 0;
    for (i, c) in hist.iter().enumerate() {
        acc += u64::from(*c);
        cdf[i] = acc as f64 / f64::from(total);
    }
    cdf
}

impl NodeImpl for HistogramMatchNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        _params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let input = named_image(&inputs, "input")?;
        let target = named_image(&inputs, "target")?;
        // One transfer LUT per RGB channel (256×256 scan, deterministic).
        let mut luts = [[0u8; 256]; 3];
        for c in 0..3 {
            let mut hist_in = [0u32; 256];
            let mut hist_tgt = [0u32; 256];
            for px in input.data.chunks_exact(4) {
                hist_in[px[c] as usize] += 1;
            }
            for px in target.data.chunks_exact(4) {
                hist_tgt[px[c] as usize] += 1;
            }
            let n_in = input.width * input.height;
            let n_tgt = target.width * target.height;
            let cdf_in = channel_cdf(&hist_in, n_in);
            let cdf_tgt = channel_cdf(&hist_tgt, n_tgt);
            for v in 0..256 {
                let t = cdf_in[v];
                // Smallest w with cdf_tgt[w] >= t (cdf_tgt[255] == 1.0 >=
                // t always, so the scan terminates).
                let mut w = 255;
                for (k, ck) in cdf_tgt.iter().enumerate() {
                    if *ck >= t {
                        w = k;
                        break;
                    }
                }
                luts[c][v] = w as u8;
            }
        }
        let mut data = Vec::with_capacity(input.data.len());
        for px in input.data.chunks_exact(4) {
            data.extend_from_slice(&[
                luts[0][px[0] as usize],
                luts[1][px[1] as usize],
                luts[2][px[2] as usize],
                px[3],
            ]);
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            input.width,
            input.height,
            data,
        )?))
    }
}

/// Registers `flood_fill`, `edge_detect`, `histogram_match`.
pub fn register_mask_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("flood_fill", Arc::new(FloodFillNode));
    registry.register("edge_detect", Arc::new(EdgeDetectNode));
    registry.register("histogram_match", Arc::new(HistogramMatchNode));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeRegistry;

    fn registry() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        register_mask_nodes(&mut r);
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
            .expect("mask op evaluates")
        {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    fn mask_input(buf: ImageBuffer) -> Vec<(String, NodeOutput)> {
        vec![("mask".into(), NodeOutput::Image(buf))]
    }

    #[test]
    fn flood_fill_two_blobs_get_discovery_order_grays() {
        // 4x4, rows 0 and 2 white (full rows → two disjoint components;
        // rows are separated by the black row 1). Discovery order: row 0
        // first → label 1 → gray (1*37+11)%255+1 = 48+1 = 49; row 2 →
        // label 2 → (74+11)%255+1 = 85+1 = 86. Background opaque black.
        // A label-id bug (e.g. 0-based ids, or a different hash) fails the
        // exact 49/86; a fill that leaks across row 1 fails the 0s.
        let r = registry();
        let mut src = ImageBuffer::filled(4, 4, [0, 0, 0, 255]).unwrap();
        for x in 0..2 {
            src.set_pixel(x, 0, [255, 255, 255, 255]);
            src.set_pixel(x, 2, [255, 255, 255, 255]);
        }
        let buf = eval_img(&r, "flood_fill", mask_input(src), &[]);
        for y in 0..4 {
            for x in 0..4 {
                let expected = if y == 0 && x < 2 {
                    [49, 49, 49, 255]
                } else if y == 2 && x < 2 {
                    [86, 86, 86, 255]
                } else {
                    [0, 0, 0, 255]
                };
                assert_eq!(buf.pixel(x, y), Some(expected), "pixel ({x}, {y})");
            }
        }
    }

    #[test]
    fn flood_fill_single_pixel_and_full_mask_are_one_component() {
        // 1x1 white → exactly 1 component → label-1 gray 49.
        let r = registry();
        let one = eval_img(
            &r,
            "flood_fill",
            mask_input(ImageBuffer::filled(1, 1, [255, 255, 255, 255]).unwrap()),
            &[],
        );
        assert_eq!(one.pixel(0, 0), Some([49, 49, 49, 255]));
        // Full 2x2 white → one component covering all (no background).
        let full = eval_img(
            &r,
            "flood_fill",
            mask_input(ImageBuffer::filled(2, 2, [255, 255, 255, 255]).unwrap()),
            &[],
        );
        assert!(
            full.data.chunks_exact(4).all(|px| px == [49, 49, 49, 255]),
            "full mask is a single component"
        );
    }

    #[test]
    fn flood_fill_is_4_neighborhood_diagonals_stay_split() {
        // 2x2 with only (0,0) and (1,1) white: diagonal touch. Under
        // 4-neighborhood these are TWO components ((0,0) → label 1 → 49,
        // (1,1) → label 2 → 86); an 8-neighborhood implementation merges
        // them into one and fails the 86.
        let r = registry();
        let mut src = ImageBuffer::filled(2, 2, [0, 0, 0, 255]).unwrap();
        src.set_pixel(0, 0, [255, 255, 255, 255]);
        src.set_pixel(1, 1, [255, 255, 255, 255]);
        let buf = eval_img(&r, "flood_fill", mask_input(src), &[]);
        assert_eq!(buf.pixel(0, 0), Some([49, 49, 49, 255]));
        assert_eq!(buf.pixel(1, 1), Some([86, 86, 86, 255]));
        assert_eq!(buf.pixel(1, 0), Some([0, 0, 0, 255]));
        assert_eq!(buf.pixel(0, 1), Some([0, 0, 0, 255]));
    }

    #[test]
    fn flood_fill_luminance_threshold_is_documented() {
        // Mid-gray 127 (sum 381, NOT > 381) is OUTSIDE; 128 (sum 384) is
        // inside. Pins the `> 127` boundary against `>=` slips.
        let r = registry();
        let src = ImageBuffer::new(2, 1, vec![127, 127, 127, 255, 128, 128, 128, 255]).unwrap();
        let buf = eval_img(&r, "flood_fill", mask_input(src), &[]);
        assert_eq!(buf.pixel(0, 0), Some([0, 0, 0, 255]), "127 is outside");
        assert_eq!(buf.pixel(1, 0), Some([49, 49, 49, 255]), "128 is inside");
    }

    #[test]
    fn edge_detect_vertical_step_is_180_at_edge_zero_beyond() {
        // 4x4, x<2 black, x>=2 white. At the edge columns the Sobel x
        // kernel sees a full step: gx = ±(1+2+1)*255 = ±1020, gy = 0
        // (rows uniform, clamped borders included), so mag = 1020 and the
        // normalized byte is round(1020/(1020√2)*255) = round(255/√2) =
        // round(180.31…) = 180. Columns 0 and 3 see a flat clamped window
        // → exactly 0. A wrong normalization (e.g. /2040, or a thresholded
        // variant) fails the 180; a wrap-mode sampler fails the 0s.
        let r = registry();
        let mut src = ImageBuffer::filled(4, 4, [0, 0, 0, 255]).unwrap();
        for y in 0..4 {
            for x in 2..4 {
                src.set_pixel(x, y, [255, 255, 255, 255]);
            }
        }
        let buf = eval_img(&r, "edge_detect", mask_input(src), &[]);
        for y in 0..4 {
            assert_eq!(buf.pixel(0, y), Some([0, 0, 0, 255]), "far black col");
            assert_eq!(buf.pixel(1, y), Some([180, 180, 180, 255]), "edge black");
            assert_eq!(buf.pixel(2, y), Some([180, 180, 180, 255]), "edge white");
            assert_eq!(buf.pixel(3, y), Some([0, 0, 0, 255]), "far white col");
        }
    }

    #[test]
    fn edge_detect_radius_clamps_and_flat_is_zero() {
        // Flat white 3x3 → zero gradient everywhere (edge-clamp keeps the
        // window uniform even at borders).
        let r = registry();
        let flat = ImageBuffer::filled(3, 3, [200, 200, 200, 255]).unwrap();
        let buf = eval_img(&r, "edge_detect", mask_input(flat), &[]);
        assert!(
            buf.data.chunks_exact(4).all(|px| px == [0, 0, 0, 255]),
            "flat field has no edges"
        );
        // Radius clamps 1..=3, NOT errors: 0 behaves as 1, 99 as 3.
        let step = {
            let mut s = ImageBuffer::filled(4, 4, [0, 0, 0, 255]).unwrap();
            for y in 0..4 {
                for x in 2..4 {
                    s.set_pixel(x, y, [255, 255, 255, 255]);
                }
            }
            s
        };
        let one = eval_img(
            &r,
            "edge_detect",
            mask_input(step.clone()),
            &[("radius".into(), ParamValue::Int(1))],
        );
        let zero = eval_img(
            &r,
            "edge_detect",
            mask_input(step.clone()),
            &[("radius".into(), ParamValue::Int(0))],
        );
        assert_eq!(zero.data, one.data, "radius 0 clamps to 1");
        let three = eval_img(
            &r,
            "edge_detect",
            mask_input(step.clone()),
            &[("radius".into(), ParamValue::Int(3))],
        );
        let huge = eval_img(
            &r,
            "edge_detect",
            mask_input(step),
            &[("radius".into(), ParamValue::Int(99))],
        );
        assert_eq!(huge.data, three.data, "radius 99 clamps to 3");
    }

    #[test]
    fn histogram_match_derived_transfer_with_per_channel_proof() {
        // Input 4x1: R = [0,85,170,255] (each once, N=4 → cdf_in 0.25 /
        // 0.5 / 0.75 / 1.0), G = 0 (cdf 1.0 at 0), B = 255 (cdf 1.0 at
        // 255), A = 200 (passthrough probe). Target 4x1: R = [0,0,255,255]
        // (cdf 0.5 up to 254, 1.0 at 255), G = 128, B = 64, A = 77.
        // R: 0 (t=.25) → w=0; 85 (t=.5) → w=0; 170 (t=.75) → w=255;
        // 255 (t=1) → w=255. G: 0 → smallest w with cdf>=1.0 → 128.
        // B: 255 → 64. A: 200 untouched (a matched alpha would give 77).
        let r = registry();
        let input = ImageBuffer::new(
            4,
            1,
            vec![
                0, 0, 255, 200, 85, 0, 255, 200, 170, 0, 255, 200, 255, 0, 255, 200,
            ],
        )
        .unwrap();
        let target = ImageBuffer::new(
            4,
            1,
            vec![
                0, 128, 64, 77, 0, 128, 64, 77, 255, 128, 64, 77, 255, 128, 64, 77,
            ],
        )
        .unwrap();
        let buf = eval_img(
            &r,
            "histogram_match",
            vec![
                ("input".into(), NodeOutput::Image(input)),
                ("target".into(), NodeOutput::Image(target)),
            ],
            &[],
        );
        assert_eq!(buf.pixel(0, 0), Some([0, 128, 64, 200]));
        assert_eq!(buf.pixel(1, 0), Some([0, 128, 64, 200]));
        assert_eq!(buf.pixel(2, 0), Some([255, 128, 64, 200]));
        assert_eq!(buf.pixel(3, 0), Some([255, 128, 64, 200]));
    }

    #[test]
    fn mask_ops_reject_bad_wiring_and_kinds() {
        let r = registry();
        let ctx = EvalContext::new(2, 2);
        let img = ImageBuffer::filled(2, 2, [9, 9, 9, 255]).unwrap();
        // Missing inputs.
        for def in ["flood_fill", "edge_detect"] {
            let imp = r.get(def).expect("registered");
            assert!(
                matches!(
                    imp.eval(vec![], &[], &ctx),
                    Err(EvalError::MissingInput { .. })
                ),
                "{def} unwired must be MissingInput"
            );
            // Uniform input is kind confusion, not a silent default.
            let uniform = vec![("mask".into(), NodeOutput::Uniform(ParamValue::Float(1.0)))];
            match imp.eval(uniform, &[], &ctx) {
                Err(EvalError::TypeMismatch { expected, got, .. }) => {
                    assert_eq!(expected, "Image");
                    assert_eq!(got, "Uniform");
                }
                other => panic!("{def}: expected TypeMismatch, got {other:?}"),
            }
        }
        // histogram_match needs BOTH inputs.
        let imp = r.get("histogram_match").expect("registered");
        let half = vec![("input".into(), NodeOutput::Image(img.clone()))];
        assert!(
            matches!(
                imp.eval(half, &[], &ctx),
                Err(EvalError::MissingInput { .. })
            ),
            "missing target must fail"
        );
        // Mistyped radius.
        let edge = r.get("edge_detect").expect("registered");
        let bad = edge.eval(
            mask_input(img),
            &[("radius".into(), ParamValue::Float(1.0))],
            &ctx,
        );
        assert!(
            matches!(bad, Err(EvalError::BadParam { .. })),
            "mistyped radius must fail, got {bad:?}"
        );
    }

    #[test]
    fn engine_end_to_end_flood_fill_into_edge_detect() {
        // Graph: uniform(white 4x4)(1) → flood_fill(2) → edge_detect(3).
        // A full-white mask is ONE component → node 2 renders every texel
        // label-1 gray [49,49,49,255]; node 3 sees a FLAT field → Sobel
        // windows are uniform (edge-clamp included) → every magnitude is
        // exactly 0. The tail must also equal a direct registry eval of
        // edge_detect on node 2's output (proves engine composition, not
        // just constants). A broken edge ("in" vs "mask" naming, dropped
        // wiring) fails to evaluate at all.
        use crate::{eval_graph, Graph, Node};
        use std::collections::HashMap;

        let mut g = Graph::new();
        g.add_node(Node {
            id: 1,
            node_def: "uniform".into(),
            params: vec![
                ("color".into(), ParamValue::Color([1.0, 1.0, 1.0])),
                ("resolution".into(), ParamValue::Vec2([4.0, 4.0])),
            ],
            canvas: None,
        });
        g.add_node(Node {
            id: 2,
            node_def: "flood_fill".into(),
            params: vec![],
            canvas: None,
        });
        g.add_node(Node {
            id: 3,
            node_def: "edge_detect".into(),
            params: vec![],
            canvas: None,
        });
        g.add_edge(crate::Edge {
            from: 1,
            to: 2,
            input: "mask".into(),
        });
        g.add_edge(crate::Edge {
            from: 2,
            to: 3,
            input: "mask".into(),
        });
        let mut registry = crate::NodeRegistry::seeded();
        register_mask_nodes(&mut registry);
        let ctx = EvalContext::new(4, 4);
        let out = eval_graph(&g, &registry, HashMap::new(), &ctx).expect("graph evaluates");
        match &out[&2] {
            NodeOutput::Image(buf) => {
                assert_eq!((buf.width, buf.height), (4, 4));
                assert!(
                    buf.data.chunks_exact(4).all(|px| px == [49, 49, 49, 255]),
                    "full mask is one component"
                );
            }
            other => panic!("expected an Image, got {other:?}"),
        }
        match &out[&3] {
            NodeOutput::Image(buf) => {
                assert_eq!((buf.width, buf.height), (4, 4));
                assert!(
                    buf.data.chunks_exact(4).all(|px| px == [0, 0, 0, 255]),
                    "flat labels have no edges"
                );
            }
            other => panic!("expected an Image, got {other:?}"),
        }
        let direct = registry
            .get("edge_detect")
            .expect("registered")
            .eval(vec![("mask".into(), out[&2].clone())], &[], &ctx)
            .expect("direct eval works");
        assert_eq!(out[&3], direct, "graph eval == direct registry eval");
    }
}

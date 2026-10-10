//! Spatial: `direction_warp`, `triplanar_blend`, `mesh_map_generator`.
//!
//! Wave-5 slice 2c (`docs/specs/node-graph-design.md`, "The node set",
//! Spatial row).
//!
//! * `direction_warp`: UV warp of an image. Input `input` (`Image`).
//!   Param `vector: Vec2` (default `[0.015, 0.0]`) — the UNIFORM offset in
//!   UV units. Optional input `vector` (`Image`): when connected it
//!   OVERRIDES the uniform param (per-pixel offset = the texel's R/G
//!   remapped from `[0,255]` to `[-1,1]` times the `strength: Float` param,
//!   default `0.05`; B/A ignored). Either way the output texel `(x, y)` is
//!   the bilinear sample of the input at `uv - offset`, edge-clamped —
//!   i.e. a FORWARD warp: positive vectors shift content toward +x/+y (a
//!   white column at `x = 0` with `vector = [0.25, 0]` on a 4-wide image
//!   lands at `x = 1`). All four channels warp identically (a resample,
//!   not a grade). Uniform warp is v1; the per-pixel `vector` INPUT (driven
//!   by a noise node for spatially-varying flow) is also wired — v2 may add
//!   scaling/curl modes. The vector map must match the input's
//!   size ([`EvalError::BadParam`] otherwise — silent resampling would
//!   hide graph bugs).
//! * `triplanar_blend`: v1 is a WEIGHTED BLEND of three same-resolution
//!   inputs (`x`, `y`, `z` — all `Image`, identical size or `BadParam`)
//!   under normalized `weights: Vec3` (default `[1,1,1]`; zero-sum weights
//!   are `BadParam`; negative weights extrapolate, honestly documented).
//!   Output texel = `(w0*x + w1*y + w2*z) / (w0+w1+w2)` per channel
//!   (alpha blends too). This is NOT a world-space projection: the real
//!   triplanar mapping (world position/normal → per-axis UVs) arrives with
//!   the mesh-bridge slice — v1 blends three ALIGNED buffers.
//! * `mesh_map_generator`: the graph-side access point for baked mesh
//!   maps. Param `map_name: Asset` (the map name string, e.g. `"AO"`).
//!   v1 pulls from [`EvalContext::external`]: the node looks up
//!   `external["mesh_map:{name}"]` and outputs it EXACTLY (clone). A
//!   missing entry — or one that isn't an `Image` — is
//!   [`EvalError::MissingInput`] with a clear message (e.g.
//!   `"mesh_map:AO not provided in context"`): v1 treats a mistyped
//!   external as missing rather than kind-confusion, because the app-side
//!   bridge always provides images. The app side (bake outputs) provides
//!   these entries; the node is the graph-side handle.

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

fn vec2_param(
    params: &[(String, ParamValue)],
    name: &str,
    default: [f32; 2],
) -> Result<[f32; 2], EvalError> {
    let v = match find_param(params, name) {
        None => default,
        Some(ParamValue::Vec2(v)) => *v,
        Some(_) => {
            return Err(EvalError::BadParam {
                node: 0,
                param: name.into(),
            });
        }
    };
    if !v[0].is_finite() || !v[1].is_finite() {
        return Err(EvalError::BadParam {
            node: 0,
            param: name.into(),
        });
    }
    Ok(v)
}

fn vec3_param(
    params: &[(String, ParamValue)],
    name: &str,
    default: [f32; 3],
) -> Result<[f32; 3], EvalError> {
    let v = match find_param(params, name) {
        None => default,
        Some(ParamValue::Vec3(v)) => *v,
        Some(_) => {
            return Err(EvalError::BadParam {
                node: 0,
                param: name.into(),
            });
        }
    };
    if !v.iter().all(|c| c.is_finite()) {
        return Err(EvalError::BadParam {
            node: 0,
            param: name.into(),
        });
    }
    Ok(v)
}

/// A required named `Image` input.
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

/// Byte-space round (bilinear results already live in 0..=255):
/// half-away-from-zero, then saturate. All warp-test values are exact,
/// so this only guards fractional blends.
fn byte_space_round(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Bilinear sample of `src` at pixel coordinates (`sx`, `sy`) with
/// edge clamping (out-of-range corners reuse the edge texel).
fn bilinear(src: &ImageBuffer, sx: f32, sy: f32) -> [f32; 4] {
    let (w, h) = (src.width as i32, src.height as i32);
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;
    let fx = (sx - x0 as f32).clamp(0.0, 1.0);
    let fy = (sy - y0 as f32).clamp(0.0, 1.0);
    let xa = x0.clamp(0, w - 1) as u32;
    let xb = (x0 + 1).clamp(0, w - 1) as u32;
    let ya = y0.clamp(0, h - 1) as u32;
    let yb = (y0 + 1).clamp(0, h - 1) as u32;
    let p00 = src.pixel(xa, ya).expect("clamped in bounds");
    let p10 = src.pixel(xb, ya).expect("clamped in bounds");
    let p01 = src.pixel(xa, yb).expect("clamped in bounds");
    let p11 = src.pixel(xb, yb).expect("clamped in bounds");
    let mut out = [0f32; 4];
    for c in 0..4 {
        let top = f32::from(p00[c]) + (f32::from(p10[c]) - f32::from(p00[c])) * fx;
        let bot = f32::from(p01[c]) + (f32::from(p11[c]) - f32::from(p01[c])) * fx;
        out[c] = top + (bot - top) * fy;
    }
    out
}

/// `direction_warp`: forward UV warp (bilinear, edge-clamped). See the
/// module docs for the uniform vs per-pixel offset paths.
pub struct DirectionWarpNode;

impl NodeImpl for DirectionWarpNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let src = named_image(&inputs, "input")?;
        let uniform = vec2_param(params, "vector", [0.015, 0.0])?;
        let strength = float_param(params, "strength", 0.05)?;
        // Optional per-pixel vector map (overrides the uniform when
        // connected — NOT added to it).
        let map: Option<&ImageBuffer> = match inputs.iter().find(|(n, _)| n == "vector") {
            None => None,
            Some((_, NodeOutput::Image(buf))) => {
                if buf.width != src.width || buf.height != src.height {
                    return Err(EvalError::BadParam {
                        node: 0,
                        param: "vector/input size mismatch".into(),
                    });
                }
                Some(buf)
            }
            Some((_, other)) => {
                return Err(EvalError::TypeMismatch {
                    node: 0,
                    expected: "Image".into(),
                    got: other.kind().into(),
                });
            }
        };
        let (w, h) = (src.width as f32, src.height as f32);
        let mut data = Vec::with_capacity(src.data.len());
        for y in 0..src.height {
            for x in 0..src.width {
                // UV offset for this texel (uniform path: the param as-is;
                // per-pixel path: R/G remapped [0,255] → [-1,1] × strength).
                let off = match map {
                    None => uniform,
                    Some(m) => {
                        let px = m.pixel(x, y).expect("same size, in bounds");
                        [
                            (f32::from(px[0]) / 255.0 * 2.0 - 1.0) * strength,
                            (f32::from(px[1]) / 255.0 * 2.0 - 1.0) * strength,
                        ]
                    }
                };
                // Forward warp: content shifts by +offset, so the sample
                // sits at uv − offset. In pixel coordinates (pixel centers
                // at (x+0.5)/w) that is exactly (x − off.x*w, …).
                let s = bilinear(src, x as f32 - off[0] * w, y as f32 - off[1] * h);
                data.extend_from_slice(&[
                    byte_space_round(s[0]),
                    byte_space_round(s[1]),
                    byte_space_round(s[2]),
                    byte_space_round(s[3]),
                ]);
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            src.width, src.height, data,
        )?))
    }
}

/// `triplanar_blend`: normalized weighted blend of three aligned buffers.
/// See the module docs (world-space projection deferred).
pub struct TriplanarBlendNode;

impl NodeImpl for TriplanarBlendNode {
    fn eval(
        &self,
        inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        _ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let x = named_image(&inputs, "x")?;
        let y = named_image(&inputs, "y")?;
        let z = named_image(&inputs, "z")?;
        if x.width != y.width || x.height != y.height || x.width != z.width || x.height != z.height
        {
            return Err(EvalError::BadParam {
                node: 0,
                param: "x/y/z size mismatch".into(),
            });
        }
        let w = vec3_param(params, "weights", [1.0, 1.0, 1.0])?;
        let sum = w[0] + w[1] + w[2];
        if sum == 0.0 {
            return Err(EvalError::BadParam {
                node: 0,
                param: "weights sum to zero".into(),
            });
        }
        let n = [w[0] / sum, w[1] / sum, w[2] / sum];
        let mut data = Vec::with_capacity(x.data.len());
        for (i, px) in x.data.chunks_exact(4).enumerate() {
            let py = &y.data[i * 4..i * 4 + 4];
            let pz = &z.data[i * 4..i * 4 + 4];
            for c in 0..4 {
                data.push(byte_round(
                    n[0] * f32::from(px[c]) / 255.0
                        + n[1] * f32::from(py[c]) / 255.0
                        + n[2] * f32::from(pz[c]) / 255.0,
                ));
            }
        }
        Ok(NodeOutput::Image(ImageBuffer::new(
            x.width, x.height, data,
        )?))
    }
}

/// `mesh_map_generator`: graph-side handle for a baked mesh map.
/// Param `map_name: Asset` (the map name, e.g. `"AO"`); outputs
/// `ctx.external["mesh_map:{name}"]` exactly. Missing (or non-image)
/// entries are [`EvalError::MissingInput`] — see the module docs.
pub struct MeshMapGeneratorNode;

impl NodeImpl for MeshMapGeneratorNode {
    fn eval(
        &self,
        _inputs: Vec<(String, NodeOutput)>,
        params: &[(String, ParamValue)],
        ctx: &EvalContext,
    ) -> Result<NodeOutput, EvalError> {
        let name = match find_param(params, "map_name") {
            Some(ParamValue::Asset(s)) => s.clone(),
            _ => {
                return Err(EvalError::BadParam {
                    node: 0,
                    param: "map_name".into(),
                });
            }
        };
        let key = format!("mesh_map:{name}");
        match ctx.external.get(&key) {
            Some(NodeOutput::Image(_)) => Ok(ctx.external[&key].clone()),
            // v1 treats a mistyped external as missing (the app bridge
            // always provides images; anything else is a bridge bug, and
            // MissingInput names the absent map).
            Some(_) | None => Err(EvalError::MissingInput {
                node: 0,
                input: format!("{key} not provided in context"),
            }),
        }
    }
}

/// Registers `direction_warp`, `triplanar_blend`, `mesh_map_generator`.
pub fn register_spatial_nodes(registry: &mut crate::NodeRegistry) {
    registry.register("direction_warp", Arc::new(DirectionWarpNode));
    registry.register("triplanar_blend", Arc::new(TriplanarBlendNode));
    registry.register("mesh_map_generator", Arc::new(MeshMapGeneratorNode));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeRegistry;

    fn registry() -> NodeRegistry {
        let mut r = NodeRegistry::new();
        register_spatial_nodes(&mut r);
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
            .expect("spatial op evaluates")
        {
            NodeOutput::Image(buf) => buf,
            other => panic!("expected an Image, got {other:?}"),
        }
    }

    fn img(name: &str, buf: ImageBuffer) -> (String, NodeOutput) {
        (name.into(), NodeOutput::Image(buf))
    }

    #[test]
    fn direction_warp_uniform_shifts_content_by_vector() {
        // 4x4, white leftmost column, vector [0.25, 0]: forward warp shifts
        // content +1 texel (0.25 UV × 4 px). Pixel-center math: output(x)
        // samples input at pixel-coordinate x − 1. x=1 → 0.0 → white
        // (the moved column). x=0 samples −1.0 → edge-CLAMPED to 0 → white
        // (the vacated column smears: a wrap-mode impl gives black here,
        // a zero-fill impl too — both fail). x=2,3 sample 1.0/2.0 → black.
        let r = registry();
        let mut src = ImageBuffer::filled(4, 4, [0, 0, 0, 255]).unwrap();
        for y in 0..4 {
            src.set_pixel(0, y, [255, 255, 255, 255]);
        }
        let buf = eval_img(
            &r,
            "direction_warp",
            vec![img("input", src)],
            &[("vector".into(), ParamValue::Vec2([0.25, 0.0]))],
        );
        for y in 0..4 {
            assert_eq!(
                buf.pixel(0, y),
                Some([255, 255, 255, 255]),
                "vacated column smears via edge-clamp ({y})"
            );
            assert_eq!(
                buf.pixel(1, y),
                Some([255, 255, 255, 255]),
                "moved column ({y})"
            );
            assert_eq!(buf.pixel(2, y), Some([0, 0, 0, 255]), "({y})");
            assert_eq!(buf.pixel(3, y), Some([0, 0, 0, 255]), "({y})");
        }
    }

    #[test]
    fn direction_warp_per_pixel_map_matches_uniform_path() {
        // Vector map of constant encoded (255,255) → remap [1,1] ×
        // strength 0.1 → offset [0.1,0.1] UV. Must equal the uniform path
        // with vector [0.1,0.1] BIT-EXACTLY (255/255 is exactly 1.0 in
        // f32, so both paths feed the sampler identical offsets — any
        // divergence, e.g. adding instead of overriding, fails). The ramp
        // image guarantees the warp actually moves bytes (output != input).
        let r = registry();
        let mut src = ImageBuffer::filled(4, 4, [0, 0, 0, 255]).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                src.set_pixel(x, y, [(x * 64) as u8, (y * 64) as u8, 0, 255]);
            }
        }
        let via_map = eval_img(
            &r,
            "direction_warp",
            vec![
                img("input", src.clone()),
                img(
                    "vector",
                    ImageBuffer::filled(4, 4, [255, 255, 0, 255]).unwrap(),
                ),
            ],
            &[("strength".into(), ParamValue::Float(0.1))],
        );
        let via_uniform = eval_img(
            &r,
            "direction_warp",
            vec![img("input", src.clone())],
            &[("vector".into(), ParamValue::Vec2([0.1, 0.1]))],
        );
        assert_eq!(
            via_map.data, via_uniform.data,
            "per-pixel [1,1]*0.1 must equal uniform [0.1,0.1]"
        );
        assert_ne!(
            via_uniform.data, src.data,
            "the 0.4-texel shift must move bytes on a ramp"
        );
    }

    #[test]
    fn triplanar_blend_weights_select_and_average_exactly() {
        // Three flat primaries. weights [1,0,0] → exactly red (normalized
        // [1,0,0]: 1.0*1.0 + 0 + 0 is f32-exact on 0/255 channels).
        // weights [1,1,1] → (255+0+0)/3 per channel: 1/3 in f32 is
        // 0.33333334, ×255 = 85.000002 → round-half-up 85. A truncating or
        // unnormalized impl fails one of the two.
        let r = registry();
        let red = ImageBuffer::filled(2, 2, [255, 0, 0, 255]).unwrap();
        let green = ImageBuffer::filled(2, 2, [0, 255, 0, 255]).unwrap();
        let blue = ImageBuffer::filled(2, 2, [0, 0, 255, 255]).unwrap();
        let inputs = || {
            vec![
                img("x", red.clone()),
                img("y", green.clone()),
                img("z", blue.clone()),
            ]
        };
        let only_x = eval_img(
            &r,
            "triplanar_blend",
            inputs(),
            &[("weights".into(), ParamValue::Vec3([1.0, 0.0, 0.0]))],
        );
        assert!(
            only_x.data.chunks_exact(4).all(|px| px == [255, 0, 0, 255]),
            "weights [1,0,0] select x exactly"
        );
        let equal = eval_img(
            &r,
            "triplanar_blend",
            inputs(),
            &[("weights".into(), ParamValue::Vec3([1.0, 1.0, 1.0]))],
        );
        assert!(
            equal.data.chunks_exact(4).all(|px| px == [85, 85, 85, 255]),
            "equal weights average to (85,85,85), got {:?}",
            equal.pixel(0, 0)
        );
        // Unnormalized weights still normalize: [2,0,0] == [1,0,0].
        let doubled = eval_img(
            &r,
            "triplanar_blend",
            inputs(),
            &[("weights".into(), ParamValue::Vec3([2.0, 0.0, 0.0]))],
        );
        assert_eq!(doubled.data, only_x.data, "weights normalize at eval");
    }

    #[test]
    fn mesh_map_generator_reads_external_or_reports_missing() {
        // Provided: outputs the entry EXACTLY (clone-equality, not a
        // resample or re-encode).
        let r = registry();
        let imp = r.get("mesh_map_generator").expect("registered");
        let ao = ImageBuffer::new(2, 1, vec![1, 2, 3, 255, 4, 5, 6, 128]).unwrap();
        let mut ctx = EvalContext::new(2, 1);
        ctx.external
            .insert("mesh_map:AO".into(), NodeOutput::Image(ao.clone()));
        let params = vec![("map_name".into(), ParamValue::Asset("AO".into()))];
        match imp.eval(vec![], &params, &ctx).expect("map resolves") {
            NodeOutput::Image(buf) => assert_eq!(buf, ao, "exact external output"),
            other => panic!("expected an Image, got {other:?}"),
        }
        // Absent: MissingInput naming the map (NOT UnknownNode or a
        // silent default — the message tells the bridge what to provide).
        let bare = EvalContext::new(2, 1);
        match imp.eval(vec![], &params, &bare) {
            Err(EvalError::MissingInput { input, .. }) => {
                assert_eq!(input, "mesh_map:AO not provided in context");
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
        // Mistyped map_name param and mistyped external entry.
        let bad_param = imp.eval(
            vec![],
            &[("map_name".into(), ParamValue::Float(1.0))],
            &bare,
        );
        assert!(
            matches!(bad_param, Err(EvalError::BadParam { .. })),
            "non-Asset map_name must fail, got {bad_param:?}"
        );
        let mut wrong_kind = EvalContext::new(2, 1);
        wrong_kind.external.insert(
            "mesh_map:AO".into(),
            NodeOutput::Uniform(ParamValue::Float(0.0)),
        );
        match imp.eval(vec![], &params, &wrong_kind) {
            Err(EvalError::MissingInput { input, .. }) => {
                assert!(
                    input.contains("mesh_map:AO"),
                    "mistyped external names the map, got {input:?}"
                );
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }

    #[test]
    fn spatial_nodes_reject_bad_wiring_and_kinds() {
        let r = registry();
        let ctx = EvalContext::new(2, 2);
        let img2 = ImageBuffer::filled(2, 2, [9, 9, 9, 255]).unwrap();
        // direction_warp without input.
        let warp = r.get("direction_warp").expect("registered");
        assert!(
            matches!(
                warp.eval(vec![], &[], &ctx),
                Err(EvalError::MissingInput { .. })
            ),
            "unwired warp must be MissingInput"
        );
        // Uniform vector MAP (not param) is kind confusion.
        let bad_map = warp.eval(
            vec![
                img("input", img2.clone()),
                ("vector".into(), NodeOutput::Uniform(ParamValue::Float(0.0))),
            ],
            &[],
            &ctx,
        );
        assert!(
            matches!(bad_map, Err(EvalError::TypeMismatch { .. })),
            "Uniform vector map must fail, got {bad_map:?}"
        );
        // Mismatched vector-map size.
        let big_map = warp.eval(
            vec![
                img("input", img2.clone()),
                img("vector", ImageBuffer::filled(3, 3, [0, 0, 0, 255]).unwrap()),
            ],
            &[],
            &ctx,
        );
        assert!(
            matches!(big_map, Err(EvalError::BadParam { .. })),
            "size-mismatched vector map must fail, got {big_map:?}"
        );
        // NaN vector param.
        let nan = warp.eval(
            vec![img("input", img2)],
            &[("vector".into(), ParamValue::Vec2([f32::NAN, 0.0]))],
            &ctx,
        );
        assert!(
            matches!(nan, Err(EvalError::BadParam { .. })),
            "NaN vector must fail, got {nan:?}"
        );
        // triplanar_blend: missing z, size mismatch, zero weights.
        let tri = r.get("triplanar_blend").expect("registered");
        let two = tri.eval(
            vec![
                img("x", ImageBuffer::filled(2, 2, [1, 2, 3, 255]).unwrap()),
                img("y", ImageBuffer::filled(2, 2, [4, 5, 6, 255]).unwrap()),
            ],
            &[],
            &ctx,
        );
        assert!(
            matches!(two, Err(EvalError::MissingInput { .. })),
            "missing z must fail, got {two:?}"
        );
        let mismatch = tri.eval(
            vec![
                img("x", ImageBuffer::filled(2, 2, [1, 2, 3, 255]).unwrap()),
                img("y", ImageBuffer::filled(3, 3, [4, 5, 6, 255]).unwrap()),
                img("z", ImageBuffer::filled(2, 2, [7, 8, 9, 255]).unwrap()),
            ],
            &[],
            &ctx,
        );
        assert!(
            matches!(mismatch, Err(EvalError::BadParam { .. })),
            "size mismatch must fail, got {mismatch:?}"
        );
        let zero_w = tri.eval(
            vec![
                img("x", ImageBuffer::filled(1, 1, [1, 2, 3, 255]).unwrap()),
                img("y", ImageBuffer::filled(1, 1, [4, 5, 6, 255]).unwrap()),
                img("z", ImageBuffer::filled(1, 1, [7, 8, 9, 255]).unwrap()),
            ],
            &[("weights".into(), ParamValue::Vec3([0.0, 0.0, 0.0]))],
            &ctx,
        );
        assert!(
            matches!(zero_w, Err(EvalError::BadParam { .. })),
            "zero-sum weights must fail, got {zero_w:?}"
        );
    }
}

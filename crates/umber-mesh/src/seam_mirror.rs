//! UV-space mirror math across seam edges (Wave-4 slice 2).
//!
//! Implements the mirror-dab emission math from
//! `docs/specs/seam-aware-stamping-design.md`: given a dab's UV position
//! near a [`SeamEdge`](crate::seam::SeamEdge), compute its image in the
//! OTHER island's UV coordinates through the `uv_a <-> uv_b` edge
//! correspondence, so the stroke paints every UV image of the same 3D
//! neighborhood.
//!
//! # Sign convention (read this before touching the formula)
//!
//! Frame the A-side edge: direction `da = uv_a[1] - uv_a[0]` of length
//! `La`, unit perpendicular `n_a = (da.y, -da.x) / La` (that is, `da`
//! rotated by -90 degrees), and likewise `db`, `Lb`, `n_b` on the B side.
//! `n_b` is the *corresponding-side* perpendicular: it is the image of
//! `n_a` under the edge correspondence, so "positive-offset side of A"
//! maps to "positive-offset side of B".
//!
//! The mirror of a point `p` near the A segment is the point in B at the
//! same normalized along-edge position, offset by the NEGATED
//! corresponding-side distance:
//!
//! ```text
//! s      = dot(p - uv_a[0], da) / La^2   (normalized edge parameter)
//! d_a    = dot(p - uv_a[0], n_a)         (signed perpendicular offset)
//! mirror = uv_b[0] + s * db - d_a * (Lb / La) * n_b
//! ```
//!
//! The negation is the whole point. A dab just INSIDE island A (near its
//! edge) must reappear just OUTSIDE island B's edge — in B's padding —
//! and vice versa. The mirror dab is stamped with the same radius as the
//! original, so its splat FOOTPRINT reaches back across B's seam line and
//! paints B's near-edge texels (the same 3D surface as the original dab)
//! with the same 3D-consistent color. Without the sign flip the mirror
//! would land on the same corresponding side — deep inside the neighbor
//! island at the mirrored offset — painting texels that have nothing to
//! do with the dab's 3D footprint while the true near-edge texels stay
//! blank (the seam-edge hole the design doc exists to close).
//!
//! Pinned ground truth (axis-aligned): edge `uv_a = [(0,0),(0,1)]`,
//! `uv_b = [(5,0),(5,1)]`. Here `n_a = n_b = (1,0)`, pointing into each
//! island. Dab `(0.1, 0.5)` (inside A, `d_a = +0.1`) mirrors to
//! `(4.9, 0.5)` (outside B, in B's padding); dab `(-0.1, 0.5)` (A's
//! padding) mirrors to `(5.1, 0.5)` (inside B). A formula yielding
//! `(5.1, ...)` for the first case has the sign wrong.
//!
//! # Gating and degeneracy
//!
//! The mirror applies only within `max_dist` of the edge SEGMENT
//! (distance to the segment, not the infinite line — extrapolation past
//! the endpoints still counts if the endpoint is close). The A side wins
//! ties: a point near both segments maps A -> B. Zero-length UV edges
//! (exact `f32` endpoint equality, hence zero length) yield `None`
//! rather than a divide-by-zero.
//!
//! # No panics
//!
//! Pure arithmetic over slices and `f32`s; no indexing, no allocation
//! failure modes, no unwrap.

use crate::seam::{SeamEdge, SeamGraph};

/// One mirrored dab position: `positions[source]` reflected across a seam
/// edge into the neighboring island's UV coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirrorMapping {
    /// Index into the input `positions` slice that produced this mirror.
    pub source: usize,
    /// The mirrored UV position (same radius/color/flow as the source dab,
    /// because it IS the same 3D paint).
    pub mirrored: [f32; 2],
}

/// Reflects UV point `p` across `edge` through the `uv_a <-> uv_b`
/// correspondence (see the module docs for the sign convention).
///
/// If `p` is within `max_dist` of the `uv_a` segment, returns its image
/// near the `uv_b` segment (A -> B); otherwise, if `p` is within
/// `max_dist` of the `uv_b` segment, returns the reverse image (B -> A).
/// Returns `None` when `p` is near neither segment or when either UV edge
/// is degenerate (exact endpoint equality or zero length).
pub fn mirror_uv(p: [f32; 2], edge: &SeamEdge, max_dist: f32) -> Option<[f32; 2]> {
    if edge.uv_a[0] == edge.uv_a[1] || edge.uv_b[0] == edge.uv_b[1] {
        return None;
    }
    let da = sub(edge.uv_a[1], edge.uv_a[0]);
    let db = sub(edge.uv_b[1], edge.uv_b[0]);
    let la_sq = dot(da, da);
    let lb_sq = dot(db, db);
    if la_sq == 0.0 || lb_sq == 0.0 {
        return None;
    }
    let (la, lb) = (la_sq.sqrt(), lb_sq.sqrt());
    if dist_to_segment(p, edge.uv_a[0], edge.uv_a[1]) <= max_dist {
        Some(map_side(p, edge.uv_a[0], da, la, edge.uv_b[0], db, lb))
    } else if dist_to_segment(p, edge.uv_b[0], edge.uv_b[1]) <= max_dist {
        Some(map_side(p, edge.uv_b[0], db, lb, edge.uv_a[0], da, la))
    } else {
        None
    }
}

/// Expands each UV dab position into its seam-mirror images.
///
/// For each position index and each edge in `graph.edges`, emits one
/// [`MirrorMapping`] whenever [`mirror_uv`] returns `Some`. Positions near
/// several edges correctly yield several mappings (several islands share
/// the 3D neighborhood); per edge per position there is at most one mirror
/// (`mirror_uv` tries the A side first, then the B side). A mirror exactly
/// equal to its source (coincident islands, on-seam dab) is skipped — it
/// would double-stamp the same texels.
pub fn mirror_positions(
    positions: &[[f32; 2]],
    graph: &SeamGraph,
    max_dist: f32,
) -> Vec<MirrorMapping> {
    let mut out = Vec::new();
    for (index, &p) in positions.iter().enumerate() {
        for edge in &graph.edges {
            if let Some(mirrored) = mirror_uv(p, edge, max_dist) {
                if mirrored == p {
                    continue;
                }
                out.push(MirrorMapping {
                    source: index,
                    mirrored,
                });
            }
        }
    }
    out
}

/// Maps `p` from the `from` island's edge frame into the `to` island:
/// same normalized along-edge parameter, negated stretch-compensated
/// perpendicular offset (the sign flip from the module docs).
///
/// `l_from` / `l_to` are the edge lengths (`|d_from|` / `|d_to|`); both
/// must be nonzero (callers guarantee this).
fn map_side(
    p: [f32; 2],
    from0: [f32; 2],
    d_from: [f32; 2],
    l_from: f32,
    to0: [f32; 2],
    d_to: [f32; 2],
    l_to: f32,
) -> [f32; 2] {
    // Normalized edge parameter: 0 at endpoint 0, 1 at endpoint 1,
    // unclamped (extrapolation past the endpoints is fine — the caller
    // already gated on segment distance). Normalized (not arc-length) so
    // the two endpoints correspond even under UV stretch.
    let s = dot(sub(p, from0), d_from) / (l_from * l_from);
    // Corresponding-side unit perpendiculars: rotate each edge by -90deg.
    let n_from = [d_from[1] / l_from, -d_from[0] / l_from];
    let n_to = [d_to[1] / l_to, -d_to[0] / l_to];
    let d = dot(sub(p, from0), n_from);
    // Negated, stretch-compensated offset: same magnitude distance on the
    // opposite corresponding side.
    let base = add(to0, scale(d_to, s));
    add(base, scale(n_to, -d * (l_to / l_from)))
}

/// Euclidean distance from `p` to the segment `a--b` (not the infinite
/// line). Zero-length segments degrade to distance-to-`a`, never NaN.
fn dist_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = sub(b, a);
    let len_sq = dot(ab, ab);
    if len_sq == 0.0 {
        return dist(p, a);
    }
    let s = dot(sub(p, a), ab) / len_sq;
    let closest = if s <= 0.0 {
        a
    } else if s >= 1.0 {
        b
    } else {
        add(a, scale(ab, s))
    };
    dist(p, closest)
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale(a: [f32; 2], s: f32) -> [f32; 2] {
    [a[0] * s, a[1] * s]
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// The brief's pinned axis-aligned fixture: island A at x in [0, ..],
    /// island B at x in [5, ..], seam the vertical segment y in [0, 1].
    fn axis_edge() -> SeamEdge {
        SeamEdge {
            tri_a: 0,
            tri_b: 1,
            uv_a: [[0.0, 0.0], [0.0, 1.0]],
            uv_b: [[5.0, 0.0], [5.0, 1.0]],
            mid_3d: Vec3::new(0.5, 0.5, 0.0),
        }
    }

    fn graph_of(edge: SeamEdge) -> SeamGraph {
        SeamGraph {
            edges: vec![edge],
            tri_touches_seam: vec![true, true],
        }
    }

    fn approx_eq(got: [f32; 2], want: [f32; 2], eps: f32) -> bool {
        (got[0] - want[0]).abs() <= eps && (got[1] - want[1]).abs() <= eps
    }

    #[test]
    fn axis_aligned_sanity_both_directions() {
        // Derivation: da = db = (0,1), La = Lb = 1, n_a = n_b = (1, 0).
        // p = (0.1, 0.5): s = 0.5, d_a = +0.1,
        //   mirror = (5,0) + 0.5*(0,1) - 0.1*(1,0) = (4.9, 0.5).
        // p = (-0.1, 0.5): s = 0.5, d_a = -0.1,
        //   mirror = (5,0) + 0.5*(0,1) + 0.1*(1,0) = (5.1, 0.5).
        let edge = axis_edge();
        let inside_a = mirror_uv([0.1, 0.5], &edge, 0.5).expect("near A must mirror");
        assert!(
            approx_eq(inside_a, [4.9, 0.5], 1e-6),
            "inside-A dab must mirror to (4.9, 0.5), got {inside_a:?}"
        );
        let padding_a = mirror_uv([-0.1, 0.5], &edge, 0.5).expect("near A must mirror");
        assert!(
            approx_eq(padding_a, [5.1, 0.5], 1e-6),
            "A-padding dab must mirror to (5.1, 0.5), got {padding_a:?}"
        );
        // Reverse: a dab inside B near the edge maps back into A's padding.
        // p = (5.1, 0.5): over db, s = 0.5, d_b = +0.1,
        //   mirror = (0,0) + 0.5*(0,1) - 0.1*(1,0) = (-0.1, 0.5).
        let inside_b = mirror_uv([5.1, 0.5], &edge, 0.5).expect("near B must mirror");
        assert!(
            approx_eq(inside_b, [-0.1, 0.5], 1e-6),
            "inside-B dab must mirror to (-0.1, 0.5), got {inside_b:?}"
        );
    }

    #[test]
    fn sign_flip_inside_a_lands_outside_b() {
        // The sign-error class: without the negation the inside-A dab
        // would land at (5.1, 0.5) — INSIDE B — instead of (4.9, 0.5).
        let edge = axis_edge();
        let mirrored = mirror_uv([0.1, 0.5], &edge, 0.5).expect("near A must mirror");
        assert!(
            mirrored[0] < 5.0,
            "inside-A mirror must fall OUTSIDE B (x < 5.0), got {mirrored:?}"
        );
        assert!(
            (5.0 - mirrored[0] - 0.1).abs() <= 1e-6,
            "mirror must preserve the offset magnitude (0.1), got {mirrored:?}"
        );
    }

    #[test]
    fn rotated_edge_45_degrees() {
        // Derivation: uv_a = [(0,0),(1,1)], uv_b = [(5,0),(6,1)].
        // da = db = (1,1), La = Lb = sqrt(2),
        // n_a = n_b = (1,-1)/sqrt(2).
        // A-side point: midpoint (0.5,0.5) pushed 0.1/sqrt(2) along +n_a:
        //   p = (0.5,0.5) + (0.1/sqrt(2))*(1,-1)/sqrt(2)
        //     = (0.5 + 0.05, 0.5 - 0.05) = (0.55, 0.45).
        // s = dot(p, da)/2 = 1.0/2 = 0.5; d_a = +0.1/sqrt(2).
        // q = (5,0) + 0.5*(1,1) = (5.5, 0.5);
        // mirror = q - (0.1/sqrt(2))*(1,-1)/sqrt(2)
        //        = (5.5 - 0.05, 0.5 + 0.05) = (5.45, 0.55).
        // (The sign bug would yield (5.55, 0.45) instead.)
        let edge = SeamEdge {
            tri_a: 0,
            tri_b: 1,
            uv_a: [[0.0, 0.0], [1.0, 1.0]],
            uv_b: [[5.0, 0.0], [6.0, 1.0]],
            mid_3d: Vec3::new(0.5, 0.5, 0.0),
        };
        let mirrored = mirror_uv([0.55, 0.45], &edge, 0.5).expect("near A must mirror");
        assert!(
            approx_eq(mirrored, [5.45, 0.55], 1e-5),
            "rotated mirror must be (5.45, 0.55), got {mirrored:?}"
        );
        // Offset magnitude preserved across the map: distance from the B
        // segment must equal the source's 0.1/sqrt(2) ~ 0.0707.
        let dist_b = dist_to_segment(mirrored, edge.uv_b[0], edge.uv_b[1]);
        assert!(
            (dist_b - 0.1 / 2.0f32.sqrt()).abs() <= 1e-5,
            "mirrored offset magnitude must be 0.1/sqrt(2), got {dist_b}"
        );
    }

    #[test]
    fn distance_gate_segment_not_line() {
        let edge = axis_edge();
        // 0.05 off the edge: comfortably inside max_dist 0.2.
        assert!(
            mirror_uv([0.05, 0.5], &edge, 0.2).is_some(),
            "point 0.05 from the segment must mirror"
        );
        // 0.5 off the edge: outside max_dist 0.2.
        assert!(
            mirror_uv([0.5, 0.5], &edge, 0.2).is_none(),
            "point 0.5 from both segments must return None"
        );
        // Past the endpoint: only 0.05 from the infinite LINE x = 0 but
        // ~0.30 from the SEGMENT endpoint (0,1) — must NOT mirror. This
        // is the test that pins segment-vs-line gating.
        assert!(
            mirror_uv([0.05, 1.3], &edge, 0.2).is_none(),
            "point past the endpoint must return None (segment, not line)"
        );
    }

    #[test]
    fn anisotropic_scale_compensates_offset() {
        // Derivation: uv_a = [(0,0),(0,1)] (La = 1),
        // uv_b = [(5,0),(5,2)] (Lb = 2). n_a = n_b = (1,0).
        // p = (0.1, 0.5): s = 0.5, d_a = +0.1,
        // q = (5,0) + 0.5*(0,2) = (5,1);
        // mirror = (5,1) - 0.1*(2/1)*(1,0) = (4.8, 1.0).
        // Offset 0.1 in A becomes offset 0.2 in B.
        let edge = SeamEdge {
            tri_a: 0,
            tri_b: 1,
            uv_a: [[0.0, 0.0], [0.0, 1.0]],
            uv_b: [[5.0, 0.0], [5.0, 2.0]],
            mid_3d: Vec3::new(0.5, 0.5, 0.0),
        };
        let mirrored = mirror_uv([0.1, 0.5], &edge, 0.5).expect("near A must mirror");
        assert!(
            approx_eq(mirrored, [4.8, 1.0], 1e-5),
            "offset 0.1 in A must mirror at offset 0.2 in B: (4.8, 1.0), got {mirrored:?}"
        );
    }

    #[test]
    fn degenerate_edges_return_none_and_are_skipped() {
        let zero_a = SeamEdge {
            tri_a: 0,
            tri_b: 1,
            uv_a: [[0.0, 0.0], [0.0, 0.0]],
            uv_b: [[5.0, 0.0], [5.0, 1.0]],
            mid_3d: Vec3::ZERO,
        };
        assert!(
            mirror_uv([0.0, 0.5], &zero_a, 1.0).is_none(),
            "zero-length uv_a must return None"
        );
        let zero_b = SeamEdge {
            tri_a: 0,
            tri_b: 1,
            uv_a: [[0.0, 0.0], [0.0, 1.0]],
            uv_b: [[5.0, 0.0], [5.0, 0.0]],
            mid_3d: Vec3::ZERO,
        };
        assert!(
            mirror_uv([5.0, 0.5], &zero_b, 1.0).is_none(),
            "zero-length uv_b must return None"
        );
        // mirror_positions skips the degenerate edge: no mappings, no panic.
        let mappings = mirror_positions(&[[0.0, 0.5], [5.0, 0.5]], &graph_of(zero_a), 1.0);
        assert!(
            mappings.is_empty(),
            "degenerate edge must yield zero mappings, got {mappings:?}"
        );
    }

    #[test]
    fn mirror_positions_fans_out_with_source_indices() {
        let graph = graph_of(axis_edge());
        let positions = [[0.1, 0.5], [0.1, 0.6], [50.0, 50.0]];
        let mappings = mirror_positions(&positions, &graph, 0.5);
        assert_eq!(
            mappings.len(),
            2,
            "two near positions mirror, the far one does not: {mappings:?}"
        );
        assert_eq!(mappings[0].source, 0);
        assert_eq!(mappings[1].source, 1);
        assert!(
            approx_eq(mappings[0].mirrored, [4.9, 0.5], 1e-6),
            "wrong mirror for position 0: {mappings:?}"
        );
        assert!(
            approx_eq(mappings[1].mirrored, [4.9, 0.6], 1e-6),
            "wrong mirror for position 1: {mappings:?}"
        );
    }

    #[test]
    fn mirror_positions_skips_mirror_equal_to_source() {
        // Coincident islands: the on-seam dab maps to itself and must be
        // skipped rather than double-stamped.
        let edge = SeamEdge {
            tri_a: 0,
            tri_b: 1,
            uv_a: [[0.0, 0.0], [0.0, 1.0]],
            uv_b: [[0.0, 0.0], [0.0, 1.0]],
            mid_3d: Vec3::ZERO,
        };
        let at_seam = mirror_uv([0.0, 0.5], &edge, 0.5).expect("on-seam maps to itself");
        assert_eq!(at_seam, [0.0, 0.5]);
        let mappings = mirror_positions(&[[0.0, 0.5]], &graph_of(edge), 0.5);
        assert!(
            mappings.is_empty(),
            "self-mirror must be skipped, got {mappings:?}"
        );
    }
}

//! High→low bake transfer CPU support: cage, distance clamps, match-by-name
//! (Wave-4 slice B).
//!
//! CPU-side helpers contracted in
//! `docs/specs/high-to-low-transfer-design.md` (§"Cage + matching"): the
//! per-texel ray configuration against the high-poly mesh needs three
//! things before the transfer pass (GPU, later slice) can run:
//!
//! - [`Cage`]: per-LOW-vertex ray origin displacement, barycentrically
//!   lerped per texel ([`Cage::lerp_at`]).
//! - [`TransferClamps`]: the front/back distance gate around the low
//!   surface ([`TransferClamps::clamps_hit`]).
//! - [`match_by_name`] / [`find_high_for_low`]: Substance-style `*`
//!   match-by-name mapping of low parts to high parts.
//!
//! # Relation to the design doc
//!
//! - The design doc sketches `Cage { offsets: Vec<Vec3> }`; this module
//!   stores `Vec<[f32; 3]>` per the slice brief (plain data, no new deps).
//! - The design doc says clamps are "measured from the ray origin: L's
//!   surface, or the cage if set". [`TransferClamps::clamps_hit`] anchors
//!   the gate on the LOW surface parameter `t_surface` (front reaches
//!   toward the ray origin, back reaches past the surface to the far
//!   side). In the canonical setup the ray origin sits exactly
//!   `front_distance` ahead of the surface, so the gate spans
//!   `[0, front_distance + back_distance]` from the origin and both
//!   wordings — including the design doc's test-plan row 5 — agree. The
//!   surface-anchored form is implemented because it stays correct when
//!   the origin push differs from `front_distance`. No disagreement found;
//!   this is the general form of the doc's wording.
//! - Skew-tolerance (cage skew compensation) is a v2 row per the design
//!   doc and is NOT implemented here: rays stay along the vertex normal.
//!
//! # No panics
//!
//! All index/slice accesses go through [`slice::get`]; malformed
//! triangles, out-of-bounds cage indices, and non-finite floats degrade to
//! `None`/`false`, never a panic.

/// Per-LOW-vertex ray origin displacement (the "cage").
///
/// `offsets[i]` displaces the ray start for low vertex `i`
/// (`origin = low_pos + normal * front_offset + lerped_cage_offset`).
/// Lerped per texel via barycentrics with [`Cage::lerp_at`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cage {
    /// One displacement per low vertex, indexed by vertex index.
    pub offsets: Vec<[f32; 3]>,
}

impl Cage {
    /// Barycentrically interpolates the cage offsets for triangle `tri`'s
    /// three vertices at `(u, v)` with `w = 1 - u - v`, matching
    /// [`RayHit`](crate::RayHit)'s convention (`w` weights vertex 0, `u`
    /// weights vertex 1, `v` weights vertex 2).
    ///
    /// `tri` holds vertex indices into [`Cage::offsets`]. Returns [`None`]
    /// (never panics) when `tri` has fewer than three entries or any index
    /// is out of bounds.
    pub fn lerp_at(&self, tri: &[u32], bary: (f32, f32)) -> Option<[f32; 3]> {
        let (u, v) = bary;
        let w = 1.0 - u - v;
        let i0 = *tri.first()? as usize;
        let i1 = *tri.get(1)? as usize;
        let i2 = *tri.get(2)? as usize;
        let (o0, o1, o2) = (
            *self.offsets.get(i0)?,
            *self.offsets.get(i1)?,
            *self.offsets.get(i2)?,
        );
        Some([
            w * o0[0] + u * o1[0] + v * o2[0],
            w * o0[1] + u * o1[1] + v * o2[1],
            w * o0[2] + u * o1[2] + v * o2[2],
        ])
    }
}

/// Front/back distance gate around the low surface.
///
/// A hit at ray parameter `hit_t` (measured from the ray ORIGIN) is valid
/// iff it falls within the clamp pair around the LOW surface point, which
/// sits at `t_surface` along the ray (the ray starts at
/// `low_pos + normal * front_offset`, so `t_surface = front_offset` for a
/// unit direction):
///
/// ```text
/// t_surface - front_distance <= hit_t <= t_surface + back_distance
/// ```
///
/// [`TransferClamps::front_distance`] reaches toward the ray origin (the
/// "in front of the low surface" side);
/// [`TransferClamps::back_distance`] reaches past the surface to the far
/// side. Anything outside is a miss (background color).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TransferClamps {
    /// Max hit distance on the ray-origin side of the low surface.
    pub front_distance: f32,
    /// Max hit distance on the far side of the low surface.
    pub back_distance: f32,
}

impl TransferClamps {
    /// Whether a hit at ray parameter `hit_t` survives the front/back
    /// gate around the low surface at `t_surface` (see the
    /// [struct-level](TransferClamps) inequality). Both bounds are
    /// inclusive. Non-finite inputs compare false, i.e. are misses — never
    /// a panic.
    pub fn clamps_hit(&self, hit_t: f32, t_surface: f32) -> bool {
        hit_t >= t_surface - self.front_distance && hit_t <= t_surface + self.back_distance
    }
}

/// Substance-style match-by-name glob (v1).
///
/// `'*'` matches any run (including empty) of name characters; there are NO
/// other glob characters — `'?'` stays literal. Matching is
/// CASE-SENSITIVE and full-anchored (the pattern must match the WHOLE
/// name, not a substring). An empty pattern matches only the empty string.
/// Byte-exact comparison (UTF-8 `'*'` cannot appear inside a multi-byte
/// sequence, so byte matching is code-point safe).
///
/// # Which side is which (the inversion rule)
///
/// The first argument is the part NAME under test and the second is the
/// glob pattern. A low part `L` maps to a high part `h` iff some pattern
/// `p` derived from `L` (e.g. low `head_low` → pattern `head*`) matches
/// the HIGH name: `match_by_name(h, p)`. In other words the caller inverts
/// the arguments — the name under test is high-side, the pattern is
/// low-side ([`find_high_for_low`] implements exactly this loop).
pub fn match_by_name(low_part: &str, pattern: &str) -> bool {
    let s = low_part.as_bytes();
    let p = pattern.as_bytes();
    let (mut si, mut pi) = (0usize, 0usize);
    // Last '*' seen in the pattern and the string position it has consumed
    // up to (classic backtracking glob; no recursion, no allocation).
    let (mut star, mut ss) = (None::<usize>, 0usize);
    while si < s.len() {
        if pi < p.len() && p[pi] == s[si] {
            // Exact byte equality only: '?' has no wildcard meaning here.
            si += 1;
            pi += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            ss = si;
            pi += 1;
        } else if let Some(st) = star {
            // Backtrack: let the last '*' swallow one more character.
            pi = st + 1;
            ss += 1;
            si = ss;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

/// Maps a low part to the first matching high part (the inversion call).
///
/// `low_part` names the LOW-poly part being resolved: the caller derives
/// `patterns` from it (e.g. `head_low` → `["head*"]`) and logs the skip
/// when this returns [`None`]. For each high part `h` in order, for each
/// pattern `p`, a match is `match_by_name(h, p)` — note the HIGH name is
/// the name under test per the [inversion rule](match_by_name). Returns
/// `Some` of the FIRST matching index into `high_parts`, or [`None`] when
/// nothing matches (including empty `patterns`/`high_parts`). Never
/// panics.
pub fn find_high_for_low(
    low_part: &str,
    high_parts: &[&str],
    patterns: &[String],
) -> Option<usize> {
    // `low_part` is caller context (pattern derivation + skip logging
    // happen upstream); the match loop itself runs high-side per the
    // inversion rule.
    let _ = low_part;
    for (i, h) in high_parts.iter().enumerate() {
        if patterns.iter().any(|p| match_by_name(h, p)) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cage_lerp_corners_midpoint_and_malformed() {
        let cage = Cage {
            offsets: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let tri = [0u32, 1, 2];
        // Corners are EXACT (w/u/v select a single vertex offset).
        assert_eq!(cage.lerp_at(&tri, (0.0, 0.0)), Some([0.0, 0.0, 0.0]));
        assert_eq!(cage.lerp_at(&tri, (1.0, 0.0)), Some([1.0, 0.0, 0.0]));
        assert_eq!(cage.lerp_at(&tri, (0.0, 1.0)), Some([0.0, 0.0, 1.0]));
        // Midpoint: w = 0.5, u = v = 0.25 — exact in binary floating point.
        assert_eq!(cage.lerp_at(&tri, (0.25, 0.25)), Some([0.25, 0.0, 0.25]));
        // Malformed indices degrade to None, never a panic.
        assert_eq!(cage.lerp_at(&[0u32, 1, 99], (0.0, 0.0)), None);
        assert_eq!(cage.lerp_at(&[0u32, 1], (0.0, 0.0)), None);
        assert_eq!(cage.lerp_at(&[], (0.0, 0.0)), None);
    }

    #[test]
    fn clamps_gate_fails_on_both_sides() {
        // Non-boundary offsets AND a nonzero surface parameter, so the
        // gate is pinned around t_surface rather than the ray origin.
        let clamps = TransferClamps {
            front_distance: 1.5,
            back_distance: 2.5,
        };
        let t_surface = 10.0;
        // Hit exactly at the surface: valid.
        assert!(clamps.clamps_hit(t_surface, t_surface));
        // Just inside the far bound: valid.
        assert!(clamps.clamps_hit(t_surface + 2.5 - 0.001, t_surface));
        // Just past the far bound: INVALID (the gate must be able to fail).
        assert!(!clamps.clamps_hit(t_surface + 2.5 + 0.001, t_surface));
        // Just inside the origin-side bound: valid.
        assert!(clamps.clamps_hit(t_surface - 1.5 + 0.001, t_surface));
        // Just past the origin-side bound: invalid.
        assert!(!clamps.clamps_hit(t_surface - 1.5 - 0.001, t_surface));
    }

    #[test]
    fn glob_anchor_case_and_empty() {
        assert!(match_by_name("head_high", "head*"));
        assert!(match_by_name("whatever 123 _-.", "*"));
        assert!(match_by_name("", "*"));
        // Full-anchored: leading content is not a substring match.
        assert!(!match_by_name("body_head", "head*"));
        assert!(!match_by_name("head_high", "*head"));
        // Case-sensitive.
        assert!(!match_by_name("head_high", "Head*"));
        // Empty pattern matches only the empty string.
        assert!(match_by_name("", ""));
        assert!(!match_by_name("head_high", ""));
        // '?' is literal in v1, not a single-char wildcard.
        assert!(match_by_name("head?", "head?"));
        assert!(!match_by_name("headX", "head?"));
    }

    #[test]
    fn mapping_first_index_semantics() {
        let highs = ["body_high", "head_high"];
        let patterns = ["head*".to_string()];
        assert_eq!(find_high_for_low("head_low", &highs, &patterns), Some(1));
        // No match: the caller logs the skip, no panic.
        let reddish = ["nope*".to_string()];
        assert_eq!(find_high_for_low("head_low", &highs, &reddish), None);
        // Multiple matches resolve to the FIRST index.
        let both = ["head_a", "head_b"];
        assert_eq!(find_high_for_low("head_low", &both, &patterns), Some(0));
        // Empty patterns never match.
        let empty: [String; 0] = [];
        assert_eq!(find_high_for_low("head_low", &highs, &empty), None);
    }
}

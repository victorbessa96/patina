//! Pulled-string ("lazy mouse") stroke stabilizer.
//!
//! See docs/research/05-brush-engine-architecture.md §§3–4 (Substance Lazy
//! Mouse, Photoshop Pulled String Mode): the painted cursor trails the raw
//! cursor on a leash of [`LazyMouse::radius_px`]. Motion inside the radius
//! snaps straight to the target (no dead-zone stamp is produced by this
//! stage itself — spacing decides stamping); motion beyond it advances at
//! most `radius_px` per update, which straightens jitter while preserving
//! deliberate direction changes.
//!
//! Pure function of its two inputs: identical `(target, current)` pairs
//! always yield the same result.

use glam::Vec2;

/// String/leash smoother: the smoothed cursor chases the raw cursor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LazyMouse {
    /// Leash length in logical pixels. Always finite and non-negative.
    pub radius_px: f32,
}

impl LazyMouse {
    /// Build a stabilizer, sanitizing non-finite or negative radii to `0.0`
    /// (which disables smoothing: output always equals the target).
    #[must_use]
    pub fn new(radius_px: f32) -> Self {
        Self {
            radius_px: sanitize_radius(radius_px),
        }
    }

    /// Update the leash length, sanitizing as in [`LazyMouse::new`].
    pub fn set_radius(&mut self, radius_px: f32) {
        self.radius_px = sanitize_radius(radius_px);
    }

    /// Step `current` toward `target` by at most [`LazyMouse::radius_px`].
    ///
    /// Returns `target` when it lies within the radius — including always
    /// when the radius is `0.0`, which disables smoothing — otherwise the
    /// point on the `current → target` segment `radius_px` away from
    /// `current`. Non-finite inputs hold `current` rather than propagating
    /// NaN downstream.
    #[must_use]
    pub fn apply(&self, target: [f32; 2], current: [f32; 2]) -> [f32; 2] {
        let t = Vec2::new(target[0], target[1]);
        let c = Vec2::new(current[0], current[1]);
        if !t.is_finite() || !c.is_finite() {
            return current;
        }
        let delta = t - c;
        let dist = delta.length();
        if self.radius_px == 0.0 || !dist.is_finite() || dist <= self.radius_px {
            return target;
        }
        // `dist > radius_px >= 0` here, so `dist > 0` and the division is safe.
        (c + delta / dist * self.radius_px).to_array()
    }
}

impl Default for LazyMouse {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Clamp a radius to a usable range: non-finite or negative becomes `0.0`.
fn sanitize_radius(radius_px: f32) -> f32 {
    if radius_px.is_finite() && radius_px > 0.0 {
        radius_px
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
        Vec2::new(a[0] - b[0], a[1] - b[1]).length()
    }

    #[test]
    fn stationary_target_converges_in_bounded_steps() {
        let lazy = LazyMouse::new(10.0);
        let target = [30.0, 0.0];
        let mut cursor = [0.0, 0.0];
        cursor = lazy.apply(target, cursor);
        assert!(dist(cursor, [10.0, 0.0]) <= EPS);
        cursor = lazy.apply(target, cursor);
        assert!(dist(cursor, [20.0, 0.0]) <= EPS);
        cursor = lazy.apply(target, cursor);
        assert!(dist(cursor, target) <= EPS, "must snap once within radius");
        cursor = lazy.apply(target, cursor);
        assert!(dist(cursor, target) <= EPS, "must rest on the target");
    }

    #[test]
    fn step_size_never_exceeds_radius() {
        let lazy = LazyMouse::new(5.0);
        let current = [0.0, 0.0];
        let next = lazy.apply([100.0, 40.0], current);
        let stepped = dist(next, current);
        assert!(
            (stepped - 5.0).abs() <= EPS,
            "far target must move exactly one radius: {stepped}"
        );
        // Direction preserved.
        let dir = Vec2::new(next[0], next[1]).normalize();
        let want = Vec2::new(100.0, 40.0).normalize();
        assert!((dir - want).length() <= EPS);
        // Near target snaps instead of stepping.
        let snap = lazy.apply([3.0, 4.0], current);
        assert!(dist(snap, [3.0, 4.0]) <= EPS);
    }

    #[test]
    fn zero_or_invalid_radius_passes_through() {
        let lazy = LazyMouse::new(0.0);
        assert!(dist(lazy.apply([50.0, 0.0], [0.0, 0.0]), [50.0, 0.0]) <= EPS);
        let bad = LazyMouse::new(f32::NAN);
        assert_eq!(bad.radius_px, 0.0);
        let neg = LazyMouse::new(-4.0);
        assert_eq!(neg.radius_px, 0.0);
        // Non-finite target never injects NaN downstream.
        let out = lazy.apply([f32::NAN, 0.0], [1.0, 2.0]);
        assert!(dist(out, [1.0, 2.0]) <= EPS);
    }
}

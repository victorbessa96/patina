//! Orbit camera: yaw/pitch/distance around a target point.
//!
//! Pure CPU math, no wgpu dependency — testable headless.

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};

/// Unprojects an NDC point (with explicit depth: z=0 near, z=1 far in
/// wgpu's directx convention) back to world space through the inverse
/// view-projection matrix.
///
/// Shared helper (Wave-4 item 7): viewport picking (`umber-app`'s
/// `pick_uv`) builds its near/far ray endpoints through this, the ground
/// grid's offscreen-test math ports it to predict world positions per
/// pixel, and the grid shader carries its own WGSL copy (`GRID_SHADER`'s
/// `world_from_ndc` — same formula, documented there). The shader copy
/// can't be deduped across the language boundary; the two Rust call
/// sites share this one function instead of each hand-rolling the
/// perspective divide.
pub fn world_from_ndc(inv_view_proj: Mat4, ndc: glam::Vec4) -> Vec3 {
    let world = inv_view_proj * ndc;
    world.truncate() / world.w
}

/// Keeps the camera from flipping over the pole (gimbal lock at the top).
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// An orbit (turntable) camera: it always looks at `target`, positioned at
/// `distance` along the direction given by `yaw`/`pitch`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    /// World-space point the camera always looks at.
    pub target: Vec3,
    /// Rotation around the world Y axis, radians.
    pub yaw: f32,
    /// Rotation up/down from the horizontal plane, radians.
    pub pitch: f32,
    /// Distance from `target` to the eye.
    pub distance: f32,
    /// Lower clamp for `distance` (see [`Self::zoom`]).
    pub min_distance: f32,
    /// Upper clamp for `distance` (see [`Self::zoom`]).
    pub max_distance: f32,
    /// Vertical field of view, radians.
    pub fov_y_radians: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.4,
            distance: 5.0,
            min_distance: 0.05,
            max_distance: 1000.0,
            fov_y_radians: 45.0_f32.to_radians(),
            near: 0.01,
            far: 1000.0,
        }
    }
}

impl OrbitCamera {
    /// Builds a camera framing the given world-space bounding box, keeping
    /// the current yaw/pitch (so re-loading a mesh doesn't spin the view).
    pub fn framing(min: Vec3, max: Vec3, yaw: f32, pitch: f32) -> Self {
        let center = (min + max) * 0.5;
        let radius = ((max - min).length() * 0.5).max(0.01);
        let fov = 45.0_f32.to_radians();
        let distance = (radius / (fov * 0.5).tan()) * 1.5;
        Self {
            target: center,
            yaw,
            pitch,
            distance,
            min_distance: radius * 0.02,
            max_distance: distance * 50.0,
            fov_y_radians: fov,
            near: (radius * 0.01).max(0.001),
            far: (distance + radius) * 20.0,
        }
    }

    /// World-space eye position.
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target
            + Vec3::new(
                self.distance * cp * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.cos(),
            )
    }

    /// World-to-view transform for the current eye/target/up.
    pub fn view_matrix(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// `aspect` is width / height; callers must guard against zero-height
    /// viewports before calling this.
    ///
    /// Uses the right-handed, `directx`-convention projection (depth range
    /// 0..1, NDC Y not flipped) — glam 0.34 moved camera math into the
    /// `glam::camera` module with per-API NDC conventions (see
    /// `glam-0.34.1/tests/camera.rs`), and `directx`'s convention is the one
    /// that matches wgpu's clip space.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        proj::directx::perspective(self.fov_y_radians, aspect, self.near, self.far)
    }

    /// Combined projection * view matrix for the given viewport aspect.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    /// Drag-orbit: `delta` is in UI pixels (or any consistent unit); scale
    /// is the caller's responsibility via `sensitivity`.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Pans the target along the camera's current right/up axes, scaled by
    /// distance so panning feels consistent regardless of zoom level.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let scale = self.distance;
        self.target += right * (delta_x * scale) + up * (delta_y * scale);
    }

    /// Wheel-zoom; `delta` > 0 zooms in. Multiplicative so it feels uniform
    /// near and far from the target.
    pub fn zoom(&mut self, delta: f32) {
        self.distance =
            (self.distance * (1.0 - delta).max(0.01)).clamp(self.min_distance, self.max_distance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_centers_on_bounds_and_keeps_orientation() {
        let cam = OrbitCamera::framing(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            0.6,
            0.4,
        );
        assert_eq!(cam.target, Vec3::ZERO);
        assert_eq!(cam.yaw, 0.6);
        assert_eq!(cam.pitch, 0.4);
        assert!(cam.distance > 0.0);
        assert!(cam.near > 0.0 && cam.near < cam.far);
    }

    #[test]
    fn framing_degenerate_bounds_does_not_panic() {
        let cam = OrbitCamera::framing(Vec3::ZERO, Vec3::ZERO, 0.0, 0.0);
        assert!(cam.distance.is_finite() && cam.distance > 0.0);
    }

    #[test]
    fn pitch_is_clamped_past_the_poles() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 10.0);
        assert!(cam.pitch <= PITCH_LIMIT);
        cam.orbit(0.0, -20.0);
        assert!(cam.pitch >= -PITCH_LIMIT);
    }

    #[test]
    fn zoom_is_clamped_to_min_max_distance() {
        let mut cam = OrbitCamera {
            min_distance: 1.0,
            max_distance: 10.0,
            distance: 5.0,
            ..Default::default()
        };
        for _ in 0..100 {
            cam.zoom(0.5);
        }
        assert!(cam.distance >= cam.min_distance);
        for _ in 0..100 {
            cam.zoom(-0.9);
        }
        assert!(cam.distance <= cam.max_distance);
    }

    #[test]
    fn view_proj_is_finite_for_sane_aspect() {
        let cam = OrbitCamera::default();
        let m = cam.view_proj(16.0 / 9.0);
        assert!(m.to_cols_array().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn world_from_ndc_roundtrips_through_view_proj() {
        let cam = OrbitCamera::default();
        let view_proj = cam.view_proj(1.0);
        let inv = view_proj.inverse();
        // World origin must survive the round trip at several depths.
        for z in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let clip = view_proj * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
            let ndc = clip.truncate() / clip.w;
            let back = world_from_ndc(inv, glam::Vec4::new(ndc.x, ndc.y, z, 1.0));
            assert!(back.is_finite());
            // At the projected depth the round trip is exact.
            if (z - ndc.z).abs() < 1e-6 {
                assert!(
                    (back - Vec3::ZERO).length() < 1e-4,
                    "round trip drifted: {back:?}"
                );
            }
        }
    }
}

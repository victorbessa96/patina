//! CPU ray-mesh intersection for viewport painting and picking.
//!
//! Möller–Trumbore over the triangle list — no acceleration structure
//! (the Wave-3 bake BVH will share this primitive). Sufficient for
//! per-pointer picking at interactive rates up to ~100k tris; a BVH
//! arrives with the bake engine's mesh-map path.

use crate::MeshData;
use glam::Vec3;

/// A ray hit: the intersected triangle index, distance along the ray,
/// and the barycentric hit point (for attribute interpolation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Triangle index (position in the index list / 3).
    pub triangle: usize,
    /// Ray parameter (`origin + dir * t`), always finite and positive.
    pub t: f32,
    /// Barycentric coordinates at the hit (u, v; w = 1-u-v).
    pub bary: (f32, f32),
}

/// Interpolates a per-vertex attribute at a hit via barycentrics.
///
/// `tri` is the raw 3-index slice for the hit's triangle.
pub fn interpolate_attr<T: Copy>(
    tri: &[u32],
    attrs: &[T],
    hit: RayHit,
    lerp: impl Fn(T, T, T, f32, f32) -> T,
) -> Option<T> {
    let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
    let (a, b, c) = (attrs.get(i0)?, attrs.get(i1)?, attrs.get(i2)?);
    Some(lerp(*a, *b, *c, hit.bary.0, hit.bary.1))
}

/// Casts `ray` against the mesh, returning the nearest hit.
///
/// `dir` need not be normalized; `t` is in the ray's own units.
pub fn ray_intersect(mesh: &MeshData, origin: Vec3, dir: Vec3) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    for (tri_idx, tri) in mesh.indices.chunks_exact(3).enumerate() {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (Some(&p0), Some(&p1), Some(&p2)) = (
            mesh.positions.get(i0),
            mesh.positions.get(i1),
            mesh.positions.get(i2),
        ) else {
            continue;
        };
        let (v0, v1, v2) = (
            Vec3::from_array(p0),
            Vec3::from_array(p1),
            Vec3::from_array(p2),
        );
        // Möller–Trumbore.
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = dir.cross(edge2);
        let a = edge1.dot(h);
        // Parallel (or degenerate) triangle.
        if a.abs() < 1e-8 {
            continue;
        }
        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(edge1);
        let v = f * dir.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = f * edge2.dot(q);
        if t <= 1e-5 {
            continue; // Behind origin or epsilon-touching.
        }
        if best.is_none_or(|b| t < b.t) {
            best = Some(RayHit {
                triangle: tri_idx,
                t,
                bary: (u, v),
            });
        }
    }
    best
}

/// The UV at a hit, barycentrically interpolated.
pub fn uv_at(mesh: &MeshData, hit: RayHit) -> Option<[f32; 2]> {
    let tri = mesh.indices.chunks_exact(3).nth(hit.triangle)?;
    interpolate_attr(tri, &mesh.uvs, hit, |a, b, c, u, v| {
        let w = 1.0 - u - v;
        [
            a[0] * w + b[0] * u + c[0] * v,
            a[1] * w + b[1] * u + c[1] * v,
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> MeshData {
        // Two-triangle quad on the z=0 plane, unit UVs.
        MeshData {
            positions: vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_names: vec![],
        }
    }

    #[test]
    fn center_hit_returns_center_uv() {
        let mesh = quad();
        let hit = ray_intersect(&mesh, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("quad must be hit from +z");
        let uv = uv_at(&mesh, hit).expect("uv present");
        assert!((uv[0] - 0.5).abs() < 1e-5 && (uv[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn corner_hits_map_to_corner_uvs() {
        let mesh = quad();
        // Aim at vertex 1's position (1,-1): its UV is (1,0).
        let hit = ray_intersect(&mesh, Vec3::new(1.0, -1.0, 5.0), Vec3::new(0.0, 0.0, -1.0))
            .expect("corner hit");
        let uv = uv_at(&mesh, hit).expect("uv");
        assert!((uv[0] - 1.0).abs() < 1e-5 && (uv[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn miss_returns_none() {
        let mesh = quad();
        assert!(ray_intersect(&mesh, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 1.0, 0.0)).is_none());
    }

    #[test]
    fn nearest_hit_wins_with_occluder() {
        let mut mesh = quad();
        // A second quad closer to the ray origin (z = 2).
        let base = mesh.positions.len() as u32;
        for p in [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.0],
        ] {
            mesh.positions.push(p);
            mesh.normals.push([0.0, 0.0, 1.0]);
            mesh.uvs.push([0.5, 0.5]);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        let hit =
            ray_intersect(&mesh, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)).expect("hit");
        assert_eq!(hit.triangle, 2, "the z=2 quad (second pair) must win");
    }
}

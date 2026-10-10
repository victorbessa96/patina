//! Median-split BVH over mesh triangles (Wave-4 high→low transfer, slice A).
//!
//! Built once per high-poly source mesh, this is the acceleration behind
//! the high→low bake transfer contracted in
//! `docs/specs/high-to-low-transfer-design.md`: per-texel rays in the
//! low-poly tangent frame are intersected against the high-poly triangle
//! soup, and brute-force Möller–Trumbore over every triangle does not
//! scale there.
//!
//! # Build
//!
//! [`build`] computes a per-triangle AABB and centroid from
//! [`MeshData`](crate::MeshData) positions via the index list (triangle
//! `k` = `indices[3k..3k+3]`, the same convention as
//! [`ray_intersect`](crate::raycast::ray_intersect)), then recurses:
//! split on the longest AABB axis at the centroid midpoint. Three stop
//! conditions: at most [`BVH_LEAF_SIZE`] triangles per leaf, a
//! centroid-split that would leave one side empty (degenerate: all
//! centroids equal on that axis — falls back to splitting at the median
//! index so both sides stay non-empty), or the [`BVH_MAX_DEPTH`] depth
//! cap.
//!
//! # Determinism
//!
//! The build is byte-deterministic: ties sort by
//! `(centroid, original triangle index)` with [`f32::total_cmp`], so
//! byte-identical inputs produce byte-identical BVHs
//! ([`Bvh::signature_debug`] exposes the full internal state for tests).
//!
//! # Traversal
//!
//! [`traverse`] is a closest-hit walk reusing [`ray_intersect`](crate::raycast::ray_intersect)'s
//! EXACT hit semantics (same Möller–Trumbore constants and bounds), so
//! BVH hits agree with brute force on every ray. Nodes whose
//! nearest-plane distance strictly exceeds the best hit's `t` are pruned.
//!
//! # No panics
//!
//! All attribute lookups go through [`slice::get`]; malformed triangles
//! (index out of bounds, `NaN` positions) are skipped at build time and
//! at traversal time exactly as [`ray_intersect`](crate::raycast::ray_intersect)
//! skips them. Internal node/child indices are bounds-checked on every
//! access.

use crate::{MeshData, RayHit};
use glam::Vec3;

/// Maximum BVH depth: [`build`] turns any node at this depth into a leaf.
pub const BVH_MAX_DEPTH: usize = 24;

/// Maximum triangle count before [`build`] splits a node (unless a split
/// would be degenerate or the depth cap is reached).
pub const BVH_LEAF_SIZE: usize = 4;

/// One BVH node: an AABB plus either two child node indices (interior)
/// or a triangle range (leaf).
#[derive(Debug, Clone, Copy, PartialEq)]
struct BvhNode {
    /// AABB minimum corner.
    min: [f32; 3],
    /// AABB maximum corner.
    max: [f32; 3],
    /// Interior: left child node index. Leaf: start index into [`Bvh::tris`].
    left: usize,
    /// Interior: right child node index. Leaf: triangle count in [`Bvh::tris`].
    right: usize,
    /// Whether this node is a leaf.
    leaf: bool,
}

impl BvhNode {
    /// An interior node (children filled in after the subtrees build).
    fn interior(min: [f32; 3], max: [f32; 3], left: usize, right: usize) -> Self {
        Self {
            min,
            max,
            left,
            right,
            leaf: false,
        }
    }

    /// A leaf node over `tris[start..start + count]`.
    fn leaf(min: [f32; 3], max: [f32; 3], start: usize, count: usize) -> Self {
        Self {
            min,
            max,
            left: start,
            right: count,
            leaf: true,
        }
    }
}

/// A deterministic median-split bounding-volume hierarchy over a mesh's
/// triangles.
///
/// Built with [`build`], queried with [`traverse`]. The triangle list
/// holds ORIGINAL mesh triangle indices (position in the index list / 3),
/// never a reordered alias, so hits report the same triangle
/// [`ray_intersect`](crate::raycast::ray_intersect) would.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bvh {
    /// All nodes; node 0 is the root. Empty when the mesh contributed no
    /// valid triangle.
    nodes: Vec<BvhNode>,
    /// Packed per-leaf original triangle indices.
    tris: Vec<usize>,
}

impl Bvh {
    /// Whether the BVH holds no node (the mesh had no valid triangle).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of nodes (1 for a single-leaf BVH, 0 when empty).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of triangle references packed in the leaves.
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    /// Maximum depth of the tree (root = 0). Never exceeds
    /// [`BVH_MAX_DEPTH`].
    pub fn max_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        // Iterative DFS; every index access is bounds-checked.
        let mut deepest = 0;
        let mut stack = vec![(0usize, 0usize)];
        while let Some((id, depth)) = stack.pop() {
            let Some(node) = self.nodes.get(id) else {
                continue;
            };
            deepest = deepest.max(depth);
            if !node.leaf {
                stack.push((node.left, depth + 1));
                stack.push((node.right, depth + 1));
            }
        }
        deepest
    }

    /// A string uniquely identifying the internal node/triangle arrays.
    ///
    /// Intended for determinism tests: two builds of byte-identical inputs
    /// must produce equal signatures.
    pub fn signature_debug(&self) -> String {
        format!("{:?}|{:?}", self.nodes, self.tris)
    }
}

/// Per-triangle build workspace: bounds, centroid, original index.
struct TriInfo {
    /// Original mesh triangle index (position in the index list / 3).
    tri: usize,
    centroid: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
}

/// Builds a median-split BVH over `mesh`'s triangles.
///
/// Splits on the longest AABB axis at the centroid midpoint, stopping at
/// [`BVH_LEAF_SIZE`] triangles per leaf, on a degenerate (empty-side)
/// centroid split — which falls back to a median-index split so both
/// sides stay non-empty — or at [`BVH_MAX_DEPTH`] depth. Malformed
/// triangles (out-of-bounds indices, `NaN` positions) are skipped, never
/// panicked on. The build is deterministic (see [`Bvh::signature_debug`]).
pub fn build(mesh: &MeshData) -> Bvh {
    let mut infos = Vec::new();
    for (tri_idx, tri) in mesh.indices.chunks_exact(3).enumerate() {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (Some(&p0), Some(&p1), Some(&p2)) = (
            mesh.positions.get(i0),
            mesh.positions.get(i1),
            mesh.positions.get(i2),
        ) else {
            continue;
        };
        if p0
            .iter()
            .chain(p1.iter())
            .chain(p2.iter())
            .any(|c| c.is_nan())
        {
            continue;
        }
        let min = [
            p0[0].min(p1[0]).min(p2[0]),
            p0[1].min(p1[1]).min(p2[1]),
            p0[2].min(p1[2]).min(p2[2]),
        ];
        let max = [
            p0[0].max(p1[0]).max(p2[0]),
            p0[1].max(p1[1]).max(p2[1]),
            p0[2].max(p1[2]).max(p2[2]),
        ];
        let centroid = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        infos.push(TriInfo {
            tri: tri_idx,
            centroid,
            min,
            max,
        });
    }

    let mut bvh = Bvh::default();
    if infos.is_empty() {
        return bvh;
    }
    build_range(&mut infos, 0, &mut bvh.nodes, &mut bvh.tris);
    bvh
}

/// Recursively builds the subtree over `items`, pushing nodes in
/// deterministic order and returning the new node's index.
fn build_range(
    items: &mut [TriInfo],
    depth: usize,
    nodes: &mut Vec<BvhNode>,
    tris: &mut Vec<usize>,
) -> usize {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for info in items.iter() {
        for a in 0..3 {
            min[a] = min[a].min(info.min[a]);
            max[a] = max[a].max(info.max[a]);
        }
    }

    // Reserve this node's slot; children (higher indices) build next, so
    // node order is deterministic for a given input.
    let id = nodes.len();
    nodes.push(BvhNode::leaf(min, max, 0, 0));

    if items.len() <= BVH_LEAF_SIZE || depth >= BVH_MAX_DEPTH {
        let start = tris.len();
        tris.extend(items.iter().map(|info| info.tri));
        nodes[id] = BvhNode::leaf(min, max, start, items.len());
        return id;
    }

    let ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0
    } else if ext[1] >= ext[2] {
        1
    } else {
        2
    };
    // Deterministic order: centroid on the split axis, ties broken by the
    // original triangle index (never a bare f32 sort key).
    items.sort_by(|a, b| {
        a.centroid[axis]
            .total_cmp(&b.centroid[axis])
            .then_with(|| a.tri.cmp(&b.tri))
    });

    let midpoint = (min[axis] + max[axis]) / 2.0;
    let mut split = items.partition_point(|info| info.centroid[axis] <= midpoint);
    if split == 0 || split == items.len() {
        // Degenerate: every centroid on one side (e.g. all equal on this
        // axis). Split at the median index so both sides are non-empty.
        split = items.len() / 2;
    }
    let (left_items, right_items) = items.split_at_mut(split);
    let left = build_range(left_items, depth + 1, nodes, tris);
    let right = build_range(right_items, depth + 1, nodes, tris);
    nodes[id] = BvhNode::interior(min, max, left, right);
    id
}

/// Slab-tests `ray` against an AABB, returning the nearest-plane distance
/// (`t_enter`, possibly negative when the origin sits inside the box), or
/// `None` on a miss.
///
/// Zero direction components are handled as parallel slabs (hit iff the
/// origin lies within the slab) so no division by zero occurs.
fn slab(origin: Vec3, dir: Vec3, min: [f32; 3], max: [f32; 3]) -> Option<f32> {
    let o = origin.to_array();
    let d = dir.to_array();
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < min[a] || o[a] > max[a] {
                return None;
            }
        } else {
            let inv = 1.0 / d[a];
            let (mut t0, mut t1) = ((min[a] - o[a]) * inv, (max[a] - o[a]) * inv);
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmin > tmax {
                return None;
            }
        }
    }
    Some(tmin)
}

/// Möller–Trumbore for one triangle with [`ray_intersect`](crate::raycast::ray_intersect)'s
/// EXACT constants and bounds (1e-8 parallel epsilon, `0.0..=1.0` `u`
/// test, `v >= 0` / `u + v <= 1`, `t > 1e-5` near-clip).
fn tri_hit(
    positions: &[[f32; 3]],
    tri: &[u32],
    tri_idx: usize,
    origin: Vec3,
    dir: Vec3,
) -> Option<RayHit> {
    let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
    let (Some(&p0), Some(&p1), Some(&p2)) =
        (positions.get(i0), positions.get(i1), positions.get(i2))
    else {
        return None;
    };
    if p0
        .iter()
        .chain(p1.iter())
        .chain(p2.iter())
        .any(|c| c.is_nan())
    {
        return None;
    }
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
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t <= 1e-5 {
        return None; // Behind origin or epsilon-touching.
    }
    Some(RayHit {
        triangle: tri_idx,
        t,
        bary: (u, v),
    })
}

/// Casts `ray` against the BVH, returning the nearest hit.
///
/// `dir` need not be normalized; `t` is in the ray's own units.
/// `positions`/`indices` are the source mesh's raw arrays (triangle `k` =
/// `indices[3k..3k+3]`); the reported [`RayHit::triangle`] is the
/// ORIGINAL mesh triangle index. Hit semantics — including the
/// smallest-`t`-wins rule with smallest-triangle-index tie-breaks — match
/// [`ray_intersect`](crate::raycast::ray_intersect) exactly, and nodes
/// whose nearest-plane distance strictly exceeds the best hit's `t` are
/// pruned (equal distances still descend, so exact-`t` ties resolve the
/// same way brute force resolves them).
pub fn traverse(
    bvh: &Bvh,
    positions: &[[f32; 3]],
    indices: &[u32],
    origin: Vec3,
    dir: Vec3,
) -> Option<RayHit> {
    if bvh.nodes.is_empty() {
        return None;
    }
    let mut best: Option<RayHit> = None;
    let mut stack = Vec::with_capacity(32);
    stack.push(0usize);
    while let Some(id) = stack.pop() {
        let Some(node) = bvh.nodes.get(id) else {
            continue;
        };
        let Some(t_enter) = slab(origin, dir, node.min, node.max) else {
            continue;
        };
        if let Some(b) = best {
            if t_enter > b.t {
                continue;
            }
        }
        if node.leaf {
            let end = node.left.saturating_add(node.right);
            let Some(range) = bvh.tris.get(node.left..end) else {
                continue;
            };
            for &tri_idx in range {
                let Some(tri) = indices.chunks_exact(3).nth(tri_idx) else {
                    continue;
                };
                if let Some(hit) = tri_hit(positions, tri, tri_idx, origin, dir) {
                    let replace = match &best {
                        None => true,
                        // Brute force visits triangles in ascending order
                        // and keeps strictly smaller `t`, so exact-`t`
                        // ties resolve to the smallest triangle index.
                        Some(b) => hit.t < b.t || (hit.t == b.t && hit.triangle < b.triangle),
                    };
                    if replace {
                        best = Some(hit);
                    }
                }
            }
        } else {
            let (Some(left), Some(right)) = (bvh.nodes.get(node.left), bvh.nodes.get(node.right))
            else {
                continue;
            };
            let lt = slab(origin, dir, left.min, left.max);
            let rt = slab(origin, dir, right.min, right.max);
            let bound = best.map(|b| b.t).unwrap_or(f32::INFINITY);
            // Push the farther child first so the nearer pops first.
            match (lt, rt) {
                (Some(a), Some(b)) => {
                    let l_ok = a <= bound;
                    let r_ok = b <= bound;
                    if l_ok && r_ok {
                        if a <= b {
                            stack.push(node.right);
                            stack.push(node.left);
                        } else {
                            stack.push(node.left);
                            stack.push(node.right);
                        }
                    } else if l_ok {
                        stack.push(node.left);
                    } else if r_ok {
                        stack.push(node.right);
                    }
                }
                (Some(a), None) => {
                    if a <= bound {
                        stack.push(node.left);
                    }
                }
                (None, Some(b)) => {
                    if b <= bound {
                        stack.push(node.right);
                    }
                }
                (None, None) => {}
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raycast::ray_intersect;

    /// Deterministic LCG (no `rand` crate): the same stream as the brief.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    /// A `[0, 1)` draw from the top 32 bits of the LCG state.
    fn rand_unit(state: &mut u64) -> f32 {
        (((lcg_next(state) >> 32) as u32) as f32) / (u32::MAX as f32)
    }

    /// A deterministic jittered grid: `cells x cells` quads (2 triangles
    /// each) over `[-2, 2]^2` with a smooth bump plus LCG jitter, so the
    /// mesh is nontrivial (non-planar, non-axis-aligned triangles).
    fn jittered_grid(cells: usize, seed: u64) -> MeshData {
        let mut rng = seed;
        let side = cells + 1;
        let step = 4.0 / cells as f32;
        let mut positions = Vec::with_capacity(side * side);
        let mut normals = Vec::with_capacity(side * side);
        let mut uvs = Vec::with_capacity(side * side);
        for j in 0..side {
            for i in 0..side {
                let x = -2.0 + i as f32 * step + (rand_unit(&mut rng) - 0.5) * 0.6 * step;
                let y = -2.0 + j as f32 * step + (rand_unit(&mut rng) - 0.5) * 0.6 * step;
                let z = 0.4 * (1.3 * x).sin() * (1.1 * y).cos() + (rand_unit(&mut rng) - 0.5) * 0.2;
                positions.push([x, y, z]);
                normals.push([0.0, 0.0, 1.0]);
                uvs.push([i as f32 / cells as f32, j as f32 / cells as f32]);
            }
        }
        let mut indices = Vec::with_capacity(cells * cells * 6);
        for j in 0..cells {
            for i in 0..cells {
                let v = (j * side + i) as u32;
                let s = side as u32;
                indices.extend_from_slice(&[v, v + 1, v + s, v + 1, v + s + 1, v + s]);
            }
        }
        MeshData {
            positions,
            normals,
            uvs,
            indices,
            material_names: vec![],
        }
    }

    /// N deterministic pseudo-random rays: origins in `-3..3`, dirs in
    /// `-1..1` (near-zero dirs skipped).
    fn test_rays(n: usize, seed: u64) -> Vec<(Vec3, Vec3)> {
        let mut rng = seed;
        let mut rays = Vec::new();
        while rays.len() < n {
            let origin = Vec3::new(
                rand_unit(&mut rng) * 6.0 - 3.0,
                rand_unit(&mut rng) * 6.0 - 3.0,
                rand_unit(&mut rng) * 6.0 - 3.0,
            );
            let dir = Vec3::new(
                rand_unit(&mut rng) * 2.0 - 1.0,
                rand_unit(&mut rng) * 2.0 - 1.0,
                rand_unit(&mut rng) * 2.0 - 1.0,
            );
            if dir.length_squared() < 1e-8 {
                continue;
            }
            rays.push((origin, dir));
        }
        rays
    }

    #[test]
    fn bvh_matches_brute_force_on_1000_rays() {
        // THE EQUIVALENCE TEST: any build or traversal bug fails this.
        let mesh = jittered_grid(11, 0xC0FFEE);
        assert!(
            mesh.triangle_count() >= 100,
            "mesh must be nontrivial, got {} tris",
            mesh.triangle_count()
        );
        let bvh = build(&mesh);
        let rays = test_rays(1000, 0xB16B00B5);
        let mut mismatches = Vec::new();
        let (mut hits, mut misses) = (0usize, 0usize);
        for (n, (origin, dir)) in rays.iter().enumerate() {
            let brute = ray_intersect(&mesh, *origin, *dir);
            let fast = traverse(&bvh, &mesh.positions, &mesh.indices, *origin, *dir);
            if brute.is_some() {
                hits += 1;
            } else {
                misses += 1;
            }
            let ok = match (fast, brute) {
                (None, None) => true,
                (Some(f), Some(b)) => {
                    f.triangle == b.triangle
                        && (f.t - b.t).abs() <= 1e-5
                        && (f.bary.0 - b.bary.0).abs() <= 1e-4
                        && (f.bary.1 - b.bary.1).abs() <= 1e-4
                }
                _ => false,
            };
            if !ok {
                mismatches.push(format!("ray {n}: bvh={fast:?} brute={brute:?}"));
            }
        }
        assert!(
            hits > 10 && misses > 10,
            "ray set must exercise both paths, got {hits} hits / {misses} misses"
        );
        assert!(
            mismatches.is_empty(),
            "{} / 1000 rays disagree; first 10:\n{}",
            mismatches.len(),
            mismatches
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn empty_and_degenerate_meshes_are_safe() {
        // Empty mesh: no vertices, no indices.
        let empty = MeshData::default();
        let bvh = build(&empty);
        assert!(bvh.is_empty());
        assert!(traverse(
            &bvh,
            &empty.positions,
            &empty.indices,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0)
        )
        .is_none());

        // All-degenerate mesh: every triangle collapses to one point.
        let flat = MeshData {
            positions: vec![[1.0, 2.0, 3.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            uvs: vec![[0.0, 0.0]],
            indices: vec![0, 0, 0, 0, 0, 0],
            material_names: vec![],
        };
        let bvh = build(&flat);
        // Degenerate triangles are skipped at build time: nothing to hit.
        assert!(traverse(
            &bvh,
            &flat.positions,
            &flat.indices,
            Vec3::new(1.0, 2.0, 8.0),
            Vec3::new(0.0, 0.0, -1.0)
        )
        .is_none());

        // Malformed mesh: indices point out of bounds — skipped, no panic.
        let bad = MeshData {
            positions: vec![[0.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            uvs: vec![[0.0, 0.0]],
            indices: vec![7, 8, 9],
            material_names: vec![],
        };
        let bvh = build(&bad);
        assert!(traverse(
            &bvh,
            &bad.positions,
            &bad.indices,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0)
        )
        .is_none());
    }

    #[test]
    fn single_triangle_matches_brute_force_semantics() {
        let mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_names: vec![],
        };
        let bvh = build(&mesh);

        // Clean hit from the front: BVH and brute force must agree exactly.
        let front = (Vec3::new(0.1, 0.1, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let brute = ray_intersect(&mesh, front.0, front.1);
        let fast = traverse(&bvh, &mesh.positions, &mesh.indices, front.0, front.1);
        assert!(brute.is_some() && fast.is_some());
        assert_eq!(fast, brute);
        assert_eq!(fast.expect("hit").triangle, 0);

        // From behind: ray_intersect's Möller–Trumbore has no backface
        // culling, so this hits too — pin that exact (double-sided)
        // semantics rather than assuming a miss.
        let back = (Vec3::new(0.1, 0.1, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let brute_back = ray_intersect(&mesh, back.0, back.1);
        assert!(brute_back.is_some(), "ray_intersect is double-sided");
        assert_eq!(
            traverse(&bvh, &mesh.positions, &mesh.indices, back.0, back.1),
            brute_back
        );

        // A genuine miss (ray parallel to the triangle plane) is None on both.
        let miss = (Vec3::new(0.1, 0.1, 1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(ray_intersect(&mesh, miss.0, miss.1).is_none());
        assert!(traverse(&bvh, &mesh.positions, &mesh.indices, miss.0, miss.1).is_none());
    }

    #[test]
    fn build_is_deterministic() {
        let mesh = jittered_grid(11, 0xDE7EC1DE);
        let a = build(&mesh);
        let b = build(&mesh);
        assert_eq!(a.signature_debug(), b.signature_debug());
        assert_eq!(a, b);
        assert!(!a.signature_debug().is_empty());
    }

    #[test]
    fn large_mesh_respects_depth_cap() {
        // 48x48 verts = 47*47*2 = 4418 tris, truncated to exactly 4096.
        let mut mesh = jittered_grid(47, 0x5EED);
        mesh.indices.truncate(4096 * 3);
        assert_eq!(mesh.triangle_count(), 4096);
        let bvh = build(&mesh);
        assert!(bvh.max_depth() <= BVH_MAX_DEPTH);
        // Spot-check: a handful of rays still agree with brute force.
        for (origin, dir) in test_rays(10, 0xF00D).iter() {
            assert_eq!(
                traverse(&bvh, &mesh.positions, &mesh.indices, *origin, *dir),
                ray_intersect(&mesh, *origin, *dir),
                "ray o={origin:?} d={dir:?}"
            );
        }
    }
}

# High→Low Bake Transfer — Wave-4 Design (§3 P0 remainder)

Wave-4 item 4 of the renewed roadmap. Written 2026-10-09 against the
tree at `fb984d1`. The audit's finding: the six merged bakers are
UV-space mesh-space bakes; the high→low transfer (Substance's "bake
from a high-poly onto a low-poly's UVs") is a distinct subsystem this
note contracts. The claw briefs for items 1-2 of the build order are
tonight-shaped; the GPU transfer pass follows the machine-load
directive.

## The problem, precisely

A low-poly mesh L (game-ready, with UVs) receives detail from a
high-poly mesh H (sculpt, no UVs needed). For each texel of L's UV
map: cast a ray, find where it lands on H, transfer H's surface data
at that hit — normal (tangent-normal map), height (along the ray),
position, AO, curvature. This is how normal/color ID/height maps get
into game assets.

Substance's parameters (the parity surface): match-by-name,
front/back distance clamps, cage (explicit per-vertex offset),
skew-tolerance, antialiasing (per-texel multi-ray supersampling),
flip-green, normal-space output (tangent/object), and the geometry
matching modes (closest-point vs ray). V1 scopes the parity subset
below; the rest stay requirements rows.

## Why the existing bakers don't cover it

The merged bakers (ao.rs et al.) bake L's OWN geometry into L's UV
plane — a flat-parameter-space ray configuration (`PlaneDesc`
origin/axes/extent). High→low needs H in the ray world: per-texel
rays in L's tangent frame (or along the cage), intersected against
H's triangle soup. That's a ray-world bake: different shader input
(binding H's positions/normals/indices as buffers), different ray
generation, same `BakeTarget` output contract.

## The architecture

**New module `umber-bake/src/transfer.rs`** + `umber-mesh/src/bvh.rs`
(the acceleration the design has promised since raycast.rs's header
comment):

1. **BVH (umber-mesh, CPU, claw slice A — tonight-shaped).** A
   median-split BVH over H's triangles: build from position
   `MeshData` (bounds via centroids, SAH-optional median split v1),
   `traverse(origin, dir) -> Option<RayHit>` reusing
   `ray_intersect`'s `RayHit` shape (triangle, t, bary). Deterministic
   build (sorted centroids, stable splits) — byte-identical BVHs for
   byte-identical inputs, the golden-test precedent. Tests: build on
   a known tetra/cube, traversal finds the exact triangle Möller-
   Trumbore finds (brute-force cross-check on randomized rays —
   BOTH must agree on every ray or the test fails), empty-mesh and
   degenerate-triangle safety.

2. **Cage + matching (umber-besh CPU, claw slice B).**
   `Cage { offsets: Vec<Vec3> }` — per-LOW-vertex ray start/end
   adjustment (lerped per texel via barycentrics). Match-by-name for
   multi-part H sets: `matches(part, pattern)` with Substance's `*`
   glob + case sensitivity, mapping L part → H part(s). Front/back
   distance clamps: hits beyond `front_distance`/`back_distance`
   (measured from the ray origin: L's surface, or the cage if set)
   are misses (background color). Skew-tolerance v1: rays stay along
   the vertex normal (no cage skew compensation — document the
   omission, Substance's skew-tolerance is a v2 row).

3. **The transfer pass (umber-gpu compute — follows load
   directive).** Per texel of L's UV: ray origin = L's position +
   normal * front_offset (cage-adjusted), direction = -normal (or
   cage direction); BVH is CPU — the GPU pass instead binds H's raw
   triangle buffers and does brute-force per-texel intersection with
   an early-out distance bound (the `PlaneDesc` bakes' precedent:
   flat brute force with bounded distance, no GPU BVH — a GPU BVH is
   the wave-5 perf row if the numbers demand it). Output per map:
   - **Tangent-normal**: H's world normal at the hit → transform
     into L's tangent frame → encode RGB.
   - **Height**: hit distance along the ray, normalized by
     `front_distance + back_distance`, centered 0.5.
   - **Position/ID/curvature**: transfer H's data (position direct;
     ID from H's part/material index; curvature re-evaluated on H
     at the hit — reuses the curvature baker's math on H's
     neighborhood).
   - **AA**: v1 = 2x2 supersample toggle; the full 4x4/xMSAA row
     stays requirements-side.

4. **App + CLI wiring.** The bakes panel gains source-mesh pickers
   (the umber-app bake_sources.rs extension), per-map toggles, the
   Substance-parity settings rows; the CLI gains
   `umber bake-transfer --low L --high H [--cage C] --maps tn,hi,id`.

## Test plan (failable or it doesn't ship)

1. **BVH brute-force equivalence**: N=1000 randomized rays over a
   nontrivial mesh; BVH hit == brute-force hit (triangle AND t,
   within 1e-5) on every ray — the strongest possible test: any
   build/traversal bug fails it.
2. **Flat-surface transfer exactness**: L = a flat quad, H = a flat
   quad displaced +1 on Z over half its area — the height map is
   EXACTLY 0.5 + normalized(+1) on the displaced region and 0.5
   outside; the tangent-normal map is exactly (128,128,255) flat
   where H is coplanar and exactly the tilt where H is sloped.
   Assert exact bytes.
3. **Cage correctness**: a cage pushing ray starts +1 along the
   normal with H beyond it — hits found where uncaged rays miss;
   assert both directions (cage on = hit, cage off = miss).
4. **Match-by-name**: `head_low` matches `head*` → the H part
   `head_high` transfers; `body_low` with no matching part →
   background color, no panic, a logged skip.
5. **Distance clamps**: H at distance 3 with front_distance 1
   back_distance 2 → hit; front 1 back 1.5 → miss (background).
   Both asserted.
6. **Golden end-to-end**: the golden-image harness bakes the
   test-pair fixture; the golden PNGs land in the repo.

## Build order

1. BVH (umber-mesh, CPU) — tonight-shaped, claw dispatch after the
   mirror-math claw merges.
2. Cage + match-by-name + clamps (CPU) — parallel slice.
3. Transfer pass (GPU compute) — after 1+2, load-directive-gated.
4. App/CLI wiring + golden tests — last.

Slices 1-2 are pure CPU, fully claw-shaped, headless-testable. The
GPU pass is the only heavy piece; its brief lands when 1-2 are
merged and CI-green.

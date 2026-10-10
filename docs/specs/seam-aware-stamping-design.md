# Seam-Aware Stamping — Wave-4 Design (§1 P0 + §6 3D-neighbor padding)

The first renewed-roadmap hunt. Written 2026-10-09 against the tree at
`68176b7`; every referenced API exists today. This note is the
implementation contract — the claw briefs and the code land on it.

## The problem, precisely

Current paint flow (umber-app/viewport → umber-mesh/raycast →
umber-gpu/paint): a pointer event ray-casts (`ray_intersect`) → the
HIT triangle's UV via `uv_at` → a paint-space `Dab` → compute splat
into the `PaintTarget`. A dab therefore lands at exactly ONE UV
location. Two failure modes:

1. **The seam-split stroke.** A stroke crossing a UV seam (two
   triangles adjacent in 3D whose UVs live in different islands)
   paints only the island the ray happened to hit. The visual result:
   the stroke "tears" at the seam — one side painted, the other blank.
2. **The seam-edge hole.** A dab whose 3D footprint overlaps the seam
   bleeds off its island's UV edge (padding region), but the
   neighboring island's edge texels — same 3D surface — never receive
   the overlapping part. Filtering/mipmap on export then samples the
   unpadded edge: halo.

Substance's answer (and Mari's): stamps near a seam land at EVERY UV
image of the same 3D neighborhood, and padding propagates through 3D
adjacency, not screen-space proximity.

## The seam graph (the one data structure this needs)

**`SeamGraph`** (new module, umber-mesh — it's mesh topology):
built once per mesh load, O(triangles):

```
struct SeamEdge {
    // The two 3D-adjacent triangles sharing edge (vi, vj) in 3D
    tri_a: u32, tri_b: u32,
    // The shared edge's UV endpoints in EACH triangle's island
    uv_a: [[f32;2]; 2],   // tri_a's island coords
    uv_b: [[f32;2]; 2],   // tri_b's island coords
    // Edge midpoint in 3D (for proximity queries)
    mid_3d: Vec3,
}
struct SeamGraph {
    edges: Vec<SeamEdge>,          // sorted by mid_3d for spatial queries
    // Per-triangle bitset: which triangles touch any seam edge
    tri_touches_seam: Vec<bool>,
}
```

Build: for each triangle pair sharing an edge in 3D (vertex-index
adjacency), compare the edge's UV endpoints between the two
triangles. **A shared 3D edge whose UVs are NOT identical
(continuous) is a seam edge** — the UV discontinuity case. Edges
whose UVs match exactly are island-internal; no entry. (This is the
standard definition; matches what the UV-island literature calls a
"cut".)

Query: `seam_edges_near(p: Vec3, r: f32)` — the seam edges within
radius r of a 3D point. With ≤ a few hundred seam edges on typical
game meshes, a linear scan per stroke START (not per dab) is fine;
build a simple uniform grid over mid_3d only if profiling demands it
(the perf protocol now exists to check).

## Stamp-time multi-hit (the fix for failure mode 1)

`Dab` today carries one paint-space position. The seam-aware change:

1. **Stroke start:** the stroke evaluator queries
   `seam_edges_near(stroke_center_3d, brush_radius +
   max_dab_spread)` once. Empty result → the stroke is seam-blind;
   zero overhead (the common case — most strokes touch no seam).
2. **Per dab near a seam (only when the step-1 set is non-empty):**
   for each nearby `SeamEdge`, compute the dab's image in the OTHER
   triangle's island: transform the dab's UV position across the edge
   correspondence (`uv_a ↔ uv_b` linear map along the edge; the
   perpendicular offset mirrors). Emit an additional `Dab` at the
   mirrored UV position — the SAME radius/color/flow, because it IS
   the same 3D paint.
3. **Compositing:** the extra dabs enter the same `splat_dabs` batch
   flow; the existing no-overlap contract (paint.rs doc header)
   applies — mirror dabs are far from their originals (different
   islands), so they never collide within a batch.

Fidelity note: edge-correspondence mirroring is exact ON the seam
line and approximate beside it (UV stretch breaks symmetry a few
texels out). Substance has the same property — the approximation is
bounded by the UV stretch factor and only affects the dab's fringe,
not its core. The alternative (re-project each mirror dab through the
other triangle's exact UV map) is the wave-4 refinement if the
mirrored result visually fails the golden tests.

## Dilate-time 3D-neighbor awareness (the fix for failure mode 2; §6 padding row)

The dilation module (umber-bake) is screen-space 8-neighbor today.
The seam-aware mode adds a **seam-aware dilate pass**:

1. Build a `SeamLink` texture (Rgba32Float, same dims as the map):
   for each seam edge, rasterize the edge's UV segment in BOTH
   islands' coordinates into the map, storing (uv_other, valid).
   Built once per bake/paint session from the `SeamGraph`.
2. The dilate compute pass, for an uncovered texel, checks the
   `SeamLink` at its own texel FIRST: if a link exists, sample the
   donor at `uv_other` (the SAME 3D surface position in the other
   island) — a true 3D-neighbor donor, not a screen-space guess.
   Screen-space 8-neighbor remains the fallback for non-seam texels.

This closes the §6 "3D-neighbor aware" row exactly as the audit
planned: seam-aware stamping builds the graph, the dilate pass
consumes it.

## Where each piece lands

| Piece | Crate | New/changed |
|---|---|---|
| `SeamGraph` build + queries | umber-mesh (new `seam.rs`) | new |
| Stroke-side seam proximity check + mirror dabs | umber-app paint wiring (stroke eval) | changed |
| `Dab` mirror emission | umber-brush wiring (DabPlan resolution) | changed |
| `SeamLink` texture build | umber-gpu (bake_shaders-adjacent pass) | new |
| Seam-aware dilate pass | umber-gpu shader + umber-bake entry | new |

## Test plan (every test must be able to fail — Pitfall 15)

1. **SeamGraph unit tests** (CPU, no GPU): a quad split across two
   UV islands at a known edge → exactly one seam edge with the
   expected endpoints; a single-island quad → zero seam edges.
2. **Mirror-dab math**: a dab at distance d from a straight seam →
   mirror dab at the same distance on the other side (exact for a
   straight, unstretched seam — assert EXACT coordinates).
3. **GPU seam-stroke test**: stroke crossing the seam of the
   two-island test mesh → BOTH islands' golden images show the dab
   at the mirrored positions (golden-image harness exists —
   umber-gpu/golden.rs).
4. **3D-neighbor dilate test**: an island-edge texel uncovered in
   map A but covered in the linked island B position → after the
   seam-aware pass, A's texel carries B's donor color (exact bytes).
5. **Perf floor**: a seam-blind stroke must add ZERO dab emissions
   (count the batch) — the common case pays nothing.

## Build order (claw-dispatchable slices)

1. `SeamGraph` (umber-mesh, pure CPU, no deps) — smallest, unblocks
   everything. Claw-friendly: self-contained, testable headless.
2. Mirror-dab emission (umber-brush + app wiring) — depends on 1.
3. `SeamLink` texture + seam-aware dilate (umber-gpu + bake) —
   depends on 1; parallel to 2.
4. Golden seam-stroke test + the perf-floor assertion — lands with
   its slice, not after.

Estimated size: 1 is a day-slice for a claw; 2 and 3 are each
claw-sized; 4 rides along. The `SeamGraph` slice can dispatch
tonight; the GPU pieces follow the machine-load directive.

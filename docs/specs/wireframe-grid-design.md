# Wireframe + Grid Overlays — Wave-4 Design (§5 P1, item 7)

Wave-4 item 7, the lightest remaining slice. Written 2026-10-09
against the tree at `ff6a417`. Two viewport overlays Substance ships
that Umber lacks entirely: triangle wireframe and the reference
ground grid.

## Wireframe overlay

**Approach — barycentric-edge fragment shading, NOT a line list.**
The mesh pass's vertex layout is position+normal; a barycentric
interpolant needs a per-vertex barycentric attribute (1,0,0 /
0,1,0 / 0,0,1 per triangle) — which requires either a duplicated-
vertex buffer (3 verts per tri, no index sharing) or a
provoking-vertex trick. The duplicated buffer is the honest v1:
the wireframe pass gets its OWN vertex buffer (built once per mesh
load from the index list — positions only + bary), its own
pipeline (same camera uniform, LINE-list-free: still triangles,
just shaded as edges), drawn AFTER the mesh pass with depth-test =
LESS-EQUAL and a depth-bias so edges win coplanar z-fights.

Fragment: `edge = min(min(bary.x, bary.y), bary.z)`; wire alpha =
`1 - smoothstep(0, line_width_px * fwidth(edge), edge)` with
`fwidth` for screen-constant line width (the standard technique —
1px lines at any zoom). Color: user-set (default 40% white);
opacity slider. Toggle: viewport View menu `Show Wireframe` (W key
shortcut) — a uniform flag on the mesh pass's fragment (when off,
alpha 0 — or skip the draw entirely; skip-draw is cheaper, do
that).

**Z-fight mitigation**: the depth-bias approach (`bias = -1e-4 *
clip.w`) or polygon-offset equivalent in the shader by nudging the
vertex toward the camera along the view ray. Test pin: render a
quad with wireframe on — the diagonal edge pixel row MUST exist
(byte-diff vs wireframe-off render at the diagonal), and the mesh
shading under the wire MUST be unchanged elsewhere (threshold
count of differing pixels ≈ the edge pixels only).

## Ground grid

**Approach — a procedural fragment-shaded ground plane, not a
mesh.** A fullscreen-quad pass (or a large quad at y=0 through the
camera): fragment computes world-pos via inverse view-proj (the
app already does NDC→world unprojection in viewport.rs's picking —
same math), then:

```
grid(uv_world) = axis lines every 1 unit (x mod 1 < w or z mod 1 < w)
                + major lines every 10 (thicker, brighter)
fade           = 1 - smoothstep(0, fade_r, length(world.xz))
```

Infinite-grid fade (Substance's look): alpha falls to zero ~30
units from origin. Color: 25% white minor / 45% major over the
existing clear color. Toggle `Show Grid` (G), draw order: grid
BEFORE mesh (mesh occludes it), depth-write OFF on the grid pass
(it's a reference, not geometry).

**Axis highlight**: x-axis line red-ish, z-axis blue-ish (the
Maya/Blender convention) — one `if abs(world.x) < w` branch each.

**Tests**: GPU-gated — (1) a camera looking straight down at the
grid: the fragment at world (0.5, 0, 0.5) is NOT a grid line
(center of a cell — assert the byte), the fragment at (1.0, 0, 0.5)
IS (on the line — assert); (2) grid-off renders byte-identical to
today's; (3) the wireframe diagonal test above. The camera math is
pure — the world-pos-from-NDC helper ports to Rust for the test's
expected-value computation (the mirrored-math pattern, proven
tonight on bent normals + ID bake).

## Where it lands

| Piece | Crate | Note |
|---|---|---|
| Wire vertex buffer build (index→dup) | umber-gpu (mesh upload path) | once per mesh load |
| Wire pipeline + shader | umber-gpu | new shader const, own pipeline |
| Grid shader + pass | umber-gpu | inverse-VP quad |
| View menu toggles + keybinds | umber-app | egui menu row |
| world-from-NDC helper | umber-gpu pub fn | reused by viewport.rs picking (dedupe!) |

One claw slice covers both overlays (same crate, same test pattern,
one commit). It's the smallest wave-4 item — a good palette-
cleanser after the bent-normals merge.

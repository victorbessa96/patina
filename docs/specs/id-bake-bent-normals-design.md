# ID Bake + Bent Normals — Wave-4 Design (§3 P0 set completion)

Wave-4 item 5 of the renewed roadmap. Written 2026-10-09 against the
tree at `0d49d83`. Completes the §3 P0 baker set the audit named
(AO, position, world normal, curvature, thickness, tangent-normal
merged; ID + bent normals were the remainder). The high→low transfer
(item 4) is separately designed (high-to-low-transfer-design.md) —
the ID bake here is the MESH-SPACE variant (bake the mesh's own
material/vertex data); the transfer variant rides item 4's pass.

## ID bake (material / vertex / mesh ID)

**What Substance ships:** three ID flavors, all per-texel integer
identifiers encoded as colors —
- **Material ID**: the material/part name hashed to a color (Substance
  hashes the name string; artists rely on stable colors across
  sessions — the hash must be deterministic across runs, NOT
  rand-based).
- **Vertex ID**: the vertex index, hashed.
- **Mesh/UV-chunk ID**: the island/part index.

**Umber v1 scope (mesh-space):** Material ID from the mesh's material
assignments + Vertex ID. MeshData carries material assignments?
CHECK: if MeshData lacks material indices, v1 hashes the TRIANGLE
INDEX (deterministic, stable, useful for masking by region even
without materials) and the material flavor lands with the high→low
pass (which binds H's parts). Be honest in the UI about which flavor
this is.

**The map format:** `Rgb10a2Unorm` is tempting but readback complexity
isn't worth it — use the same `Rgba8Unorm` `BakeTarget` as the merged
bakers. Hash → sRGB byte triple: `hash = fnv1a(name)` (or triangle
index) → `[h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF, 255]`. The
FNV-1a of the material/triangle identity is the ID; the color IS the
hash. Deterministic by construction; collisions are cosmetic (two
materials sharing a color), never structural.

**Implementation shape:** a `bake_shaders::ID_BAKE_SHADER` compute
pass — per texel: fetch the triangle index (the position map's
coverage data — the existing bakers already produce a per-texel
triangle/coverage buffer; reuse it), hash, encode. The shader is
~15 lines. New module `umber-bake/src/id.rs` following curvature.rs's
pattern (params struct, bake fn over BakeTarget, error enum, tests).

**Tests:** a two-material test mesh (or two-triangle mesh for the
triangle-hash flavor): texels in triangle 0's UV region hash to
EXACTLY the expected bytes (compute the FNV in the test and compare
against the baked output — the hash is computed twice, once in Rust
once in WGSL; assert byte equality). Coverage boundary: uncovered
texels are background (0,0,0,0). Determinism: bake twice, byte-
identical.

## Bent normals (from AO)

**What it is:** the average unoccluded direction — for each texel,
the mean direction of the visible hemisphere from the AO baker's
sample set. Used for adjusting bent-normal lighting (the "bent
normal" trick: light the surface with the bent vector instead of the
geometric normal in deferred composits).

**Umber v1:** derive from the AO baker's occlusion samples — the AO
pass already casts N rays per texel and accumulates visibility;
extend the accumulation to also sum the UNOCCLUDED ray directions
(dot > 0 samples that hit nothing), normalize the sum, encode. If the
AO shader keeps per-sample hit/miss (it does — visibility
accumulation), the bent normal is a second accumulator in the same
pass: `bent = normalize(sum of unoccluded dirs)`, `alpha =
|sum| / ray_count` as a confidence channel.

**Where it lands:** `umber-bake/src/ao.rs` grows the accumulator +
a bent-normal output toggle on `AoBakeParams` (bake AO and bent in
one pass — Substance does exactly this, the maps share samples).
Output encoding: RGB = bent * 0.5 + 0.5, A = confidence. When a
texel's sum is zero (fully occluded): RGB = geometric normal * 0.5 +
0.5, A = 0.

**Tests:** a single-quad fixture: bent normal == the quad's normal
exactly (nothing occludes, all rays unoccluded, the mean IS the
hemisphere axis — WAIT, it's the mean of the sampled directions, not
the normal; for a hemisphere-sampled set the mean is the normal ONLY
for symmetric sampling. The AO baker's sample pattern is a fixed
hemisphere spiral — the test asserts the EXACT expected mean for that
fixed pattern, computed in Rust by mirroring the WGSL sample
generation. Pin it; don't assume symmetry). Occluding plane fixture:
the quad's texels near the occluder have bent normals tilted AWAY
from the occluder — assert the tilt direction (dot with the away-
vector > 0) and confidence < 1. Determinism: double-bake byte-
identical.

## Build order (both slices claw-shaped, GPU tests on the real adapter per the wiring claw's precedent)

1. ID bake (shader + module + hash tests) — smaller, no AO coupling.
2. Bent normals (AO accumulator extension + the fixed-pattern mean
   test + occluder test) — touches the merged AO shader; cross-review
   the WGSL diff carefully (the AO golden tests must stay green).

Both land in umber-bake only; the bakes panel + CLI rows ride the
app-wiring slice after (the panel's per-map checklist exists — the
new maps are checklist entries, not new UI patterns).

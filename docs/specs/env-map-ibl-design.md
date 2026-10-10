# Environment-Map IBL — Wave-4 Design (§5 P0)

Wave-4 item 6. Written 2026-10-09 against the tree at `2139df2`.
The audit's finding: §5's P0 shipped a procedural stand-in —
`environment_irradiance(n)` mixes two constants by the normal's Y.
This note contracts the image-based version: an HDR environment
texture actually driving the ambient term.

## The gap, precisely

Shaders.rs has TWO `environment_irradiance` call sites (the viewport
mesh pass ~line 50, a second pass ~line 215) plus `SUN_COLOR`/`SKY_-
COLOR` constants. The real IBL needs:

1. **An env-map asset path**: load an equirectangular HDR (EXR —
   umber-export's exr.rs already reads EXR; umber-app can reuse the
   reader) or a tonemapped LDR PNG fallback. Default asset: ship a
   small neutral studio equirect (generated, not licensed — a
   procedural overcast gradient rendered to EXR at 1024x512) so the
   app never blocks on a missing file.
2. **An irradiance integration**: for diffuse IBL, the env map must
   be pre-integrated (convolved over the hemisphere) — a 1024x512
   equirect sampled per-fragment with a 64-sample hemisphere loop is
   ~65k texture fetches per pixel: NOT viable per-frame. The
   standard answer: pre-filter at load time ONCE into a small
   irradiance map (32x16 equirect is plenty for diffuse — the
   low-frequency result).
3. **The specular tier**: v1 scopes DIFFUSE ONLY. A prefiltered
   mip-chain + BRDF-LUT (the Split-Sum approximation) is the classic
   wave-5 follow-up; the design names it but does not build it. The
   roughmetal parameters exist in OpenPBR's model already — the
   viewport shader just won't consume the specular term v1.

## Implementation shape

**umber-gpu**: a new `EnvIrradiance` resource — load equirect →
compute-shader convolve (hemisphere sample loop over the equirect,
~5k samples per output texel, once at load) → 32x16 Rgba16Float
irradiance texture → bind as a texture_2d<f32> in the mesh pass.
`environment_irradiance(n)` becomes `textureSample(irradiance_map,
sampler, equirect_uv(n))` with equirect_uv(n) = (atan2(n.z, n.x) /
2π + 0.5, asin(n.y)/π + 0.5). The procedural fn stays as the
fallback when no map is bound (uniform flag) — the two call sites
both route through one helper; the UV view pass keeps the procedural
path (flat 2D display wants unlit).

**umber-app**: a File > Load Environment… menu row → file dialog →
the EXR/PNG read → `EnvIrradiance::new(device, queue, pixels)`. The
asset-default load happens at startup if no user env is set.

**Tests (GPU-gated, the adapter precedent)**:
1. **Convolve correctness**: a synthetic equirect with a single
   bright pixel-band at a known direction; the irradiance output at
   the pole-aligned texel ≈ the band's radiance scaled by the
   hemisphere coverage — computed in Rust (mirror the convolve
   loop) and byte-asserted within unorm quantization. This is the
   mirrored-math test pattern that worked for bent normals.
2. **Sampler math**: equirect_uv for the six axis directions
   (+X, -X, +Y, -Y, +Z, -Z) = the exact expected UV pairs (a pure
   fn ported to Rust, exact asserts).
3. **Fallback regression**: no map bound → output byte-identical to
   today's procedural render (the golden tests pin this).
4. **End-to-end**: golden viewport render with the neutral studio
   env vs. the procedural — the images must DIFFER (a real env
   changes the ambient tint); assert a nonzero pixel delta over a
   threshold count. This is the can-fail test: an IBL that silently
   no-ops fails it.

## Build order

1. equirect_uv + EnvIrradiance convolve (umber-gpu, compute) —
   claw-shaped; the convolve shader + its mirrored-math test.
2. App wiring (menu row, default asset, fallback flag) — rides 1.
3. The generated neutral-studio EXR asset — a small Python/CLI
   generator script committed to assets/gen/ (deterministic output,
   the binary committed too).

The wave-5 specular tier (prefiltered mips + BRDF LUT) is named in
the audit's wave-5 list; this design doesn't block on it.

# Wave-3 Delivery Audit — §6 Export Pipeline & §9 Color Management

Requirement-by-requirement truth of what shipped against the wave-3
tags in requirements.md. Written 2026-10-09 19:45; every claim points
at a commit. The requirements doc stays the contract — this is the
delivery-side audit trail.

## §6 Export pipeline

| Requirement | State | Evidence / notes |
|---|---|---|
| Template-driven export, naming tokens | **Partial** | The token ENGINE is complete (expand_template, unknown tokens pass through — tested); the DRIVER feeds only `$textureSet` today (d64a65c). $mesh/$udim/$colorSpace/$srcMap/$layerName need driver-side sources — the CLI export knows mesh name; wiring is mechanical once painted maps exist. Presets serialize as files (to_json/from_json, roundtripped). |
| Channel packing, per-output slot mapping | **Done** | `ChannelWiring {map, slot}` per output channel (a1490ad fixed the bare-slot design gap); ORM/gltf-metR packings verified end-to-end in driver tests (5eb9cff). |
| Engine presets (glTF/Unreal/Unity/Blender) | **Done** | All four + JSON round-trip (a1490ad); Unreal ORM map-index wiring, Unity smoothness source, Blender passthroughs pinned by tests. |
| Bit depths 8/16/32F; PNG/EXR/TIFF/JPEG | **Done** | PNG8 (wave-2), PNG16 (73d8dc0, u16 + 16-bit sRGB curve), EXR 32F (f3b6bd9), TIFF + JPEG (73d8dc0 — JPEG alpha-dropped, documented: YCbCr has no alpha plane). Dithering: DONE (6edd477) — Floyd-Steinberg `dither_quantize_rgba8` at the f32→u8 boundary, alpha excluded, deterministic. |
| Normal-convention conversion (DX/GL Y-flip) | **Done** | Driver flips green for normal outputs only, decode-verified 200→55 (5eb9cff); the TBN bake takes directx_y_flip (2d7fa0a). |
| Padding: dilation + fill modes + 3D-neighbor | **Mostly done** | Dilation: done — 8-neighbor ping-pong, flag-driven, Bessa's transfer fix (b5a2e2d). Infinite dilation: DONE (a04e243) — `dilate_map_filled`, w+h bound covers every connected texel. Fill modes: DONE (0694344) — `fill_uncovered` for default-color, transparent mode is the documented no-op default. **3D-neighbor-aware (triangle-adjacency) padding: not built** (screen-space 8-neighbor only) — needs the seam-graph pass. |
| 8K export | Deferred (P1 wave 5) | Not started — as planned. |

## §9 Color management

| Requirement | State | Evidence / notes |
|---|---|---|
| Scene-linear working space, OCIO v2 configs incl. ACES 2.0 built-ins | **Bridge done, real OCIO pending** | Survey (65fc8be): ocio-rs 0.2.1 = OCIO 2.5.2 with built-in ACES 2.0 CG/Studio — no config vendoring needed. Stub-gated bridge landed (6b1e1fe): enumeration + creation + names pinned, stub-contract tests. The bundled C++ build (real configs) is the manual-dispatch step per the load directive. Working space today: the paint pipeline's linear rgba8unorm. |
| Per-channel CM flags (baseColor yes; data = no) | **Done** | umber-color's ColorManagement vocabulary (wave-1/2) + the driver's per-output transfer rule (5eb9cff): color maps sRGB, data maps linear — Bessa's cross-review catch enforced it through the dilate path too (b5a2e2d). |
| ICC profile embed on export | **Done** | png-crate native iCCP + bundled 2576-byte sRGB constant; lcms2 dropped (1e3a139, DECISIONS [19:20]). Roundtrip byte-identical, pixels preserved — tested. |
| Display transform via GPU LUT (OCIO shader extraction → 3D texture) | **Not started** | API path confirmed by probe (GpuShaderDesc::create — bfe4019); the survey documents the wgpu mapping. Lands with the display-transform panel. |

## Honest gaps (the next hunts)

1. ~~Dithering option (§6)~~ — DONE (6edd477).
2. Driver token sources beyond $textureSet (§6) — mechanical after painted maps.
3. ~~Infinite dilation + fill modes~~ — DONE (a04e243, 0694344). Remaining in the padding row: 3D-neighbor-aware padding (needs the seam-graph pass).
4. Real-OCIO bundled build + golden tests against the CPU reference (§9).
5. GPU display LUT via shader extraction (§9) — the viewport HDR path.

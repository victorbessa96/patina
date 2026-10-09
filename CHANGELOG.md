# Changelog

All notable changes to Umber are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); commits are
the source of truth.

## [Unreleased] — Wave 3 (2026-10-09)

The bake & export pipeline: six mesh-map bakers, UV padding, engine
presets, the export driver, the headless CLI, and the app panels —
workspace 268 tests green, CI on Windows and Linux for every commit.

### Added — bake engine

- §6 default-color fill mode — fill_uncovered completes the padding row (`0694344`)
- §6 infinite dilation — dilate_map_filled, the full-coverage entry point (`a04e243`)
- §6 dithering option — Floyd-Steinberg at the f32->u8 boundary (`6edd477`)
- export feeds the real baked tangent normal (`d64a65c`)
- bake-all tangent-space normal — 6 maps, the full baker surface (`83099d7`)
- tangent-space normal bake — screen-space TBN from the position pair (`2d7fa0a`)
- --dilate N flag on bake-ao + bake-all — uniform seam-padding control (`f2c29ee`)
- bake-all world-space normal — 6 maps per invocation (`eff0a8f`)
- world-space normal bake — readback of the position pass's normal texture (`4b83a03`)
- Bakes panel in the app — bake engine exposed in the UI (`af8b31b`)
- bake-all dilation post-pass — seam-free AO export (`6b728ae`)
- UV-padding dilation — 8-neighbor nearest-donor ping-pong passes (`6df5824`)
- bake-all completes the P0 surface — thickness map wired (`f466e0d`)
- thickness bake — inward ray-cast min-distance; P0 baker set complete (`ed23d0b`)
- bake-all covers the full landed bake surface — curvature + position maps (`2335e00`)
- curvature bake — screen-space signed curvature from the position/normal pair (`f012420`)
- CLI bake-all — batch bake with TextureSetName_map convention naming (`5327294`)
- mesh-map naming convention — TextureSetName_map parse + format (`fd58791`)
- headless bake — umber-cli bake-ao: mesh → GPU AO bake → PNG (`74393f6`)
- UV→world position map + mesh AO — real bakes replace the parameter plane (`8dea62d`)
- AO bake engine slice — compute-shader raycast core (plane-bake proving the path) (`620847f`)

### Added — export pipeline

- §9 ICC profile embed — iCCP chunk via the png crate, no lcms2 (`1e3a139`)
- §9 ocio bridge — stub-gated built-in ACES 2.0 config surface (`6b1e1fe`)
- export feeds the real baked tangent normal (`d64a65c`)
- Export dialog in the app — presets callable from the UI (`51ae037`)
- CLI export command — presets callable headless, §10 path closed (`51f0295`)
- export driver — presets to files, the §6 pipeline glue (`5eb9cff`)
- format writers complete — TIFF, JPEG, 16-bit PNG (`73d8dc0`)
- engine presets complete — Unreal ORM, Unity HDRP/URP, Blender Principled (`a1490ad`)
- bake-all dilation post-pass — seam-free AO export (`6b728ae`)
- EXR float export — 32-bit float maps via the exr crate (`f3b6bd9`)
- export preset engine — packing, conventions, formats, glTF preset (`fafb4c7`)
- mesh-map naming convention — TextureSetName_map parse + format (`fd58791`)
- headless bake — umber-cli bake-ao: mesh → GPU AO bake → PNG (`74393f6`)

### Added — app & CLI

- .umber project I/O in the app — Open/Save Project in the File menu (`98e4c89`)
- bake-all tangent-space normal — 6 maps, the full baker surface (`83099d7`)
- --dilate N flag on bake-ao + bake-all — uniform seam-padding control (`f2c29ee`)
- Export dialog in the app — presets callable from the UI (`51ae037`)
- bake-all world-space normal — 6 maps per invocation (`eff0a8f`)
- CLI export command — presets callable headless, §10 path closed (`51f0295`)
- Bakes panel in the app — bake engine exposed in the UI (`af8b31b`)
- bake-all dilation post-pass — seam-free AO export (`6b728ae`)
- bake-all completes the P0 surface — thickness map wired (`f466e0d`)
- bake-all covers the full landed bake surface — curvature + position maps (`2335e00`)
- CLI bake-all — batch bake with TextureSetName_map convention naming (`5327294`)
- headless bake — umber-cli bake-ao: mesh → GPU AO bake → PNG (`74393f6`)

### Added — earlier waves

- wave-2-tail→3): 3D-viewport painting — Ctrl+drag ray-picks the mesh and paints at the hit UV (`708ae8a`)
- wave-2-tail): texture tile pool — lazy virtual-texture allocation over 512² PaintTarget tiles (`ce70c42`)
- wave-2-tail): color display transforms — CPU reference path (OCIO-lite) (`a30efe1`)
- wave-2): File → Export Painted Map (PNG) — paint target to deliverable file (`4e7bb39`)
- wave-2): OpenPBR Surface viewport shader — real-time subset, GGX specular + coat + metal lerp + emission (third-claw scope, orchestrator-landed) (`0071bb9`)
- wave-2): OpenPBR params foundation + display-orientation fix (claw salvage, post-504) (`bf208a0`)
- wave-2): PNG export — paint-target readback to file, with sRGB transfer (`c97e8cb`)
- wave-2): document UI — Layers + History panels on the real layer stack + undo journal (`e1321a1`)
- wave-2): live paint canvas in the UV view — paint target presented beneath the wireframe (`8a1c142`)
- wave-2): texture-display callback — paint target presented in-panel (quad pipeline + CallbackTrait mirror) (`089ee44`)
- wave-2): paint input path — UV-view strokes through conditioner+adapter into PaintThread (`a08af15`)
- wave-2): paint-thread slab — command channel + segment/dispatch orchestration (claude claw + nemotron review + orchestrator fixes) (`1dd4dff`)
- wave-2): dab adapter — brush stamps to paint dabs (the stroke→splat bridge, brush side) (`ec10258`)
- wave-2): GPU paint target + dab compositing compute pass — the paint core (claude claw + nemotron cross-review) (`8875362`)
- wave-2): 2D UV view panel — wireframe of loaded mesh's UV triangles on the 0..1 square (`8128dff`)
- wave-2): stroke conditioning (umber-brush) + layer stack/undo/.umber format (umber-core) (`579eb09`)
- wave-2): golden-image test harness — headless render target + tolerance comparison (umber-gpu) (`ef12bcc`)
- wave-1): stylus Windows Ink backend skeleton (claude claw) (`905a389`)
- wave-1): IBL environment lighting — hemispherical sky/ground irradiance in the viewport shader (`342e8f1`)
- wave-1): wgpu viewport render pass (claude claw + cross-review integration) (`93e0140`)
- wave-1): real glTF/GLB + FBX loaders (opencode claw + cross-review integration) (`fa393d2`)
- wave-1): 11-crate workspace skeleton — builds, 12 tests green, clippy+fmt clean (`ba9fecc`)

### Fixed

- wave-3): Open Project clears the undo journal with the stack swap (`bcb5983`)
- wave-3): bake-all --dilate writes each map with its base transfer (`b5a2e2d`)
- wave-2): adapt UV-view canvas call site to the simplified TextureDisplay::callback signature (`a0599c3`)
- wave-2): texture-display vertex stride — uv attribute offset 16→8 (`ac343df`)
- wave-2): request TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES on the eframe device (`c2dbf79`)
- wave-2): umber-core hardening round — path-traversal rejection + case-collision validation (claude claw, post-landing review) (`fc50afd`)
- clone URL in CONTRIBUTING (`6c8e309`)
- gitignore inline-comment bug (Cargo.lock pattern was a literal); real comments now (`b4a2da0`)


# Adobe Substance 3D Painter — Exhaustive Feature & Workflow Inventory (as of 2026-10-09)

Reference target for an open-source Rust-based 3D texture-painting DCC (Windows + Linux). All claims sourced; URLs inline. Version numbers as of Oct 2026: current release **12.1.5 (2026/09/15)**; the modern bar is defined by the 12.x series (OpenPBR default, Vulkan, skew-map baking).

Primary sources:
- Docs home: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/home
- All-versions changelog (12.1.5 → 0.1.0-beta): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/release-notes/all-changes
- Python API: https://experienceleague.adobe.com/en/docs/substance-3d-dev/painter-python/api/api-overview
- Shader API: https://adobedocs.github.io/painter-shader-api/api/

---

## 1. Painting Tools

Tool list (official): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/tool-list

**Core tools**
- **Paint brush** — default tool; stamp-based stroke engine. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/paint-brush
- **Eraser** — sets layer alpha to zero rather than truly deleting paint (strokes recompute; filters can recover erased info). Per-channel enable. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/eraser
- **Path tools** (added 9.0.0, Jun 2023; massively expanded 9.1.0, 11.0.0, 11.1.0): curve on mesh surface with modes — Paint along path, Erase along path, Smudge along path, **Filled path** (11.0.0), **Ribbon path** (11.1.0 — stretches a single image/SBSAR along path with start/middle/end segments, per-vertex size/opacity). Per-vertex pressure editing, tangent editing (smooth/corner/custom/broken, ALT to break, CTRL to scale), angle constraint, snap-to-polygons, path panel (rename/delete/copy/paste/duplicate/visibility), path presets + favorites, Path display settings (handle size, width, colors, normals/tangents/direction arrows). Paths only work in 3D space, not UV/screen space. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/path-tools/path
- **Projection** — screen-space material/texture projection (camera-aligned); transformation via S+LMB rotate (SHIFT snaps 90°), S+RMB zoom, S+MMB pan; plus **Physical projection** (particle-based). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/projection
- **Polygon Fill** — pixel-mask fill tool (not a selection tool) with 4 modes: Triangle, Polygon, Mesh (connected sub-mesh), UV chunk/island. Hotkey 4; X inverts mask color. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/polygon-fill
- **Smudge** — stretch/mix/blur; non-destructive when used on a PassThrough layer. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/smudge-tool
- **Clone** — V sets source; "Clone source behavior" toggle (source follows stroke vs. fixed); non-destructive with PassThrough layer. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/clone-tool
- **Material Picker** (P) — temporary tool; copies material/channel properties from mesh surface, reverts to previous tool after pick.
- **Quick Mask** — Y to paint temporary mask, U to paint through it, I to invert, Y again to reset. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/quick-mask

**Brush engine parameters** (Paint tool Properties): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/paint-brush
- Size (pen-pressure bindable), **Flow** (per-stamp intensity, pressure-bindable), **Stroke Opacity** (global end-of-stroke max opacity, NOT pressure-bindable; "A" continues a previous stroke), **Spacing** (stamp distance), Angle, **Follow Path** (orient stamps to stroke direction; needs ≥2 stamps), **Size Jitter / Flow Jitter / Angle Jitter / Position Jitter** (0–1 randomization per stamp), **Alignment** (Camera | Tangent-Wrap default | Tangent-Planar | UV), **Backface Culling** (angle-threshold), **Size Space** (Object default / Viewport / Texture).
- **Alpha** — grayscale stamp mask; bitmap or Substance (.sbsar); exposed Substance param `hardness` auto-binds to the Hardness shortcut.
- **Physics (particles)** — enabled via "Physical" tool mode or particle presets; particle system is **PopcornFX** (.pkfx files, Emitter + Receiver saved into preset). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/presets/creating-particles-presets/creating-particles-presets
- **Stencil** — global viewport-space grayscale mask: S+LMB rotate (SHIFT snap 90°), S+RMB scale, S+MMB move, N hold to temporarily disable, reset button; tiling modes: No Tiling (default)/Horizontal/Vertical/H+V. 12.0.2 fixed stencil preview resolution. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/paint-brush
- **Material** section — per-channel enable toggles; Material mode loads a Substance/preset to drive multiple channels at once.
- **Dynamic Strokes** — Substance-driven per-stamp variation (alpha/substance changes per stamp; start/middle/end behavior, distance/size/spacing props). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/dynamic-strokes/dynamic-strokes
- **Lazy Mouse** — contextual-toolbar toggle; radius = smoothing distance between cursor and painted stamps. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/lazy-mouse
- **Presets** — Brush preset (brush params), Material preset (channels + Substance), Tool preset (both + tool type); stored in Assets folder, portable; ABR (Photoshop brush) import supported; tool presets savable from right-click in Properties. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/presets/presets

**Fill/projection modes (Fill layers & effects)**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/fill-projections/fill-projections
- Fill (match per UV Tile), UV projection, Tri-planar, Planar, Spherical, Cylindrical (with cap/cylinder-cap handling; "Backface Culling" renamed 8.3.0), **Warp projection** (interactive grid; **Warp to Geometry** auto-conform added 12.0.0 — grid wraps to mesh, points stick to surface, deformations preserved). Tiling values up to >128 (8.2.0), physical-size scaling for UV fill (8.3.0/11.1.0 displacement units).

**Symmetry**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/symmetry/symmetry
- **Mirror**: axis + axis-position offset, manipulator, Show Plane / Show Intersection / Show Cursor / Hide While Painting / Manipulator Size. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/symmetry/mirror-symmetry
- **Radial**: axis, count, Flip copy (U/V), axis position, show axis/intersection/cursor, manipulator. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/symmetry/radial-symmetry
- Fill-layer/effect symmetry (11.1.0, Nov 2025): symmetry in layer Properties, enabled for Tri-planar/Planar/Sphere/Cylindrical/Warp projections; Python-exposed. 2D-view paint/stencil projection does NOT support symmetry.

**Straight-line**: no dedicated line tool; straight strokes via Path tool + angle constraint ("constrain angle when creating a new point", 11.0.0) and tangent snapping.

---

## 2. Layer + Channel System

**Layer stack**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/layer-stack
- Bottom layer computed first; folder content processed before same-level siblings. Layers are **multi-channel**; paint tool writes to all channels enabled in the tool's Material section regardless of which channel the stack is viewing. Per-channel **blending mode + opacity** via top-left channel dropdown. Layer types: **Paint layer** (brush/particle-paintable), **Fill layer** (material-driven, not paintable, with projection settings), **Folder** (organizational; can carry masks/effects).
- Actions: Add Effect, Create Mask (white / black / bitmap / color-selection / height-combination), New Paint Layer, New Fill Layer, Add Smart Material (mini-shelf), New Folder, Delete. Ctrl+drop a material = fill layer with mask.
- **Layer instancing**: copy source layer, "Paste as instance" (Ctrl+Shift+V); only source editable; instance across texture sets in one action; instance cycle detection → broken instances disabled; icons navigate source↔instances. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/layer-instancing
- **Flatten layers (12.0.0, Mar 2026)**: Ctrl/Cmd+G group → Ctrl/Cmd+M flatten → new Fill layer with baked bitmaps, source group auto-disabled (optionally saved as Smart Material); "Flatten all instances across Texture Sets" (12.1.0); export flattened layers/masks/groups to disk, batch capable; default filename pattern `$textureSet_$layerName_$srcMap(.$udim)`; EXR for height/normal, PNG otherwise; 1px padding hardcoded; flattened images tagged & searchable in Assets panel, stored in .spp. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/flatten-layers
- Blending-mode copy/paste per layer ("Blending options" right-click), apply-blend-to-all-channels (8.2.0).

**Masks & effects**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/masking-and-effects
- Masks always grayscale; ALT+click to solo-view mask; SHIFT+click to toggle; copy/paste mask content; invert mask background (keeps effects). Re-adding/removing a mask destroys it and its effects. Effect stacks per content and per mask; colored underline indicates effects. Drag-drop effects into stack creates PassThrough layer.
- **Effect types**: Generator, Paint, Fill, Levels, Compare Mask, Filter, **Anchor Point**. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/effects
- **Anchor Points**: expose any layer/mask for reuse by Fill layers, Fill effects, and Substance filter inputs; same-texture-set only; must be BELOW referencing layer; reference list shown in Properties; jump-to-anchor. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/anchor-point
- **Geometry mask** (secondary mask): include/exclude by mesh names or UV Tiles; faster than paint masks, non-destructive on re-import, lets you paint on occluded geometry via "Hide excluded geometry"; copy/paste/include-all/exclude-all; Python API for include/exclude modes (12.1.0). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/geometry-mask
- **Quick Mask** (see §1).
- **Smart Masks**: right-click a mask → Create smart mask; drag onto layer creates black mask if none exists; accumulates effects; CTRL+drop replaces whole effect stack. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/smart-materials-and-masks

**Full blending-mode list** (per-channel, computed in linear space; HSV modes noted): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/blending-modes
1. Normal 2. **Passthrough** 3. **Disable** 4. **Replace** 5. Multiply 6. Divide 7. Inverse Divide 8. Darken (Min) 9. Lighten (Max) 10. Linear Dodge (Add) 11. Subtract 12. Inverse Subtract 13. Difference 14. Exclusion 15. Signed Addition (AddSub) 16. Overlay 17. Screen 18. Linear Burn 19. Color Burn 20. Color Dodge 21. Soft Light 22. Hard Light 23. Vivid Light 24. Linear Light 25. Pin Light 26. Tint (HSV) 27. Saturation (HSV) 28. Color (HSV) 29. Value (HSV) 30. **Normal Map Combine** (Whiteout) 31. **Normal Map Detail** (Reoriented Normal Mapping; default for normal channel) 32. **Normal Map Inverse Detail**.
- All blending performed in **linear gamma** internally.

**Channels per Texture Set**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/texture-set/texture-set-settings
- Standard channel vocabulary (from color-management table): Base color, Metallic, Roughness, Normal, Height, Ambient occlusion, Opacity, Anisotropy angle, Anisotropy level, Coat color/normal/opacity/roughness/specular level, Diffuse, Glossiness, Specular, Specular edge color, Specular level, Ior, Reflection, Displacement, Scattering, Scattering color, Sheen color/opacity/roughness, Translucency, Transmissive, Blending mask, plus **User channels (User0–15, renameable; limit raised 7.2.0 to 16)**. OpenPBR 1.1 channel set is now default (12.1.0): e.g. specular roughness, transmission depth/color, scatter, thin-film, fuzz, emission, etc. — full param list: https://experienceleague.adobe.com/en/docs/substance-3d/general-knowledge/openpbr/openpbr-overview
- Add/remove channels at any time; add-channel popup categorizes Supported (shader-usable) / Unsupported / User channels; **multi-channel select window + Apply to all Texture Sets** added 12.1.0. Paint data survives channel removal (recoverable on re-add).
- **Per-channel storage formats**: sRGB8, L8, RGB8, L16, RGB16, L16F, RGB16F, L32F, RGB32F. Storage type ≠ color space: base color stays gamma-corrected, roughness stays raw.
- **Color-managed channels** (fixed except user channels): Yes = base color, coat color, diffuse, scattering color, sheen color, specular, specular edge color, transmissive. Everything else = No (data). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/color-management/color-management
- **Resolution**: dynamic, non-destructive; lock icon toggles non-square. **Max in-app 4096²; max export 8192² (GPU-dependent; ≥1.5–2.5GB VRAM required)**. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/getting-started/project-creation
- **Mixing settings**: Normal mixing (Replace | Combine default — detail-oriented function), Height-to-normal method (Sharp | Smooth/Sobel default), AO mixing (Replace | Multiply default), **UV padding** (3D Space Neighbor default — samples across UV seams; 2D Space Neighbor — per-island copy; normal channel forced to 2D variant).
- **Shader instance** per Texture Set; per-Texture Set shader overrides via Texture Set list. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/texture-set/texture-set-list
- **Texture Set switching**: Texture Set List window shows all material IDs; selected/visible/hidden/disabled states; focus mode (isolate); show/hide all, invert; rename (feeds export naming), reset name, per-set description; shader instance creation; reassignment of texture sets; sub-stacks shown for Material Layering workflow. Only one texture set editable at a time. 12.1.0 adds viewport warning when painting on another texture set.
- **UV Tiles (UDIM)**: per-texture-set multi-tile workflow; tiles grouped per material; paint seamlessly across tiles in a set; per-tile resolution overrides; image sequences; custom tile names used at export (11.0.0). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/uv-tiles/uv-tiles
- **Dynamic Material Layering**: shader-declared sub-stacks (e.g. pbr-material-layering: 4 materials, 3 masks); channels defined in shader; 32-texture channel limit (Windows); material inputs via shader `//: materials` metadata; `param auto Material.channel` binding; export with "Export shaders parameters" → JSON describing stacks/materials/params, importable back. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/dynamic-material-layering

**Normal map painting**: dedicated normal blending modes (Detail default, Inverse Detail, Combine); texture-set "replace" mixing to paint over baked normal; normal color-space override (DirectX Y− default per resource; OpenGL Y+ via color space menu). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/advanced-channel-painting/normal-map-painting

---

## 3. Baking Engine

Docs root: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/baking/baking

**Map types (Mesh map bakers)**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/baking-mode/mesh-map-bakers
- **Ambient occlusion** — Secondary rays, Min/Max occluder distance, Relative to bounding box, Spread angle, Distribution, Ignore backface, Self occlusion (per-mesh selection), Attenuation, Ground plane + offset.
- **Curvature** — Method, Secondary rays, Sampling radius, Relative to bounding box, Self intersection, Auto tonemapping per UV tile (or manual Min/Max).
- **Position** — Mode (all-axis/single-axis), Axis, Normalization (bounding box/sphere/none), Normalization scale.
- **Thickness** — Secondary rays, Min/Max occluder distance, Relative to bounding box, Spread angle, Distribution, Self occlusion, Normalization.
- **Height baker** (added 8.1.0) — Normalization, Scaling divisor (manual).
- **Bent normals** (added 8.1.0) — Secondary rays, Min/Max occluder distance, Relative to bounding box, Spread angle, Distribution, Ignore backface, Self occlusion.
- **Opacity baker** (8.1.0).
- **ID map** — Color source: Vertex Color / Material Color / File ID / Mesh ID (Polygroup); Color generator: Random / Hue shift / Grayscale.
- **Normal (from mesh / from high-poly)**, **World Space Normal** — via common settings + matching.
- **Skew correction painting (12.1.0)**: paint skew map on low-poly mesh with brush/eraser/polygon fill; grayscale value picker; **Edge Protection** (Edge distance, Edge contrast); **Skew Base Normal Mode: mesh or per-triangle** (12.1.1); Skew preview shader + direction-vector visuals; auto rebake per map.
https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/baking/mesh-map-settings

**Common settings**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/baking/mesh-map-settings
- **Output size** (X/Y, lock for non-square), **Dilation width**, **Apply diffusion**.
- High-poly params: Use Low poly as high poly; multiple high-def meshes; **Cage generation**: Distance-based (front/rear inflate), **Automatic (experimental, 11.0.0)**, Custom file (same vertex count required); **Ignore backface**; **Match**: Always | By mesh name (low/high suffixes); **Antialiasing** (renamed "Supersampling" ×N in 8.3.0); bakers library version (3.15.4 in 11.1.1, 3.22.2 in 12.0.3); GPU raytracing incl. AMD RT (11.1.0); AO parity CPU/GPU fixed 12.0.0.

**Workflow (Baking mode, introduced 8.3.0 Jan 2023; reworked 12.1.0)**: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/baking/how-to-bake-mesh-maps
1. Switch to Baking mode (F8 / croissant icon / Mode menu). 2. Check Texture Sets / UV Tiles to bake. 3. Check bakers in Mesh Map Bakers panel. 4. Common settings. 5. Adjust cage (front/rear distances; red spots on intersection). 6. Bake (single Bake button showing "Texture Sets × UV Tiles × maps" count, 12.1.0). 7. Baking Log errors with jump-to-setting buttons.
- Per-map controls: check, viewport preview, quick-bake single map, **Auto-rebake toggle** (12.1.0), **Sync settings across texture sets** (per-map and common); copy/paste baker settings across texture sets; check-status copy (apply checked to more/all texture sets); drag across checkboxes; hard-edge-without-UV-seam error visualization; cage/high-poly/low-poly visibility toggles; baking uses neutral material; engine computation paused during bake.

---

## 4. Procedural System (.sbsar runtime)

- **Generators** (masks from mesh maps: Position/Curvature/WS normal etc.): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/generators/generator — mostly monochrome; usable in masks or layers; custom generators authored in Designer via "Painter Generator template".
- **Filters**: Effect-stack Substance filters (blur, highpass, FXAA, pixelate, posterize, smoothstep, threshold — 10.1.0; stylization, quantize, anisotropic kuwahara, bevel smooth, directional distance, grayscale conversion — 11.0.0; embroidery/decal filter, fill-area mask — 10.1.0; MatFx comic/watercolor/oil historical). Filters have blending mode + opacity (8.2.0).
- **Smart Materials**: folder-based presets (right-click folder → Create smart material); drag-drop anywhere in stack; stored on disk; **once added, the stack loses track of which smart material was used**; per-resource updates via Resources Updater plugin; save to specific location via Python (12.0.3). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/smart-materials-and-masks
- **Smart Masks**: effect-stack presets (see §2).
- **Substance runtime model**: .sbsar = compiled Substance archive (7z container per ArchiveTeam: http://justsolve.archiveteam.org/wiki/Substance). Parameters exposed at runtime; presets queryable via Python (`internal_properties`, list presets + values, 10.0.0); resolution override for Substance resources in tools/fills (11.1.0); graph input/output ColorSpace property (9.1.0); new engine map inputs `mesh_hard_edges` / `mesh_hard_edges_triangle` (12.1.0); auto-plumb of baked maps (AO/Curvature) into effect inputs; "use texture" toggles per input; Auto-update of modified .sbsar assets (11.0.0, off by default). Substance Engine versions shipped recently: 9.0 (9.0.0), 9.0.3 (9.1.0), 9.1.2 (10.0.0), 9.1.3 (10.1.0), 9.2.5 (11.1.0), 9.3.4 (12.0.0), 9.4.3 (12.0.3), 9.4.5 (12.1.1), 9.4.6 (12.1.3).
- **LICENSING (critical for an open-source clone)**:
  - Substance **Materials SDK** (the C++ engine that executes .sbsar) is distributed via Adobe Developer Console under **Adobe Developer Additional Terms** (https://wwwimages2.adobe.com/content/dam/cc/en/legal/servicetou/Adobe-Developer-Additional-Terms_en-US_20230622.pdf, referenced at https://community.adobe.com/questions-64/substance-3d-sdk-licence-631723). SDK page: https://developer.adobe.com/substance3d-sdk/ (Materials SDK downloadable; Models SDK "coming soon"; partnership contact substance-3d-partnership@adobe.com).
  - Adobe staff answer on community: "yes feel free to use this in your game!" but closed-source redistribution requires reading the Additional Terms; no FOSS-compatible license exists — **the engine is proprietary and NOT redistributable under open-source terms**; integrating and shipping the SDK DLLs inside an open-source app is legally unresolved and requires Adobe partnership/legal review. https://community.adobe.com/questions-59/substance-sdk-licensing-terms-629480
  - Substance 3D **Assets** product terms (assets ≠ engine, but relevant to content shipped in-app): perpetual-ish license to use/modify assets only inside Modified/Larger Works; no standalone redistribution; **no ML training on assets**. https://www.adobe.com/cc-shared/assets/pdf/legal/servicetou/adobe-substance-3d-assets-product-specific-terms-20250422.pdf
  - Practical consequence for the clone: .sbsar execution cannot be bundled; alternatives are required (own procedural runtime, or interop via sbsrender CLI which ships with Designer — see §10).
  - **Substance 3D Connector** (Send-To interop framework) IS open source, Apache 2.0: https://github.com/adobe/substance-3d-connector (LICENSE: https://github.com/adobe/substance-3d-connector/blob/main/LICENSE)
  - **OpenPBR**: the shading model is open (ASWF-hosted, Apache 2.0, Adobe's production implementation open-sourced: https://github.com/adobe/openpbr-bsdf). Overview: https://experienceleague.adobe.com/en/docs/substance-3d/general-knowledge/openpbr/openpbr-overview; FAQ: https://experienceleague.adobe.com/en/docs/substance-3d/general-knowledge/openpbr/openpbr-faq

---

## 5. Viewport & Shaders

- **Viewport structure**: Contextual toolbar (top), 3D view, 2D view (UV), progress bar; layout modes 3D/2D, 3D only, 2D only, Swap; perspective/orthographic; free/constrained camera rotation; Alt+LMB orbit, Alt+MMB pan, Alt+RMB zoom, Alt+Shift snap ortho angles. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/viewport/viewport
- **View modes (Display Settings)**: **Lighting** (full PBR w/ shadows), **Single channel** (solo/unlit per-channel view; unlit toggle, HDR scale, ±color HDR view, R/G/B/A component isolation), **Mesh maps** (baked-map solo view). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/display-settings/display-settings
- **Default shaders**: OpenPBR 1.1 (default since 12.1.0, replacing ASM as first option in New Project), **Adobe Standard Material (ASM)** (7.2.0: anisotropy, clear coat, SSS, specular edge color, sheen, opacity/translucency 9.1.0), legacy **pbr-metal-rough**, **pbr-spec-gloss**, coated/SSG variants, **pbr-material-layering** (dynamic material layering), non-PBR, toon, skin, iray-compatible MDLs, **custom .glsl shaders**. Shader list per docs + API: https://adobedocs.github.io/painter-shader-api/api/ ; custom shader authoring: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/custom-shaders ; PBR Metal Rough reference: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/scripting-and-development/shader-api-reference/shaders-shader-api/pbr-metal-rough-shader-api
- **Shader API**: GLSL fragment ("surface") shader — `void shade(V2F inputs)`; engine params (channels, additional maps, camera), rendering states, custom params/tweaks, `uniform_specialization` qualifier, embedded libs (lib-pbr, lib-sss, lib-pom (parallax occlusion), lib-coat, lib-sheen, lib-bent-normal, lib-emissive, lib-env, lib-normal, lib-alpha-test, lib-sampler, lib-sparse, lib-utils, lib-vectors, lib-random, lib-bayer), QML custom-ui metadata, stacks/materials metadata for material layering. https://adobedocs.github.io/painter-shader-api/api/
- **Shader settings window**: shader file picker, instance name, restore defaults, separate undo/redo stack for shader params; **Displacement + Tessellation preview**: Source channel (Height/Displacement), Scale unit (Normalized / Scene / **Physical size (cm)** — 11.1.0 real-world displacement), Scale amount; Tessellation: Subdivision mode Uniform (count 1–32) | Edge length (max length). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/shader-settings/shader-settings
- **Iray (render mode)**: GPU path-traced renderer (NVIDIA); F10 in / F9 out, camera icon, Mode menu; status (rendering/paused/done), resolution, scene size, iterations, time; Min/Max sample, Max time, **Caustic sampler**, **Firefly filter**, resolution override, Save Render, Share to ArtStation; OpenPBR 1.1 MDL for Iray added 12.1.0; shares Display settings with viewport. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/iray-renderer/iray-renderer and https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/iray-renderer/iray-settings
- **Display settings**: Environment map (HDR; opacity, exposure EV, rotation, blur, alignment World/Local, color-space override), Shadows (mode intensive/average/lightweight, opacity), Viewport settings (Anisotropic filtering 0/2/4/8/16 spp, MipMap bias, camera frame, hide-stencil-while-painting, stencil opacity, projection preview channel, mesh wireframe show/color/opacity, grid show/axis/color/opacity), Post-effects. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/display-settings/environment-settings and https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/display-settings/viewport-settings
- **Post-effects (new stack, 12.0.0)**: Depth of field (custom bokeh), Bloom, Glare, Lens flare, Lateral aberration, Vignette, Sharpen, Film grain, Tone mapping (updated mapper), Color correction; stack-ordered, per-effect toggle; default post-effect assets integrated in library. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/release-notes/version-12-0
- Baking-mode visualizations: cage wireframe, high/low mesh visibility, implicit cage display, hard-edge errors (8.3.0).
- Rendering backend: **full Vulkan on Windows/Linux (11.1.0)**, Metal on macOS (11.0.0, Intel Macs dropped), SVT sparse virtual textures (AMD hardware sparse 8.3.0), TAA on by default (9.1.0), SSS default on (9.1.0).

---

## 6. Export

- **Export window**: Settings tab (texture-set checklist w/ global overrides, check/uncheck/invert all), Output directory, Output template, File type (or "Based on output template" per-texture format), Size (per texture set / 128–**8192**; 8K needs >1.5GB VRAM), Padding (No padding passthrough / Dilation infinite / Dilation+transparent / Dilation+default color / Dilation+diffusion), per-map format/bitdepth overrides, color-space column, dithering for 8-bit banding, USD asset export checkbox (texture folder + .usda + optional assembled .usd incl. mesh; used in Omniverse). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/export/export-window/export-settings
- **Output template system**: Default (editable/duplicable) + Predefined (non-editable) templates. Templates saved as individual files (assets/export-presets), auto-embedded into project on save; naming flags **$mesh**, **$textureSet**, `/` folders (folders→PSD groups), **$colorSpace** (color-managed export), `$udim`, `$srcMap`, `$layerName` (flatten export). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/export/output-templates/creating-export-presets
- **Channel packing**: per-output-map slots accept RGB, individual R/G/B, Alpha, Grayscale of any input (input maps / mesh maps / converted maps). Converted maps: Normal OpenGL/DirectX, Mixed AO, Diffuse, Specular, Glossiness, Unity4 Diffuse/Gloss, Reflection, 1/ior, Glossiness², f0. Random color cue per dropped input. Missing channels get neutral defaults (height→gray).
- **Predefined templates**: 2D View; Document channels+Normal+AO (with/without alpha); Sketchfab; Substance 3D Stager; USDz (Apple AR); glTF PBR Metal Roughness; glTF PBR Metal Roughness + Displacement (experimental). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/export/output-templates/default-output-templates/predefined-presets
- **Default (editable) templates** (~30): PBR Metallic Roughness (+ PSD-group variant), PBR Specular Glossiness (+ converted variant), Non-PBR Specular Glossiness, **Unreal Engine (Packed)** (ORM packing; renamed 11.0.0), Unreal Engine SSS (Packed), Unity HDRP (Metallic Standard / Specular), Unity URP (Metallic Standard / Specular — smoothness conversion), Amazon Lumberyard, CryEngine 3, Dota 2, Arnold (AiStandard) + UDIM legacy, Corona, Keyshot / Keyshot 9+, Maxwell (MR/SG), Redshift, Renderman (pxrDisney/pxrSurface), Vray Next (MR/SG) + UDIM legacy, **Blender (Principled BSDF)**, Shade 3D, Roblox (MaterialVariant / SurfaceAppearance), Lens Studio, Spark AR Studio, **Mesh Maps** (exports baked slots; grayscale export update 11.1.0). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/export/output-templates/default-output-templates/default-presets
- **File formats & bit depths**: bmp, exr (16F/32F), gif, hdr (32F), ico, j2k/jp2/jng (8+dither/16), jpeg, jpeg-xr (8/16/32F), pbm, pfm (32F), pgm, png (8/8+d/16), ppm, psd (8/16; container — maps as layers), targa, tiff (8/16/32F), wbmp, webp, xpm. Dithering available at 8-bit.
- **Normal conventions**: project-level DirectX (X+,Y−,Z+) vs OpenGL (X+,Y+,Z+) — Unreal→DirectX, Unity→OpenGL; "Compute tangent space per fragment" (Unreal on, Unity off / HDRP on). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/getting-started/project-creation
- **UDIM export**: UV Tile workflow; per-tile textures; UDIM numbering; custom tile names at export (11.0.0); glTF export with physical size; Send-to panel (11.0.0) for one-click Send to Photoshop/Stager etc.
- **OpenPBR export** (12.1.0): naming-convention dropdown in Export window (defaults to OpenPBR when any shader uses it); channels regrouped by category for template creation (12.1.1).

---

## 7. Import Formats

- **Mesh/scene import (Painter)**: FBX (in/out), OBJ (in/out), **USD** (in/out; scope/variants, subdivision levels, frame selection; .geo.usd openable 12.0.0), glTF/GLB (in/out; normal-map flip option on import 10.1.0; alpha blending fixes 12.1.0), DAE (in/out), PLY (in/out), **ABC/Alembic (in)**; cameras importable (with limitations: 3ds Max physical cams, ortho cams in Alembic). Ecosystem format matrix: https://experienceleague.adobe.com/en/docs/substance-3d/general-knowledge/ecosystem/import-and-export-formats
- **Materials import**: SBSAR (in), MDL (in/out), GLSL shader files (in). MaterialX: not in Painter (Designer only).
- **Bitmaps**: png, tiff, exr, hdr, jpeg, etc.; PSD import (with layers support), **AI (native Illustrator, artboards, 10.0.0)**, **SVG vector (9.1.0)**, ABR (Photoshop brushes), **fonts .ttf/.otf as Text resources (10.0.0)** — embeddable-only fonts enforced.
- **Mesh maps import**: naming convention `TextureSetName_MeshMapName` → ambient_occlusion, curvature, normal_base, world_space_normals, position, thickness, id (also opacity, height, bent_normals via bakers).
- **Auto-unwrap** on import for missing UVs (seams/islands/packing per-mesh or recompute-all; margin 0–1%; orientation unconstrained/align-with-mesh; texel-density-driven UV tile count 9.1.0; avoid-elongated-islands 7.4.0; lock orientation 8.3.0; **Hard Surface unwrap mode 12.1.0** — orthographic, low-distortion layouts for mechanical meshes). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/automatic-uv-unwrapping
- **Project creation**: New Project window (File>New, Ctrl+N) — file, template (.spt), resolution, normal format, compute-tangent-per-fragment, UV Tile workflow (per-material tiles vs legacy split), import cameras, import baked maps, physical-size settings (mesh unit scale / custom / auto-switch fill scaling), color management (Legacy sRGB/linear default, OCIO or ACE/ICC). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/getting-started/project-creation
- **.spp project file**: **HDF5 binary container** (ArchiveTeam: http://justsolve.archiveteam.org/wiki/Substance); **not publicly documented by Adobe** — no spec, no open parser. App version stored inside .spp since 8.2 (queryable via Python `last_saved_substance_painter_version()`). Corrupt-file warnings exist ("project processed as text file"). **A clone must define its own project format; .spp compat is out of reach.** Lossless compression on 16-bit images to shrink projects (9.1.0).
- Project templates `.spt` shareable; Resources Updater plugin for asset refresh; "Remove unused resources" cleanup.

---

## 8. UI Layout (what a clone must mirror)

- **Main structure** (default layout): Tools toolbar (vertical, left) → Tools list https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/painting/paint-tools/tool-list; **Docks toolbar** (vertical, right) — quick dock open/close; **Plugins toolbar**; **Contextual toolbar** (immovable, top of viewport — tool params + fixed right-side viewport display shortcuts); central **Viewport** (3D+2D); **Properties** window (tool/brush/layer params; right-click viewport also opens it). Toolbars doc: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/toolbars ; Properties: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/properties
- **Assets window (formerly Shelf; redesigned 7.2.0)**: Starter assets + Your assets; filter area (category icons, saved searches, pin/favorites, breadcrumbs, list view, drag-out libraries); custom library locations on disk; import window for adding resources; "delete/reload/rename in user library"; auto-import on drag-drop; particle-brush tag. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/assets/assets
- **Layer stack window** (see §2) incl. channel dropdown, action icons, mask/effect stacks, geometry-mask icon states.
- **Texture Set list window** (see §2 end).
- **Shaders settings window** (see §5), **Display settings window** (environment/camera/viewport sections + post effects), **History window**, **Export dialog** (Settings + Output templates tabs), **Baking mode windows** (Mesh Map Bakers panel, Mesh Map Settings, Baking Log; reworked 12.1.0), **Path panel** (9.0.0).
- **Main menus**: File (New/Open/Save/Save as/Import/Export/Recent/Project configuration), Edit (Undo/Redo/Reimport mesh Ctrl+Shift+R), Mode (Painting/Baking/Iray switching), Window (dock management, save/load/reset UI layouts — 8.2.0; Painting vs Rendering layouts saved separately), Help (docs, Python/Shader API changelogs, plugins).
- **Onboarding**: Welcome panel (first-run, 8.2.0), What's New panel, Home screen.
- UI layouts saveable/loadable via `substance_painter.ui` scripting; high-DPI scaling fixed 8.2.0; translations: EN/DE/FR/ZH-CN (7.2.0).
- System requirements (Steam 2026 listing): Windows 10/11 + GTX 1060-class min; Ubuntu 22.04 LTS, RTX 2060 Super/RX 5700 XT min, RTX 3080/RX 6800 XT recommended; macOS 13+ Apple silicon (min macOS raised to Ventura in 12.1.0). https://store.steampowered.com/app/4329260/Substance_3D_Painter_2026/

---

## 9. Workflow Features

- **History window**: session-global project action list; click any entry to time-travel; branching replaces redo list; **history is session-only (not saved in .spp)**; shader params have their own independent undo stack. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/history ; "value" undo entries renamed to parameter names (12.1.0).
- **Drag & drop** (9.1.0 wave): external assets into layer stack, textures from Assets into stack, generators/filters onto mesh, smart masks as layers, single-channel images onto fill effects, CTRL/ALT modifiers to control effect/layer creation; decal drop aligns rotation to camera (10.0.0).
- **Symmetry**: brush + fill (11.1.0) mirror/radial (see §1).
- **Fill projects**: no special "fill project" mode; fill via Fill layers/fill effects + Polygon fill + Fill area mask/color filter (10.1.0).
- **Dynamic material layering** (see §2).
- **Camera management**: camera presets, imported cameras, default camera manipulation via Python (9.1.0). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/viewport/camera-management
- **Interop**: Send to Photoshop (layer stack export; masks), Send to Stager, Send to After Effects (Ae 24.1 beta, 9.1.0), Adobe Bridge integration, receive from Designer/Sampler (7.2.0), Substance 3D Assets panel in-app (10.0.0, ~20k assets, download manager), Substance 3D Connector (open-source Send-To framework, 2024), USD round-trip.
- **Color management**: OCIO 2 (7.4.0) + ACE/ICC (8.1.0); per-channel color-managed flags; display-transform dropdown per viewport; export color-space tags; OCIO default color-space override for color picker (12.0.2); ACE 7.0 (12.1.0).
- **Auto-update of modified assets** (.sbsar/.glsl/.ai/.svg) across project (11.0.0, optional, env-var to disable).
- **Performance features**: optimized layer thumbnails (10.1.0), parallelized brush stroke computation (11.0.0), lossless 16-bit compression (9.1.0), UI grouping of tool properties (11.1.0), only-first-channel enabled on new fills (11.0.0), partial UV-tile texture computation (11.1.1), async saving for UV-tile bakes (11.1.2).

---

## 10. Automation & Scripting

- **Python API** (`substance_painter` package, current doc version 0.3.5, Python 3.13 since 12.0.0): modules — application, async_utils, **baking** (parameters, launch/cancel, baker selection, curvature method, sync across texture sets, auto-cage 11.0.0), colormanagement, **display**, **event**, exception, **export** (list predefined + library presets, retrieve preset content, export textures/mesh), **js** (call JS from Python), **layerstack** (create/edit layers, masks, effects, fill projections, blending modes, uniform colors/resources, anchor points, smart masks, instanced layers, color selection, levels, compare mask, text resources, vector sources, scoped modifications w/ single undo), logging, **project** (open/save/create w/ USD params, project config, auto-update, last-saved version), properties, **resource** (shelf add/save smart materials & masks 11.0.0/12.0.3), **source** (Font/Vector sources), **textureset** (channels, per-UV-tile resolution, names/descriptions 11.0.0, geometry mask include/exclude 12.1.0), **ui** (dock widgets, QML). Overview: https://experienceleague.adobe.com/en/docs/substance-3d-dev/painter-python/api/api-overview ; module index: https://experienceleague.adobe.com/en/docs/substance-3d-dev/painter-python/api/module-index ; mirror: https://adobedocs.github.io/painter-python-api/
- **JavaScript/QML plugin API**: JS API v1.1.20 (12.0.0); plugins in `javascript/plugins` (10.1.0); custom QML shader UIs; default plugins: **Autosave**, **Resources Updater**. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/features/plugins/plugins
- **Command line**: `--help/-?/-h`, `--version`, `--mesh <file> [project.spp]`, `--mesh-map <TextureSetName_map>` (repeated; ambient_occlusion/curvature/normal_base/world_space_normals/position/thickness/id), `--split-by-udim`, `--export-path`, `--vram-budget <MB>`, `--disable-version-checking`, `--enable-remote-scripting` (remote control/headless via scripting). https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/pipeline-and-integration/configuration/command-lines ; remote control tutorial: https://experienceleague.adobe.com/en/docs/substance-3d-dev/painter-python/tutorials/remote-control
- **Substance Automation Toolkit (SAT)**: command-line tools (sbsbaker, sbscooker, sbsrender, sbsmutator, sbsupdater, sbsar_unpack, sbsdeps...) + PySBS python package for .sbs/.sbsar creation/modification/dependency management. Distribution via general license was closed/merged in 2022 (community threads on availability); `sbsrender` etc. effectively ship with Designer subscriptions; docs remain live: https://adobedocs.github.io/substance-automation-toolkit/ ; availability discussion: https://community.adobe.com/questions-50/automation-api-substance-discontinued-625968 . SAT governs .sbsar batch rendering — the pipeline a clone would need to interoperate with, not reimplement.
- **CCD plugins (11.1.0)**: Creative Cloud Desktop plugin distribution for Painter (public Python API plugins, e.g. Random Seed Batch Changer). https://blog.adobe.com/en/publish/2025/11/18/substance-3d-painter-update-adds-ribbon-tool-real-world-displacement

---

## 11. Changelog Highlights 2023 → 2026 (defines the modern bar)

Source: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/release-notes/all-changes (+ per-version pages)

- **8.1.0 (Jun 2022)**: ACE/ICC color management, physical-size material scaling, Height/Bent Normals/Opacity bakers, eyedropper overhaul, Qt 5.15.8, Python 3.9.
- **8.2.0 (Oct 2022)**: Welcome/What's-New panels, **export to SBSAR** (layer stacks), effects on folders, blending copy/paste, tiling >128, HDPI fixes, save/load UI layouts.
- **8.3.0 (Jan 2023)**: **New Baking mode (F8)** with Mesh Map Bakers/Settings/Baking Log windows, cage visualization, sync across texture sets, supersampling rename; **USD import/export** (scope/variants, subdivision, frames, USD export checkbox, mesh export); physical size for UV projection; Python baking module; AMD SVT; app version stored in .spp.
- **9.0.0 (Jun 2023)**: **Paint-along-Path tool** (3D curves, per-vertex pressure, tangents, marquee selection, path panel, paint/erase/smudge modes), dynamic strokes props, engine 9.0.
- **9.1.0 (Nov 2023)**: **SVG import** + transparency, Send to After Effects, big **drag-and-drop wave**, path manipulators/tangent copy-paste, ASM opacity/translucency/absorption channels, TAA + SSS on by default, colorSpace property from graphs, texel-density UV-tile control.
- **10.0.0 (May 2024)** — 10th anniversary: **native Illustrator (.ai artboards)**, **Substance 3D Assets panel in-app**, **Text resource (system fonts)**, **Python layer-stack editing** (30+ API additions: blending modes, fill settings, anchor points, masks, instanced layers, export presets), precise manipulator mode.
- **10.1.0 (Sep 2024)**: Fill area mask/color filter, embroidery decal filter, 6 generic filters (FXAA, pixelate, highpass, posterize, smoothstep, threshold), USD material/shader import + ASM USD export, VFX Platform 2024 (Python 3.11, OpenEXR 3.2, OCIO 2.3.2, OpenSubdiv 3.6), RedHat Linux, engine 9.1.3.
- **11.0.0 (Mar 2025)**: **Auto-update modified assets**, **Filled Path tool**, path overhaul (polygon snap, type switching, angle constraint, scale/rotate vertices, close-shape click, focus mode), **experimental Auto-cage baking**, 6 new filters + 3 texture generators (Tile Random, Triangle Grid, Scratches), Unreal template/preset renames, Python (smart material/mask save, auto-cage, texture set names), **Metal renderer on macOS, Intel Mac dropped**, "Send to" panel, export window unified.
- **11.1.0 (Nov 2025)**: **Ribbon tool** (seamless stretched SBSAR along path, start/middle/end segments, 75 presets, self-overlap blending, per-vertex opacity/size), **fill-layer symmetry**, **physical-size (cm) displacement**, **full Vulkan (Win/Linux)**, faster bakers + AMD raytracing, tool-properties regrouping, engine 9.2.5, CCD plugin support. https://blog.adobe.com/en/publish/2025/11/18/substance-3d-painter-update-adds-ribbon-tool-real-world-displacement
- **12.0.0 (Mar 2026)**: **Flatten layers + export flattened**, **Warp to geometry** (auto-warping decals), **new post-effects stack** (DoF/bokeh, bloom, glare, lens flare, lateral aberration, vignette, sharpen, film grain, new tone mapper, color correction), redesigned New Project/Project Configuration, reimport-mesh improvements, USD 25.05, Qt 6.8.6, Python 3.13, JS API 1.1.20, engine 9.3.4. https://blog.adobe.com/en/publish/2026/03/09/more-control-less-friction-texturing-workflows-latest-substance-3d-innovations
- **12.1.0 (Jun 2026)**: **Skew map painting** (brush/eraser/polyfill, edge protection, grayscale picker, skew preview shader, auto rebake, redesigned mesh-map list & baking UI), **OpenPBR 1.1 as default shader/workflow** (new project templates, USD import/export of OpenPBR, export naming convention, Iray MDL), **Hard-surface auto-unwrap**, multi-channel add/remove window, flatten across instances, engine inputs mesh_hard_edges(+_triangle), ACE 7.0, macOS 13 min. https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/release-notes/version-12-1 ; SIGGRAPH 2026 announcement: https://blog.adobe.com/en/publish/2026/07/21/adobe-substance-3d-unveils-new-innovations-deliver-faster-workflows-openpbr-everywhere-digital-twins-scale
- 12.1.1–12.1.5 (Jul–Sep 2026): skew base-normal mode (mesh/per-triangle), OpenPBR export regrouping, engine 9.4.5/9.4.6, skew/baking/projection/dynamic-stroke fixes, network export fixes.
- Content cadence: per-release filter/generator/alpha/material drops (e.g. 6 filters 11.0.0; 75 Ribbon presets 11.1.0; 848 materials + 2736 models added to 3D Assets in year to Jul 2026; 14k+ materials converting to OpenPBR).

---

## 12. AI Features

- **Painter itself has NO in-app generative AI as of 12.1.5.** Changelog grep shows zero "AI/generative/Firefly" features in Painter releases; the "AI" string appears only in an unrelated changelog line. Painter's 10th-anniversary AI messaging (May 2024) covered **Sampler** (Text to Texture, Text to Pattern via Adobe Firefly) and **Stager** (Generative Background) — not Painter. https://blog.adobe.com/en/publish/2024/05/29/painter-10th-anniversary-generative-ai-updates-for-substance-3d
- Sampler's Firefly-powered generative suite (Text to Texture, Text to Pattern, Image to Texture) is the ecosystem's AI surface, with results imported into Painter as ordinary materials/SBSAR: https://experienceleague.adobe.com/en/docs/substance-3d-sampler/using/features-and-workflows/generative-workflows
- AI-adjacent Painter capabilities: Text resource (font rendering, not AI), USD/digital-twin pipelines with Firefly context (e.g. LG case study references Substance 3D + Firefly, Jan 2026: https://blog.adobe.com/en/publish/2026/01/15/how-lg-household-healthcare-scales-digital-twin-use-substance-3d-firefly).
- Substance 3D Assets terms **prohibit using assets/outputs to train ML systems** — relevant constraint if a clone ever sources Adobe assets. https://www.adobe.com/cc-shared/assets/pdf/legal/servicetou/adobe-substance-3d-assets-product-specific-terms-20250422.pdf

---

## 13. Pricing & Licensing (positioning context)

- **Substance 3D Texturing** plan (Painter + Designer + Sampler + unmetered 3D Assets): **$24.99/mo or $249.88/yr** since 25 Mar 2025 (raised from $19.99/$219.88 — first Texturing price change since 2015). https://blog.adobe.com/en/publish/2025/02/20/substance-3d-innovations-pricing-updates and https://www.cgchannel.com/2025/02/adobe-to-raise-the-price-of-substance-3d-subscriptions/
- **Substance 3D Collection** (adds Stager + Modeler): $59.99/mo / $599.88/yr (from $49.99/$549.88); Teams $1,439.88/yr (from $1,199.88); Enterprise on enquiry. Plans page: https://www.adobe.com/products/substance3d/plans.html
- **Substance 3D Indie Bundle (Steam, GDC 2025)**: $24.99/mo, Painter+Designer+Modeler+curated assets. https://blog.adobe.com/en/publish/2025/03/17/create-faster-export-smarter-whats-new-in-substance-3d-painter
- **Perpetual licenses**: Painter/Designer ~$199.99 via Steam (Painter 2024 listing https://store.steampowered.com/app/2718190/ ; Painter 2026 listing https://store.steampowered.com/app/4329260/ — includes updates through the version year); Modeler $149.99; Sampler perpetual discontinued 2024. https://www.cgchannel.com/2025/02/adobe-to-raise-the-price-of-substance-3d-subscriptions/
- Unmetered Substance 3D Assets access (Jan 2025) bundled with subscriptions: https://blog.adobe.com/en/publish/2025/01/29/introducing-unmetered-access-to-substance-3d-assets
- Education: free for students/faculty via Adobe Edu processes; EDU asset terms have 30-day post-subscription grace period.

---

## Clone-Critical Notes (synthesis)

1. **.spp is an undocumented HDF5 container** — no interop possible; define an open project format (the clone's UVP).
2. **.sbsar execution requires Adobe's proprietary engine** under closed terms (Adobe Developer Additional Terms); not FOSS-bundleable. The procedural layer must be re-invented (node graph runtime) or delegated to external sbsrender/Designer installs.
3. **OpenPBR (Apache 2.0, ASWF)** is the legitimate open spec to target — Adobe's own open-source implementation exists (github.com/adobe/openpbr-bsdf); MaterialX is the interchange vehicle.
4. The modern bar: GPU Vulkan/Metal viewport, 8K export, UDIM, OCIO/ICC color pipeline, baking mode with skew painting + auto-rebake + auto-cage, path/ribbon procedural brushes, non-destructive layer/mask/effect/anchor-point graph, per-channel blending, template-driven channel-packed export, full Python+JS scripting, headless/remote automation.
5. Painter's moat is workflow density (layer stack semantics + smart materials + baking UX), not any single algorithm.

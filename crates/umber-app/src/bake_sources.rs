//! Shared bake-sourced map construction: the raw AO bake, the
//! flat-normal placeholder, and the painted Base Color bridge, used by
//! both the Bakes panel and the Export dialog so neither reimplements
//! the other's GPU call.
//!
//! `bake_ao` is the AO arm of `BakesPanel::run_all`, lifted out so the
//! Export dialog can build an `umber_export::MapSet` without re-deriving
//! the bake params; the panel still applies its own dilation afterward
//! (this module returns the raw bake, undilated — see
//! `LANDING_NOTES_EXPORT_DIALOG.md` for why the export path also stays
//! undilated, matching `umber-cli export`).
//!
//! Painted-maps bridge (wave-5, audit remainder #1): when a paint
//! session is live, the export's Base Color output maps to
//! [`painted_base_color`] instead of the flat placeholder —
//! paint-what-you-export. V1 scope is Base Color ONLY: normal/AO/etc.
//! keep their baked sources (paint-per-channel compositing is a later
//! slice).

use anyhow::Context as _;

use crate::paint_state::PaintState;

/// Which pixel source feeds the export's Base Color output — surfaced
/// in the Export dialog so the user always sees what the export
/// consumed (honesty in UI: a painted export must say painted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseColorSource {
    /// Live paint-target readback (paint-what-you-export).
    Painted,
    /// Procedural graph output (the wave-5 slice-5 bridge): carries the
    /// output node's id so the badge names what it consumed.
    Graph(u64),
    /// Flat placeholder: no paint session, no graph output, a failed
    /// readback, or a size mismatch — the honest fallback, never silent.
    FlatPlaceholder,
}

/// Pure three-way selection for the Base Color source (no GPU/IO): the
/// painted readback wins when correctly sized, else the graph output
/// (naming its node), else the flat placeholder. Extracted so the
/// dialog's badge and the driver can never disagree about priority.
pub fn pick_base_color_source(painted_len_ok: bool, graph: Option<(u64, bool)>) -> BaseColorSource {
    if painted_len_ok {
        BaseColorSource::Painted
    } else if let Some((node, len_ok)) = graph {
        if len_ok {
            BaseColorSource::Graph(node)
        } else {
            BaseColorSource::FlatPlaceholder
        }
    } else {
        BaseColorSource::FlatPlaceholder
    }
}

/// Reads the live paint target back as tightly-packed RGBA8 bytes
/// (`width×height×4`, the format `umber_export::png::write_png`
/// consumes) for the export's Base Color output.
///
/// Same-thread by construction: `PaintThread` owns no background
/// thread — it drains its command queue on `process_pending` (called
/// once per frame from the app), so borrowing `paint_target()` here
/// races with nothing. The display callback and the File > Export-PNG
/// path read back the same way. Note the readback reflects the last
/// drained frame: dabs staged but not yet processed are not included.
///
/// Returns `None` when there is no session (`paint` is `None`) or the
/// GPU readback fails (logged) — the caller falls back to
/// [`flat_base_color_rgba8`] via [`apply_base_color`].
///
/// Reads the ACTIVE tile (source-compat with the single-target callers);
/// the Export dialog now drives the per-tile [`painted_base_color_tile`],
/// so this entry point is retained for its tests and future
/// single-target callers.
#[allow(dead_code)]
pub fn painted_base_color(
    paint: Option<&PaintState>,
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
) -> Option<Vec<u8>> {
    read_back(paint?.paint_target(), device, queue)
}

/// The per-tile painted source (UDIM slice 5): `tile`'s paint target
/// read back as RGBA8, via [`PaintState::tile_target`]. `None` when there
/// is no session, nothing ever routed to `tile`, or the readback fails
/// (logged) — the caller then falls back to the flat placeholder.
/// Same-thread reasoning as [`painted_base_color`].
pub fn painted_base_color_tile(
    paint: Option<&PaintState>,
    tile: u16,
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
) -> Option<Vec<u8>> {
    read_back(paint?.tile_target(tile)?, device, queue)
}

/// Shared readback for both painted sources.
fn read_back(
    target: &umber_gpu::paint::PaintTarget,
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
) -> Option<Vec<u8>> {
    match target.read_back_rgba8(device, queue) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            log::warn!("painted base-color readback failed: {err:#}");
            None
        }
    }
}

/// A flat opaque-white Base Color map — the placeholder the export
/// uses until a paint session exists (white = glTF's default
/// `baseColorFactor`, i.e. "no paint" in a neutral, documented form).
pub fn flat_base_color_rgba8(size: u32) -> Vec<u8> {
    vec![255u8; (size as usize) * (size as usize) * 4]
}

/// Attaches the Base Color map to an export-ready [`umber_export::MapSet`]:
/// the painted readback when it exists at exactly `size²×4` bytes,
/// else the flat placeholder — the pure half of the painted bridge
/// (no GPU access), so it is directly testable.
///
/// V1 scope: only Base Color comes from paint; normal/AO/etc. keep
/// their baked sources (paint-per-channel is a later slice).
///
/// Retained as the two-source convenience (the painted bridge's
/// original entry point, still covered by its tests); the dialog drives
/// the three-way [`apply_base_color_with_graph`].
#[allow(dead_code)]
pub fn apply_base_color(
    set: &mut umber_export::MapSet,
    size: u32,
    painted: Option<Vec<u8>>,
) -> BaseColorSource {
    apply_base_color_with_graph(set, size, painted, None)
}

/// Three-way Base Color attach (the graph bridge): painted first, then
/// the graph output `(output node id, bytes)` when correctly sized,
/// else the flat placeholder. Priority preserves the painted bridge
/// exactly — a live session still wins over a graph.
pub fn apply_base_color_with_graph(
    set: &mut umber_export::MapSet,
    size: u32,
    painted: Option<Vec<u8>>,
    graph: Option<(u64, Vec<u8>)>,
) -> BaseColorSource {
    let expected = (size as usize) * (size as usize) * 4;
    let painted_len_ok = painted.as_ref().is_some_and(|b| b.len() == expected);
    let graph_len_ok = graph.as_ref().is_some_and(|(_, b)| b.len() == expected);
    let source = pick_base_color_source(
        painted_len_ok,
        graph.as_ref().map(|(node, _)| (*node, graph_len_ok)),
    );
    match source {
        BaseColorSource::Painted => {
            set.set(
                umber_export::MapKind::BaseColor,
                painted.expect("painted length checked"),
            );
        }
        BaseColorSource::Graph(_) => {
            set.set(
                umber_export::MapKind::BaseColor,
                graph.map(|(_, b)| b).expect("graph length checked"),
            );
        }
        BaseColorSource::FlatPlaceholder => {
            set.set(
                umber_export::MapKind::BaseColor,
                flat_base_color_rgba8(size),
            );
        }
    }
    source
}

/// Bakes ambient occlusion over `mesh` at `size²` with `rays` hemisphere
/// samples. Returns the raw RGBA8 bake (coverage alpha, no dilation) —
/// callers that want seam-filled output (the Bakes panel) dilate it
/// themselves; callers that mirror `umber-cli export` (the Export
/// dialog) use it as-is.
pub fn bake_ao(
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
    mesh: &umber_mesh::MeshData,
    size: u32,
    rays: u32,
) -> anyhow::Result<Vec<u8>> {
    let mut params = umber_bake::AoBakeParams::new(
        // max_distance/bias per the bake tests' convention; the plane is
        // unused by the mesh-fed path (the mesh's position map supplies
        // ray origins).
        10.0,
        0.01,
        umber_bake::PlaneDesc::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0],
        ),
    );
    params.rays = rays;
    umber_bake::ao::bake_ao_mesh(device, queue, mesh, size, size, &params).context("ao bake")
}

/// A flat tangent-space-up normal map (`[128, 128, 255, 255]` at every
/// texel) — the placeholder `umber-cli export` writes until the paint
/// pipeline grows a real normal source (painted or baked) to export
/// instead.
pub fn flat_normal_rgba8(size: u32) -> Vec<u8> {
    let texels = (size as usize) * (size as usize);
    let mut out = Vec::with_capacity(texels * 4);
    for _ in 0..texels {
        out.extend_from_slice(&[128, 128, 255, 255]);
    }
    out
}

/// Builds the export-ready [`umber_export::MapSet`] from an already-baked
/// AO buffer plus the flat-normal placeholder — the pure half of the
/// composition (no GPU access), so it is directly testable.
pub fn map_set_from(ao: Vec<u8>, size: u32) -> umber_export::MapSet {
    let mut set = umber_export::MapSet::new(size);
    set.set(umber_export::MapKind::AmbientOcclusion, ao);
    set.set(umber_export::MapKind::Normal, flat_normal_rgba8(size));
    set
}

/// Bakes AO and composes it with the flat-normal placeholder into one
/// [`umber_export::MapSet`] — the GPU-touching entry point the Export
/// dialog calls.
pub fn bake_export_map_set(
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
    mesh: &umber_mesh::MeshData,
    size: u32,
    rays: u32,
) -> anyhow::Result<umber_export::MapSet> {
    let ao = bake_ao(device, queue, mesh, size, rays)?;
    Ok(map_set_from(ao, size))
}

/// [`bake_export_map_set`] for one UDIM tile of a multi-tile mesh: AO
/// baked over the tile's own UV window (`umber_bake::ao::bake_ao_tile` —
/// the whole-mesh bake only rasterizes `[0, 1]`, i.e. tile 1001) plus
/// the flat-normal placeholder. Errors when the tile owns no triangles
/// (the bake core rejects the empty filtered mesh) — callers skip
/// geometry-less tiles first.
pub fn bake_export_map_set_tile(
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
    mesh: &umber_mesh::MeshData,
    size: u32,
    rays: u32,
    tile: u16,
) -> anyhow::Result<umber_export::MapSet> {
    let mut params = umber_bake::AoBakeParams::new(
        // Same params as `bake_ao`; `bake_ao_tile` swaps in the tile's
        // window plane.
        10.0,
        0.01,
        umber_bake::PlaneDesc::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0],
        ),
    );
    params.rays = rays;
    let ao = umber_bake::ao::bake_ao_tile(device, queue, mesh, size, size, &params, tile)
        .with_context(|| format!("ao bake, tile {tile}"))?;
    Ok(map_set_from(ao, size))
}

/// One tile's export-ready maps and which source fed its Base Color.
#[derive(Debug)]
pub struct TileMapSet {
    /// The UDIM tile number (`$udim` for this tile's files).
    pub tile: u16,
    /// The tile's maps (baked half + Base Color).
    pub set: umber_export::MapSet,
    /// What fed this tile's Base Color.
    pub base_source: BaseColorSource,
}

/// The per-tile map-set assembly (UDIM slice 5), the painted bridge's
/// per-tile variant: for each tile, `baked(tile)` supplies the baked half
/// (AO + normal) and `painted(tile)` that tile's paint readback, attached
/// as Base Color through [`apply_base_color_with_graph`]. The graph output
/// is a single `[0, 1]` image, so it only reaches tile 1001
/// ([`umber_mesh::FIRST_TILE`]); other tiles fall back to flat when their
/// readback is missing. With `tiles == [1001]` this is exactly the
/// pre-UDIM single-set assembly.
///
/// Injected sources keep the loop GPU-free and testable; the Export
/// dialog passes the real bakes and [`painted_base_color_tile`].
///
/// # Errors
///
/// The first `baked` failure (tile named in its context).
pub fn assemble_tile_map_sets(
    tiles: &[u16],
    size: u32,
    mut baked: impl FnMut(u16) -> anyhow::Result<umber_export::MapSet>,
    mut painted: impl FnMut(u16) -> Option<Vec<u8>>,
    graph: Option<(u64, Vec<u8>)>,
) -> anyhow::Result<Vec<TileMapSet>> {
    let mut graph = graph;
    let mut out = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        let mut set = baked(tile)?;
        let tile_graph = if tile == umber_mesh::FIRST_TILE {
            graph.take()
        } else {
            None
        };
        let base_source = apply_base_color_with_graph(&mut set, size, painted(tile), tile_graph);
        out.push(TileMapSet {
            tile,
            set,
            base_source,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_normal_is_up_everywhere() {
        let bytes = flat_normal_rgba8(2);
        assert_eq!(bytes.len(), 2 * 2 * 4);
        for texel in bytes.chunks_exact(4) {
            assert_eq!(texel, [128, 128, 255, 255]);
        }
    }

    #[test]
    fn flat_normal_zero_size_is_empty() {
        assert!(flat_normal_rgba8(0).is_empty());
    }

    #[test]
    fn map_set_from_carries_ao_and_normal_only() {
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let set = map_set_from(ao, size);
        let kinds: Vec<umber_export::MapKind> = set.maps_iter().collect();
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&umber_export::MapKind::AmbientOcclusion));
        assert!(kinds.contains(&umber_export::MapKind::Normal));
        assert!(!kinds.contains(&umber_export::MapKind::BaseColor));
    }

    #[test]
    fn flat_base_color_is_opaque_white() {
        let bytes = flat_base_color_rgba8(2);
        assert_eq!(bytes.len(), 2 * 2 * 4);
        for texel in bytes.chunks_exact(4) {
            assert_eq!(texel, [255, 255, 255, 255]);
        }
    }

    #[test]
    fn flat_base_color_zero_size_is_empty() {
        assert!(flat_base_color_rgba8(0).is_empty());
    }

    #[test]
    fn apply_base_color_falls_back_to_flat_without_paint() {
        // The empty-session path, headless: no painted bytes → the flat
        // placeholder feeds Base Color (never a missing map).
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let mut set = map_set_from(ao, size);
        let source = apply_base_color(&mut set, size, None);
        assert_eq!(source, BaseColorSource::FlatPlaceholder);
        let kinds: Vec<umber_export::MapKind> = set.maps_iter().collect();
        assert!(kinds.contains(&umber_export::MapKind::BaseColor));
        // White survives the driver's sRGB curve (255 → 255) and is
        // uniform under the PNG reader's row flip, so the fallback is
        // provable byte-exact through a real driver run.
        let preset = umber_export::ExportPreset {
            name: "base probe".into(),
            outputs: vec![umber_export::presets::OutputSpec {
                filename: "$textureSet_base.png".into(),
                maps: vec![(umber_export::MapKind::BaseColor, vec![])],
                channels: [
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::R),
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::G),
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::B),
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::A),
                ],
                normal_convention: umber_export::presets::NormalConvention::Opengl,
                format: umber_export::presets::OutputFormat::Png8,
            }],
        };
        let out = std::env::temp_dir().join(format!(
            "umber-bake-sources-flat-base-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&out);
        let written = umber_export::run_preset(
            &preset,
            &set,
            &umber_export::TokenSources::new("Flat"),
            &out,
        )
        .expect("flat base color must satisfy the driver");
        assert_eq!(written.len(), 1);
        let (_, _, decoded) = umber_export::png::read_png_rgba8(&written[0]).expect("decodes");
        assert_eq!(decoded.len(), (size * size * 4) as usize);
        for texel in decoded.chunks_exact(4) {
            assert_eq!(texel, [255, 255, 255, 255]);
        }
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn apply_base_color_rejects_wrong_sized_paint() {
        // A corrupt/short readback must not reach the driver (which
        // would fail SizeMismatch mid-export): it falls back to flat.
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let mut set = map_set_from(ao, size);
        let source = apply_base_color(&mut set, size, Some(vec![1u8; 7]));
        assert_eq!(source, BaseColorSource::FlatPlaceholder);
        assert!(
            set.maps_iter()
                .any(|k| k == umber_export::MapKind::BaseColor),
            "fallback still provides Base Color"
        );
    }

    #[test]
    fn apply_base_color_prefers_correctly_sized_paint() {
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let mut set = map_set_from(ao, size);
        let painted = vec![7u8; (size * size * 4) as usize];
        let source = apply_base_color(&mut set, size, Some(painted));
        assert_eq!(source, BaseColorSource::Painted);
    }

    #[test]
    fn pick_base_color_source_prefers_painted_then_graph_then_flat() {
        // Pure-logic priority probe (headless): painted wins, else the
        // graph (naming its node), else flat. A priority regression
        // FAILS here (e.g. graph stealing a live session).
        assert_eq!(
            pick_base_color_source(true, Some((7, true))),
            BaseColorSource::Painted
        );
        assert_eq!(
            pick_base_color_source(false, Some((7, true))),
            BaseColorSource::Graph(7)
        );
        assert_eq!(
            pick_base_color_source(false, Some((7, false))),
            BaseColorSource::FlatPlaceholder
        );
        assert_eq!(
            pick_base_color_source(false, None),
            BaseColorSource::FlatPlaceholder
        );
        assert_eq!(pick_base_color_source(true, None), BaseColorSource::Painted);
    }

    #[test]
    fn apply_base_color_with_graph_feeds_the_driver() {
        // Graph bytes at exactly size²×4 land on Base Color with the
        // Graph(node) source; wrong-sized graph bytes fall back to flat.
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let mut set = map_set_from(ao, size);
        let graph_bytes = vec![3u8; (size * size * 4) as usize];
        let source = apply_base_color_with_graph(&mut set, size, None, Some((11, graph_bytes)));
        assert_eq!(source, BaseColorSource::Graph(11));

        let mut set = map_set_from(vec![9u8; (size * size * 4) as usize], size);
        let source = apply_base_color_with_graph(&mut set, size, None, Some((11, vec![1u8; 7])));
        assert_eq!(source, BaseColorSource::FlatPlaceholder);

        // Painted still wins over a valid graph (the painted bridge is
        // preserved, not replaced).
        let mut set = map_set_from(vec![9u8; (size * size * 4) as usize], size);
        let source = apply_base_color_with_graph(
            &mut set,
            size,
            Some(vec![7u8; (size * size * 4) as usize]),
            Some((11, vec![3u8; (size * size * 4) as usize])),
        );
        assert_eq!(source, BaseColorSource::Painted);
    }

    /// Requests a paint-capable device (graceful skip where no wgpu
    /// adapter exists — the same pattern as `paint_state`'s GPU tests).
    /// Returns `None` when the test must be skipped.
    fn try_request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("skipping: no wgpu adapter available");
            return None;
        };
        if !adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            eprintln!("skipping: adapter lacks TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES");
            return None;
        }
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            ..Default::default()
        }))
        .ok()
    }

    /// The IEC 61966-2-1 linear→sRGB curve at 8-bit precision — the same
    /// math `umber_export::png` applies to color outputs, restated here
    /// so the test pins the transfer instead of importing it.
    fn srgb_u8(linear: u8) -> u8 {
        let l = linear as f32 / 255.0;
        let s = if l <= 0.003_130_8 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    }

    #[test]
    fn painted_pixels_survive_into_the_exported_png() {
        // THE CAN-FAIL CORE (GPU-gated): stage real dabs through a
        // PaintState, read back via the new bridge path, export through
        // the real preset driver, decode the PNG, and prove the painted
        // pixels arrived. A bridge that silently exports the flat
        // placeholder FAILS here (the export would be uniform white).
        let Some((device, queue)) = try_request_device() else {
            return;
        };
        let mut paint = PaintState::new(device.clone(), queue.clone()).expect("paint state builds");
        // March across the target center: 60 events converge the stroke
        // conditioner well past its spacing threshold, laying a dense
        // brick-red run (the default brush color [0.85, 0.2, 0.1]).
        paint.begin_stroke(egui::pos2(0.35, 0.5));
        for i in 1..=60u32 {
            paint.extend_stroke(egui::pos2(0.35 + 0.005 * i as f32, 0.5 + 0.001 * i as f32));
        }
        paint.end_stroke();
        let stats = paint.process_pending().expect("drain succeeds");
        assert!(
            stats.dabs_composited > 0,
            "stroke must composite dabs, got {stats:?}"
        );

        let (w, h) = paint.paint_target().dimensions();
        assert_eq!((w, h), (512, 512));
        let readback =
            painted_base_color(Some(&paint), &device, &queue).expect("live session reads back");
        assert_eq!(readback.len(), w as usize * h as usize * 4);
        // Distinctive paint landed: opaque reddish texels (the default
        // brush color ~[217, 51, 26]), not transparent black, not white.
        assert!(
            readback
                .chunks_exact(4)
                .any(|px| { px[0] > 150 && px[0] > px[1].saturating_mul(2) && px[3] > 150 }),
            "readback must contain opaque reddish paint"
        );

        // Through the real export path: dummy AO + painted Base Color,
        // glTF preset filtered to its satisfiable outputs (which must
        // now include baseColor — the pre-bridge dialog only kept
        // normal).
        let size = w;
        let ao = vec![200u8; readback.len()];
        let mut map_set = map_set_from(ao, size);
        let source = apply_base_color(&mut map_set, size, Some(readback.clone()));
        assert_eq!(source, BaseColorSource::Painted);
        let preset = umber_export::ExportPreset::gltf_metal_rough();
        let available: Vec<umber_export::MapKind> = map_set.maps_iter().collect();
        let (filtered, _) = crate::export_dialog::filter_satisfiable(&preset, &available);
        assert!(
            filtered.outputs.iter().any(|o| o
                .maps
                .iter()
                .any(|(k, _)| *k == umber_export::MapKind::BaseColor)),
            "painted Base Color must keep the baseColor output"
        );
        let out = std::env::temp_dir().join(format!(
            "umber-bake-sources-painted-bridge-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&out);
        let written = umber_export::run_preset(
            &filtered,
            &map_set,
            &umber_export::TokenSources::new("Bridge"),
            &out,
        )
        .expect("painted map set satisfies the filtered preset");
        let base_path = written
            .iter()
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("baseColor"))
            })
            .expect("a baseColor output was written");
        let (dw, dh, decoded) =
            umber_export::png::read_png_rgba8(base_path).expect("exported PNG decodes");
        assert_eq!((dw, dh), (w, h));
        assert_eq!(decoded.len(), readback.len());
        // Byte-exact modulo the two DOCUMENTED transforms: the PNG
        // reader returns GPU row order (vertical flip) and the driver
        // encodes color outputs with the sRGB transfer (alpha linear).
        for row in 0..h as usize {
            for col in 0..w as usize {
                let s = &readback[(row * w as usize + col) * 4..][..4];
                let d = &decoded[((h as usize - 1 - row) * w as usize + col) * 4..][..4];
                assert_eq!(
                    [srgb_u8(s[0]), srgb_u8(s[1]), srgb_u8(s[2]), s[3]],
                    d,
                    "painted texel ({col}, {row}) must survive the export"
                );
            }
        }
        // And the distinctive color is visibly there (not flat white).
        assert!(
            decoded.chunks_exact(4).any(|px| px[0] > 200 && px[1] < 150),
            "exported PNG must contain the painted red"
        );
        std::fs::remove_dir_all(&out).ok();
    }

    // --- Per-tile export (UDIM slice 5) ---

    #[test]
    fn assemble_routes_each_tiles_paint_and_keeps_graph_on_1001() {
        // Headless probe of the per-tile loop: each tile's Base Color is
        // ITS painted bytes; a tile with no readback falls back to flat,
        // never to the graph (the graph image is the [0,1] tile's).
        let size = 1;
        let texel = |v: u8| vec![v, v, v, 255];
        let sets = assemble_tile_map_sets(
            &[1001, 1002, 1003],
            size,
            |_| Ok(map_set_from(texel(9), size)),
            |tile| match tile {
                1001 => Some(texel(11)),
                1002 => Some(texel(22)),
                _ => None,
            },
            Some((5, texel(77))),
        )
        .expect("assembly succeeds");
        let tiles: Vec<u16> = sets.iter().map(|t| t.tile).collect();
        assert_eq!(tiles, vec![1001, 1002, 1003]);
        assert_eq!(sets[0].base_source, BaseColorSource::Painted);
        assert_eq!(sets[1].base_source, BaseColorSource::Painted);
        assert_eq!(sets[2].base_source, BaseColorSource::FlatPlaceholder);

        // Decode each set's Base Color through the driver to see the bytes.
        let preset = umber_export::ExportPreset {
            name: "base probe".into(),
            outputs: vec![umber_export::presets::OutputSpec {
                filename: "$textureSet_base.png".into(),
                maps: vec![(umber_export::MapKind::BaseColor, vec![])],
                channels: [
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::R),
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::G),
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::B),
                    umber_export::presets::ChannelWiring::new(0, umber_export::ChannelSlot::A),
                ],
                normal_convention: umber_export::presets::NormalConvention::Opengl,
                format: umber_export::presets::OutputFormat::Png8,
            }],
        };
        let out = std::env::temp_dir().join(format!(
            "umber-bake-sources-assemble-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&out);
        let pairs: Vec<(u16, &umber_export::MapSet)> =
            sets.iter().map(|t| (t.tile, &t.set)).collect();
        let written = umber_export::run_preset_tiled(
            &preset,
            &pairs,
            &umber_export::TokenSources::new("T"),
            &out,
        )
        .expect("assembled sets satisfy the driver");
        let reds: Vec<u8> = written
            .iter()
            .map(|p| umber_export::png::read_png_rgba8(p).expect("decodes").2[0])
            .collect();
        assert_eq!(reds, vec![srgb_u8(11), srgb_u8(22), 255]);

        // Without paint, the graph reaches tile 1001 only.
        let sets = assemble_tile_map_sets(
            &[1001, 1002],
            size,
            |_| Ok(map_set_from(texel(9), size)),
            |_| None,
            Some((5, texel(77))),
        )
        .expect("assembly succeeds");
        assert_eq!(sets[0].base_source, BaseColorSource::Graph(5));
        assert_eq!(sets[1].base_source, BaseColorSource::FlatPlaceholder);
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn assemble_propagates_a_bake_failure() {
        let err = assemble_tile_map_sets(
            &[1001, 1002],
            1,
            |tile| {
                if tile == 1002 {
                    anyhow::bail!("tile {tile} has no geometry")
                }
                Ok(map_set_from(vec![9, 9, 9, 255], 1))
            },
            |_| None,
            None,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("1002"));
    }

    /// Paints a horizontal run at UV height `v` from `u0` (60 events, the
    /// painted-bridge test's convergence regime), then drains.
    fn paint_run(paint: &mut PaintState, u0: f32, v: f32) {
        paint.begin_stroke(egui::pos2(u0, v));
        for i in 1..=60u32 {
            paint.extend_stroke(egui::pos2(u0 + 0.005 * i as f32, v));
        }
        paint.end_stroke();
        let stats = paint.process_pending().expect("drain succeeds");
        assert!(stats.dabs_composited > 0, "stroke must composite dabs");
    }

    fn is_reddish(px: &[u8]) -> bool {
        px[3] > 150 && px[0] > 150 && px[2] < 100
    }

    fn is_bluish(px: &[u8]) -> bool {
        px[3] > 150 && px[2] > 150 && px[0] < 100
    }

    #[test]
    fn each_tiles_paint_exports_to_its_own_file() {
        // THE CAN-FAIL CORE (GPU-gated): red in tile 1001, blue in tile
        // 1002, assembled per tile, exported, decoded. A bridge that
        // reads the active tile for every tile (both files blue), or
        // tile 1001 for every tile (both red), or one target for both
        // FAILS here: each PNG must hold ITS color and not the other's.
        let Some((device, queue)) = try_request_device() else {
            return;
        };
        let mut paint = PaintState::new(device.clone(), queue.clone()).expect("paint state builds");
        paint.set_mesh(&crate::paint_state::two_tile_strip());
        paint.set_brush_color([0.85, 0.2, 0.1, 1.0]);
        paint_run(&mut paint, 0.35, 0.5); // tile 1001
        paint.set_brush_color([0.1, 0.2, 0.85, 1.0]);
        paint_run(&mut paint, 1.35, 0.5); // tile 1002
        assert_eq!(paint.tiles_present(), vec![1001, 1002]);

        let (w, h) = paint.paint_target().dimensions();
        let size = w;
        assert_eq!(w, h);
        let sets = assemble_tile_map_sets(
            &paint.tiles_present(),
            size,
            |_| Ok(map_set_from(vec![200u8; (size * size * 4) as usize], size)),
            |tile| painted_base_color_tile(Some(&paint), tile, &device, &queue),
            None,
        )
        .expect("assembly succeeds");
        assert!(sets
            .iter()
            .all(|t| t.base_source == BaseColorSource::Painted));

        let available: Vec<umber_export::MapKind> = sets[0].set.maps_iter().collect();
        let (filtered, _) = crate::export_dialog::filter_satisfiable(
            &umber_export::ExportPreset::gltf_metal_rough(),
            &available,
        );
        let out = std::env::temp_dir().join(format!(
            "umber-bake-sources-per-tile-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&out);
        let pairs: Vec<(u16, &umber_export::MapSet)> =
            sets.iter().map(|t| (t.tile, &t.set)).collect();
        let written = umber_export::run_preset_tiled(
            &filtered,
            &pairs,
            &umber_export::TokenSources::new("Strip"),
            &out,
        )
        .expect("per-tile sets satisfy the filtered preset");

        let decode = |name: &str| {
            let path = written
                .iter()
                .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
                .unwrap_or_else(|| panic!("{name} was not written: {written:?}"));
            umber_export::png::read_png_rgba8(path)
                .expect("exported PNG decodes")
                .2
        };
        let tile_1001 = decode("Strip_baseColor_1001.png");
        let tile_1002 = decode("Strip_baseColor_1002.png");
        assert!(
            tile_1001.chunks_exact(4).any(is_reddish),
            "tile 1001 must carry its red paint"
        );
        assert!(
            !tile_1001.chunks_exact(4).any(is_bluish),
            "tile 1001 must not carry tile 1002's blue"
        );
        assert!(
            tile_1002.chunks_exact(4).any(is_bluish),
            "tile 1002 must carry its blue paint"
        );
        assert!(
            !tile_1002.chunks_exact(4).any(is_reddish),
            "tile 1002 must not carry tile 1001's red"
        );
        std::fs::remove_dir_all(&out).ok();
    }
}

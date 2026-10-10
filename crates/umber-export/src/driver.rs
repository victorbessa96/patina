//! The export driver (requirements.md §6): runs an [`ExportPreset`]
//! against a set of source maps and writes the packed output files —
//! the glue that turns the preset engine + format writers + naming
//! convention into a working pipeline.
//!
//! Composition: [`MapSet`] carries RGBA8 source maps keyed by
//! [`MapKind`]; [`run_preset`] packs each [`OutputSpec`] texel-by-texel
//! ([`pack_texel`]), applies the normal-convention conversion, expands
//! the filename template with the §6 tokens, and writes through the
//! format writers. The driver is pure CPU — callable from the CLI, the
//! app, or tests.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::png::Transfer;
use crate::presets::{
    convert_normal, pack_texel, ExportPreset, MapKind, NormalConvention, OutputFormat, Texel,
};
use crate::{exr, formats};

/// Whether a map kind is color-managed (sRGB transfer at export) or
/// data (linear passthrough) — requirements §9: "baseColor yes;
/// roughness/metallic/normal = data".
fn map_kind_is_color(kind: MapKind) -> bool {
    matches!(kind, MapKind::BaseColor | MapKind::Emissive)
}

/// Everything that can go wrong driving an export.
#[derive(Debug, Error)]
pub enum ExportError {
    /// An output references a map the set doesn't carry.
    #[error("output {output:?} needs map {map:?} but the set doesn't have it")]
    MissingMap {
        /// The failing output's filename template.
        output: String,
        /// The missing map kind.
        map: MapKind,
    },
    /// A source map's size doesn't match the requested export size.
    #[error("map {map:?} is {actual} texels; expected {expected} ({size}x{size})")]
    SizeMismatch {
        /// The offending map kind.
        map: MapKind,
        /// Actual texel count.
        actual: usize,
        /// Expected texel count.
        expected: usize,
        /// The square export size.
        size: u32,
    },
    /// The underlying format writer failed.
    #[error("write failed: {0}")]
    Write(String),
}

impl From<crate::png::PngError> for ExportError {
    fn from(err: crate::png::PngError) -> Self {
        Self::Write(err.to_string())
    }
}

impl From<exr::ExrError> for ExportError {
    fn from(err: exr::ExrError) -> Self {
        Self::Write(err.to_string())
    }
}

impl From<formats::TiffError> for ExportError {
    fn from(err: formats::TiffError) -> Self {
        Self::Write(err.to_string())
    }
}

impl From<formats::JpegError> for ExportError {
    fn from(err: formats::JpegError) -> Self {
        Self::Write(err.to_string())
    }
}

/// Source maps for one texture set, keyed by kind. All maps must share
/// the export resolution (validated at run time).
#[derive(Debug, Clone, Default)]
pub struct MapSet {
    maps: Vec<(MapKind, Vec<u8>)>,
    size: u32,
}

impl MapSet {
    /// An empty set at the given square resolution.
    pub fn new(size: u32) -> Self {
        Self {
            maps: Vec::new(),
            size,
        }
    }

    /// Adds (or replaces) a map. The buffer must be `size×size×4`
    /// RGBA8 — enforced at [`run_preset`] time, not here, so partial
    /// sets can be built incrementally.
    pub fn set(&mut self, kind: MapKind, rgba8: Vec<u8>) {
        if let Some(slot) = self.maps.iter_mut().find(|(k, _)| *k == kind) {
            slot.1 = rgba8;
        } else {
            self.maps.push((kind, rgba8));
        }
    }

    fn get(&self, kind: MapKind) -> Option<&[u8]> {
        self.maps
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, bytes)| bytes.as_slice())
    }

    /// The kinds currently held (for callers filtering presets to
    /// satisfiable outputs).
    pub fn maps_iter(&self) -> impl Iterator<Item = MapKind> + '_ {
        self.maps.iter().map(|(k, _)| *k)
    }
}

/// Single-tile UDIM code: every export lands on tile 1001 until the
/// wave-5 multi-tile texture sets arrive (requirements §2: UDIM is a
/// wave-4/5 item; the token must still expand today, so it expands to
/// the only tile that exists).
pub const SINGLE_TILE_UDIM: &str = "1001";

/// Driver-side sources for the §6 naming tokens, carried per export run.
///
/// `$textureSet` was the only token the driver fed (d64a65c); the rest
/// passed through verbatim. Every field here is static per run except
/// `$srcMap`/`$colorSpace`, which the driver derives per output (see
/// [`src_map_token`] / [`output_color_space`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSources<'a> {
    /// `$textureSet` — the texture-set name (mesh material or file stem).
    pub texture_set: &'a str,
    /// `$mesh` — the loaded mesh's file stem ([`mesh_stem`]); `""` when
    /// unknown, in which case the placeholder stays literal per
    /// [`crate::expand_template`]'s empty-value contract.
    pub mesh: &'a str,
    /// `$layerName` — v1: the topmost layer's display name (`""` when
    /// the document has no layers; single-layer painting until the
    /// layer-compositor export lands).
    pub layer_name: &'a str,
    /// `$udim` — v1: always [`SINGLE_TILE_UDIM`].
    pub udim: &'a str,
}

impl<'a> TokenSources<'a> {
    /// Minimal sources: texture set only, mesh/layer empty (the
    /// headless/CLI shape — filenames using only `$textureSet` expand
    /// exactly as before, pinned by test).
    pub fn new(texture_set: &'a str) -> Self {
        Self {
            texture_set,
            mesh: "",
            layer_name: "",
            udim: SINGLE_TILE_UDIM,
        }
    }

    /// Expands the static (run-level) tokens in `template`, leaving the
    /// per-output `$srcMap`/`$colorSpace` placeholders untouched — for
    /// display strings (the Export dialog's skipped-output lines) where
    /// no output spec exists.
    pub fn expand_static(&self, template: &str) -> String {
        crate::expand_template(
            template,
            &[
                ("textureSet", self.texture_set),
                ("mesh", self.mesh),
                ("layerName", self.layer_name),
                ("udim", self.udim),
            ],
        )
    }

    /// Expands every §6 token in one output's filename: the static
    /// sources plus that output's `$srcMap` ([`src_map_token`]) and
    /// `$colorSpace` ([`output_color_space`]).
    pub fn expand_output(&self, output: &crate::presets::OutputSpec) -> String {
        let src_map = src_map_token(output);
        let color_space = output_color_space(output);
        crate::expand_template(
            &output.filename,
            &[
                ("textureSet", self.texture_set),
                ("mesh", self.mesh),
                ("layerName", self.layer_name),
                ("udim", self.udim),
                ("srcMap", &src_map),
                ("colorSpace", color_space),
            ],
        )
    }
}

/// The `$mesh` source: a mesh path's file stem (`sword.obj` → `sword`);
/// `""` when there is no stem — the placeholder then stays literal.
pub fn mesh_stem(path: &Path) -> &str {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("")
}

/// The `$srcMap` value for one output: the single input map's naming
/// token, or — for packed multi-map outputs (ORM etc.) — the input
/// tokens joined with `_` in wiring order. No inputs → `""` (the
/// placeholder stays literal).
pub fn src_map_token(output: &crate::presets::OutputSpec) -> String {
    output
        .maps
        .iter()
        .map(|(kind, _)| kind.token())
        .collect::<Vec<_>>()
        .join("_")
}

/// The `$colorSpace` value for one output: `sRGB` when any input is a
/// color-managed map (baseColor/emissive — the §9 rule [`run_preset`]
/// already applies to the PNG transfer), `linear` for pure data
/// outputs. Same predicate as the transfer choice, so the token never
/// disagrees with the file's actual encoding.
pub fn output_color_space(output: &crate::presets::OutputSpec) -> &'static str {
    if output.maps.iter().any(|(kind, _)| map_kind_is_color(*kind)) {
        "sRGB"
    } else {
        "linear"
    }
}

/// JPEG quality used when a preset output picks JPEG (§6 previews).
pub const DEFAULT_JPEG_QUALITY: u8 = 90;

/// The working-space normal convention: maps are authored OpenGL
/// (+Y up, the glTF/Blender/Unity convention); presets asking for
/// [`NormalConvention::Directx`] get the green flip at export.
pub const WORKING_NORMAL_CONVENTION: NormalConvention = NormalConvention::Opengl;

/// Runs `preset` against `set`, writing outputs under `out_dir`.
/// Returns the written paths in output order.
///
/// Filenames expand every §6 token from `sources` (per-output `$srcMap`
/// / `$colorSpace` derived by the driver); unknown or empty-valued
/// tokens pass through verbatim per the template engine's contract.
///
/// # Errors
///
/// [`ExportError::MissingMap`] when an output needs a map the set
/// lacks; [`ExportError::SizeMismatch`] when a map's buffer is the
/// wrong length; [`ExportError::Write`] from the format writers.
pub fn run_preset(
    preset: &ExportPreset,
    set: &MapSet,
    sources: &TokenSources<'_>,
    out_dir: &Path,
) -> Result<Vec<PathBuf>, ExportError> {
    std::fs::create_dir_all(out_dir).map_err(|e| ExportError::Write(e.to_string()))?;
    let size = set.size;
    let texels = size as usize * size as usize;

    // Pass 1 — validate EVERY output's maps (presence + size) before
    // a single byte is written: a preset either exports whole or
    // leaves the output dir untouched (no partial exports).
    for output in &preset.outputs {
        for (kind, _) in &output.maps {
            let bytes = set.get(*kind).ok_or(ExportError::MissingMap {
                output: output.filename.clone(),
                map: *kind,
            })?;
            if bytes.len() != texels * 4 {
                return Err(ExportError::SizeMismatch {
                    map: *kind,
                    actual: bytes.len() / 4,
                    expected: texels,
                    size,
                });
            }
        }
    }

    let mut written = Vec::new();
    for output in &preset.outputs {
        // Validation happened in pass 1 — this loop only packs + writes.
        let map_slices: Vec<&[u8]> = output
            .maps
            .iter()
            .map(|(kind, _)| set.get(*kind).expect("validated in pass 1"))
            .collect();

        // Pack the whole map: for each texel index, build the input
        // texel list (one per map), pack, and — for normal maps —
        // apply the convention conversion.
        let mut packed = vec![0u8; texels * 4];
        let is_normal = output.maps.iter().any(|(kind, _)| *kind == MapKind::Normal);
        let needs_flip = is_normal && output.normal_convention != WORKING_NORMAL_CONVENTION;

        for t in 0..texels {
            let texel_inputs: Vec<Texel> = map_slices
                .iter()
                .map(|bytes| {
                    let i = t * 4;
                    Texel::from([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
                })
                .collect();
            let mut out = pack_texel(output, &texel_inputs);
            if needs_flip {
                let texel = Texel::from(out);
                out =
                    convert_normal(texel, WORKING_NORMAL_CONVENTION, output.normal_convention).rgba;
            }
            packed[t * 4..t * 4 + 4].copy_from_slice(&out);
        }

        // Expand the filename's §6 tokens from the driver-side
        // sources (per-output $srcMap/$colorSpace derived here; empty
        // sources pass through verbatim per the template engine's
        // contract).
        let filename = sources.expand_output(output);
        let path = out_dir.join(filename);

        // Per-output transfer: color-managed maps (baseColor/emissive)
        // take the sRGB curve; data maps pass through linear — the §9
        // per-channel color-management rule applied at the output
        // level (a packed output with any color input is sRGB; pure
        // data outputs — rough/metal/normal/AO — stay linear).
        let transfer = if output.maps.iter().any(|(kind, _)| map_kind_is_color(*kind)) {
            Transfer::Srgb
        } else {
            Transfer::Linear
        };

        match output.format {
            OutputFormat::Png8 => crate::png::write_png(&path, size, size, &packed, transfer)?,
            OutputFormat::Png16 => {
                // Up-shift the u8 sources to u16 (×257: 0->0, 255->65535).
                let mut u16s = Vec::with_capacity(texels * 4);
                for b in &packed {
                    u16s.push(u16::from(*b) * 257);
                }
                crate::png::write_png16(&path, size, size, &u16s, transfer)?;
            }
            OutputFormat::Exr32F => {
                // Widen u8 sources to f32 in [0,1] — the driver's
                // contract (EXR output from 8-bit sources is
                // lossless-but-quantized; true float sources arrive
                // when the paint pipeline grows f32 readbacks).
                let mut f32s = Vec::with_capacity(texels * 4);
                for b in &packed {
                    f32s.push(f32::from(*b) / 255.0);
                }
                exr::write_exr_f32(&path, size, size, &f32s)?;
            }
            OutputFormat::Tiff => {
                formats::write_tiff_rgba8(&path, size, size, &packed, Transfer::Srgb)?
            }
            OutputFormat::Jpeg => {
                formats::write_jpeg_rgba8(&path, size, size, &packed, DEFAULT_JPEG_QUALITY)?
            }
        }
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::ChannelSlot;

    fn solid(size: u32, value: u8) -> Vec<u8> {
        vec![value; (size * size * 4) as usize]
    }

    #[test]
    fn gltf_preset_writes_three_named_outputs() {
        let size = 4;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 200));
        set.set(MapKind::Metallic, solid(size, 255));
        set.set(MapKind::Roughness, solid(size, 64));
        set.set(MapKind::Normal, solid(size, 128));

        let out = std::env::temp_dir().join(format!("umber-export-drv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(
            &ExportPreset::gltf_metal_rough(),
            &set,
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        assert_eq!(written.len(), 3);
        assert!(written[0].ends_with("Sword_baseColor.png"));
        assert!(written[1].ends_with("Sword_metallicRoughness.png"));
        assert!(written[2].ends_with("Sword_normal.png"));
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn directx_preset_flips_normals_only() {
        // Unreal preset: normal green channel flipped vs the source.
        let size = 2;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 100));
        set.set(MapKind::AmbientOcclusion, solid(size, 230));
        set.set(MapKind::Roughness, solid(size, 128));
        set.set(MapKind::Metallic, solid(size, 64));
        // OpenGL normal: g = 200 -> DirectX: 255-200 = 55.
        let mut normal = solid(size, 0);
        for px in normal.chunks_exact_mut(4) {
            px[1] = 200;
        }
        set.set(MapKind::Normal, normal);

        let out = std::env::temp_dir().join(format!("umber-export-drv-dx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(
            &ExportPreset::unreal_orm(),
            &set,
            &TokenSources::new("Hull"),
            &out,
        )
        .unwrap();
        // The normal output is third; decode and check green = 55.
        let normal_path = written
            .iter()
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("_normal.png"))
            })
            .unwrap();
        let decoder = ::png::Decoder::new(std::fs::File::open(normal_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut bytes = vec![0u8; reader.output_buffer_size()];
        let _ = reader.next_frame(&mut bytes).unwrap();
        assert_eq!(bytes[1], 55, "green flipped 200 -> 55");
        // The ORM output must NOT be flipped (non-normal outputs skip
        // the conversion).
        let orm_path = written
            .iter()
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("ORM.png"))
            })
            .unwrap();
        let decoder = ::png::Decoder::new(std::fs::File::open(orm_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut orm_bytes = vec![0u8; reader.output_buffer_size()];
        let _ = reader.next_frame(&mut orm_bytes).unwrap();
        assert_eq!(orm_bytes[0], 230, "R = AO gray unflipped");
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn missing_map_fails_the_output_cleanly() {
        let size = 2;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 1)); // metallic/roughness missing
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-miss-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let err = run_preset(
            &ExportPreset::gltf_metal_rough(),
            &set,
            &TokenSources::new("X"),
            &out,
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::MissingMap { .. }));
        // No OUTPUT FILES may exist (the directory itself is created
        // up front — checking files, not the dir).
        let files = std::fs::read_dir(&out).map(|d| d.count()).unwrap_or(0);
        assert_eq!(files, 0, "nothing written on failure");
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn size_mismatch_is_caught_before_writing() {
        let size = 2;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, vec![0u8; 15]); // wrong: 2*2*4 = 16
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-size-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let err = run_preset(
            &ExportPreset::gltf_metal_rough(),
            &set,
            &TokenSources::new("X"),
            &out,
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::SizeMismatch { .. }));
    }

    #[test]
    fn channel_wiring_reaches_the_driver() {
        // A preset whose output packs two maps' grays into R/G —
        // the driver must honor per-channel map indices end to end.
        let size = 1;
        let mut set = MapSet::new(size);
        set.set(MapKind::Roughness, vec![10, 10, 10, 255]);
        set.set(MapKind::Metallic, vec![20, 20, 20, 255]);
        let preset = ExportPreset {
            name: "wiring probe".into(),
            outputs: vec![crate::presets::OutputSpec {
                filename: "$textureSet_wiring.png".into(),
                maps: vec![(MapKind::Roughness, vec![]), (MapKind::Metallic, vec![])],
                channels: [
                    crate::presets::ChannelWiring::new(0, ChannelSlot::Gray),
                    crate::presets::ChannelWiring::new(1, ChannelSlot::Gray),
                    crate::presets::ChannelWiring::new(1, ChannelSlot::Gray),
                    crate::presets::ChannelWiring::new(0, ChannelSlot::A),
                ],
                normal_convention: NormalConvention::Opengl,
                format: OutputFormat::Png8,
            }],
        };
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-wire-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        run_preset(&preset, &set, &TokenSources::new("W"), &out).unwrap();
        let path = out.join("W_wiring.png");
        let decoder = ::png::Decoder::new(std::fs::File::open(&path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut bytes = vec![0u8; reader.output_buffer_size()];
        let _ = reader.next_frame(&mut bytes).unwrap();
        assert_eq!(bytes[0], 10, "R = roughness gray");
        assert_eq!(bytes[1], 20, "G = metallic gray");
        assert_eq!(bytes[2], 20, "B = metallic gray");
        assert_eq!(bytes[3], 255, "A = roughness alpha");
        std::fs::remove_dir_all(&out).unwrap();
    }

    fn passthrough_output(filename: &str, kind: MapKind) -> crate::presets::OutputSpec {
        crate::presets::OutputSpec {
            filename: filename.into(),
            maps: vec![(kind, vec![])],
            channels: [
                crate::presets::ChannelWiring::new(0, ChannelSlot::R),
                crate::presets::ChannelWiring::new(0, ChannelSlot::G),
                crate::presets::ChannelWiring::new(0, ChannelSlot::B),
                crate::presets::ChannelWiring::new(0, ChannelSlot::A),
            ],
            normal_convention: NormalConvention::Opengl,
            format: OutputFormat::Png8,
        }
    }

    #[test]
    fn every_token_expands_to_its_source_exactly() {
        // Populated case for ALL six tokens: one BaseColor passthrough
        // whose filename names each token once.
        let size = 2;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 200));
        let preset = ExportPreset {
            name: "token probe".into(),
            outputs: vec![passthrough_output(
                "$mesh_$textureSet_$srcMap_$colorSpace_$layerName_$udim.png",
                MapKind::BaseColor,
            )],
        };
        let sources = TokenSources {
            texture_set: "Blade",
            mesh: "Sword",
            layer_name: "Paint 1",
            udim: SINGLE_TILE_UDIM,
        };
        let out = std::env::temp_dir().join(format!("umber-export-drv-tok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(&preset, &set, &sources, &out).unwrap();
        assert_eq!(written.len(), 1);
        assert!(
            written[0].ends_with("Sword_Blade_baseColor_sRGB_Paint 1_1001.png"),
            "every token expanded, got {}",
            written[0].display()
        );
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn empty_sources_leave_their_placeholders_literal() {
        // Empty-source case for $mesh/$layerName (the engine's
        // pass-through contract); driver-known tokens ($srcMap,
        // $colorSpace, $udim, $textureSet) still expand.
        let size = 2;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 200));
        let preset = ExportPreset {
            name: "token probe".into(),
            outputs: vec![passthrough_output(
                "$mesh_$textureSet_$srcMap_$colorSpace_$layerName_$udim.png",
                MapKind::BaseColor,
            )],
        };
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-tokempty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(&preset, &set, &TokenSources::new("Blade"), &out).unwrap();
        assert_eq!(written.len(), 1);
        assert!(
            written[0].ends_with("$mesh_Blade_baseColor_sRGB_$layerName_1001.png"),
            "empty sources stay literal, got {}",
            written[0].display()
        );
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn packed_output_joins_src_map_tokens_and_reports_linear() {
        // Multi-map $srcMap joins wiring-order tokens; a pure-data
        // output's $colorSpace is linear.
        let size = 1;
        let mut set = MapSet::new(size);
        set.set(MapKind::AmbientOcclusion, vec![230, 230, 230, 255]);
        set.set(MapKind::Roughness, vec![128, 128, 128, 255]);
        set.set(MapKind::Metallic, vec![64, 64, 64, 255]);
        let preset = ExportPreset {
            name: "orm probe".into(),
            outputs: vec![crate::presets::OutputSpec {
                filename: "$textureSet_$srcMap_$colorSpace.png".into(),
                maps: vec![
                    (MapKind::AmbientOcclusion, vec![]),
                    (MapKind::Roughness, vec![]),
                    (MapKind::Metallic, vec![]),
                ],
                channels: [
                    crate::presets::ChannelWiring::new(0, ChannelSlot::Gray),
                    crate::presets::ChannelWiring::new(1, ChannelSlot::Gray),
                    crate::presets::ChannelWiring::new(2, ChannelSlot::Gray),
                    crate::presets::ChannelWiring::new(0, ChannelSlot::A),
                ],
                normal_convention: NormalConvention::Opengl,
                format: OutputFormat::Png8,
            }],
        };
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-tokorm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(&preset, &set, &TokenSources::new("Hull"), &out).unwrap();
        assert_eq!(written.len(), 1);
        assert!(
            written[0].ends_with("Hull_ambient_occlusion_roughness_metallic_linear.png"),
            "got {}",
            written[0].display()
        );
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn populated_sources_dont_rename_texture_set_only_templates() {
        // CLI regression: today's built-in filenames name only
        // $textureSet, so a fully-populated source set must write
        // byte-identical names to the minimal one.
        let size = 2;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 200));
        set.set(MapKind::Metallic, solid(size, 255));
        set.set(MapKind::Roughness, solid(size, 64));
        set.set(MapKind::Normal, solid(size, 128));
        let preset = ExportPreset::gltf_metal_rough();
        let full = TokenSources {
            texture_set: "Sword",
            mesh: "sword",
            layer_name: "Paint 1",
            udim: SINGLE_TILE_UDIM,
        };
        let minimal = TokenSources::new("Sword");
        for (tag, sources) in [("full", full), ("minimal", minimal)] {
            let out = std::env::temp_dir().join(format!(
                "umber-export-drv-tokcompat-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&out);
            let written = run_preset(&preset, &set, &sources, &out).unwrap();
            let names: Vec<String> = written
                .iter()
                .map(|p| {
                    p.file_name()
                        .expect("written file has a name")
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            assert_eq!(
                names,
                vec![
                    "Sword_baseColor.png",
                    "Sword_metallicRoughness.png",
                    "Sword_normal.png",
                ],
                "{tag} sources renamed a textureSet-only template"
            );
            std::fs::remove_dir_all(&out).unwrap();
        }
    }

    #[test]
    fn mesh_stem_takes_the_file_stem() {
        assert_eq!(mesh_stem(std::path::Path::new("/m/sword.obj")), "sword");
        assert_eq!(mesh_stem(std::path::Path::new("blade.glb")), "blade");
        assert_eq!(
            mesh_stem(std::path::Path::new("/m/my.set.v2.fbx")),
            "my.set.v2"
        );
    }

    #[test]
    fn expand_static_leaves_per_output_tokens_for_expand_output() {
        let sources = TokenSources {
            texture_set: "Blade",
            mesh: "Sword",
            layer_name: "Paint 1",
            udim: SINGLE_TILE_UDIM,
        };
        assert_eq!(
            sources.expand_static("$mesh/$textureSet_$layerName_$udim_$srcMap"),
            "Sword/Blade_Paint 1_1001_$srcMap"
        );
    }
}

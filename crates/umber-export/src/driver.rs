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
use umber_graph::mtlx_doc::{
    to_mtlx_document, MtlxColorSpace, MtlxTexture, SurfaceParams, TextureMap, UDIM_TOKEN,
};

use crate::presets::{
    convert_normal, pack_texel, ChannelSlot, ChannelWiring, ExportPreset, MapKind,
    NormalConvention, OutputFormat, OutputSpec, Texel,
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
    /// A tiled run listed the same UDIM tile twice (its outputs would
    /// overwrite each other).
    #[error("tile {0} appears more than once in the tiled export")]
    DuplicateTile(u16),
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

    /// The set's square resolution.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Every held map with its bytes, in insertion order — the
    /// document-format writers (PSD) need the full set.
    pub fn maps_with_bytes(&self) -> impl Iterator<Item = (MapKind, &[u8])> + '_ {
        self.maps.iter().map(|(k, b)| (*k, b.as_slice()))
    }
}

/// The single-tile `$udim` value: tile 1001, the default every
/// [`TokenSources::new`] / [`run_preset`] run carries. Retired as THE
/// `$udim` source by wave-5's UDIM slice 5 — multi-tile runs go through
/// [`run_preset_tiled`], which expands `$udim` to each output tile's own
/// number — and kept as the single-tile default (byte-identical names).
pub const SINGLE_TILE_UDIM: &str = "1001";

/// The `$udim` placeholder as it appears in filename templates.
const UDIM_PLACEHOLDER: &str = "$udim";

/// The filename template one tile's output is expanded from in a tiled
/// run (several tiles, or a lone tile other than 1001): the template verbatim when it already names `$udim`,
/// else `_$udim` inserted before the extension (`$textureSet_baseColor.png`
/// → `$textureSet_baseColor_$udim.png`; no extension → appended). Without
/// this every built-in preset (none name `$udim`) would write each tile
/// to the same path, the last tile silently overwriting the rest.
///
/// Single-tile (lone 1001) runs never call this: their templates expand
/// unchanged.
pub fn tiled_template(template: &str) -> String {
    if template.contains(UDIM_PLACEHOLDER) {
        return template.to_string();
    }
    // The extension dot must live in the final path component ('/' makes
    // subfolders per `template_to_path`), and a leading dot is a hidden
    // file name, not an extension.
    let name_start = template.rfind('/').map_or(0, |i| i + 1);
    match template[name_start..].rfind('.') {
        Some(dot) if dot > 0 => {
            let at = name_start + dot;
            format!("{}_{UDIM_PLACEHOLDER}{}", &template[..at], &template[at..])
        }
        _ => format!("{template}_{UDIM_PLACEHOLDER}"),
    }
}

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
    /// `$udim` — the tile number of the output being written.
    /// [`run_preset`] uses this value as-is (single-tile runs:
    /// [`SINGLE_TILE_UDIM`]); [`run_preset_tiled`] overrides it per
    /// output tile with that tile's number.
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
        self.expand_output_from(output, &output.filename)
    }

    /// [`Self::expand_output`] over an explicit `template` (the tiled
    /// driver passes [`tiled_template`]'s form of `output.filename`).
    fn expand_output_from(&self, output: &crate::presets::OutputSpec, template: &str) -> String {
        let src_map = src_map_token(output);
        let color_space = output_color_space(output);
        crate::expand_template(
            template,
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

/// The transfer one output's bytes are ACTUALLY written with — the
/// single source for both the writers and the `.mtlx` image colorspace.
/// PNG follows the §9 per-channel rule at the output level (any
/// color-managed input — baseColor/emissive — takes the sRGB curve; pure
/// data outputs — rough/metal/normal/AO — stay linear); TIFF is always
/// sRGB-curved; EXR and JPEG write the source bytes as-is.
fn output_transfer(output: &OutputSpec) -> Transfer {
    match output.format {
        OutputFormat::Png8 | OutputFormat::Png16 => {
            if output.maps.iter().any(|(kind, _)| map_kind_is_color(*kind)) {
                Transfer::Srgb
            } else {
                Transfer::Linear
            }
        }
        OutputFormat::Tiff => Transfer::Srgb,
        OutputFormat::Exr32F | OutputFormat::Jpeg => Transfer::Linear,
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
/// With [`ExportPreset::materialx`] on, `<textureSet>.mtlx` is written
/// last (and listed last) — see [`run_preset_with_surface`]; this entry
/// point carries the OpenPBR nodedef defaults for the untextured inputs
/// (this crate cannot see the app's material).
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
    run_preset_with_surface(preset, set, sources, out_dir, &SurfaceParams::default())
}

/// [`run_preset`] with the material's untextured OpenPBR values for the
/// `.mtlx` (the app passes its viewport material via
/// [`SurfaceParams::from_gpu_slots`]). Without
/// [`ExportPreset::materialx`] identical to [`run_preset`].
///
/// # Errors
///
/// As [`run_preset`].
pub fn run_preset_with_surface(
    preset: &ExportPreset,
    set: &MapSet,
    sources: &TokenSources<'_>,
    out_dir: &Path,
    surface: &SurfaceParams,
) -> Result<Vec<PathBuf>, ExportError> {
    std::fs::create_dir_all(out_dir).map_err(|e| ExportError::Write(e.to_string()))?;
    validate_set(preset, set)?;
    let mut written = Vec::new();
    write_outputs(preset, set, sources, out_dir, false, &mut written)?;
    if preset.materialx {
        // The same expansion write_outputs used: the document names the
        // exact files just written.
        let doc = materialx_document(preset, sources.texture_set, surface, |output| {
            sources.expand_output(output)
        });
        written.push(write_materialx(out_dir, sources.texture_set, &doc)?);
    }
    Ok(written)
}

/// Runs `preset` once per UDIM tile — the multi-tile export (wave-5
/// UDIM slice 5). `tiles` pairs each tile number with ITS map set (each
/// tile's own bytes and resolution); outputs are written tile by tile in
/// list order, each tile in preset output order.
///
/// `$udim` expands to the CURRENT tile's number for every output
/// (`sources.udim` is overridden per tile). With more than one tile — or
/// a lone tile other than 1001 — templates that don't name `$udim` get
/// `_$udim` before the extension ([`tiled_template`]) so tiles never
/// overwrite each other and a tile's files always say which tile they
/// are; the lone-1001 run expands templates unchanged, so
/// `[(1001, set)]` writes exactly the files [`run_preset`] writes
/// (pinned by test). An empty `tiles` writes nothing.
///
/// With [`ExportPreset::materialx`] on, ONE `<textureSet>.mtlx` is
/// written after every tile: tile-named runs reference the files through
/// MaterialX's `<UDIM>` token (the renderer resolves it per UV tile); the
/// lone-1001 run references the concrete names. Untextured inputs carry
/// the nodedef defaults (see [`run_preset_tiled_with_surface`]).
///
/// # Errors
///
/// As [`run_preset`] — validated across EVERY tile before any file is
/// written (a tiled export is whole or nothing) — plus
/// [`ExportError::DuplicateTile`] when a tile is listed twice.
pub fn run_preset_tiled(
    preset: &ExportPreset,
    tiles: &[(u16, &MapSet)],
    sources: &TokenSources<'_>,
    out_dir: &Path,
) -> Result<Vec<PathBuf>, ExportError> {
    run_preset_tiled_with_surface(preset, tiles, sources, out_dir, &SurfaceParams::default())
}

/// [`run_preset_tiled`] with the material's untextured OpenPBR values
/// for the `.mtlx` (as [`run_preset_with_surface`]).
///
/// # Errors
///
/// As [`run_preset_tiled`].
pub fn run_preset_tiled_with_surface(
    preset: &ExportPreset,
    tiles: &[(u16, &MapSet)],
    sources: &TokenSources<'_>,
    out_dir: &Path,
    surface: &SurfaceParams,
) -> Result<Vec<PathBuf>, ExportError> {
    for (i, (tile, _)) in tiles.iter().enumerate() {
        if tiles[..i].iter().any(|(seen, _)| seen == tile) {
            return Err(ExportError::DuplicateTile(*tile));
        }
    }
    std::fs::create_dir_all(out_dir).map_err(|e| ExportError::Write(e.to_string()))?;
    for (_, set) in tiles {
        validate_set(preset, set)?;
    }
    let tile_names = tiles.len() > 1 || tiles.iter().any(|(tile, _)| *tile != FIRST_TILE);
    let mut written = Vec::new();
    for (tile, set) in tiles {
        let udim = tile.to_string();
        let tile_sources = TokenSources {
            udim: &udim,
            ..*sources
        };
        write_outputs(
            preset,
            set,
            &tile_sources,
            out_dir,
            tile_names,
            &mut written,
        )?;
    }
    if preset.materialx && !tiles.is_empty() {
        let doc = if tile_names {
            let udim_sources = TokenSources {
                udim: UDIM_TOKEN,
                ..*sources
            };
            materialx_document(preset, sources.texture_set, surface, |output| {
                udim_sources.expand_output_from(output, &tiled_template(&output.filename))
            })
        } else {
            // The lone 1001 tile: its files carry concrete names.
            let udim = tiles[0].0.to_string();
            let tile_sources = TokenSources {
                udim: &udim,
                ..*sources
            };
            materialx_document(preset, sources.texture_set, surface, |output| {
                tile_sources.expand_output(output)
            })
        };
        written.push(write_materialx(out_dir, sources.texture_set, &doc)?);
    }
    Ok(written)
}

/// The `.mtlx` side of a run: one [`MtlxTexture`] per OpenPBR-wirable
/// map of every output, `file_name` resolving each output's file name
/// (relative to `out_dir`, where the document is written).
///
/// Per output map: a passthrough output (R/G/B straight from that map)
/// references the whole image; a packed output (ORM, metallicRoughness)
/// references one channel through an `<extract>` — scalar maps only.
/// The colorspace is the output's real transfer ([`output_transfer`]).
/// A DirectX-convention normal is NOT wired: MaterialX's `normalmap`
/// expects +Y (OpenGL) tangent space and has no green-flip input.
fn materialx_document(
    preset: &ExportPreset,
    texture_set: &str,
    surface: &SurfaceParams,
    file_name: impl Fn(&OutputSpec) -> String,
) -> String {
    let mut textures = Vec::new();
    for output in &preset.outputs {
        let file = file_name(output);
        let colorspace = match output_transfer(output) {
            Transfer::Srgb => MtlxColorSpace::SrgbTexture,
            Transfer::Linear => MtlxColorSpace::LinRec709,
        };
        for (index, (kind, _)) in output.maps.iter().enumerate() {
            let map = texture_map(*kind);
            let Some((_, ty)) = map.openpbr_input() else {
                continue; // AO / height: no OpenPBR input.
            };
            if map == TextureMap::Normal && output.normal_convention != NormalConvention::Opengl {
                continue;
            }
            let channel = if is_passthrough(output, index) {
                None
            } else if ty == "float" {
                match packed_channel(output, index) {
                    Some(c) => Some(c),
                    None => continue, // wired in no channel.
                }
            } else {
                continue; // a color/vector map can't be one channel.
            };
            textures.push(MtlxTexture {
                map,
                file: file.clone(),
                colorspace,
                channel,
            });
        }
    }
    to_mtlx_document(texture_set, surface, &textures)
}

/// Writes `<textureSet>.mtlx` under `out_dir`, returning its path.
fn write_materialx(out_dir: &Path, texture_set: &str, doc: &str) -> Result<PathBuf, ExportError> {
    let path = out_dir.join(format!("{texture_set}.mtlx"));
    std::fs::write(&path, doc).map_err(|e| ExportError::Write(e.to_string()))?;
    Ok(path)
}

/// The graph-side mirror of a [`MapKind`] (exhaustive: a new kind fails
/// to compile here until it is mapped).
fn texture_map(kind: MapKind) -> TextureMap {
    match kind {
        MapKind::BaseColor => TextureMap::BaseColor,
        MapKind::Roughness => TextureMap::Roughness,
        MapKind::Metallic => TextureMap::Metallic,
        MapKind::AmbientOcclusion => TextureMap::AmbientOcclusion,
        MapKind::Normal => TextureMap::Normal,
        MapKind::Height => TextureMap::Height,
        MapKind::Opacity => TextureMap::Opacity,
        MapKind::Emissive => TextureMap::Emissive,
    }
}

/// Whether output R/G/B are map `index`'s own R/G/B (the file IS the map).
fn is_passthrough(output: &OutputSpec, index: usize) -> bool {
    output.channels[0] == ChannelWiring::new(index, ChannelSlot::R)
        && output.channels[1] == ChannelWiring::new(index, ChannelSlot::G)
        && output.channels[2] == ChannelWiring::new(index, ChannelSlot::B)
}

/// The output channel carrying map `index` in a packed output: its Gray
/// slot first (the packing presets' scalar wire — glTF roughness = B, not
/// the R it also feeds), then any of R/G/B, then A.
fn packed_channel(output: &OutputSpec, index: usize) -> Option<u8> {
    let ch = &output.channels;
    (0..4)
        .find(|&i| ch[i].map == index && ch[i].slot == ChannelSlot::Gray)
        .or_else(|| (0..3).find(|&i| ch[i].map == index))
        .or_else(|| (ch[3].map == index).then_some(3))
        .map(|i| i as u8)
}

/// Tile 1001 as a number — the single tile whose lone export keeps
/// pre-UDIM names (mirrors `umber_mesh::udim::FIRST_TILE`; this crate
/// stays mesh-free).
const FIRST_TILE: u16 = 1001;

/// Pass 1 — validates EVERY output's maps (presence + size) before a
/// single byte is written: a preset either exports whole or leaves the
/// output dir untouched (no partial exports).
fn validate_set(preset: &ExportPreset, set: &MapSet) -> Result<(), ExportError> {
    let size = set.size;
    let texels = size as usize * size as usize;
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
    Ok(())
}

/// Pass 2 — packs and writes every output of `preset` from `set`
/// (already validated by [`validate_set`]), appending the paths to
/// `written`. `tile_names` routes each filename through
/// [`tiled_template`] before token expansion.
fn write_outputs(
    preset: &ExportPreset,
    set: &MapSet,
    sources: &TokenSources<'_>,
    out_dir: &Path,
    tile_names: bool,
    written: &mut Vec<PathBuf>,
) -> Result<(), ExportError> {
    let size = set.size;
    let texels = size as usize * size as usize;
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

        // One input list reused across texels: a per-texel Vec was 67M
        // heap allocations per output at 8K.
        let mut texel_inputs: Vec<Texel> = Vec::with_capacity(map_slices.len());
        for t in 0..texels {
            texel_inputs.clear();
            texel_inputs.extend(map_slices.iter().map(|bytes| {
                let i = t * 4;
                Texel::from([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
            }));
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
        // contract). Tiled runs first make the template tile-unique.
        let filename = if tile_names {
            sources.expand_output_from(output, &tiled_template(&output.filename))
        } else {
            sources.expand_output(output)
        };
        let path = out_dir.join(filename);

        let transfer = output_transfer(output);

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
            OutputFormat::Tiff => formats::write_tiff_rgba8(&path, size, size, &packed, transfer)?,
            OutputFormat::Jpeg => {
                formats::write_jpeg_rgba8(&path, size, size, &packed, DEFAULT_JPEG_QUALITY)?
            }
        }
        written.push(path);
    }
    Ok(())
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
            materialx: false,
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
            materialx: false,
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
            materialx: false,
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
            materialx: false,
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

    // --- Per-tile export (UDIM slice 5) ---

    fn file_names(written: &[PathBuf]) -> Vec<String> {
        written
            .iter()
            .map(|p| {
                p.file_name()
                    .expect("written file has a name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    fn gltf_ready_set(size: u32, base: u8) -> MapSet {
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, base));
        set.set(MapKind::Metallic, solid(size, 255));
        set.set(MapKind::Roughness, solid(size, 64));
        set.set(MapKind::Normal, solid(size, 128));
        set
    }

    #[test]
    fn udim_token_expands_each_tiles_number() {
        // Test core 5: a two-tile run over a `$udim` template writes one
        // file per tile, each named with ITS tile's number (exact names).
        let size = 2;
        let mut set_a = MapSet::new(size);
        set_a.set(MapKind::BaseColor, solid(size, 10));
        let mut set_b = MapSet::new(size);
        set_b.set(MapKind::BaseColor, solid(size, 20));
        let preset = ExportPreset {
            name: "udim probe".into(),
            materialx: false,
            outputs: vec![passthrough_output(
                "$textureSet_$srcMap_$udim.png",
                MapKind::BaseColor,
            )],
        };
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-udim2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset_tiled(
            &preset,
            &[(1001, &set_a), (1002, &set_b)],
            &TokenSources::new("Blade"),
            &out,
        )
        .unwrap();
        assert_eq!(
            file_names(&written),
            vec!["Blade_baseColor_1001.png", "Blade_baseColor_1002.png"]
        );
        // Each file carries ITS tile's bytes (10 vs 20 linear → distinct
        // after the sRGB curve): no tile wrote the other's map.
        let (_, _, a) = crate::png::read_png_rgba8(&written[0]).unwrap();
        let (_, _, b) = crate::png::read_png_rgba8(&written[1]).unwrap();
        assert_ne!(a[0], b[0], "per-tile bytes must differ");
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn tiled_run_without_udim_token_never_overwrites() {
        // Every built-in preset names only $textureSet: a two-tile run
        // must still write tile-unique paths (`_$udim` before the
        // extension), never one tile over the other.
        let size = 2;
        let set_a = gltf_ready_set(size, 10);
        let set_b = gltf_ready_set(size, 20);
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-udimgltf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset_tiled(
            &ExportPreset::gltf_metal_rough(),
            &[(1001, &set_a), (1002, &set_b)],
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        assert_eq!(
            file_names(&written),
            vec![
                "Sword_baseColor_1001.png",
                "Sword_metallicRoughness_1001.png",
                "Sword_normal_1001.png",
                "Sword_baseColor_1002.png",
                "Sword_metallicRoughness_1002.png",
                "Sword_normal_1002.png",
            ]
        );
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), 6);
        std::fs::remove_dir_all(&out).unwrap();

        // A lone non-1001 tile (the CLI's `--tile 1002`) still names its
        // tile; only the lone-1001 run keeps pre-UDIM names.
        let written = run_preset_tiled(
            &ExportPreset::gltf_metal_rough(),
            &[(1002, &set_b)],
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        assert_eq!(
            file_names(&written),
            vec![
                "Sword_baseColor_1002.png",
                "Sword_metallicRoughness_1002.png",
                "Sword_normal_1002.png",
            ]
        );
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn single_tile_tiled_run_matches_run_preset_byte_for_byte() {
        // Single-tile regression: tiles = [1001] writes the same names
        // (the token suite's fixture names) AND the same bytes as today's
        // run_preset — for a $textureSet-only preset and a $udim one.
        let size = 2;
        let set = gltf_ready_set(size, 200);
        let udim_preset = ExportPreset {
            name: "token probe".into(),
            materialx: false,
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
        for (tag, preset, expected) in [
            (
                "gltf",
                ExportPreset::gltf_metal_rough(),
                vec![
                    "Blade_baseColor.png",
                    "Blade_metallicRoughness.png",
                    "Blade_normal.png",
                ],
            ),
            (
                "udim",
                udim_preset,
                vec!["Sword_Blade_baseColor_sRGB_Paint 1_1001.png"],
            ),
        ] {
            let base = std::env::temp_dir().join(format!(
                "umber-export-drv-single-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&base);
            let old = run_preset(&preset, &set, &sources, &base.join("old")).unwrap();
            let new =
                run_preset_tiled(&preset, &[(1001, &set)], &sources, &base.join("new")).unwrap();
            assert_eq!(file_names(&old), expected, "{tag}: run_preset names");
            assert_eq!(file_names(&new), expected, "{tag}: tiled names");
            for (o, n) in old.iter().zip(&new) {
                assert_eq!(
                    std::fs::read(o).unwrap(),
                    std::fs::read(n).unwrap(),
                    "{tag}: {} differs byte-wise",
                    o.display()
                );
            }
            std::fs::remove_dir_all(&base).unwrap();
        }
    }

    #[test]
    fn tiled_run_validates_every_tile_before_writing() {
        // Tile 1002's set lacks Metallic: the run fails MissingMap and
        // tile 1001 (valid, listed first) wrote nothing either.
        let size = 2;
        let good = gltf_ready_set(size, 10);
        let mut bad = MapSet::new(size);
        bad.set(MapKind::BaseColor, solid(size, 20));
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-udimval-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let err = run_preset_tiled(
            &ExportPreset::gltf_metal_rough(),
            &[(1001, &good), (1002, &bad)],
            &TokenSources::new("X"),
            &out,
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::MissingMap { .. }));
        let files = std::fs::read_dir(&out).map(|d| d.count()).unwrap_or(0);
        assert_eq!(files, 0, "no tile may write when any tile fails");
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn tiled_run_rejects_a_duplicate_tile() {
        let set = gltf_ready_set(1, 10);
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-udimdup-{}", std::process::id()));
        let err = run_preset_tiled(
            &ExportPreset::gltf_metal_rough(),
            &[(1002, &set), (1002, &set)],
            &TokenSources::new("X"),
            &out,
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::DuplicateTile(1002)));
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn tiled_template_inserts_udim_before_the_extension() {
        assert_eq!(
            tiled_template("$textureSet_baseColor.png"),
            "$textureSet_baseColor_$udim.png"
        );
        // Already tile-aware: untouched.
        assert_eq!(
            tiled_template("$textureSet.$udim.png"),
            "$textureSet.$udim.png"
        );
        // The extension is the LAST dot of the final path component.
        assert_eq!(
            tiled_template("out.v2/$textureSet_n.tar.gz"),
            "out.v2/$textureSet_n.tar_$udim.gz"
        );
        // No extension (or a dotted folder only): appended.
        assert_eq!(tiled_template("v1.0/$textureSet"), "v1.0/$textureSet_$udim");
        assert_eq!(tiled_template(".hidden"), ".hidden_$udim");
    }

    // --- MaterialX document (docs/specs/mtlx-export-design.md) ---

    /// Every image `file` value in a `.mtlx`, unescaped, in document order.
    fn mtlx_files(doc: &str) -> Vec<String> {
        let key = "type=\"filename\" value=\"";
        doc.match_indices(key)
            .map(|(at, _)| {
                let rest = &doc[at + key.len()..];
                rest[..rest.find('"').unwrap()]
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&amp;", "&")
            })
            .collect()
    }

    /// The `<extract>` body reading channel `index` of image `image`.
    fn extract_of(image: &str, index: u8) -> String {
        format!(
            "<input name=\"in\" type=\"color3\" nodename=\"{image}\" />\n    \
             <input name=\"index\" type=\"integer\" value=\"{index}\" />"
        )
    }

    fn with_materialx(mut preset: ExportPreset) -> ExportPreset {
        preset.materialx = true;
        preset
    }

    #[test]
    fn materialx_run_writes_the_mtlx_naming_the_written_files() {
        // (e) The preset integration: the .mtlx lands beside the PNGs and
        // every image file it names is a file this run wrote.
        let set = gltf_ready_set(2, 200);
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-mtlx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(
            &with_materialx(ExportPreset::gltf_metal_rough()),
            &set,
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        assert_eq!(
            file_names(&written),
            vec![
                "Sword_baseColor.png",
                "Sword_metallicRoughness.png",
                "Sword_normal.png",
                "Sword.mtlx",
            ]
        );
        let doc = std::fs::read_to_string(&written[3]).unwrap();
        // base_color, base_metalness (G), specular_roughness (B), normal.
        assert_eq!(
            mtlx_files(&doc),
            vec![
                "Sword_baseColor.png",
                "Sword_metallicRoughness.png",
                "Sword_metallicRoughness.png",
                "Sword_normal.png",
            ]
        );
        for file in mtlx_files(&doc) {
            assert!(out.join(&file).is_file(), "{file} was not written");
        }
        // The packed channels follow the preset's wiring (glTF: G = metal,
        // B = rough); the colorspaces follow the written transfer.
        assert!(doc.contains(&extract_of("Sword_base_metalness_image", 1)));
        assert!(doc.contains(&extract_of("Sword_specular_roughness_image", 2)));
        assert!(doc.contains("value=\"Sword_baseColor.png\" colorspace=\"srgb_texture\""));
        assert!(doc.contains("value=\"Sword_metallicRoughness.png\" colorspace=\"lin_rec709\""));
        assert!(doc.contains("value=\"Sword_normal.png\" colorspace=\"lin_rec709\""));
        // The graph reader accepts it (no painter nodes inside → empty).
        let (graph, _, _) = umber_graph::mtlx::from_mtlx(&doc).expect("parses");
        assert!(graph.nodes.is_empty());
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn materialx_off_writes_no_mtlx() {
        let set = gltf_ready_set(2, 200);
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-mtlxoff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        run_preset(
            &ExportPreset::gltf_metal_rough(),
            &set,
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        assert!(!out.join("Sword.mtlx").exists());
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn materialx_tiled_run_uses_the_udim_token_and_single_tile_stays_concrete() {
        // (d) through the real driver: a two-tile run's document names the
        // files with MaterialX's <UDIM> token — each tile's substitution is
        // a file the run wrote; the lone-1001 run names concrete files.
        let set_a = gltf_ready_set(2, 10);
        let set_b = gltf_ready_set(2, 20);
        let preset = with_materialx(ExportPreset::gltf_metal_rough());
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-mtlxudim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset_tiled(
            &preset,
            &[(1001, &set_a), (1002, &set_b)],
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        assert_eq!(written.len(), 7, "6 tile files + one .mtlx");
        assert!(written[6].ends_with("Sword.mtlx"));
        let doc = std::fs::read_to_string(&written[6]).unwrap();
        let files = mtlx_files(&doc);
        assert_eq!(files[0], "Sword_baseColor_<UDIM>.png");
        assert!(doc.contains("value=\"Sword_baseColor_&lt;UDIM&gt;.png\""));
        for file in &files {
            assert!(file.contains(UDIM_TOKEN), "{file} must be tiled");
            for tile in ["1001", "1002"] {
                let concrete = file.replace(UDIM_TOKEN, tile);
                assert!(out.join(&concrete).is_file(), "{concrete} was not written");
            }
        }
        std::fs::remove_dir_all(&out).unwrap();

        let written = run_preset_tiled(
            &preset,
            &[(1001, &set_a)],
            &TokenSources::new("Sword"),
            &out,
        )
        .unwrap();
        let doc = std::fs::read_to_string(written.last().unwrap()).unwrap();
        assert!(!doc.contains("UDIM"));
        assert_eq!(mtlx_files(&doc)[0], "Sword_baseColor.png");
        for file in mtlx_files(&doc) {
            assert!(out.join(&file).is_file(), "{file} was not written");
        }
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn materialx_orm_skips_ao_and_the_directx_normal() {
        // Unreal: AO has no OpenPBR input; ORM's G/B carry rough/metal;
        // the DirectX normal is not wired (normalmap expects +Y).
        let size = 2;
        let mut set = gltf_ready_set(size, 100);
        set.set(MapKind::AmbientOcclusion, solid(size, 230));
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-mtlxorm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(
            &with_materialx(ExportPreset::unreal_orm()),
            &set,
            &TokenSources::new("Hull"),
            &out,
        )
        .unwrap();
        let doc = std::fs::read_to_string(written.last().unwrap()).unwrap();
        assert_eq!(
            mtlx_files(&doc),
            vec![
                "Hull_basecolor.png",
                "Hull_OpenglORM.png",
                "Hull_OpenglORM.png"
            ]
        );
        assert!(doc.contains(&extract_of("Hull_specular_roughness_image", 1)));
        assert!(doc.contains(&extract_of("Hull_base_metalness_image", 2)));
        assert!(!doc.contains("geometry_normal"));
        assert!(!doc.contains("normalmap"));
        std::fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn materialx_colorspace_follows_the_written_transfer() {
        // TIFF is always sRGB-curved; EXR never — a roughness map in each
        // is tagged by its bytes, not by the §9 kind default.
        let mut tiff = passthrough_output("$textureSet_rough.tif", MapKind::Roughness);
        tiff.format = OutputFormat::Tiff;
        let mut exr = passthrough_output("$textureSet_base.exr", MapKind::BaseColor);
        exr.format = OutputFormat::Exr32F;
        let preset = ExportPreset {
            name: "transfer probe".into(),
            outputs: vec![tiff, exr],
            materialx: true,
        };
        let mut set = MapSet::new(2);
        set.set(MapKind::Roughness, solid(2, 64));
        set.set(MapKind::BaseColor, solid(2, 64));
        let out =
            std::env::temp_dir().join(format!("umber-export-drv-mtlxcs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let written = run_preset(&preset, &set, &TokenSources::new("T"), &out).unwrap();
        let doc = std::fs::read_to_string(written.last().unwrap()).unwrap();
        assert!(doc.contains("value=\"T_rough.tif\" colorspace=\"srgb_texture\""));
        assert!(doc.contains("value=\"T_base.exr\" colorspace=\"lin_rec709\""));
        std::fs::remove_dir_all(&out).unwrap();
    }
}

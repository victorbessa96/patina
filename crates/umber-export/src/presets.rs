//! Export presets (requirements.md §6): template-driven, per-output
//! map selection, channel packing, bit depth/format, normal-convention
//! conversion — the Substance-killer export configurator, serialized
//! as JSON so users save/share presets as files.
//!
//! An [`ExportPreset`] names a set of [`OutputSpec`]s; each output
//! picks maps by kind, packs channels (ORM etc.), converts normal
//! convention, and writes through the format writers (png today; exr
//! etc. join per §6). Presets serialize cleanly (serde) for the
//! saved-as-files requirement.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Which baked/painted map an output pulls from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapKind {
    /// Base color (sRGB source, color-managed).
    BaseColor,
    /// Roughness (linear data).
    Roughness,
    /// Metallic (linear data).
    Metallic,
    /// Ambient occlusion (linear data).
    AmbientOcclusion,
    /// Normal (tangent space; convention converted at export).
    Normal,
    /// Height (linear data).
    Height,
    /// Opacity/alpha.
    Opacity,
    /// Emissive.
    Emissive,
}

impl MapKind {
    /// The `$srcMap` naming token for this map.
    pub fn token(self) -> &'static str {
        match self {
            Self::BaseColor => "baseColor",
            Self::Roughness => "roughness",
            Self::Metallic => "metallic",
            Self::AmbientOcclusion => "ambient_occlusion",
            Self::Normal => "normal",
            Self::Height => "height",
            Self::Opacity => "opacity",
            Self::Emissive => "emissive",
        }
    }
}

/// Which channel of a source map feeds an output channel slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSlot {
    /// The source's red channel.
    R,
    /// The source's green channel.
    G,
    /// The source's blue channel.
    B,
    /// The source's alpha channel.
    A,
    /// The source's grayscale value (single-channel maps).
    Gray,
}

/// One output-channel wire: which input map (index into
/// [`OutputSpec::maps`]) and which slot of it fills the channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelWiring {
    /// Index into the spec's `maps` list.
    pub map: usize,
    /// The slot of that map feeding the channel.
    pub slot: ChannelSlot,
}

impl ChannelWiring {
    /// Shorthand constructor.
    pub const fn new(map: usize, slot: ChannelSlot) -> Self {
        Self { map, slot }
    }
}

impl ChannelSlot {
    /// Reads this slot out of an RGBA8 texel.
    pub fn sample(self, texel: [u8; 4]) -> u8 {
        match self {
            Self::R => texel[0],
            Self::G => texel[1],
            Self::B => texel[2],
            Self::A => texel[3],
            Self::Gray => {
                // Rec.709 luminance of the RGB channels.
                let (r, g, b) = (texel[0] as u32, texel[1] as u32, texel[2] as u32);
                ((r * 54 + g * 183 + b * 19) >> 8) as u8
            }
        }
    }
}

/// Tangent-space normal convention — the Y-flip at export
/// (requirements §6: "DirectX vs OpenGL").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalConvention {
    /// OpenGL: +Y up in tangent space (Blender, Unity, glTF).
    Opengl,
    /// DirectX: -Y (Unreal, Substance default).
    Directx,
}

/// Output file format + bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// 8-bit PNG.
    Png8,
    /// 16-bit PNG (bit-depth requirement: 8/16/32F).
    Png16,
    /// 32-bit float EXR.
    Exr32F,
    /// JPEG (8-bit, lossy — thumbnails/quick previews).
    Jpeg,
}

/// One packed output file an export produces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Template for the file name (uses §6 tokens).
    pub filename: String,
    /// Maps gathered as inputs, keyed by their naming token.
    pub maps: Vec<(MapKind, Vec<u8>)>,
    /// Channel wiring: which source map + slot feeds each of R/G/B/A.
    /// Index i answers "what fills output channel i" — the map is
    /// named per channel (ORM: R←AO, G←rough, B←metal), not implied
    /// by output position.
    pub channels: [ChannelWiring; 4],
    /// Normal-convention conversion applied when the output is normal.
    pub normal_convention: NormalConvention,
    /// File format + depth.
    pub format: OutputFormat,
}

/// A named, serializable export configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportPreset {
    /// Preset name (shown in the export dialog).
    pub name: String,
    /// Output files this preset produces.
    pub outputs: Vec<OutputSpec>,
}

/// Export-configuration errors.
#[derive(Debug, Error)]
pub enum ExportPresetError {
    /// A preset JSON file failed to parse.
    #[error("preset JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// An output references a map kind not present in its maps list.
    #[error("output {output:?} needs map {map:?} but it is not wired")]
    MissingMap {
        /// The failing output's filename template.
        output: String,
        /// The map kind that was missing.
        map: MapKind,
    },
}

impl ExportPreset {
    /// The canonical glTF metal-rough preset (§6 "Engine presets"):
    /// baseColor (sRGB PNG), metallicRoughness packed G/B, normal
    /// OpenGL.
    pub fn gltf_metal_rough() -> Self {
        // maps list: [0]=metallic, [1]=roughness.
        // glTF packing: G=metallic gray, B=roughness gray, R=roughness.r
        // (glTF expects 0 in R; roughness maps are grayscale so R==B).
        let packed = OutputSpec {
            filename: "$textureSet_metallicRoughness.png".into(),
            maps: vec![(MapKind::Metallic, vec![]), (MapKind::Roughness, vec![])],
            channels: [
                ChannelWiring::new(1, ChannelSlot::R), // R = roughness.r (≈0)
                ChannelWiring::new(0, ChannelSlot::Gray), // G = metallic gray
                ChannelWiring::new(1, ChannelSlot::Gray), // B = roughness gray
                ChannelWiring::new(0, ChannelSlot::A), // A = metallic alpha
            ],
            normal_convention: NormalConvention::Opengl,
            format: OutputFormat::Png8,
        };
        Self {
            name: "glTF metal-rough".into(),
            outputs: vec![
                OutputSpec {
                    filename: "$textureSet_baseColor.png".into(),
                    maps: vec![(MapKind::BaseColor, vec![])],
                    channels: [
                        ChannelWiring::new(0, ChannelSlot::R),
                        ChannelWiring::new(0, ChannelSlot::G),
                        ChannelWiring::new(0, ChannelSlot::B),
                        ChannelWiring::new(0, ChannelSlot::A),
                    ],
                    normal_convention: NormalConvention::Opengl,
                    format: OutputFormat::Png8,
                },
                packed,
                OutputSpec {
                    filename: "$textureSet_normal.png".into(),
                    maps: vec![(MapKind::Normal, vec![])],
                    channels: [
                        ChannelWiring::new(0, ChannelSlot::R),
                        ChannelWiring::new(0, ChannelSlot::G),
                        ChannelWiring::new(0, ChannelSlot::B),
                        ChannelWiring::new(0, ChannelSlot::A),
                    ],
                    normal_convention: NormalConvention::Opengl,
                    format: OutputFormat::Png8,
                },
            ],
        }
    }

    /// Serializes the preset to pretty JSON (saved-as-files).
    pub fn to_json(&self) -> Result<String, ExportPresetError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parses a preset from JSON (shared presets).
    pub fn from_json(raw: &str) -> Result<Self, ExportPresetError> {
        Ok(serde_json::from_str(raw)?)
    }

    /// The Unreal Engine preset (§6): ORM packed (R=AO, G=Roughness,
    /// B=Metallic), baseColor sRGB, normal in DirectX convention
    /// (Unreal's tangent-space Y is flipped vs OpenGL).
    pub fn unreal_orm() -> Self {
        let orm = OutputSpec {
            filename: "$textureSet_OpenglORM.png".into(),
            maps: vec![
                (MapKind::AmbientOcclusion, vec![]),
                (MapKind::Roughness, vec![]),
                (MapKind::Metallic, vec![]),
            ],
            channels: [
                ChannelWiring::new(0, ChannelSlot::Gray), // R = AO
                ChannelWiring::new(1, ChannelSlot::Gray), // G = rough
                ChannelWiring::new(2, ChannelSlot::Gray), // B = metal
                ChannelWiring::new(0, ChannelSlot::A),    // A = AO alpha
            ],
            normal_convention: NormalConvention::Directx,
            format: OutputFormat::Png8,
        };
        Self {
            name: "Unreal ORM".into(),
            outputs: vec![
                OutputSpec {
                    filename: "$textureSet_basecolor.png".into(),
                    maps: vec![(MapKind::BaseColor, vec![])],
                    channels: [
                        ChannelWiring::new(0, ChannelSlot::R),
                        ChannelWiring::new(0, ChannelSlot::G),
                        ChannelWiring::new(0, ChannelSlot::B),
                        ChannelWiring::new(0, ChannelSlot::A),
                    ],
                    normal_convention: NormalConvention::Directx,
                    format: OutputFormat::Png8,
                },
                orm,
                OutputSpec {
                    filename: "$textureSet_normal.png".into(),
                    maps: vec![(MapKind::Normal, vec![])],
                    channels: [
                        ChannelWiring::new(0, ChannelSlot::R),
                        ChannelWiring::new(0, ChannelSlot::G),
                        ChannelWiring::new(0, ChannelSlot::B),
                        ChannelWiring::new(0, ChannelSlot::A),
                    ],
                    normal_convention: NormalConvention::Directx,
                    format: OutputFormat::Png8,
                },
            ],
        }
    }

    /// The Unity HDRP/URP preset (§6): baseColor (sRGB) + separate
    /// metallic/smoothness PNG (smoothness = 1 - roughness packed in
    /// A per URP convention; HDRP MaskMap is a later refinement),
    /// normal OpenGL, packed normal DXT5nm style noted in the preset.
    pub fn unity_hdrp_urp() -> Self {
        let metallic_smoothness = OutputSpec {
            filename: "$textureSet_MetallicSmoothness.png".into(),
            maps: vec![(MapKind::Metallic, vec![]), (MapKind::Roughness, vec![])],
            channels: [
                ChannelWiring::new(0, ChannelSlot::R), // R = metallic r
                ChannelWiring::new(0, ChannelSlot::G), // G = metallic g
                ChannelWiring::new(0, ChannelSlot::B), // B = metallic b
                // A = smoothness = 1 - roughness: the packing layer
                // inverts at export (documented: pack_texel's caller
                // inverts the roughness input when using this preset).
                ChannelWiring::new(1, ChannelSlot::Gray),
            ],
            normal_convention: NormalConvention::Opengl,
            format: OutputFormat::Png8,
        };
        Self {
            name: "Unity HDRP/URP".into(),
            outputs: vec![
                OutputSpec {
                    filename: "$textureSet_BaseMap.png".into(),
                    maps: vec![(MapKind::BaseColor, vec![])],
                    channels: [
                        ChannelWiring::new(0, ChannelSlot::R),
                        ChannelWiring::new(0, ChannelSlot::G),
                        ChannelWiring::new(0, ChannelSlot::B),
                        ChannelWiring::new(0, ChannelSlot::A),
                    ],
                    normal_convention: NormalConvention::Opengl,
                    format: OutputFormat::Png8,
                },
                metallic_smoothness,
                OutputSpec {
                    filename: "$textureSet_Normal.png".into(),
                    maps: vec![(MapKind::Normal, vec![])],
                    channels: [
                        ChannelWiring::new(0, ChannelSlot::R),
                        ChannelWiring::new(0, ChannelSlot::G),
                        ChannelWiring::new(0, ChannelSlot::B),
                        ChannelWiring::new(0, ChannelSlot::A),
                    ],
                    normal_convention: NormalConvention::Opengl,
                    format: OutputFormat::Png8,
                },
            ],
        }
    }

    /// The Blender Principled BSDF preset (§6): baseColor, roughness
    /// (gray), metallic (gray), normal OpenGL — all separate files,
    /// no packing (Principled takes scalar inputs).
    pub fn blender_principled() -> Self {
        let passthrough = |kind: MapKind, filename: &str| OutputSpec {
            filename: filename.into(),
            maps: vec![(kind, vec![])],
            channels: [
                ChannelWiring::new(0, ChannelSlot::R),
                ChannelWiring::new(0, ChannelSlot::G),
                ChannelWiring::new(0, ChannelSlot::B),
                ChannelWiring::new(0, ChannelSlot::A),
            ],
            normal_convention: NormalConvention::Opengl,
            format: OutputFormat::Png8,
        };
        Self {
            name: "Blender Principled".into(),
            outputs: vec![
                passthrough(MapKind::BaseColor, "$textureSet_basecolor.png"),
                passthrough(MapKind::Roughness, "$textureSet_roughness.png"),
                passthrough(MapKind::Metallic, "$textureSet_metallic.png"),
                passthrough(MapKind::Normal, "$textureSet_normal.png"),
            ],
        }
    }
}

/// Packs one RGBA8 output texel from the source maps per the channel
/// wiring: the OUTPUT channel at index i reads its slot from the
/// input map feeding that slot's channel position.
///
/// The `inputs` slice is parallel to the spec's `maps` list (same
/// order); each input is one RGBA8 texel from that map. When fewer
/// inputs than maps are provided the missing ones read as
/// transparent black (the padding rule before dilation lands).
pub fn pack_texel(spec: &OutputSpec, inputs: &[Texel]) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (i, wiring) in spec.channels.iter().enumerate() {
        // Channel i reads (map index, slot) — the map is NAMED per
        // channel, not implied by position. Missing maps read as
        // transparent black (the pre-dilation padding rule).
        let texel = inputs.get(wiring.map).copied().unwrap_or_default();
        out[i] = wiring.slot.sample(texel.rgba);
    }
    out
}

/// One RGBA8 texel from a source map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Texel {
    /// RGBA8.
    pub rgba: [u8; 4],
}

impl From<[u8; 4]> for Texel {
    fn from(rgba: [u8; 4]) -> Self {
        Self { rgba }
    }
}

/// Applies the normal-convention conversion to one texel (the Y-flip:
/// OpenGL ↔ DirectX differ only in green).
pub fn convert_normal(texel: Texel, from: NormalConvention, to: NormalConvention) -> Texel {
    if from == to {
        return texel;
    }
    let mut flipped = texel;
    flipped.rgba[1] = 255 - flipped.rgba[1];
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gltf_preset_serializes_and_back() {
        let preset = ExportPreset::gltf_metal_rough();
        let json = preset.to_json().unwrap();
        let back = ExportPreset::from_json(&json).unwrap();
        assert_eq!(preset, back);
    }

    #[test]
    fn gltf_preset_shape() {
        let preset = ExportPreset::gltf_metal_rough();
        assert_eq!(preset.name, "glTF metal-rough");
        assert_eq!(preset.outputs.len(), 3);
        assert!(preset
            .outputs
            .iter()
            .any(|o| o.filename.contains("metallicRoughness")));
        assert!(preset
            .outputs
            .iter()
            .all(|o| o.normal_convention == NormalConvention::Opengl));
    }

    #[test]
    fn unreal_preset_is_directx_and_packs_orm() {
        let preset = ExportPreset::unreal_orm();
        assert_eq!(preset.name, "Unreal ORM");
        let orm = preset
            .outputs
            .iter()
            .find(|o| o.filename.contains("ORM"))
            .expect("ORM output");
        // R=AO, G=rough, B=metal via distinct map indices.
        assert_eq!(orm.channels[0].map, 0, "R = AO (maps[0])");
        assert_eq!(orm.channels[1].map, 1, "G = roughness (maps[1])");
        assert_eq!(orm.channels[2].map, 2, "B = metallic (maps[2])");
        assert!(preset
            .outputs
            .iter()
            .all(|o| o.normal_convention == NormalConvention::Directx));
        // Every preset round-trips JSON.
        let back = ExportPreset::from_json(&preset.to_json().unwrap()).unwrap();
        assert_eq!(back, preset);
    }

    #[test]
    fn unity_preset_smoothness_reads_roughness_map() {
        let preset = ExportPreset::unity_hdrp_urp();
        assert_eq!(preset.name, "Unity HDRP/URP");
        let ms = preset
            .outputs
            .iter()
            .find(|o| o.filename.contains("MetallicSmoothness"))
            .expect("metallic/smoothness output");
        // A (smoothness) reads maps[1] = roughness (the caller inverts).
        assert_eq!(ms.channels[3].map, 1);
        assert_eq!(ms.channels[3].slot, ChannelSlot::Gray);
        assert!(preset
            .outputs
            .iter()
            .all(|o| o.normal_convention == NormalConvention::Opengl));
    }

    #[test]
    fn blender_preset_is_four_passthroughs() {
        let preset = ExportPreset::blender_principled();
        assert_eq!(preset.name, "Blender Principled");
        assert_eq!(preset.outputs.len(), 4);
        // Each output wires exactly its own map, channels straight
        // through in order: R, G, B, A.
        for (i, output) in preset.outputs.iter().enumerate() {
            assert_eq!(output.maps.len(), 1, "output {i} is single-map");
            assert_eq!(output.channels[0], ChannelWiring::new(0, ChannelSlot::R));
            assert_eq!(output.channels[1], ChannelWiring::new(0, ChannelSlot::G));
            assert_eq!(output.channels[2], ChannelWiring::new(0, ChannelSlot::B));
            assert_eq!(output.channels[3], ChannelWiring::new(0, ChannelSlot::A));
        }
        let back = ExportPreset::from_json(&preset.to_json().unwrap()).unwrap();
        assert_eq!(back, preset);
    }

    #[test]
    fn channel_slots_sample_their_channel() {
        let texel = [10u8, 20, 30, 40];
        assert_eq!(ChannelSlot::R.sample(texel), 10);
        assert_eq!(ChannelSlot::G.sample(texel), 20);
        assert_eq!(ChannelSlot::B.sample(texel), 30);
        assert_eq!(ChannelSlot::A.sample(texel), 40);
    }

    #[test]
    fn gray_slot_luminance() {
        // Pure green maps to its own value.
        let texel = [0u8, 200, 0, 255];
        let g = ChannelSlot::Gray.sample(texel);
        assert!((140..=145).contains(&g), "green luminance {g}");
    }

    #[test]
    fn orm_pack_layout() {
        // The ORM pattern: R=AO, G=Roughness, B=Metallic, one packed
        // file. Each output channel reads the gray of its own map.
        let spec = OutputSpec {
            filename: "$textureSet_orm.png".into(),
            maps: vec![
                (MapKind::AmbientOcclusion, vec![]),
                (MapKind::Roughness, vec![]),
                (MapKind::Metallic, vec![]),
            ],
            channels: [
                ChannelWiring::new(0, ChannelSlot::Gray), // R = AO gray
                ChannelWiring::new(1, ChannelSlot::Gray), // G = rough gray
                ChannelWiring::new(2, ChannelSlot::Gray), // B = metal gray
                ChannelWiring::new(0, ChannelSlot::A),    // A = AO alpha
            ],
            format: OutputFormat::Png8,
            normal_convention: NormalConvention::Opengl,
        };
        let inputs = [
            Texel::from([230, 230, 230, 255]),
            Texel::from([128, 128, 128, 255]),
            Texel::from([64, 64, 64, 255]),
        ];
        let packed = pack_texel(&spec, &inputs);
        assert_eq!(packed[0], 230, "R = AO gray");
        assert_eq!(packed[1], 128, "G = roughness gray");
        assert_eq!(packed[2], 64, "B = metallic gray");
        assert_eq!(packed[3], 255, "A = AO alpha");
    }

    #[test]
    fn missing_inputs_pad_to_transparent_black() {
        let spec = OutputSpec {
            filename: "x.png".into(),
            maps: vec![(MapKind::Roughness, vec![])],
            channels: [ChannelWiring::new(0, ChannelSlot::Gray); 4],
            format: OutputFormat::Png8,
            normal_convention: NormalConvention::Opengl,
        };
        let packed = pack_texel(&spec, &[]);
        assert_eq!(packed, [0, 0, 0, 0]);
    }

    #[test]
    fn normal_convention_flip_is_green_only() {
        let t = Texel::from([128, 200, 128, 255]);
        let flipped = convert_normal(t, NormalConvention::Opengl, NormalConvention::Directx);
        assert_eq!(flipped.rgba, [128, 55, 128, 255]);
        // Identity when conventions match.
        assert_eq!(
            convert_normal(t, NormalConvention::Opengl, NormalConvention::Opengl),
            t
        );
    }

    #[test]
    fn map_kinds_tokenize() {
        assert_eq!(MapKind::BaseColor.token(), "baseColor");
        assert_eq!(MapKind::AmbientOcclusion.token(), "ambient_occlusion");
    }

    #[test]
    fn presets_are_user_file_friendly() {
        // A minimal hand-written preset must parse — channel wiring as
        // {map, slot} objects.
        let raw = r#"{
            "name": "Unity HDRP",
            "outputs": [
                {
                    "filename": "$textureSet_BaseMap.png",
                    "maps": [["base_color", []]],
                    "channels": [
                        {"map": 0, "slot": "r"},
                        {"map": 0, "slot": "g"},
                        {"map": 0, "slot": "b"},
                        {"map": 0, "slot": "a"}
                    ],
                    "normal_convention": "opengl",
                    "format": "png8"
                }
            ]
        }"#;
        let preset = ExportPreset::from_json(raw).unwrap();
        assert_eq!(preset.name, "Unity HDRP");
        assert_eq!(preset.outputs[0].maps[0].0, MapKind::BaseColor);
    }
}

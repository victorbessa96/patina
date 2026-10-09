//! Mesh-map naming convention: `TextureSetName_map` file naming both
//! ways — parsing imported map files into (texture-set, map kind) and
//! formatting exports (requirements §7: "Mesh-map import by naming
//! convention (TextureSetName_map)").
//!
//! Matches Substance's convention: `<set>_ambient_occlusion.png` is an
//! AO map for texture set `<set>`; a file with no recognized map suffix
//! is not a mesh map (it's a plain texture — return `None`, don't
//! guess). The recognized suffixes mirror
//! `umber_bake::BakeMap::file_stem_suffix` (the P0 bakers) plus the
//! engine-specific extras (`_basecolor`, `_normal`, `_roughness`, …)
//! that Painter itself recognizes on import.

use crate::MeshData;

/// The mesh-map kinds recognized by the naming convention (a subset of
/// `umber_bake::BakeMap` — the two vocabularies intentionally overlap
/// but live in separate crates; this one is the IMPORT side and must
/// not depend on the bake engine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshMapKind {
    /// `<set>_ambient_occlusion`
    AmbientOcclusion,
    /// `<set>_curvature`
    Curvature,
    /// `<set>_normal_base` (baked) — distinct from a user normal map
    /// (`<set>_normal`), which is not a mesh map.
    NormalBase,
    /// `<set>_position`
    Position,
    /// `<set>_thickness`
    Thickness,
    /// `<set>_id`
    Id,
    /// `<set>_height`
    Height,
    /// `<set>_bent_normal`
    BentNormal,
    /// `<set>_world_space_normal`
    WorldSpaceNormal,
}

impl MeshMapKind {
    /// The `_suffix` this kind parses from / formats with.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::AmbientOcclusion => "ambient_occlusion",
            Self::Curvature => "curvature",
            Self::NormalBase => "normal_base",
            Self::Position => "position",
            Self::Thickness => "thickness",
            Self::Id => "id",
            Self::Height => "height",
            Self::BentNormal => "bent_normal",
            Self::WorldSpaceNormal => "world_space_normal",
        }
    }
}

/// A parsed mesh-map file name: the owning texture set + the map kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshMapName {
    /// The texture set the map belongs to (may differ from the mesh name).
    pub texture_set: String,
    /// Which map this is.
    pub kind: MeshMapKind,
}

/// Parses a file stem by the mesh-map convention.
///
/// `path` may be a full path (only the file stem is inspected).
/// Returns `None` when the stem carries no recognized map suffix — the
/// caller treats it as a plain texture, never an error.
pub fn parse_mesh_map(path: &std::path::Path) -> Option<MeshMapName> {
    let stem = path.file_stem()?.to_str()?;
    parse_mesh_map_stem(stem)
}

/// Stem-only variant of [`parse_mesh_map`].
pub fn parse_mesh_map_stem(stem: &str) -> Option<MeshMapName> {
    // Longest-suffix-first so `_normal_base` wins over a hypothetical
    // `_base` (none today, but the scan must be unambiguous).
    const KINDS: &[MeshMapKind] = &[
        MeshMapKind::AmbientOcclusion,
        MeshMapKind::Curvature,
        MeshMapKind::NormalBase,
        MeshMapKind::Position,
        MeshMapKind::Thickness,
        MeshMapKind::Id,
        MeshMapKind::Height,
        MeshMapKind::BentNormal,
        MeshMapKind::WorldSpaceNormal,
    ];
    // Sort by suffix length descending once (const array, could be
    // pre-sorted; done at runtime for clarity).
    let mut kinds: Vec<&MeshMapKind> = KINDS.iter().collect();
    kinds.sort_by_key(|k| std::cmp::Reverse(k.suffix().len()));
    for kind in kinds {
        let suffix = format!("_{}", kind.suffix());
        if let Some(set) = stem.strip_suffix(&suffix) {
            if set.is_empty() {
                // `_ambient_occlusion.png` — no set name: not ours.
                return None;
            }
            return Some(MeshMapName {
                texture_set: set.to_string(),
                kind: *kind,
            });
        }
    }
    None
}

/// Formats an export path for a baked mesh map: `<set>_<suffix>` under
/// `dir`, with the given extension. Used by the export presets and the
/// CLI bake commands.
pub fn format_mesh_map(
    dir: &std::path::Path,
    texture_set: &str,
    kind: MeshMapKind,
    ext: &str,
) -> std::path::PathBuf {
    dir.join(format!("{}_{}.{}", texture_set, kind.suffix(), ext))
}

/// The texture-set name derived from a mesh: the file stem, or the
/// material name when the mesh carries exactly one (Painter uses
/// material names; the file stem is the fallback).
pub fn texture_set_name(mesh_path: &std::path::Path, mesh: &MeshData) -> String {
    if mesh.material_names.len() == 1 {
        return mesh.material_names[0].clone();
    }
    mesh_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("TextureSet")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_bake_suffix() {
        for (stem, kind) in [
            ("Sword_ambient_occlusion", MeshMapKind::AmbientOcclusion),
            ("Blade_curvature", MeshMapKind::Curvature),
            ("Hull_normal_base", MeshMapKind::NormalBase),
            ("x_position", MeshMapKind::Position),
            ("rock_thickness", MeshMapKind::Thickness),
            ("crate_id", MeshMapKind::Id),
            ("terr_height", MeshMapKind::Height),
            ("bot_bent_normal", MeshMapKind::BentNormal),
            ("ship_world_space_normal", MeshMapKind::WorldSpaceNormal),
        ] {
            let parsed = parse_mesh_map_stem(stem).expect("parses");
            assert_eq!(parsed.kind, kind, "{stem}");
            assert!(!parsed.texture_set.is_empty());
        }
    }

    #[test]
    fn set_name_survives_underscores() {
        // `my_set_ao`: the SET itself contains underscores.
        let parsed = parse_mesh_map_stem("my_rock_thickness").expect("parses");
        assert_eq!(parsed.texture_set, "my_rock");
        assert_eq!(parsed.kind, MeshMapKind::Thickness);
    }

    #[test]
    fn plain_textures_return_none() {
        // No recognized suffix — a plain albedo, not a mesh map.
        assert!(parse_mesh_map_stem("wood_basecolor").is_none());
        assert!(parse_mesh_map_stem("noise").is_none());
    }

    #[test]
    fn bare_suffix_without_set_is_rejected() {
        assert!(parse_mesh_map_stem("_curvature").is_none());
    }

    #[test]
    fn full_path_parses_stem_only() {
        let p = std::path::Path::new("/assets/maps/Sword_ambient_occlusion.png");
        let parsed = parse_mesh_map(p).expect("parses");
        assert_eq!(parsed.texture_set, "Sword");
        assert_eq!(parsed.kind, MeshMapKind::AmbientOcclusion);
    }

    #[test]
    fn format_roundtrips_through_parse() {
        let path = format_mesh_map(
            std::path::Path::new("/out"),
            "Sword",
            MeshMapKind::AmbientOcclusion,
            "png",
        );
        let parsed = parse_mesh_map(&path).expect("roundtrip");
        assert_eq!(parsed.texture_set, "Sword");
        assert_eq!(parsed.kind, MeshMapKind::AmbientOcclusion);
    }

    #[test]
    fn texture_set_prefers_single_material_name() {
        let mesh = MeshData {
            material_names: vec!["Blade".into()],
            ..MeshData::default()
        };
        assert_eq!(
            texture_set_name(std::path::Path::new("/m/sword.obj"), &mesh),
            "Blade"
        );
    }

    #[test]
    fn texture_set_falls_back_to_file_stem() {
        let mesh = MeshData::default();
        assert_eq!(
            texture_set_name(std::path::Path::new("/m/sword.obj"), &mesh),
            "sword"
        );
    }
}

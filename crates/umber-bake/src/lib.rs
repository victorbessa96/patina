//! umber-bake — mesh-map bakers.
//!
//! Wave 3 scope (docs/specs/requirements.md §3): AO, normal (mesh +
//! high→low), curvature, position, thickness, ID, with cage controls,
//! dilation, and golden-image CI verification on lavapipe/WARP.
//! Compute passes share the umber-gpu tile pool.

pub mod ao;
pub mod curvature;
pub mod dilation;
pub mod normal_map;
pub mod position;
pub mod thickness;

pub use ao::{AoBakeError, AoBakeParams, BakeTarget, PlaneDesc};
pub use curvature::{CurvatureBakeError, CurvatureParams};
pub use dilation::{DilateError, DilateParams};
pub use normal_map::{TangentNormalBakeError, TangentNormalParams};
pub use thickness::{ThicknessBakeError, ThicknessParams};

/// The bake map types, named per Substance conventions for the
/// `TextureSetName_map` import naming (requirements §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeMap {
    AmbientOcclusion,
    Normal,
    Curvature,
    Position,
    Thickness,
    Id,
    Height,
    BentNormal,
    WorldSpaceNormal,
}

impl BakeMap {
    /// The conventional on-disk suffix for imported mesh maps
    /// (`TextureSetName_ambient_occlusion.png` etc.).
    pub fn file_stem_suffix(self) -> &'static str {
        match self {
            BakeMap::AmbientOcclusion => "ambient_occlusion",
            BakeMap::Normal => "normal_base",
            BakeMap::Curvature => "curvature",
            BakeMap::Position => "position",
            BakeMap::Thickness => "thickness",
            BakeMap::Id => "id",
            BakeMap::Height => "height",
            BakeMap::BentNormal => "bent_normal",
            BakeMap::WorldSpaceNormal => "world_space_normal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_naming_matches_substance_convention() {
        assert_eq!(
            BakeMap::AmbientOcclusion.file_stem_suffix(),
            "ambient_occlusion"
        );
        assert_eq!(
            BakeMap::WorldSpaceNormal.file_stem_suffix(),
            "world_space_normal"
        );
    }
}

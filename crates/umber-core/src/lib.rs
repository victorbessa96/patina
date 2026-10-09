//! umber-core — the Umber document model.
//!
//! Pure Rust, no GPU dependency: texture sets, channel stacks, layers,
//! masks, and the undo journal. Everything here must be testable headless
//! in CI (SPEC.md acceptance criterion: CPU-path round-trip determinism).
//!
//! Wave 2 fills in the layer stack ([`layers`]), the command-pattern undo
//! journal ([`undo`]), and the `.umber` project format ([`project`]) on top
//! of the Wave 1 texture-set/channel skeleton below
//! (docs/specs/requirements.md §2, §8).

use glam::Vec3;
use serde::{Deserialize, Serialize};

pub mod layers;
pub mod project;
pub mod undo;

pub use layers::{BlendMode, Layer, LayerCommand, LayerKind, LayerMask, LayerStack};
pub use project::{
    ProjectError, ProjectModel, ProjectSettings, TextureSetLayers, CURRENT_PROJECT_VERSION,
};
pub use undo::{Command, UndoStack};

/// A named channel of a texture set (e.g. baseColor, roughness, normal).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Channel {
    pub name: String,
    pub kind: ChannelKind,
}

/// How a channel's values are stored and color-managed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelKind {
    /// sRGB-encoded color data (baseColor, emissive color).
    Color,
    /// Linear data (roughness, metallic, height, normal, AO).
    Data,
}

/// One texture set: a resolution plus a stack of channels.
/// Wave 1: model skeleton; painting + per-channel blending land in Wave 2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureSet {
    pub name: String,
    pub resolution: u32,
    pub channels: Vec<Channel>,
}

impl TextureSet {
    /// The default OpenPBR-style starting set for a new project.
    pub fn new_default(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            resolution: 2048,
            channels: vec![
                Channel {
                    name: "baseColor".into(),
                    kind: ChannelKind::Color,
                },
                Channel {
                    name: "roughness".into(),
                    kind: ChannelKind::Data,
                },
                Channel {
                    name: "metallic".into(),
                    kind: ChannelKind::Data,
                },
                Channel {
                    name: "normal".into(),
                    kind: ChannelKind::Data,
                },
                Channel {
                    name: "height".into(),
                    kind: ChannelKind::Data,
                },
            ],
        }
    }
}

/// A flat 3D bounding box, in the mesh's own coordinate space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_texture_set_has_openpbr_channels() {
        let ts = TextureSet::new_default("Body");
        assert_eq!(ts.channels.len(), 5);
        assert_eq!(ts.channels[0].name, "baseColor");
        assert_eq!(ts.channels[0].kind, ChannelKind::Color);
        assert_eq!(ts.channels[2].name, "metallic");
        assert_eq!(ts.channels[2].kind, ChannelKind::Data);
    }
}

//! umber-color — color management integration.
//!
//! Wave 2 scope (docs/specs/requirements.md §9): OCIO v2 configs via
//! ocio-rs (vendored), scene-linear working space, display transforms via
//! GPU LUTs, per-channel CM flags (baseColor managed; roughness/normal = data).
//!
//! Wave 1 scope: the per-channel CM flag vocabulary.
//! Wave 2: the CPU display-transform reference path ([`display`]) — the
//! spec the GPU shader and the exporter validate against.

pub mod display;

pub use display::{apply_display, invert_display, DisplayTransform};

/// Whether a channel's values are color-managed or raw data
/// (mirrors umber-core's ChannelKind; duplicated at the crate boundary
/// so umber-color stays standalone-usable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorManagement {
    /// sRGB-encoded, managed through OCIO on display.
    Managed,
    /// Linear data — displayed raw, exported without display transforms.
    Data,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cm_vocabulary() {
        assert_ne!(ColorManagement::Managed, ColorManagement::Data);
    }
}

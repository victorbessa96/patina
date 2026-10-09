//! OpenPBR Surface material parameters for the real-time viewport shader.
//!
//! Defines the GPU-side uniform structure matching the OpenPBR Surface
//! parameter reference (v1.1.1). Only the real-time subset is included:
//! - Base (weight, color, metalness)
//! - Specular (weight, color, roughness, IOR)
//! - Coat (weight, color, roughness, IOR)
//! - Emission (weight, color) — using luminance * color as combined weight
//! - Geometry (opacity)
//!
//! Deferred (not in this subset): subsurface, transmission, thin-film, fuzz,
//! anisotropy parameters, base_diffuse_roughness, coat_roughness_anisotropy,
//! coat_darkening, geometry_thin_walled, normal/tangent maps.
//!
//! # Layout contract
//!
//! WGSL uniform-buffer structs align `vec3<f32>` members to 16 bytes, so a
//! Rust struct with `[f32; 3]` fields cannot byte-match a naive WGSL
//! mirror (the original layout here was 100 bytes of interleaved padding
//! vs a 56-byte expectation — caught by its own alignment test). The
//! layout is now **WGSL-first**: six `vec4<f32>` slots (96 bytes total,
//! every slot 16-byte aligned, zero internal padding), mirrored in Rust
//! as six `[f32; 4]` fields in the same order. The WGSL struct in
//! `shaders.rs::OPENPBR_SHADER` must keep the same slot order.

use bytemuck::{Pod, Zeroable};

/// OpenPBR Surface parameters for the real-time viewport shader.
///
/// All defaults come from the MaterialX reference implementation
/// (`docs/claw-artifacts/openpbr/open_pbr_surface.mtlx`) and spec v1.1.1.
///
/// Slots (each a `vec4<f32>` in WGSL, mirrored as `[f32; 4]` here):
/// - `base`:     `.xyz` = base_color, `.w` = base_weight
/// - `surface`:  `.x` = base_metalness, `.y` = specular_roughness,
///   `.z` = coat_roughness, `.w` = geometry_opacity
/// - `specular`: `.xyz` = specular_color, `.w` = specular_weight
/// - `coat`:     `.xyz` = coat_color, `.w` = coat_weight
/// - `emission`: `.xyz` = emission (luminance × color), `.w` unused
/// - `iors`:     `.x` = specular_ior, `.y` = coat_ior, `.zw` unused
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OpenPbrParams {
    /// Base color (rgb) + base weight (w). Defaults: (0.8, 0.8, 0.8, 1.0).
    pub base: [f32; 4],
    /// Metalness, specular roughness, coat roughness, geometry opacity.
    /// Defaults: (0.0, 0.3, 0.0, 1.0).
    pub surface: [f32; 4],
    /// Specular color (rgb) + weight (w). Defaults: (1, 1, 1, 1.0).
    pub specular: [f32; 4],
    /// Coat color (rgb) + weight (w). Defaults: (1, 1, 1, 0.0).
    pub coat: [f32; 4],
    /// Emission rgb (luminance premultiplied) + unused. Defaults: 0.
    pub emission: [f32; 4],
    /// Specular IOR (x, default 1.5), coat IOR (y, default 1.6), unused.
    pub iors: [f32; 4],
}

impl Default for OpenPbrParams {
    fn default() -> Self {
        Self {
            base: [0.8, 0.8, 0.8, 1.0],
            surface: [0.0, 0.3, 0.0, 1.0],
            specular: [1.0, 1.0, 1.0, 1.0],
            coat: [1.0, 1.0, 1.0, 0.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            iors: [1.5, 1.6, 0.0, 0.0],
        }
    }
}

impl OpenPbrParams {
    /// Returns the byte representation for uniform-buffer upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openpbr_params_defaults_match_mtlx() {
        let p = OpenPbrParams::default();
        assert_eq!(p.base, [0.8, 0.8, 0.8, 1.0]); // base_color 0.8 + weight 1.0
        assert_eq!(p.surface[0], 0.0); // base_metalness
        assert_eq!(p.surface[1], 0.3); // specular_roughness
        assert_eq!(p.surface[2], 0.0); // coat_roughness
        assert_eq!(p.surface[3], 1.0); // geometry_opacity
        assert_eq!(p.specular, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(p.coat, [1.0, 1.0, 1.0, 0.0]); // coat weight 0
        assert_eq!(p.emission, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(p.iors[0], 1.5); // specular_ior
        assert_eq!(p.iors[1], 1.6); // coat_ior
    }

    #[test]
    fn openpbr_params_size_is_wgsl_uniform_aligned() {
        // Six vec4 slots = 96 bytes, every field 16-byte aligned — matches
        // the WGSL uniform layout with zero internal padding.
        assert_eq!(std::mem::size_of::<OpenPbrParams>(), 96);
        assert_eq!(std::mem::align_of::<OpenPbrParams>(), 4);
        // Byte-level: first slot is base_color.rgb + base_weight.
        let p = OpenPbrParams::default();
        let bytes = p.as_bytes();
        assert_eq!(bytes.len(), 96);
        assert_eq!(&bytes[0..4], &0.8f32.to_ne_bytes());
        assert_eq!(&bytes[12..16], &1.0f32.to_ne_bytes());
    }

    #[test]
    fn openpbr_params_pod_zeroable() {
        let zeroed = OpenPbrParams::zeroed();
        assert_eq!(zeroed.base, [0.0; 4]);
        assert_eq!(zeroed.iors, [0.0; 4]);
    }
}

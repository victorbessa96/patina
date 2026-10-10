//! The GPU display LUT: the viewer chain (exposure → view → gamma, the
//! app's Display panel) baked into a 256×1 RGBA8 texture that the two
//! display passes — the viewport mesh pass (`shaders::MESH_SHADER`) and
//! the UV view's paint-target display (`texture_display`) — read as
//! their LAST step before the color target. See
//! `LANDING_NOTES_DISPLAY_PANEL.md`.
//!
//! Crate boundary: this crate never sees `umber_color::DisplaySettings`.
//! The bytes ARE the interface — the app tabulates the chain
//! (`umber_color::build_display_lut`) and uploads the result here with
//! [`DisplayLut::upload`].
//!
//! Format + sampling contract:
//! - `Rgba8Unorm` (NOT `…Srgb`): entries are final display bytes. The
//!   eframe surface is a non-sRGB `Rgba8Unorm`/`Bgra8Unorm`
//!   (`egui_wgpu::preferred_framebuffer_format`), so the shader's output
//!   value is the displayed byte — no second encode.
//! - 256×1 `D2` (not `D1`): mirrors every other texture binding here and
//!   sidesteps 1D quirks on the GL backend.
//! - Read with `textureLoad` at `round(clamp(c, 0, 1) · 255)` — nearest,
//!   exact, NO sampler. Each channel reads its own entry's matching
//!   component. For the 8-bit paint target this is bit-exact (the input
//!   IS one of the 256 levels); the identity table then returns the
//!   input value unchanged, which is the golden's no-regression contract.
//! - Group 1, binding 0 of both consumer pipelines (group 0 stays each
//!   pass's own set — the same additive-group shape the wireframe pass
//!   uses for its color uniform).
//!
//! Honest v1 limits: the input domain clamps to 0..1 (negative exposure
//! cannot recover mesh-pass values above 1 — the chain sees them
//! clipped), and 256 nearest levels on linear input band in the
//! shadows under the sRGB/Rec.709 views (linear 1/255 already maps to
//! byte 13). The overlays (ground grid, wireframe) are not transformed.

/// Entries in the LUT (one per 8-bit input level).
pub const DISPLAY_LUT_ENTRIES: u32 = 256;
/// Byte length of a LUT upload: [`DISPLAY_LUT_ENTRIES`] RGBA8 texels,
/// entry-major. Matches `umber_color::DISPLAY_LUT_BYTES`.
pub const DISPLAY_LUT_BYTES: usize = DISPLAY_LUT_ENTRIES as usize * 4;

/// WGSL for the LUT read — the shader-side contract above, pasted
/// verbatim into `shaders::MESH_SHADER` and the texture-display shader
/// (WGSL has no includes here; `shaders_embed_the_lut_snippet_verbatim`
/// pins both copies to this text so they cannot drift).
pub const DISPLAY_LUT_WGSL: &str = r#"// ---- display LUT (crate::display_lut — keep verbatim) ----
@group(1) @binding(0) var display_lut: texture_2d<f32>;

fn display_lut_index(c: f32) -> i32 {
    return i32(floor(clamp(c, 0.0, 1.0) * 255.0 + 0.5));
}

fn apply_display_lut(color: vec3<f32>) -> vec3<f32> {
    let r = textureLoad(display_lut, vec2<i32>(display_lut_index(color.r), 0), 0).r;
    let g = textureLoad(display_lut, vec2<i32>(display_lut_index(color.g), 0), 0).g;
    let b = textureLoad(display_lut, vec2<i32>(display_lut_index(color.b), 0), 0).b;
    return vec3<f32>(r, g, b);
}
// ---- end display LUT ----"#;

/// The identity table: entry `i` = `(i, i, i, 255)`. What the context's
/// fallback LUT holds (bound whenever a consumer passes `None`), and
/// byte-equal to `umber_color::build_display_lut` of the default
/// (Raw / 0 EV / gamma 1) settings — the app's tests pin that equality.
pub fn identity_lut_bytes() -> [u8; DISPLAY_LUT_BYTES] {
    let mut lut = [0u8; DISPLAY_LUT_BYTES];
    for (i, entry) in lut.chunks_exact_mut(4).enumerate() {
        let b = i as u8;
        entry.copy_from_slice(&[b, b, b, 255]);
    }
    lut
}

/// The group-1 layout both consumers share: one unfilterable-float D2
/// texture (`textureLoad` only — no sampler binding).
pub(crate) fn display_lut_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_display_lut_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    })
}

/// A display LUT on the device: the 256×1 texture plus its group-1 bind
/// group. Created once; [`Self::upload`] rewrites the texels in place,
/// so a settings change never rebuilds a bind group or a pipeline.
///
/// The app owns one (opaque — it never names a wgpu type) and hands
/// `Some(&lut)` to [`crate::MeshBuffers::paint_callback`] and
/// [`crate::texture_display::TextureDisplay::callback`]. Cloned handles
/// (wgpu resources are `Arc`-backed), so callbacks keep it alive.
#[derive(Clone)]
pub struct DisplayLut {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl DisplayLut {
    /// Creates a LUT on `gpu`'s device holding `bytes` (entry-major
    /// RGBA8, see [`DISPLAY_LUT_BYTES`]).
    pub fn new(gpu: &crate::GpuContext, bytes: &[u8; DISPLAY_LUT_BYTES]) -> Self {
        Self::with_layout(&gpu.device, &gpu.queue, gpu.display_lut_layout(), bytes)
    }

    /// [`Self::new`] against an explicit layout — `GpuContext::new` builds
    /// its identity fallback through this before the context exists.
    pub(crate) fn with_layout(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        bytes: &[u8; DISPLAY_LUT_BYTES],
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("umber_display_lut"),
            size: lut_extent(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_display_lut_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let lut = Self {
            texture,
            bind_group,
        };
        lut.write(queue, bytes);
        lut
    }

    /// Replaces the LUT's contents (the app's rebuild-on-dirty path).
    /// Staged on the queue: it lands before the next submit, so callbacks
    /// built earlier in the same frame already draw with the new table.
    pub fn upload(&self, gpu: &crate::GpuContext, bytes: &[u8; DISPLAY_LUT_BYTES]) {
        self.write(&gpu.queue, bytes);
    }

    fn write(&self, queue: &wgpu::Queue, bytes: &[u8; DISPLAY_LUT_BYTES]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            // write_texture has no 256-byte row alignment rule (only
            // buffer→texture copies do); one 1024-byte row.
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(DISPLAY_LUT_BYTES as u32),
                rows_per_image: Some(1),
            },
            lut_extent(),
        );
    }

    /// The group-1 bind group the consumer pipelines set.
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

fn lut_extent() -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: DISPLAY_LUT_ENTRIES,
        height: 1,
        depth_or_array_layers: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_bytes_are_the_identity_table() {
        let lut = identity_lut_bytes();
        assert_eq!(lut.len(), 1024);
        for (i, entry) in lut.chunks_exact(4).enumerate() {
            let b = i as u8;
            assert_eq!(entry, [b, b, b, 255], "entry {i}");
        }
    }

    #[test]
    fn shaders_embed_the_lut_snippet_verbatim() {
        // Both consumers paste the same read; a one-sided edit (index
        // rounding, binding slot, channel pick) must fail here, not as
        // a silent per-view mismatch on screen.
        assert!(crate::shaders::MESH_SHADER.contains(DISPLAY_LUT_WGSL));
        assert!(crate::texture_display::TEXTURE_DISPLAY_SHADER.contains(DISPLAY_LUT_WGSL));
    }

    #[test]
    fn lut_index_math_round_trips_every_byte() {
        // CPU mirror of `display_lut_index` on the unorm8 decode of each
        // byte: the 8-bit paint target's level i must address entry i
        // (the exactness the golden relies on).
        let index = |c: f32| (c.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as i32;
        for i in 0..=255u8 {
            assert_eq!(index(f32::from(i) / 255.0), i32::from(i));
        }
        assert_eq!(index(-0.5), 0);
        assert_eq!(index(7.0), 255);
    }
}

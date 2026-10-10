//! Tangent-space normal-map baking from a position/normal map.
//!
//! Wave 3 bake path (docs/specs/requirements.md §3): where [`crate::ao`]
//! raycasts occlusion against the mesh, this module converts the mesh's
//! own UV→world position map (see [`crate::position`]) into a
//! tangent-space normal map — per-texel world normals transformed by a
//! screen-space tangent frame built from the position map's own
//! x-derivative, through
//! `umber_gpu::bake_shaders::TANGENT_NORMAL_BAKE_SHADER`. See
//! `LANDING_NOTES_NORMAL_MAP.md` for the screen-space TBN derivation, its
//! UV-alignment assumption, the wave-4 per-texel UV-derivative plan, and
//! the DirectX/OpenGL flip convention.
//!
//! Composition mirrors [`crate::ao::bake_ao_mesh`] exactly: rasterize the
//! mesh's own UV layout into a world-position + face-normal map (see
//! [`crate::position::bake_position_and_normal`]), then bind both textures
//! read-only into the second pass — only the second pass differs (a
//! per-texel TBN transform instead of hemisphere raycasting, so no
//! triangle buffer, no raycount uniform, no atomics).

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::TANGENT_NORMAL_BAKE_SHADER;
use umber_mesh::MeshData;

use crate::ao::{AoBakeError, BakeTarget};
use crate::position::{bake_position_and_normal, PositionMapError};

/// Parameters for [`bake_tangent_normal_mesh`].
#[derive(Debug, Clone, Copy)]
pub struct TangentNormalParams {
    /// When `false` (default), bakes the OpenGL convention (green up =
    /// `+Y` up). When `true`, inverts the green channel after encoding
    /// for the DirectX convention (green down = `-Y` up).
    pub directx_y_flip: bool,
}

impl TangentNormalParams {
    /// Builds params with an explicit [`TangentNormalParams::directx_y_flip`]
    /// choice.
    pub fn new(directx_y_flip: bool) -> Self {
        Self { directx_y_flip }
    }
}

impl Default for TangentNormalParams {
    /// OpenGL convention ([`TangentNormalParams::directx_y_flip`] is
    /// `false`).
    fn default() -> Self {
        Self::new(false)
    }
}

/// Errors from [`bake_tangent_normal_mesh`].
#[derive(Debug, thiserror::Error)]
pub enum TangentNormalBakeError {
    /// The requested bake-target width or height was zero.
    #[error("bake target dimensions must be non-zero (got {width}x{height})")]
    EmptyTarget {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// [`bake_tangent_normal_mesh`]'s position-map pass (see
    /// [`crate::position`]) failed before the TBN transform ever started.
    #[error("position map: {0}")]
    PositionMap(#[from] PositionMapError),
    /// Reading the baked texture back to CPU memory failed.
    #[error("bake-target readback failed: {0}")]
    Readback(String),
}

/// GPU-side params uniform, byte-identical to `TANGENT_NORMAL_BAKE_SHADER`'s
/// WGSL `TangentNormalParams` struct (`width`, `height`, `flip_y` as a
/// `u32` word — WGSL uniforms have no bools — plus one `u32` pad — 16
/// bytes, already a multiple of WGSL's 16-byte uniform alignment, so no
/// further padding is needed).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TangentNormalUniform {
    width: u32,
    height: u32,
    flip_y: u32,
    _pad: u32,
}

const _: () = assert!(
    size_of::<TangentNormalUniform>() == 16,
    "TangentNormalUniform must match TANGENT_NORMAL_BAKE_SHADER's WGSL TangentNormalParams layout"
);

/// Maps [`bake_tangent_normal_mesh`]'s dimensions and
/// [`TangentNormalParams`] onto the GPU uniform word-for-word
/// (`directx_y_flip` becomes `1`/`0` — factored out so the bool-to-word
/// mapping is unit-testable without a device).
fn uniform_for(width: u32, height: u32, params: &TangentNormalParams) -> TangentNormalUniform {
    TangentNormalUniform {
        width,
        height,
        flip_y: u32::from(params.directx_y_flip),
        _pad: 0,
    }
}

/// Checks [`bake_tangent_normal_mesh`]'s target dimensions independently of
/// any GPU call, so the validation itself is unit-testable without a
/// device — matching `ao::validate`'s convention. Mesh validation (empty
/// mesh, over the triangle budget, malformed indices) is left to
/// [`bake_position_and_normal`] and surfaces as
/// [`TangentNormalBakeError::PositionMap`].
fn validate(width: u32, height: u32) -> Result<(), TangentNormalBakeError> {
    if width == 0 || height == 0 {
        return Err(TangentNormalBakeError::EmptyTarget { width, height });
    }
    Ok(())
}

/// Bakes a tangent-space normal map for `mesh` against *itself*: first
/// rasterizes `mesh`'s own UV layout into a world-position + face-normal
/// map (see [`crate::position::bake_position_and_normal`]), then
/// transforms each covered texel's world normal by a screen-space tangent
/// frame built from the position map's own x-derivative — the same
/// two-pass composition [`crate::ao::bake_ao_mesh`] uses, with the
/// hemisphere-raycast second pass swapped for the TBN transform (see
/// `umber_gpu::bake_shaders::TANGENT_NORMAL_BAKE_SHADER`'s doc comment for
/// the frame derivation, its UV-alignment assumption, and the encoding).
///
/// Returns the full `width * height * 4` RGBA8 bytes in row-major order:
/// `rgb` is `tangent_normal * 0.5 + 0.5` (flat-in-tangent-space is
/// `(128, 128, 255)`; `params.directx_y_flip` inverts green for the
/// DirectX convention), and alpha is coverage (`0` means the position pass
/// found no UV triangle covering that texel, `255` means a tangent normal
/// was actually baked). Collapsing to RGB would make "uncovered"
/// indistinguishable from "flat black" — the same reason `bake_ao_mesh`
/// returns full RGBA8.
///
/// Builds a fresh compute pipeline on every call rather than caching one
/// across bakes — reasonable for this slice's one-shot entry point, the
/// same tradeoff `ao::run` documents.
///
/// # Errors
///
/// Returns [`TangentNormalBakeError::EmptyTarget`] if `width`/`height` is
/// zero, [`TangentNormalBakeError::PositionMap`] wrapping whatever
/// [`crate::position::bake_position_and_normal`] rejected (empty mesh,
/// over the triangle budget, zero-sized target, malformed indices, or a
/// GPU readback failure), or [`TangentNormalBakeError::Readback`] if the
/// GPU readback fails.
pub fn bake_tangent_normal_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
    params: &TangentNormalParams,
) -> Result<Vec<u8>, TangentNormalBakeError> {
    profiling::scope!("bake_pass");
    validate(width, height)?;

    let position_map = bake_position_and_normal(device, queue, mesh, width, height)?;

    let uniform = uniform_for(width, height, params);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_tangent_normal_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Reuses `ao::BakeTarget`'s `Rgba8Unorm` texture/view/readback
    // machinery (including the 256-byte-row-pitch de-padding) rather than
    // re-inventing a second storage-texture type — the same reuse `ao`
    // itself makes over `umber_gpu::PaintTarget`.
    let target = BakeTarget::new(device, width, height);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_tangent_normal_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TANGENT_NORMAL_BAKE_SHADER)),
    });

    // `TANGENT_NORMAL_BAKE_SHADER`'s binding set mirrors `bake_ao_mesh`'s
    // `cs_main_from_position` shape — write-only output texture plus the
    // two read-only position/normal textures — minus the triangle storage
    // buffer and raycast uniform this pass has no use for, so the bindings
    // pack contiguously instead of inheriting the AO pass's numbering (the
    // same packing `CURVATURE_BAKE_SHADER` uses).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_tangent_normal_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<TangentNormalUniform>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_bake_tangent_normal_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_tangent_normal_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_tangent_normal_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(target.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&position_map.position_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&position_map.normal_view),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_tangent_normal_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_tangent_normal_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per texel, matching `cs_main_from_position`'s
        // dispatch convention (`TANGENT_NORMAL_BAKE_SHADER` runs
        // `@workgroup_size(1)`, so `workgroup_id.xy` is directly the texel
        // coordinate).
        pass.dispatch_workgroups(width, height, 1);
    }
    queue.submit(Some(encoder.finish()));

    target.read_back_rgba8(device, queue).map_err(|e| match e {
        AoBakeError::Readback(msg) => TangentNormalBakeError::Readback(msg),
        other => TangentNormalBakeError::Readback(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_normal_uniform_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<TangentNormalUniform>(), 16);
    }

    #[test]
    fn tangent_normal_params_default_is_opengl() {
        let params = TangentNormalParams::default();
        assert!(!params.directx_y_flip);
        assert!(!TangentNormalParams::new(false).directx_y_flip);
        assert!(TangentNormalParams::new(true).directx_y_flip);
    }

    #[test]
    fn uniform_for_maps_flip_bool_to_word() {
        let off = uniform_for(32, 16, &TangentNormalParams::new(false));
        assert_eq!((off.width, off.height, off.flip_y), (32, 16, 0));
        let on = uniform_for(32, 16, &TangentNormalParams::new(true));
        assert_eq!((on.width, on.height, on.flip_y), (32, 16, 1));
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let err = validate(0, 32).unwrap_err();
        assert!(matches!(
            err,
            TangentNormalBakeError::EmptyTarget {
                width: 0,
                height: 32
            }
        ));
    }

    #[test]
    fn validate_accepts_well_formed_inputs() {
        assert!(validate(32, 32).is_ok());
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;
        use crate::position::{bake_world_normal_map, PositionMapParams};

        /// Requests a plain default device — this pass needs no special
        /// feature (write-only `Rgba8Unorm` storage is core WebGPU, and the
        /// `Rgba32Float` inputs are sampled read-only, not read-written —
        /// see `bake_shaders::TANGENT_NORMAL_BAKE_SHADER`'s doc comment).
        /// Skips gracefully (mirroring `umber_gpu::paint`'s test
        /// convention) if no adapter is available in this environment.
        fn try_request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
            let instance = wgpu::Instance::default();
            let Ok(adapter) = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            ) else {
                eprintln!("skipping: no wgpu adapter available");
                return None;
            };
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
        }

        /// Two-triangle quad spanning world `[-1, 1]` with full-`[0, 1]`
        /// UVs — the same shape as `position::tests::full_uv_quad`,
        /// reproduced here since that fixture is private to its own test
        /// module. The quad's UV→world map is the exact affine
        /// `pos = (2u - 1, 2v - 1, 0)`, so `dP/dx = +x` and the
        /// screen-space frame is `T = +x`, `B = +y`, `N = +z`.
        fn full_uv_quad() -> MeshData {
            MeshData {
                positions: vec![
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                normals: vec![[0.0, 0.0, 1.0]; 4],
                uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                indices: vec![0, 1, 2, 0, 2, 3],
                material_names: vec![],
            }
        }

        /// Slanted quad: the same full-`[0, 1]` UV chart as
        /// [`full_uv_quad`], but the world positions vary in `x` and `z`
        /// (`pos = (2u - 1, 0, 2v - 1)` — a horizontal plane) with the
        /// winding below giving the face normal `(0, -1, 0)`. The world
        /// normal map is `y`-dominant while the tangent-space bake must
        /// stay blue-dominant, proving the TBN transform actually ran
        /// rather than passing the world normal through.
        fn slanted_quad() -> MeshData {
            MeshData {
                positions: vec![
                    [-1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                normals: vec![[0.0, -1.0, 0.0]; 4],
                uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                indices: vec![0, 1, 2, 0, 2, 3],
                material_names: vec![],
            }
        }

        /// Reads texel `(x, y)` of a `width`-wide RGBA8 `Vec<u8>` (as
        /// returned by [`bake_tangent_normal_mesh`]).
        fn texel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = ((y * width + x) * 4) as usize;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        /// Reads texel `(x, y)` of a `width`-wide RGBA-`f32` `Vec<f32>`
        /// (as returned by [`bake_world_normal_map`]).
        fn texel_f32(map: &[f32], width: u32, x: u32, y: u32) -> [f32; 4] {
            let i = ((y * width + x) * 4) as usize;
            [map[i], map[i + 1], map[i + 2], map[i + 3]]
        }

        const SIZE: u32 = 32;

        /// A flat quad facing `+z` has tangent normal `(0, 0, 1)` in its
        /// own screen-space frame (`T = +x`, `B = +y` from `dP/dx`), which
        /// encodes to `(128, 128, 255)`. Red/green assert the two-value
        /// band, not one exact byte: `0.5` (`127.5` in byte units) rounds
        /// to `127` or `128` depending on the driver's `Rgba8Unorm`
        /// quantization, while blue (`1.0`) and alpha are exact. Bounds
        /// everywhere else would hide a dropped transform; exactness here
        /// is analytic (the dots are `0 ± 1e-7` in `f32`).
        #[test]
        fn flat_quad_bakes_plus_z_tangent_everywhere() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = TangentNormalParams::default();
            let bytes =
                bake_tangent_normal_mesh(&device, &queue, &full_uv_quad(), SIZE, SIZE, &params)
                    .expect("bake should succeed");
            assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize);

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let px = texel(&bytes, SIZE, x, y);
                    assert_eq!(px[3], 255, "texel ({x}, {y}) must be covered");
                    assert!(
                        (127..=128).contains(&px[0]),
                        "texel ({x}, {y}) red should be ~128 (tangent x ~= 0): {}",
                        px[0]
                    );
                    assert!(
                        (127..=128).contains(&px[1]),
                        "texel ({x}, {y}) green should be ~128 (tangent y ~= 0): {}",
                        px[1]
                    );
                    assert_eq!(
                        px[2], 255,
                        "texel ({x}, {y}) blue should be 255 (tangent z ~= 1): {}",
                        px[2]
                    );
                }
            }
        }

        /// The DirectX flip inverts the *encoded* green channel, but a
        /// flat quad's tangent `y` is `0 ± 1e-7`, i.e. encoded `0.5` — a
        /// fixed point of `g = 1 - g` up to `Rgba8Unorm` quantization. So
        /// the analytically-correct expectation on this geometry is
        /// near-identity (within one quantization step), not the
        /// `255 - 128 = 127` byte inversion a nonzero tangent `y` would
        /// show; that case needs wave-4 smooth normals (see
        /// `LANDING_NOTES_NORMAL_MAP.md`). The test pins the flip path as
        /// executed-but-harmless here: same coverage, same blue, green
        /// within one step of the unflipped bake.
        #[test]
        fn directx_flip_is_near_identity_on_flat_quad() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let plain = bake_tangent_normal_mesh(
                &device,
                &queue,
                &full_uv_quad(),
                SIZE,
                SIZE,
                &TangentNormalParams::new(false),
            )
            .expect("bake should succeed");
            let flipped = bake_tangent_normal_mesh(
                &device,
                &queue,
                &full_uv_quad(),
                SIZE,
                SIZE,
                &TangentNormalParams::new(true),
            )
            .expect("bake should succeed");
            assert_eq!(plain.len(), flipped.len());

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let a = texel(&plain, SIZE, x, y);
                    let b = texel(&flipped, SIZE, x, y);
                    assert_eq!(a[3], 255, "texel ({x}, {y}) must be covered");
                    assert_eq!(b[3], 255, "flipped texel ({x}, {y}) must be covered");
                    assert_eq!(b[2], 255, "flipped texel ({x}, {y}) blue must stay 255");
                    for c in 0..2 {
                        assert!(
                            (i16::from(a[c]) - i16::from(b[c])).abs() <= 1,
                            "flipped texel ({x}, {y}) channel {c} should match unflipped {:?} within one step: {}",
                            a,
                            b[c]
                        );
                    }
                }
            }
        }

        /// A slanted quad (world normal `(0, -1, 0)`) must still bake
        /// blue-dominant in tangent space — the facet's own normal is
        /// `(0, 0, 1)` in its own frame by construction — while the world
        /// normal map reads `y`-dominant. Together the two halves prove
        /// the TBN transform ran: a world-normal passthrough would read
        /// green-dominant instead of blue.
        #[test]
        fn slanted_quad_stays_blue_dominant_in_tangent_space() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = TangentNormalParams::default();
            let bytes =
                bake_tangent_normal_mesh(&device, &queue, &slanted_quad(), SIZE, SIZE, &params)
                    .expect("bake should succeed");
            assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize);

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let px = texel(&bytes, SIZE, x, y);
                    assert_eq!(px[3], 255, "texel ({x}, {y}) must be covered");
                    assert!(
                        px[2] > 190,
                        "texel ({x}, {y}) blue should stay dominant (near-flat in tangent space): {}",
                        px[2]
                    );
                    assert!(
                        (127..=128).contains(&px[0]),
                        "texel ({x}, {y}) red should be ~128: {}",
                        px[0]
                    );
                    assert!(
                        (127..=128).contains(&px[1]),
                        "texel ({x}, {y}) green should be ~128: {}",
                        px[1]
                    );
                }
            }

            // The world map of the same mesh is y-dominant, not z: the
            // tangent bake above differs from it, so the transform
            // actually happened.
            let world = bake_world_normal_map(
                &device,
                &queue,
                &slanted_quad(),
                &PositionMapParams {
                    width: SIZE,
                    height: SIZE,
                },
            )
            .expect("world bake should succeed");
            let n = texel_f32(&world, SIZE, SIZE / 2, SIZE / 2);
            assert_eq!(n[3], 1.0, "center texel must be covered");
            assert!(
                n[0].abs() < 1e-4 && (n[1] + 1.0).abs() < 1e-4 && n[2].abs() < 1e-4,
                "world normal should be (0, -1, 0), got {n:?}"
            );
        }

        #[test]
        fn bake_tangent_normal_mesh_rejects_zero_sized_target() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = TangentNormalParams::default();
            let err = bake_tangent_normal_mesh(&device, &queue, &full_uv_quad(), 0, 32, &params)
                .unwrap_err();
            assert!(matches!(
                err,
                TangentNormalBakeError::EmptyTarget {
                    width: 0,
                    height: 32
                }
            ));
        }

        #[test]
        fn bake_tangent_normal_mesh_propagates_position_map_errors() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = TangentNormalParams::default();
            let err =
                bake_tangent_normal_mesh(&device, &queue, &MeshData::default(), 8, 8, &params)
                    .unwrap_err();
            assert!(matches!(
                err,
                TangentNormalBakeError::PositionMap(crate::position::PositionMapError::EmptyMesh)
            ));
        }
    }
}

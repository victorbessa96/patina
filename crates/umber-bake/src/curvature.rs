//! Screen-space curvature baking from a position/normal map.
//!
//! Wave 3 bake path (docs/specs/requirements.md §3): where `ao` raycasts
//! occlusion against the mesh, this module estimates *signed mean
//! curvature* per texel from the mesh's own UV→world position map (see
//! [`crate::position`]) — no raycast, no acceleration structure, just the
//! 4-neighborhood's normals and positions through
//! `umber_gpu::bake_shaders::CURVATURE_BAKE_SHADER`. See
//! `LANDING_NOTES_CURVATURE.md` for the estimator's sources, the MeshLab
//! sign convention (convex = negative = darker), and the UV-seam artifact
//! this estimator cannot avoid.

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::CURVATURE_BAKE_SHADER;
use umber_mesh::MeshData;

use crate::ao::{AoBakeError, BakeTarget};
use crate::position::{bake_position_and_normal, PositionMapError};

/// Parameters for [`bake_curvature_mesh`].
#[derive(Debug, Clone, Copy)]
pub struct CurvatureParams {
    /// Gain applied to the averaged directional curvature before the
    /// `[-1, 1]` clamp. `1.0` maps the estimator's raw output directly;
    /// raise it to push soft bevels toward full black/white, lower it to
    /// keep gentle curvature in the mid-gray range.
    pub strength: f32,
}

impl CurvatureParams {
    /// The default gain used by [`CurvatureParams::new`]: the estimator's
    /// raw output, unscaled.
    pub const DEFAULT_STRENGTH: f32 = 1.0;

    /// Builds params with [`CurvatureParams::DEFAULT_STRENGTH`] gain.
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }
}

impl Default for CurvatureParams {
    fn default() -> Self {
        Self::new(Self::DEFAULT_STRENGTH)
    }
}

/// Errors from [`bake_curvature_mesh`].
#[derive(Debug, thiserror::Error)]
pub enum CurvatureBakeError {
    /// The requested bake-target width or height was zero.
    #[error("bake target dimensions must be non-zero (got {width}x{height})")]
    EmptyTarget {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// [`bake_curvature_mesh`]'s position-map pass (see
    /// [`crate::position`]) failed before curvature estimation ever started.
    #[error("position map: {0}")]
    PositionMap(#[from] PositionMapError),
    /// Reading the baked texture back to CPU memory failed.
    #[error("bake-target readback failed: {0}")]
    Readback(String),
}

/// GPU-side params uniform, byte-identical to `CURVATURE_BAKE_SHADER`'s
/// WGSL `CurvatureParams` struct (`width`, `height`, `strength`, one `f32`
/// pad — 16 bytes, already a multiple of WGSL's 16-byte uniform alignment,
/// so no further padding is needed).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CurvatureUniform {
    width: u32,
    height: u32,
    strength: f32,
    _pad: f32,
}

const _: () = assert!(
    size_of::<CurvatureUniform>() == 16,
    "CurvatureUniform must match CURVATURE_BAKE_SHADER's WGSL CurvatureParams layout"
);

/// Checks [`bake_curvature_mesh`]'s target dimensions independently of any
/// GPU call, so the validation itself is unit-testable without a device —
/// matching `ao::validate`'s convention. Mesh validation (empty mesh, over
/// the triangle budget, malformed indices) is left to
/// [`bake_position_and_normal`] and surfaces as
/// [`CurvatureBakeError::PositionMap`].
fn validate(width: u32, height: u32) -> Result<(), CurvatureBakeError> {
    if width == 0 || height == 0 {
        return Err(CurvatureBakeError::EmptyTarget { width, height });
    }
    Ok(())
}

/// Bakes signed screen-space curvature for `mesh` against *itself*: first
/// rasterizes `mesh`'s own UV layout into a world-position + face-normal
/// map (see [`crate::position::bake_position_and_normal`]), then estimates
/// per-texel curvature from each covered texel's 4-neighborhood — the same
/// two-pass composition [`crate::ao::bake_ao_mesh`] uses, with the
/// hemisphere-raycast second pass swapped for the curvature estimator (see
/// `umber_gpu::bake_shaders::CURVATURE_BAKE_SHADER`'s doc comment for the
/// estimator and its sign convention).
///
/// Returns the full `width * height * 4` RGBA8 bytes in row-major order:
/// `rgb` is `(curvature + 1) / 2` grayscale (mid-gray is flat, darker is
/// convex, brighter is concave — MeshLab sign convention), and alpha is
/// coverage (`0` means the position pass found no UV triangle covering
/// that texel, `255` means curvature was actually estimated). Collapsing
/// to a single channel would make "uncovered" indistinguishable from
/// "maximally convex" — the same reason `bake_ao_mesh` returns full RGBA8.
///
/// Builds a fresh compute pipeline on every call rather than caching one
/// across bakes — reasonable for this slice's one-shot entry point, the
/// same tradeoff `ao::run` documents.
///
/// # Errors
///
/// Returns [`CurvatureBakeError::EmptyTarget`] if `width`/`height` is zero,
/// [`CurvatureBakeError::PositionMap`] wrapping whatever
/// [`crate::position::bake_position_and_normal`] rejected (empty mesh,
/// over the triangle budget, zero-sized target, malformed indices, or a
/// GPU readback failure), or [`CurvatureBakeError::Readback`] if the GPU
/// readback fails.
pub fn bake_curvature_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
    params: &CurvatureParams,
) -> Result<Vec<u8>, CurvatureBakeError> {
    profiling::scope!("bake_pass");
    validate(width, height)?;

    let position_map = bake_position_and_normal(device, queue, mesh, width, height)?;

    let uniform = CurvatureUniform {
        width,
        height,
        strength: params.strength,
        _pad: 0.0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_curvature_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Reuses `ao::BakeTarget`'s `Rgba8Unorm` texture/view/readback
    // machinery (including the 256-byte-row-pitch de-padding) rather than
    // re-inventing a second storage-texture type — the same reuse `ao`
    // itself makes over `umber_gpu::PaintTarget`.
    let target = BakeTarget::new(device, width, height);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_curvature_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CURVATURE_BAKE_SHADER)),
    });

    // `CURVATURE_BAKE_SHADER`'s binding set mirrors `bake_ao_mesh`'s
    // `cs_main_from_position` shape — write-only output texture plus the
    // two read-only position/normal textures — minus the triangle storage
    // buffer and raycast uniform this pass has no use for, so the bindings
    // pack contiguously instead of inheriting the AO pass's numbering.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_curvature_bind_group_layout"),
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
                    min_binding_size: NonZeroU64::new(size_of::<CurvatureUniform>() as u64),
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
        label: Some("umber_bake_curvature_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_curvature_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_curvature_bind_group"),
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
        label: Some("umber_bake_curvature_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_curvature_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per texel, matching `cs_main_from_position`'s
        // dispatch convention (`CURVATURE_BAKE_SHADER` runs
        // `@workgroup_size(1)`, so `workgroup_id.xy` is directly the texel
        // coordinate).
        pass.dispatch_workgroups(width, height, 1);
    }
    queue.submit(Some(encoder.finish()));

    target.read_back_rgba8(device, queue).map_err(|e| match e {
        AoBakeError::Readback(msg) => CurvatureBakeError::Readback(msg),
        other => CurvatureBakeError::Readback(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curvature_uniform_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<CurvatureUniform>(), 16);
    }

    #[test]
    fn curvature_params_new_uses_default_strength() {
        let params = CurvatureParams::new(2.5);
        assert_eq!(params.strength, 2.5);
        assert_eq!(CurvatureParams::DEFAULT_STRENGTH, 1.0);
        assert_eq!(CurvatureParams::default().strength, 1.0);
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let err = validate(0, 32).unwrap_err();
        assert!(matches!(
            err,
            CurvatureBakeError::EmptyTarget {
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

        /// Requests a plain default device — this pass needs no special
        /// feature (write-only `Rgba8Unorm` storage is core WebGPU, and the
        /// `Rgba32Float` inputs are sampled read-only, not read-written —
        /// see `bake_shaders::CURVATURE_BAKE_SHADER`'s doc comment). Skips
        /// gracefully (mirroring `umber_gpu::paint`'s test convention) if
        /// no adapter is available in this environment.
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
        /// UVs — the same shape as `position::tests::gpu`'s `full_uv_quad`,
        /// reproduced here since that fixture is private to its own test
        /// module.
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

        /// "Tent" (roof) mesh: two quads meeting at a convex ridge along
        /// world `x = 0`, extruded along `y`. The ridge is a straight UV
        /// line at `u = 0.5` (interior, shared by both quads — no UV seam),
        /// so texels adjacent to the ridge see a neighbor across it with a
        /// ~90°-different face normal, while texels farther out sit on a
        /// perfectly flat slope. Winding gives outward (upward-facing)
        /// normals on both slopes: left `(-0.707, 0, 0.707)`, right
        /// `(0.707, 0, 0.707)`.
        fn tent_mesh() -> MeshData {
            MeshData {
                positions: vec![
                    [-1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [0.0, -1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                normals: vec![[0.0, 0.0, 1.0]; 6],
                uvs: vec![
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 0.0],
                    [0.5, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
                indices: vec![0, 3, 1, 0, 2, 3, 2, 4, 5, 2, 5, 3],
                material_names: vec![],
            }
        }

        /// Reads texel `(x, y)` of a `width`-wide RGBA8 `Vec<u8>` (as
        /// returned by [`bake_curvature_mesh`]).
        fn texel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = ((y * width + x) * 4) as usize;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        const SIZE: u32 = 32;

        /// A flat quad has zero curvature everywhere: every covered texel
        /// must read mid-gray (`(0 + 1) / 2 -> 127 or 128` after `Rgba8Unorm`
        /// quantization) with full coverage alpha. Bounds, not exact
        /// values: the band admits either rounding of `127.5` plus float
        /// slack, while still catching a sign flip (0/255) or a dropped
        /// coverage flag.
        #[test]
        fn flat_quad_bakes_mid_gray_everywhere() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = CurvatureParams::default();
            let bytes = bake_curvature_mesh(&device, &queue, &full_uv_quad(), SIZE, SIZE, &params)
                .expect("bake should succeed");
            assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize);

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let px = texel(&bytes, SIZE, x, y);
                    assert_eq!(px[3], 255, "texel ({x}, {y}) must be covered");
                    for (c, channel) in px.iter().enumerate().take(3) {
                        assert!(
                            (120..=136).contains(channel),
                            "texel ({x}, {y}) channel {c} should be ~128 (zero curvature): {}",
                            channel
                        );
                    }
                }
            }
        }

        /// The tent's convex ridge must bake clearly darker than its flat
        /// slopes. At 32x32 the ridge (`u = 0.5`) falls exactly on the
        /// boundary between texel columns 15 and 16, so both columns sit
        /// one texel off the crease and each sees an across-ridge neighbor
        /// whose face normal differs by ~90°: `|dn| ~= 1.41` over a
        /// `~0.06`-world-unit step gives `k ~= 22`, saturating the `[-1, 1]`
        /// clamp to pure convex-black even after averaging with the three
        /// same-slope neighbors. Slope texels far from the ridge (columns 4
        /// and 27) have identical normals all around and must read mid-gray
        /// like the flat-quad test. All bounds, not exact values.
        #[test]
        fn tent_crease_bakes_darker_than_flat_slopes() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = CurvatureParams::default();
            let bytes = bake_curvature_mesh(&device, &queue, &tent_mesh(), SIZE, SIZE, &params)
                .expect("bake should succeed");
            assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize);

            // Flat slopes first: mid-gray with full coverage, same band as
            // the flat-quad test (these texels' neighborhoods never touch
            // the ridge).
            for (x, y) in [(4, 16), (27, 16), (4, 8), (27, 24)] {
                let px = texel(&bytes, SIZE, x, y);
                assert_eq!(px[3], 255, "slope texel ({x}, {y}) must be covered");
                assert!(
                    (120..=136).contains(&px[0]),
                    "slope texel ({x}, {y}) should be ~128 (flat): {}",
                    px[0]
                );
            }

            // Crease columns on both sides of the ridge, sampled at several
            // rows so the test doesn't hinge on one texel: convex means
            // darker than 100 on the MeshLab convention this pass uses.
            let mut crease_max: u8 = 0;
            for y in [8, 16, 24] {
                for x in [15, 16] {
                    let px = texel(&bytes, SIZE, x, y);
                    assert_eq!(px[3], 255, "crease texel ({x}, {y}) must be covered");
                    crease_max = crease_max.max(px[0]);
                    assert!(
                        px[0] < 100,
                        "crease texel ({x}, {y}) should be clearly darker than mid-gray: {}",
                        px[0]
                    );
                }
            }

            // The required difference, stated directly: even the brightest
            // crease texel must sit well below a flat slope texel.
            let flat = texel(&bytes, SIZE, 4, 16)[0];
            assert!(
                i16::from(flat) - i16::from(crease_max) >= 28,
                "crease ({crease_max}) must be clearly darker than flat slope ({flat})"
            );
        }

        #[test]
        fn bake_curvature_mesh_rejects_zero_sized_target() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = CurvatureParams::default();
            let err =
                bake_curvature_mesh(&device, &queue, &full_uv_quad(), 0, 32, &params).unwrap_err();
            assert!(matches!(
                err,
                CurvatureBakeError::EmptyTarget {
                    width: 0,
                    height: 32
                }
            ));
        }

        #[test]
        fn bake_curvature_mesh_propagates_position_map_errors() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = CurvatureParams::default();
            let err = bake_curvature_mesh(&device, &queue, &MeshData::default(), 8, 8, &params)
                .unwrap_err();
            assert!(matches!(
                err,
                CurvatureBakeError::PositionMap(crate::position::PositionMapError::EmptyMesh)
            ));
        }
    }
}

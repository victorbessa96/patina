//! Thickness (local solid depth) baking via inward compute-shader raycasting.
//!
//! Wave 3 bake path (docs/specs/requirements.md §3): where [`crate::ao`]
//! raycasts occlusion outward into the scene, this module measures how much
//! solid sits *behind* each texel by raycasting into the mesh — the
//! distance from a texel's surface point to the opposite face, through
//! `umber_gpu::bake_shaders::THICKNESS_BAKE_SHADER`. See
//! `LANDING_NOTES_THICKNESS.md` for the estimator's sources, the
//! miss-means-thick convention (open meshes read thick), and the
//! ray-count/bias tradeoffs.
//!
//! Composition mirrors [`crate::ao::bake_ao_mesh`] exactly: rasterize the
//! mesh's own UV layout into a world-position + face-normal map (see
//! [`crate::position::bake_position_and_normal`]), then bind both textures
//! read-only into the second pass — only the second pass differs (minimum
//! inward hit distance instead of outward occlusion fraction, so the
//! workgroup fan-out collapses to a serial per-texel loop; see the shader's
//! doc comment for why there is no atomic float-min).

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use glam::Vec3;
use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::THICKNESS_BAKE_SHADER;
use umber_mesh::MeshData;

use crate::ao::{AoBakeError, BakeTarget};
use crate::position::{bake_position_and_normal, PositionMapError};

/// Parameters for [`bake_thickness_mesh`].
#[derive(Debug, Clone, Copy)]
pub struct ThicknessParams {
    /// Maximum ray travel distance, in world units; geometry beyond this
    /// distance along a ray is treated as if it weren't there (that ray
    /// simply never lowers the texel's minimum, and a texel whose every
    /// ray misses reads `1.0` — see the module docs).
    pub max_distance: f32,
    /// Offset *into* the mesh (along the negative normal) the ray origin is
    /// nudged by, so an inward ray never re-hits its own originating
    /// surface at `t ≈ 0` — the mirror of [`crate::ao::AoBakeParams`]'s
    /// outward bias.
    pub bias: f32,
    /// Inward hemisphere rays cast per texel.
    pub rays: u32,
}

impl ThicknessParams {
    /// The default hemisphere ray count, matching
    /// [`crate::ao::AoBakeParams::DEFAULT_RAYS`].
    pub const DEFAULT_RAYS: u32 = 16;
    /// The default maximum ray travel distance in world units.
    pub const DEFAULT_MAX_DISTANCE: f32 = 10.0;
    /// The default inward ray-origin offset in world units.
    pub const DEFAULT_BIAS: f32 = 0.01;

    /// Builds params with [`ThicknessParams::DEFAULT_RAYS`] rays; override
    /// the `rays` field afterward (struct-update syntax) for a different
    /// sample count — the same convention
    /// [`crate::ao::AoBakeParams::new`] uses.
    pub fn new(max_distance: f32, bias: f32) -> Self {
        Self {
            max_distance,
            bias,
            rays: Self::DEFAULT_RAYS,
        }
    }
}

impl Default for ThicknessParams {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_DISTANCE, Self::DEFAULT_BIAS)
    }
}

/// Errors from [`bake_thickness_mesh`].
#[derive(Debug, thiserror::Error)]
pub enum ThicknessBakeError {
    /// `params.rays` was zero.
    #[error("ray count must be at least 1")]
    InvalidRayCount,
    /// The requested bake-target width or height was zero.
    #[error("bake target dimensions must be non-zero (got {width}x{height})")]
    EmptyTarget {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// [`bake_thickness_mesh`]'s position-map pass (see
    /// [`crate::position`]) failed before thickness raycasting ever started.
    #[error("position map: {0}")]
    PositionMap(#[from] PositionMapError),
    /// Reading the baked texture back to CPU memory failed.
    #[error("bake-target readback failed: {0}")]
    Readback(String),
}

/// GPU-side triangle layout: three vertex positions plus a face normal,
/// each padded to `vec4<f32>` to match `THICKNESS_BAKE_SHADER`'s WGSL `Tri`
/// struct — byte-identical to `ao`'s private `GpuTriangle`, since both
/// shaders share the same ray-machinery input.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    normal: [f32; 4],
}

const _: () = assert!(
    size_of::<GpuTriangle>() == 64,
    "GpuTriangle must match THICKNESS_BAKE_SHADER's WGSL Tri layout"
);

/// GPU-side params uniform, byte-identical to `THICKNESS_BAKE_SHADER`'s
/// WGSL `ThicknessParams` struct (`rays`, `max_distance`, `bias`,
/// `tri_count` — four 4-byte scalars, 16 bytes, already a multiple of
/// WGSL's 16-byte uniform-struct alignment, so no padding field is needed).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ThicknessUniform {
    rays: u32,
    max_distance: f32,
    bias: f32,
    tri_count: u32,
}

const _: () = assert!(
    size_of::<ThicknessUniform>() == 16,
    "ThicknessUniform must match THICKNESS_BAKE_SHADER's WGSL ThicknessParams layout"
);

/// Builds the GPU triangle list from `mesh`'s positions/indices. The face
/// normal is recomputed from the triangle's own vertex positions (not read
/// from `mesh.normals`) so it is always available even for meshes imported
/// without vertex normals — the same choice `ao::build_triangles` makes,
/// for the same reason. This runs only after
/// [`bake_position_and_normal`] has already validated the index list, so
/// the indices here are known in-range (mirroring `ao::bake_ao_mesh`'s
/// ordering, which builds its triangle buffer after the position pass for
/// exactly this reason).
fn build_triangles(mesh: &MeshData) -> Vec<GpuTriangle> {
    mesh.indices
        .chunks_exact(3)
        .map(|idx| {
            let p0 = mesh.positions[idx[0] as usize];
            let p1 = mesh.positions[idx[1] as usize];
            let p2 = mesh.positions[idx[2] as usize];
            let a = Vec3::from(p0);
            let b = Vec3::from(p1);
            let c = Vec3::from(p2);
            let normal = (b - a).cross(c - a).normalize_or_zero();
            GpuTriangle {
                v0: [p0[0], p0[1], p0[2], 0.0],
                v1: [p1[0], p1[1], p1[2], 0.0],
                v2: [p2[0], p2[1], p2[2], 0.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
            }
        })
        .collect()
}

/// Checks [`bake_thickness_mesh`]'s ray count and target dimensions
/// independently of any GPU call, so the validation itself is unit-testable
/// without a device — matching `curvature::validate`'s convention. Mesh
/// validation (empty mesh, over the triangle budget, malformed indices) is
/// left to [`bake_position_and_normal`] and surfaces as
/// [`ThicknessBakeError::PositionMap`].
fn validate(params: &ThicknessParams, width: u32, height: u32) -> Result<(), ThicknessBakeError> {
    if params.rays == 0 {
        return Err(ThicknessBakeError::InvalidRayCount);
    }
    if width == 0 || height == 0 {
        return Err(ThicknessBakeError::EmptyTarget { width, height });
    }
    Ok(())
}

/// Bakes local thickness for `mesh` against *itself*: first rasterizes
/// `mesh`'s own UV layout into a world-position + face-normal map (see
/// [`crate::position::bake_position_and_normal`]), then casts an inward
/// hemisphere from every covered texel against `mesh`'s own triangle list
/// and records the nearest opposite-face hit — the same two-pass
/// composition [`crate::ao::bake_ao_mesh`] uses, with the outward
/// occlusion-fraction pass swapped for the inward minimum-distance pass
/// (see `umber_gpu::bake_shaders::THICKNESS_BAKE_SHADER`'s doc comment for
/// the estimator, the inward bias, and the miss-means-thick convention).
///
/// Returns the full `width * height * 4` RGBA8 bytes in row-major order:
/// `rgb` is `clamp(min_hit / max_distance, 0, 1)` grayscale (black is
/// paper-thin, white is at-or-beyond `max_distance` — including the
/// all-rays-missed case, which reads as maximally thick), and alpha is
/// coverage (`0` means the position pass found no UV triangle covering
/// that texel, `255` means thickness was actually measured). Collapsing to
/// a single channel would make "uncovered" indistinguishable from
/// "zero thickness" — the same reason `bake_ao_mesh` returns full RGBA8.
///
/// Builds a fresh compute pipeline on every call rather than caching one
/// across bakes — reasonable for this slice's one-shot entry point, the
/// same tradeoff `ao::run` documents.
///
/// # Errors
///
/// Returns [`ThicknessBakeError::InvalidRayCount`] if `params.rays == 0`,
/// [`ThicknessBakeError::EmptyTarget`] if `width`/`height` is zero,
/// [`ThicknessBakeError::PositionMap`] wrapping whatever
/// [`crate::position::bake_position_and_normal`] rejected (empty mesh,
/// over the triangle budget, zero-sized target, malformed indices, or a
/// GPU readback failure), or [`ThicknessBakeError::Readback`] if the GPU
/// readback fails.
pub fn bake_thickness_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
    params: &ThicknessParams,
) -> Result<Vec<u8>, ThicknessBakeError> {
    validate(params, width, height)?;

    let position_map = bake_position_and_normal(device, queue, mesh, width, height)?;

    let triangles = build_triangles(mesh);
    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_thickness_triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let uniform = ThicknessUniform {
        rays: params.rays,
        max_distance: params.max_distance,
        bias: params.bias,
        tri_count: triangles.len() as u32,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_thickness_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Reuses `ao::BakeTarget`'s `Rgba8Unorm` texture/view/readback
    // machinery (including the 256-byte-row-pitch de-padding) rather than
    // re-inventing a second storage-texture type — the same reuse `curvature`
    // makes over `ao`.
    let target = BakeTarget::new(device, width, height);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_thickness_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(THICKNESS_BAKE_SHADER)),
    });

    // `THICKNESS_BAKE_SHADER`'s binding set mirrors `bake_ao_mesh`'s
    // `cs_main_from_position` shape — write-only output texture, raycast
    // uniform, triangle storage buffer, plus the two read-only
    // position/normal textures — including the skipped binding `3` (the
    // plane-path `dims` uniform this mesh-fed pass has no use for), so the
    // two raycast passes stay grep-comparable.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_thickness_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuTriangle>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<ThicknessUniform>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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
        label: Some("umber_bake_thickness_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_thickness_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_thickness_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &triangle_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(target.view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&position_map.position_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&position_map.normal_view),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_thickness_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_thickness_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per texel, matching `cs_main_from_position`'s
        // dispatch convention (`THICKNESS_BAKE_SHADER` runs
        // `@workgroup_size(1)`, so `workgroup_id.xy` is directly the texel
        // coordinate).
        pass.dispatch_workgroups(width, height, 1);
    }
    queue.submit(Some(encoder.finish()));

    target.read_back_rgba8(device, queue).map_err(|e| match e {
        AoBakeError::Readback(msg) => ThicknessBakeError::Readback(msg),
        other => ThicknessBakeError::Readback(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_mesh() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0]; 3],
            indices: vec![0, 1, 2],
            material_names: vec!["m".into()],
        }
    }

    #[test]
    fn gpu_triangle_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<GpuTriangle>(), 64);
    }

    #[test]
    fn thickness_uniform_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<ThicknessUniform>(), 16);
    }

    #[test]
    fn thickness_params_new_defaults_to_sixteen_rays() {
        let params = ThicknessParams::new(10.0, 0.01);
        assert_eq!(params.rays, ThicknessParams::DEFAULT_RAYS);
        assert_eq!(params.rays, 16);
        assert_eq!(params.max_distance, 10.0);
        assert_eq!(params.bias, 0.01);
    }

    #[test]
    fn thickness_params_default_matches_new_with_defaults() {
        let params = ThicknessParams::default();
        assert_eq!(params.rays, ThicknessParams::DEFAULT_RAYS);
        assert_eq!(params.max_distance, ThicknessParams::DEFAULT_MAX_DISTANCE);
        assert_eq!(params.bias, ThicknessParams::DEFAULT_BIAS);
    }

    #[test]
    fn build_triangles_computes_face_normal_from_positions() {
        let mesh = unit_mesh();
        let tris = build_triangles(&mesh);
        assert_eq!(tris.len(), 1);
        // Right-angle triangle in the XY plane, CCW winding -> +Z normal.
        assert!(
            (tris[0].normal[2] - 1.0).abs() < 1e-6,
            "{:?}",
            tris[0].normal
        );
        assert_eq!(tris[0].v0, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(tris[0].v1, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(tris[0].v2, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn validate_rejects_zero_rays() {
        let mut params = ThicknessParams::new(10.0, 0.01);
        params.rays = 0;
        let err = validate(&params, 32, 32).unwrap_err();
        assert!(matches!(err, ThicknessBakeError::InvalidRayCount));
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let params = ThicknessParams::new(10.0, 0.01);
        let err = validate(&params, 0, 32).unwrap_err();
        assert!(matches!(
            err,
            ThicknessBakeError::EmptyTarget {
                width: 0,
                height: 32
            }
        ));
    }

    #[test]
    fn validate_accepts_well_formed_inputs() {
        let params = ThicknessParams::new(10.0, 0.01);
        assert!(validate(&params, 32, 32).is_ok());
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        /// Requests a plain default device — this pass needs no special
        /// feature (write-only `Rgba8Unorm` storage is core WebGPU, and the
        /// `Rgba32Float` inputs are sampled read-only, not read-written —
        /// see `bake_shaders::THICKNESS_BAKE_SHADER`'s doc comment). Skips
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

        /// "Slab": the baked front face (a `[-1, 1]` quad at `z = 0` with
        /// full `[0, 1]` UVs, `+Z` face normal) plus the opposite face (a
        /// larger `[-2, 2]` quad at `z = -2`, i.e. a known `2.0` gap behind
        /// the front face along its inward `-Z` direction) with all four
        /// UVs collapsed to a single point — a zero-UV-area pair that
        /// `POSITION_BAKE_SHADER`'s `abs(denom) >= DET_EPS` check skips
        /// entirely, so it never claims a position-map texel despite being
        /// part of the same `MeshData` (the same trick
        /// `ao::tests::gpu`'s `quad_with_floating_occluder` uses to keep
        /// raycast-only geometry out of the position map).
        ///
        /// The back face is deliberately oversized (`±2` vs the front's
        /// `±1`) so even the most tilted inward rays from front-face edge
        /// texels still land inside it: ray 0 (the most inward ray,
        /// `cos(theta) = 0.96875`) drifts only `~0.51` world units sideways
        /// over the `~2.0` gap, so from any front texel (`|x|, |y| <= 1`)
        /// it lands within `±1.51 < ±2` — every covered texel is guaranteed
        /// at least that hit, and the minimum across the ray set is the
        /// near-vertical distance `~2.05 / 10.0 ~= 0.205` regardless of
        /// position. See `LANDING_NOTES_THICKNESS.md` for the derivation.
        fn slab_mesh() -> MeshData {
            MeshData {
                positions: vec![
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-2.0, -2.0, -2.0],
                    [2.0, -2.0, -2.0],
                    [2.0, 2.0, -2.0],
                    [-2.0, 2.0, -2.0],
                ],
                normals: vec![[0.0, 0.0, 1.0]; 8],
                uvs: vec![
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                material_names: vec![],
            }
        }

        /// Reads texel `(x, y)` of a `width`-wide RGBA8 `Vec<u8>` (as
        /// returned by [`bake_thickness_mesh`]).
        fn texel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = ((y * width + x) * 4) as usize;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        const SIZE: u32 = 32;

        /// A parallel-face slab of known gap `2.0` with `max_distance =
        /// 10.0` must bake `~0.205` (`~52/255`) on every covered texel: the
        /// near-vertical ray-0 hit (`2.0 / cos(theta_0)`, `cos(theta_0) =
        /// 0.96875`) is the minimum across the set for these parallel
        /// planes, and the oversized back face guarantees it hits from
        /// every texel (see `slab_mesh`). Bounds, not exact values: the
        /// `[20, 90]` band (`0.08..0.35`) admits tilt/quantization slack
        /// while pinning the result clearly away from both `0` (no false
        /// thin) and `255` (no false thick) — and, crucially, far below the
        /// open quad's `255`, so the two tests distinguish slab from open.
        #[test]
        fn slab_bakes_gap_over_max_distance_everywhere() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = ThicknessParams {
                rays: 16,
                ..ThicknessParams::new(10.0, 0.01)
            };
            let bytes = bake_thickness_mesh(&device, &queue, &slab_mesh(), SIZE, SIZE, &params)
                .expect("bake should succeed");
            assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize);

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let px = texel(&bytes, SIZE, x, y);
                    assert_eq!(px[3], 255, "texel ({x}, {y}) must be covered");
                    assert!(
                        (20..=90).contains(&px[0]),
                        "texel ({x}, {y}) should read ~0.205 (gap 2.0 / max 10.0): {}",
                        px[0]
                    );
                    assert_eq!(px[0], px[1], "texel ({x}, {y}) must be grayscale");
                    assert_eq!(px[0], px[2], "texel ({x}, {y}) must be grayscale");
                }
            }
        }

        /// An open single quad has no opposite face: every inward ray
        /// misses, so every covered texel must read `1.0` (`255`) — the
        /// miss-means-thick convention. Same mesh shape, same params, same
        /// size as the slab test; the only difference is the missing back
        /// face, so together the two tests prove the baker actually
        /// measures the gap rather than returning a constant.
        #[test]
        fn open_quad_bakes_full_thickness_everywhere() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = ThicknessParams {
                rays: 16,
                ..ThicknessParams::new(10.0, 0.01)
            };
            let bytes = bake_thickness_mesh(&device, &queue, &full_uv_quad(), SIZE, SIZE, &params)
                .expect("bake should succeed");
            assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize);

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let px = texel(&bytes, SIZE, x, y);
                    assert_eq!(px[3], 255, "texel ({x}, {y}) must be covered");
                    assert_eq!(
                        px[0], 255,
                        "open texel ({x}, {y}) should read 1.0 (all rays miss): {}",
                        px[0]
                    );
                }
            }
        }

        #[test]
        fn bake_thickness_mesh_rejects_zero_rays_without_touching_the_gpu_pipeline() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let mut params = ThicknessParams::new(10.0, 0.01);
            params.rays = 0;
            let err =
                bake_thickness_mesh(&device, &queue, &full_uv_quad(), 8, 8, &params).unwrap_err();
            assert!(matches!(err, ThicknessBakeError::InvalidRayCount));
        }

        #[test]
        fn bake_thickness_mesh_rejects_zero_sized_target() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = ThicknessParams::default();
            let err =
                bake_thickness_mesh(&device, &queue, &full_uv_quad(), 0, 32, &params).unwrap_err();
            assert!(matches!(
                err,
                ThicknessBakeError::EmptyTarget {
                    width: 0,
                    height: 32
                }
            ));
        }

        #[test]
        fn bake_thickness_mesh_propagates_position_map_errors() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = ThicknessParams::default();
            let err = bake_thickness_mesh(&device, &queue, &MeshData::default(), 8, 8, &params)
                .unwrap_err();
            assert!(matches!(
                err,
                ThicknessBakeError::PositionMap(crate::position::PositionMapError::EmptyMesh)
            ));
        }
    }
}

//! High→low bake transfer via compute-shader raycasting.
//!
//! Wave-4 item 4, final slice (`docs/specs/high-to-low-transfer-design.md`
//! §"The architecture" point 3): where [`crate::ao`] raycasts a mesh against
//! *itself*, this module bakes a HIGH-poly mesh's surface data into a
//! LOW-poly mesh's UV map — one ray per texel of the LOW map, brute-forced
//! against the HIGH triangle list through
//! `umber_gpu::bake_shaders::TRANSFER_BAKE_SHADER`.
//!
//! Composition mirrors [`crate::ao::bake_ao_mesh`] exactly: rasterize the
//! LOW mesh's own UV layout into a world-position + face-normal map (see
//! [`crate::position::bake_position_and_normal`]), then bind both textures
//! read-only into the second pass — only the second pass differs (a single
//! `-normal` ray per texel against the HIGH mesh's triangles instead of an
//! outward hemisphere against the mesh itself; see that shader's doc comment
//! for the ray math, the clamp gate, and both output encodings).
//!
//! # What v1 is (read this before extending)
//!
//! - **Uniform-selected map**: one shader, one output texture per bake
//!   invocation; [`TransferParams::map`] picks height vs. world normal —
//!   simpler than two shaders, and the bake fn runs the pass once per map.
//! - **World-space normals**: the HIGH face normal is written as-is. The
//!   tangent-space transform (per-texel UV-derivative TBN) is the deferred
//!   fold — see `TRANSFER_BAKE_SHADER`'s doc comment.
//! - **No cage buffer**: [`umber_mesh::bake_support::Cage`] stays CPU-side
//!   for callers; lerping per-vertex offsets mid-shader needs a cage buffer
//!   the app-wiring slice adds later. The front/back
//!   [`umber_mesh::bake_support::TransferClamps`] DO enter v1 (the shader
//!   mirrors `TransferClamps::clamps_hit`; the CPU type itself is not
//!   re-declared here).
//! - **Empty HIGH bakes background**: a HIGH mesh with no triangles means
//!   every texel misses, so the pass short-circuits to an all-zeros map
//!   without touching the GPU dispatch (which also sidesteps binding a
//!   zero-length triangle storage buffer).
//!
//! # Output encoding
//!
//! Full `width * height * 4` RGBA8 bytes in row-major order (the same reason
//! `bake_ao_mesh` returns full RGBA8, not one channel): alpha `0` means
//! background — either the position pass found no LOW UV triangle covering
//! that texel or the transfer ray missed/failed the clamps — alpha `255`
//! means the texel carries transferred HIGH data.

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use glam::Vec3;
use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::TRANSFER_BAKE_SHADER;
use umber_mesh::MeshData;

use crate::ao::{AoBakeError, BakeTarget};
use crate::position::{bake_position_and_normal, PositionMapError, MAX_TRIS_PER_BAKE};

/// Which HIGH-mesh map [`bake_transfer_mesh`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMap {
    /// Normalized hit distance along the ray, centered on the LOW surface
    /// (see `TRANSFER_BAKE_SHADER`'s doc comment for the formula and why the
    /// surface value is `front / (front + back)`, not unconditionally 0.5).
    Height,
    /// The HIGH face normal at the hit, WORLD-space (`n * 0.5 + 0.5`) — the
    /// "object space" normal-space option; tangent-space is the deferred TBN
    /// fold.
    WorldNormal,
}

impl TransferMap {
    /// The `map_mode` uniform word (`TRANSFER_BAKE_SHADER`'s contract:
    /// `0 = height`, `1 = world normal`).
    fn as_u32(self) -> u32 {
        match self {
            TransferMap::Height => 0,
            TransferMap::WorldNormal => 1,
        }
    }
}

/// Parameters for [`bake_transfer_mesh`].
#[derive(Debug, Clone, Copy)]
pub struct TransferParams {
    /// Max hit distance on the ray-origin side of the LOW surface (toward
    /// `origin = low_pos + normal * front_offset`).
    pub front_distance: f32,
    /// Max hit distance on the far side of the LOW surface (past it, along
    /// `-normal`).
    pub back_distance: f32,
    /// Ray-origin push off the LOW surface along `+normal`, in world units.
    pub front_offset: f32,
    /// Which map to bake (see [`TransferMap`]).
    pub map: TransferMap,
    /// Output width in texels.
    pub width: u32,
    /// Output height in texels.
    pub height: u32,
}

impl TransferParams {
    /// Builds params baking the [`TransferMap::Height`] map; chain
    /// [`TransferParams::with_map`] for the world-normal output — the same
    /// `new` + `with_*` convention [`crate::ao::AoBakeParams`] uses for its
    /// bent-normal toggle.
    pub fn new(
        front_distance: f32,
        back_distance: f32,
        front_offset: f32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            front_distance,
            back_distance,
            front_offset,
            map: TransferMap::Height,
            width,
            height,
        }
    }

    /// Returns these params baking `map` instead (see [`TransferMap`]).
    pub fn with_map(mut self, map: TransferMap) -> Self {
        self.map = map;
        self
    }
}

/// Errors from [`bake_transfer_mesh`].
#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    /// `low` has no triangles to bake from.
    #[error("low mesh has no triangles to bake from")]
    Empty,
    /// `high` has more triangles than [`MAX_TRIS_PER_BAKE`].
    #[error("high mesh has {count} triangles, over the {max} budget for this bake path")]
    TooManyHighTris {
        /// The HIGH mesh's actual triangle count.
        count: usize,
        /// The ceiling it exceeded ([`MAX_TRIS_PER_BAKE`]).
        max: u32,
    },
    /// The requested bake-target width or height was zero.
    #[error("bake target dimensions must be non-zero (got {width}x{height})")]
    EmptyTarget {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The front/back clamp pair was negative, non-finite, or totaled to a
    /// non-positive range (which would bake `NaN` heights).
    #[error(
        "transfer clamps must be finite, non-negative, with positive total range \
         (got front {front_distance}, back {back_distance})"
    )]
    InvalidClamps {
        /// The rejected front distance.
        front_distance: f32,
        /// The rejected back distance.
        back_distance: f32,
    },
    /// A HIGH triangle's vertex index ran past the HIGH mesh's positions.
    /// (The LOW mesh's malformed indices surface as
    /// [`TransferError::PositionMap`] instead — the position pass validates
    /// the LOW side; the HIGH side never passes through it.)
    #[error(
        "high triangle {triangle} references position index {index}, \
         but the high mesh only has {len} positions"
    )]
    MalformedHighMesh {
        /// Triangle index (position in the index list / 3).
        triangle: usize,
        /// The out-of-range index.
        index: u32,
        /// The HIGH positions array's actual length.
        len: usize,
    },
    /// [`bake_transfer_mesh`]'s LOW position-map pass (see
    /// [`crate::position`]) failed before transfer raycasting ever started.
    #[error("position map: {0}")]
    PositionMap(#[from] PositionMapError),
    /// Reading the baked texture back to CPU memory failed.
    #[error("bake-target readback failed: {0}")]
    Readback(String),
}

/// GPU-side HIGH-triangle layout: three vertex positions plus a face normal,
/// each padded to `vec4<f32>` to match `TRANSFER_BAKE_SHADER`'s WGSL `Tri`
/// struct — byte-identical to `ao`/`thickness`'s private `GpuTriangle`, since
/// all three shaders share the same brute-force ray-machinery input (NOT
/// `position`'s 80-byte `GpuPosTri`: the HIGH mesh needs no UVs per the
/// design doc, and that builder errors on missing UVs).
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
    "GpuTriangle must match TRANSFER_BAKE_SHADER's WGSL Tri layout"
);

/// GPU-side params uniform, byte-identical to `TRANSFER_BAKE_SHADER`'s WGSL
/// `TransferParams` struct (`front_distance`, `back_distance`,
/// `front_offset`, `high_tri_count`, `map_mode`, `width`, `height`, one
/// `u32` pad — 32 bytes, already a multiple of WGSL's 16-byte
/// uniform-struct alignment, so no further padding is needed; field ORDER
/// here must match the WGSL struct order).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransferUniform {
    front_distance: f32,
    back_distance: f32,
    front_offset: f32,
    high_tri_count: u32,
    map_mode: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

const _: () = assert!(
    size_of::<TransferUniform>() == 32,
    "TransferUniform must match TRANSFER_BAKE_SHADER's WGSL TransferParams layout"
);

/// Maps [`TransferParams`] onto the GPU uniform word-for-word (`map` becomes
/// `0`/`1` via [`TransferMap::as_u32`] — factored out so the enum-to-word
/// mapping is unit-testable without a device, matching
/// `ao::uniform_for`'s convention).
fn uniform_for(params: &TransferParams, high_tri_count: u32) -> TransferUniform {
    TransferUniform {
        front_distance: params.front_distance,
        back_distance: params.back_distance,
        front_offset: params.front_offset,
        high_tri_count,
        map_mode: params.map.as_u32(),
        width: params.width,
        height: params.height,
        _pad: 0,
    }
}

/// Checks [`bake_transfer_mesh`]'s inputs independently of any GPU call, so
/// the validation itself is unit-testable without a device — matching
/// `id::validate`'s convention.
///
/// Check order is deliberate: target dimensions first (they size the
/// background short-circuit too), then the LOW mesh (empty LOW fails the
/// same way regardless of the HIGH side), then the HIGH triangle budget,
/// then the clamp pair. Residual [`PositionMapError`]s from the position
/// pass itself (LOW over the triangle budget, malformed LOW indices, GPU
/// readback) surface as [`TransferError::PositionMap`] — mirroring
/// `thickness::validate`, which likewise leaves mesh validation to the
/// position pass.
fn validate(low: &MeshData, high: &MeshData, params: &TransferParams) -> Result<(), TransferError> {
    if params.width == 0 || params.height == 0 {
        return Err(TransferError::EmptyTarget {
            width: params.width,
            height: params.height,
        });
    }
    if low.triangle_count() == 0 {
        return Err(TransferError::Empty);
    }
    let high_tris = high.triangle_count();
    if high_tris > MAX_TRIS_PER_BAKE as usize {
        return Err(TransferError::TooManyHighTris {
            count: high_tris,
            max: MAX_TRIS_PER_BAKE,
        });
    }
    let (front, back) = (params.front_distance, params.back_distance);
    // NaN/non-finite rejected first, so the final `<=` comparison below is
    // over finite floats (total order — no partial-ord negation involved).
    if !front.is_finite() || !back.is_finite() || front < 0.0 || back < 0.0 || front + back <= 0.0 {
        return Err(TransferError::InvalidClamps {
            front_distance: front,
            back_distance: back,
        });
    }
    Ok(())
}

/// Builds the GPU HIGH-triangle list from `high`'s positions/indices. The
/// face normal is recomputed from the triangle's own vertex positions (not
/// read from `high.normals`) so it is always available even for HIGH meshes
/// imported without vertex normals — the same choice `ao::build_triangles`
/// makes, for the same reason.
///
/// # Errors
///
/// Returns [`TransferError::MalformedHighMesh`] instead of panicking if an
/// index runs past `high.positions` (a corrupt or hand-built HIGH mesh),
/// rather than the slice-index panic a direct `[i]` lookup would produce —
/// the HIGH side never passes through the position pass's validation, so
/// this builder checks its own bounds.
fn build_high_triangles(high: &MeshData) -> Result<Vec<GpuTriangle>, TransferError> {
    high.indices
        .chunks_exact(3)
        .enumerate()
        .map(|(triangle, idx)| {
            let pos = |i: u32| -> Result<[f32; 3], TransferError> {
                high.positions
                    .get(i as usize)
                    .copied()
                    .ok_or(TransferError::MalformedHighMesh {
                        triangle,
                        index: i,
                        len: high.positions.len(),
                    })
            };
            let p0 = pos(idx[0])?;
            let p1 = pos(idx[1])?;
            let p2 = pos(idx[2])?;

            let a = Vec3::from(p0);
            let b = Vec3::from(p1);
            let c = Vec3::from(p2);
            let normal = (b - a).cross(c - a).normalize_or_zero();

            Ok(GpuTriangle {
                v0: [p0[0], p0[1], p0[2], 0.0],
                v1: [p1[0], p1[1], p1[2], 0.0],
                v2: [p2[0], p2[1], p2[2], 0.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
            })
        })
        .collect()
}

/// Bakes HIGH-mesh surface data into LOW-mesh UV space: first rasterizes
/// `low`'s own UV layout into a world-position + face-normal map (see
/// [`crate::position::bake_position_and_normal`]), then casts one `-normal`
/// ray per covered LOW texel against `high`'s triangle list, gating hits by
/// the front/back clamps — the same two-pass composition
/// [`crate::ao::bake_ao_mesh`] uses, with the self-occlusion hemisphere
/// swapped for the cross-mesh transfer ray (see
/// `umber_gpu::bake_shaders::TRANSFER_BAKE_SHADER`'s doc comment for the ray
/// math, the gate, and both output encodings).
///
/// Returns the full `width * height * 4` RGBA8 bytes in row-major order:
/// height bakes grayscale `h` (`0` is the front-clamp extreme, `1` the back;
/// a hit exactly on the LOW surface reads `front / (front + back)`);
/// world-normal bakes `n * 0.5 + 0.5`. Alpha is coverage-or-hit (`0` means
/// the position pass found no LOW triangle covering that texel, or the ray
/// missed/failed the clamps; `255` means HIGH data transferred). Collapsing
/// to fewer channels would make "background" indistinguishable from real
/// data — the same reason `bake_ao_mesh` returns full RGBA8.
///
/// Builds a fresh compute pipeline on every call rather than caching one
/// across bakes — reasonable for this slice's one-shot entry point, the
/// same tradeoff `ao::run` documents.
///
/// # Errors
///
/// Returns [`TransferError::EmptyTarget`] if `params.width`/`params.height`
/// is zero, [`TransferError::Empty`] if `low` has no triangles,
/// [`TransferError::TooManyHighTris`] if `high` exceeds
/// [`MAX_TRIS_PER_BAKE`], [`TransferError::InvalidClamps`] on a
/// negative/non-finite/zero-total clamp pair,
/// [`TransferError::MalformedHighMesh`] if a HIGH triangle indexes past
/// `high.positions`, [`TransferError::PositionMap`] wrapping whatever
/// [`crate::position::bake_position_and_normal`] rejected for the LOW side,
/// or [`TransferError::Readback`] if the GPU readback fails.
pub fn bake_transfer_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    low: &MeshData,
    high: &MeshData,
    params: &TransferParams,
) -> Result<Vec<u8>, TransferError> {
    profiling::scope!("bake_pass");
    validate(low, high, params)?;
    let width = params.width;
    let height = params.height;

    let position_map = bake_position_and_normal(device, queue, low, width, height)?;

    let triangles = build_high_triangles(high)?;
    if triangles.is_empty() {
        // No HIGH geometry: every transfer ray misses, so the map is
        // background everywhere (see this module's doc header).
        return Ok(vec![0u8; width as usize * height as usize * 4]);
    }
    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_transfer_high_triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let uniform = uniform_for(params, triangles.len() as u32);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_transfer_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Reuses `ao::BakeTarget`'s `Rgba8Unorm` texture/view/readback
    // machinery (including the 256-byte-row-pitch de-padding) rather than
    // re-inventing a second storage-texture type — the same reuse `id` and
    // `thickness` make over `ao`.
    let target = BakeTarget::new(device, width, height);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_transfer_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TRANSFER_BAKE_SHADER)),
    });

    // `TRANSFER_BAKE_SHADER`'s binding set mirrors `bake_ao_mesh`'s
    // `cs_main_from_position` shape — triangle storage buffer, write-only
    // output texture, raycast uniform, plus the two read-only
    // position/normal textures — renumbered contiguously (the AO pass's
    // skipped binding `3` is its plane-path `dims` uniform, which no mesh-fed
    // pass needs).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_transfer_bind_group_layout"),
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
                    min_binding_size: NonZeroU64::new(size_of::<TransferUniform>() as u64),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_bake_transfer_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_transfer_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_transfer_bind_group"),
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
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&position_map.position_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&position_map.normal_view),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_transfer_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_transfer_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per texel, matching `cs_main_from_position`'s
        // dispatch convention (`TRANSFER_BAKE_SHADER` runs
        // `@workgroup_size(1)`, so `workgroup_id.xy` is directly the texel
        // coordinate).
        pass.dispatch_workgroups(width, height, 1);
    }
    queue.submit(Some(encoder.finish()));

    target.read_back_rgba8(device, queue).map_err(|e| match e {
        AoBakeError::Readback(msg) => TransferError::Readback(msg),
        other => TransferError::Readback(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer_params(map: TransferMap) -> TransferParams {
        TransferParams::new(1.0, 2.0, 0.5, 32, 32).with_map(map)
    }

    fn low_quad() -> MeshData {
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

    fn high_triangle(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> MeshData {
        MeshData {
            positions: vec![v0, v1, v2],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2],
            material_names: vec![],
        }
    }

    #[test]
    fn transfer_map_selects_the_documented_uniform_words() {
        assert_eq!(TransferMap::Height.as_u32(), 0);
        assert_eq!(TransferMap::WorldNormal.as_u32(), 1);
    }

    #[test]
    fn transfer_uniform_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<TransferUniform>(), 32);
    }

    #[test]
    fn gpu_triangle_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<GpuTriangle>(), 64);
    }

    #[test]
    fn transfer_params_new_defaults_to_height() {
        let params = TransferParams::new(1.0, 2.0, 0.5, 32, 32);
        assert_eq!(params.map, TransferMap::Height);
        assert_eq!(params.front_distance, 1.0);
        assert_eq!(params.back_distance, 2.0);
        assert_eq!(params.front_offset, 0.5);
        assert_eq!((params.width, params.height), (32, 32));
    }

    #[test]
    fn with_map_builder_sets_the_map_and_preserves_the_other_fields() {
        let params = TransferParams::new(1.0, 2.0, 0.5, 32, 32).with_map(TransferMap::WorldNormal);
        assert_eq!(params.map, TransferMap::WorldNormal);
        assert_eq!(params.front_distance, 1.0);
        assert_eq!(params.back_distance, 2.0);
        assert_eq!(params.front_offset, 0.5);
        assert_eq!((params.width, params.height), (32, 32));
        assert_eq!(
            TransferParams::new(1.0, 2.0, 0.5, 32, 32)
                .with_map(TransferMap::Height)
                .map,
            TransferMap::Height
        );
    }

    #[test]
    fn uniform_for_maps_fields_word_for_word() {
        let params = transfer_params(TransferMap::WorldNormal);
        let uniform = uniform_for(&params, 7);
        assert_eq!(
            (
                uniform.front_distance,
                uniform.back_distance,
                uniform.front_offset,
                uniform.high_tri_count,
                uniform.map_mode,
                uniform.width,
                uniform.height
            ),
            (1.0, 2.0, 0.5, 7, 1, 32, 32)
        );
        let height = uniform_for(&transfer_params(TransferMap::Height), 3);
        assert_eq!((height.map_mode, height.high_tri_count), (0, 3));
    }

    #[test]
    fn build_high_triangles_computes_face_normal_from_positions() {
        let high = high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let tris = build_high_triangles(&high).expect("valid high mesh");
        assert_eq!(tris.len(), 1);
        assert!(
            (tris[0].normal[2] - 1.0).abs() < 1e-6,
            "{:?}",
            tris[0].normal
        );
        assert_eq!(tris[0].v0, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn build_high_triangles_needs_no_uvs_or_normals() {
        // The HIGH mesh carries positions only (the design doc: "no UVs
        // needed") — the builder must not touch `uvs`/`normals` at all.
        let mut high = high_triangle([0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]);
        high.uvs = vec![];
        high.normals = vec![];
        let tris = build_high_triangles(&high).expect("uv-less high mesh must build");
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn build_high_triangles_rejects_out_of_range_position_index() {
        let mut high = high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        high.indices = vec![0, 1, 9];
        let err = build_high_triangles(&high).unwrap_err();
        assert!(matches!(
            err,
            TransferError::MalformedHighMesh { index: 9, .. }
        ));
    }

    #[test]
    fn validate_rejects_empty_low() {
        let err = validate(
            &MeshData::default(),
            &high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            &transfer_params(TransferMap::Height),
        )
        .unwrap_err();
        assert!(matches!(err, TransferError::Empty));
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let params = TransferParams::new(1.0, 2.0, 0.5, 0, 32);
        let err = validate(
            &low_quad(),
            &high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            &params,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            TransferError::EmptyTarget {
                width: 0,
                height: 32
            }
        ));
    }

    #[test]
    fn validate_rejects_high_over_the_triangle_budget() {
        let mut high = high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Triangle count comes from the index list alone, so three shared
        // vertices suffice to exceed the budget.
        high.indices = [0, 1, 2].repeat(MAX_TRIS_PER_BAKE as usize + 1);
        let err = validate(&low_quad(), &high, &transfer_params(TransferMap::Height)).unwrap_err();
        assert!(matches!(
            err,
            TransferError::TooManyHighTris {
                max: MAX_TRIS_PER_BAKE,
                ..
            }
        ));
    }

    #[test]
    fn validate_rejects_degenerate_clamp_pairs() {
        let low = low_quad();
        let high = high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Negative front.
        let err = validate(&low, &high, &TransferParams::new(-1.0, 2.0, 0.5, 8, 8)).unwrap_err();
        assert!(matches!(err, TransferError::InvalidClamps { .. }));
        // Zero total range (would bake NaN heights: 0/0 at the surface).
        let err = validate(&low, &high, &TransferParams::new(0.0, 0.0, 0.5, 8, 8)).unwrap_err();
        assert!(matches!(err, TransferError::InvalidClamps { .. }));
        // Non-finite back.
        let err =
            validate(&low, &high, &TransferParams::new(1.0, f32::NAN, 0.5, 8, 8)).unwrap_err();
        assert!(matches!(err, TransferError::InvalidClamps { .. }));
    }

    #[test]
    fn validate_accepts_well_formed_inputs() {
        assert!(validate(
            &low_quad(),
            &high_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            &transfer_params(TransferMap::Height),
        )
        .is_ok());
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        /// Requests a plain default device — `TRANSFER_BAKE_SHADER` needs no
        /// special feature (write-only `Rgba8Unorm` storage is core WebGPU,
        /// the same reason `AO_BAKE_SHADER` uses `write`, not `read_write`).
        /// Skips gracefully (mirroring `umber_gpu::paint`'s test convention)
        /// if no adapter is available in this environment.
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

        /// Reads texel `(x, y)` of a `width`-wide RGBA8 map.
        fn texel(map: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = ((y * width + x) * 4) as usize;
            [map[i], map[i + 1], map[i + 2], map[i + 3]]
        }

        /// Flat LOW quad at `z = 0` with full `[0, 1]` UVs (normal `+z`):
        /// every texel is covered, `low_pos = (2u - 1, 2v - 1, 0)`.
        fn low_quad() -> MeshData {
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

        /// HIGH fixture with a coplanar half and a sloped half sharing the
        /// `x = 0` edge: `x < 0` is flat at `z = 0` (coplanar with the LOW
        /// quad); `x > 0` is a ramp descending `z = -x` to `z = -1` at
        /// `x = 1` (one unit past the LOW surface ALONG the transfer ray —
        /// the brief's "+1 displacement" realized on the hittable side: a
        /// `+z` displacement would sit BEHIND the ray origin at
        /// `z = front_offset`, i.e. at negative `t`, which Möller–Trumbore
        /// rejects — see the height test's sign-convention note).
        ///
        /// Winding is `+z`-facing on both halves (flat: `(0, 0, 2)`; ramp:
        /// `(2, 0, 2)` — both normalize to `+z`-hemisphere normals), though
        /// the shader is double-sided either way.
        fn high_coplanar_ramp() -> MeshData {
            MeshData {
                positions: vec![
                    [-1.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                normals: vec![],
                uvs: vec![],
                indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                material_names: vec![],
            }
        }

        /// LOW quad whose UVs fill `[0.5, 1]` instead of `[0, 1]`: the
        /// `[0, 0.5)` UV region is never covered by any triangle, exercising
        /// the background path (mirrors `position`'s `half_uv_quad`).
        fn half_uv_low() -> MeshData {
            let mut mesh = low_quad();
            mesh.uvs = vec![[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]];
            mesh
        }

        /// Flat HIGH quad at `z = -2.5` spanning the LOW quad's full `xy`
        /// footprint: every transfer ray travels `t = 3.0` from an origin at
        /// `z = front_offset = 0.5` (`0.5 - (-2.5)`), i.e. the HIGH sits at
        /// distance 3 from the ray origin for the clamp tests.
        fn high_at_minus_two_point_five() -> MeshData {
            MeshData {
                positions: vec![
                    [-1.0, -1.0, -2.5],
                    [1.0, -1.0, -2.5],
                    [1.0, 1.0, -2.5],
                    [-1.0, 1.0, -2.5],
                ],
                normals: vec![],
                uvs: vec![],
                indices: vec![0, 1, 2, 0, 2, 3],
                material_names: vec![],
            }
        }

        /// Quantizes one `0..=1` float channel the way `Rgba8Unorm` storage
        /// does away from exact `.5`-in-byte-units boundaries:
        /// `(v * 255 + 0.5).floor()`. Every height expectation below pins its
        /// distance to the nearest boundary first (see
        /// `ao`'s `encode_unorm` + margin-test precedent); the exact-`0.5`
        /// normal channels instead assert the documented `127..=128` driver
        /// band (see `normal_map`'s flat-quad precedent).
        fn encode_unorm(v: f64) -> u8 {
            (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
        }

        /// THE FLAT-SURFACE EXACTNESS TEST (design test-plan item 2):
        /// `front_offset = 0.5`, `front_distance = 1`, `back_distance = 2`.
        ///
        /// Sign convention (distances are measured ALONG the ray, whose
        /// direction is `-normal`, i.e. INTO the surface): the ray origin
        /// sits at `z = +0.5`, the LOW surface crosses the ray at
        /// `t_surface = front_offset = 0.5`, and the clamp gate spans
        /// `[0.5 - 1, 0.5 + 2] = [-0.5, 2.5]`. A HIGH point `d` units past
        /// the surface along the ray hits at `hit_t = t_surface + d`
        /// (NOT `t_surface - d`: the brief's parenthetical had the sign
        /// flipped — `-t` runs toward the ray origin, where Möller–Trumbore's
        /// `t > 1e-5` near-clip rejects behind-origin intersections).
        ///
        /// Height formula: `h = (hit_t - t_surface + front) / (front + back)`.
        /// Over the coplanar half `hit_t == t_surface`, so
        /// `h = front / (front + back) = 1/3` — the centered value `0.5`
        /// holds ONLY for symmetric clamps (`front == back`); with `1`/`2`
        /// the surface value is exactly `1/3` (`255/3 = 85.0`, no
        /// 127-vs-128 ambiguity at all).
        ///
        /// Over the ramp half the HIGH plane is `z = -x`, so a ray at world
        /// `x` hits at `t = 0.5 + x` and `h = (x + 1) / 3` with
        /// `x = 2u - 1`, `u = (i + 0.5) / 32`:
        ///
        /// ```text
        /// col 16: u = 33/64, x = 1/32,  h = 1.03125/3 = 0.34375,  *255 = 87.65625  -> 88
        /// col 23: u = 47/64, x = 15/32, h = 1.46875/3 = 0.4895833 *255 = 124.84375 -> 125
        /// col 31: u = 63/64, x = 31/32, h = 1.96875/3 = 0.65625,  *255 = 167.34375 -> 167
        /// ```
        ///
        /// (each `*255` product sits `>= 0.15` clear of a `.5` quantization
        /// boundary, so CPU/GPU last-ulp differences cannot flip a byte).
        #[test]
        fn flat_surface_height_is_byte_exact_on_both_halves() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = TransferParams::new(1.0, 2.0, 0.5, 32, 32);
            let map =
                bake_transfer_mesh(&device, &queue, &low_quad(), &high_coplanar_ramp(), &params)
                    .expect("bake should succeed");
            assert_eq!(map.len(), 32 * 32 * 4);

            // Coplanar half (columns 0..16, x < 0): h == 1/3 -> 85, alpha 255.
            for y in 0..32 {
                for x in 0..16 {
                    assert_eq!(
                        texel(&map, 32, x, y),
                        [85, 85, 85, 255],
                        "coplanar texel ({x}, {y}): h = (0.5 - 0.5 + 1)/3 = 1/3 -> 85"
                    );
                }
            }

            // Ramp half: per-column derived bytes (h = (x + 1)/3, x = 2u - 1).
            let ramp_columns = [(16u32, 88u8), (23, 125), (31, 167)];
            for (col, byte) in ramp_columns {
                let u = (f64::from(col) + 0.5) / 32.0;
                let h = (2.0 * u - 1.0 + 1.0) / 3.0;
                let scaled = h * 255.0;
                assert!(
                    (scaled - scaled.round()).abs() > 0.15,
                    "col {col}: h*255 = {scaled} too close to a boundary for a byte-exact assert"
                );
                assert_eq!(
                    encode_unorm(h),
                    byte,
                    "col {col}: mirror mismatch (h = {h})"
                );
                for y in 0..32 {
                    assert_eq!(
                        texel(&map, 32, col, y),
                        [byte, byte, byte, 255],
                        "ramp texel ({col}, {y}): h = {h} -> {byte}"
                    );
                }
            }
        }

        /// Distance-clamp hit/miss pair (design test-plan item 5): the HIGH
        /// quad at `z = -2.5` is hit at `t = 3.0` (distance 3 from the ray
        /// origin at `z = 0.5`). With `front = 3, back = 3` the gate is
        /// `[0.5 - 3, 0.5 + 3] = [-2.5, 3.5] ∋ 3.0` → hit, baking
        /// `h = (3.0 - 0.5 + 3) / 6 = 5.5/6 = 0.9166…`, `*255 = 233.75`
        /// → `234` (0.25 clear of the 233.5/234.5 boundaries). Tightening to
        /// `back = 2` moves the far gate to `2.5 < 3.0` — the SAME geometry
        /// now misses everywhere → background bytes.
        #[test]
        fn clamp_gate_hits_then_misses_the_same_geometry() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let hit = bake_transfer_mesh(
                &device,
                &queue,
                &low_quad(),
                &high_at_minus_two_point_five(),
                &TransferParams::new(3.0, 3.0, 0.5, 32, 32),
            )
            .expect("bake should succeed");
            assert_eq!(hit.len(), 32 * 32 * 4);
            for y in 0..32 {
                for x in 0..32 {
                    assert_eq!(
                        texel(&hit, 32, x, y),
                        [234, 234, 234, 255],
                        "texel ({x}, {y}): h = 5.5/6 -> 234"
                    );
                }
            }

            let miss = bake_transfer_mesh(
                &device,
                &queue,
                &low_quad(),
                &high_at_minus_two_point_five(),
                &TransferParams::new(3.0, 2.0, 0.5, 32, 32),
            )
            .expect("bake should succeed");
            assert_eq!(miss.len(), 32 * 32 * 4);
            assert!(
                miss.iter().all(|b| *b == 0),
                "tightened back clamp must miss everywhere: background (0,0,0,0)"
            );
        }

        /// Uncovered LOW texels bake background `(0, 0, 0, 0)` on either map.
        #[test]
        fn uncovered_low_texels_bake_background() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            // Texel (4, 28) sits at u ~= 0.14, v ~= 0.14 — outside the
            // half-UV footprint (mirrors `position`'s coverage test); texel
            // (28, 4) at u, v ~= 0.86 is covered and must carry a hit.
            for map in [TransferMap::Height, TransferMap::WorldNormal] {
                let bytes = bake_transfer_mesh(
                    &device,
                    &queue,
                    &half_uv_low(),
                    &high_coplanar_ramp(),
                    &TransferParams::new(1.0, 2.0, 0.5, 32, 32).with_map(map),
                )
                .expect("bake should succeed");
                assert_eq!(
                    texel(&bytes, 32, 4, 28),
                    [0, 0, 0, 0],
                    "uncovered texel must read fully zero ({map:?})"
                );
                assert_eq!(
                    texel(&bytes, 32, 28, 4)[3],
                    255,
                    "covered texel must carry a hit ({map:?})"
                );
            }
        }

        /// World-normal map (design test-plan item 2, second half): the
        /// coplanar half bakes `+z` → `(128, 128, 255)`-equivalent; the ramp
        /// half bakes the slope normal `(2, 0, 2)/|(2, 0, 2)| =
        /// `(0.7071…, 0, 0.7071…)` → x/z: `0.85355… * 255 = 217.656… → 218`
        /// (0.34 clear of any boundary), y: exact `0.5`.
        ///
        /// The `0.5` channels (x/y coplanar, y ramp) assert the documented
        /// `127..=128` driver band, not one exact byte: `0.5` is `127.5` in
        /// byte units, whose rounding is driver-dependent (see `normal_map`'s
        /// flat-quad test precedent). The `1.0`/`0.7071` channels assert
        /// exact bytes.
        #[test]
        fn world_normal_map_encodes_both_regions() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let map = bake_transfer_mesh(
                &device,
                &queue,
                &low_quad(),
                &high_coplanar_ramp(),
                &TransferParams::new(1.0, 2.0, 0.5, 32, 32).with_map(TransferMap::WorldNormal),
            )
            .expect("bake should succeed");
            assert_eq!(map.len(), 32 * 32 * 4);

            for y in 0..32 {
                for x in 0..16 {
                    let px = texel(&map, 32, x, y);
                    assert!(
                        (127..=128).contains(&px[0]) && (127..=128).contains(&px[1]),
                        "coplanar texel ({x}, {y}) xy should be ~128 (+z): {px:?}"
                    );
                    assert_eq!(
                        px[2], 255,
                        "coplanar texel ({x}, {y}) z should be 255: {px:?}"
                    );
                    assert_eq!(px[3], 255, "coplanar texel ({x}, {y}) alpha: {px:?}");
                }
                for x in 16..32 {
                    let px = texel(&map, 32, x, y);
                    assert_eq!(px[0], 218, "ramp texel ({x}, {y}) x should be 218: {px:?}");
                    assert!(
                        (127..=128).contains(&px[1]),
                        "ramp texel ({x}, {y}) y should be ~128: {px:?}"
                    );
                    assert_eq!(px[2], 218, "ramp texel ({x}, {y}) z should be 218: {px:?}");
                    assert_eq!(px[3], 255, "ramp texel ({x}, {y}) alpha: {px:?}");
                }
            }
        }

        /// Double-bakes are byte-identical (no RNG anywhere in the path: the
        /// position pass is deterministic and the transfer ray loop keeps the
        /// first nearest hit in buffer order).
        #[test]
        fn transfer_bake_is_deterministic() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            for map in [TransferMap::Height, TransferMap::WorldNormal] {
                let params = TransferParams::new(1.0, 2.0, 0.5, 32, 32).with_map(map);
                let first = bake_transfer_mesh(
                    &device,
                    &queue,
                    &low_quad(),
                    &high_coplanar_ramp(),
                    &params,
                )
                .expect("bake should succeed");
                let second = bake_transfer_mesh(
                    &device,
                    &queue,
                    &low_quad(),
                    &high_coplanar_ramp(),
                    &params,
                )
                .expect("bake should succeed");
                assert_eq!(
                    first, second,
                    "double bake must be byte-identical ({map:?})"
                );
            }
        }

        /// An empty HIGH mesh bakes background without dispatching (every
        /// transfer ray misses by construction).
        #[test]
        fn empty_high_mesh_bakes_background() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let map = bake_transfer_mesh(
                &device,
                &queue,
                &low_quad(),
                &MeshData::default(),
                &TransferParams::new(1.0, 2.0, 0.5, 8, 8),
            )
            .expect("bake should succeed");
            assert_eq!(map.len(), 8 * 8 * 4);
            assert!(
                map.iter().all(|b| *b == 0),
                "empty HIGH must bake background everywhere"
            );
        }
    }
}

//! UV→world-position mapping: rasterizes a mesh's own UV layout into a
//! world-space position texture (plus a face-normal texture), the input a
//! real mesh-fed bake needs instead of the Wave-3 AO slice's stand-in
//! parameter plane (see `ao.rs`/`LANDING_NOTES_AO.md`).
//!
//! See `LANDING_NOTES_POSITION.md` for the shared-memory-chunking and
//! normal-output deviations from the task sketch, and
//! `umber_gpu::bake_shaders::POSITION_BAKE_SHADER`'s doc comment for the
//! shader-side rationale.

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use glam::Vec3;
use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::POSITION_BAKE_SHADER;
use umber_mesh::MeshData;

/// Rust-side ceiling on mesh triangle count for [`bake_position_map`],
/// enforced by [`validate`] before any GPU work happens.
///
/// This is **not** a workgroup-shared-memory array size — see
/// `POSITION_BAKE_SHADER`'s doc comment ("Shared-memory triangle
/// batching") for why a literal `array<_, 4096>` in workgroup storage is
/// infeasible on real hardware (measured: a plain
/// `wgpu::DeviceDescriptor::default()` device caps
/// `max_compute_workgroup_storage_size` at `16384` bytes, and even this
/// sandbox's adapter maximum is only `65536`). The shader instead streams
/// triangles through shared memory in fixed-size batches; `4096` survives
/// purely as "how big a mesh is this bake path willing to brute-force,"
/// matching `ao::run`'s existing few-thousand-triangle budget.
pub const MAX_TRIS_PER_BAKE: u32 = 4096;

/// Errors from [`bake_position_map`] (and the crate-internal GPU bake it
/// shares with `ao::bake_ao_mesh`).
#[derive(Debug, thiserror::Error)]
pub enum PositionMapError {
    /// `mesh` has no triangles to rasterize.
    #[error("mesh has no triangles to rasterize")]
    EmptyMesh,
    /// `mesh` has more triangles than [`MAX_TRIS_PER_BAKE`].
    #[error("mesh has {count} triangles, over the {max} budget for this bake path")]
    TooManyTriangles {
        /// The mesh's actual triangle count.
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
    /// A triangle's vertex index ran past the mesh's attribute arrays.
    #[error(
        "triangle {triangle} references {attr} index {index}, but the mesh only has {len} {attr}"
    )]
    MalformedMesh {
        /// Triangle index (position in the index list / 3).
        triangle: usize,
        /// The out-of-range index.
        index: u32,
        /// The attribute array's actual length.
        len: usize,
        /// Which attribute array was short (`"positions"` or `"uvs"`).
        attr: &'static str,
    },
    /// Reading the baked position texture back to CPU memory failed.
    #[error("position-map readback failed: {0}")]
    Readback(String),
}

/// Parameters for [`bake_position_map`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionMapParams {
    /// Output width in texels.
    pub width: u32,
    /// Output height in texels.
    pub height: u32,
}

/// GPU-side triangle layout matching `POSITION_BAKE_SHADER`'s WGSL
/// `PosTri` struct byte-for-byte: three world-space vertex positions (UV
/// x stashed in each one's unused `w`), the three UVs' `y` components
/// packed into a fourth `vec4`, and the triangle's face normal in a
/// fifth — every field already 16 bytes, so (like `ao::GpuTriangle`) no
/// extra padding fields are needed to match WGSL's `vec4<f32>` alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPosTri {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    uv_y: [f32; 4],
    normal: [f32; 4],
}

const _: () = assert!(
    size_of::<GpuPosTri>() == 80,
    "GpuPosTri must match POSITION_BAKE_SHADER's WGSL PosTri layout"
);

/// GPU-side params uniform, byte-identical to `POSITION_BAKE_SHADER`'s
/// WGSL `PositionParams` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PositionUniform {
    dims: [u32; 2],
    tri_count: u32,
    _pad: u32,
}

const _: () = assert!(
    size_of::<PositionUniform>() == 16,
    "PositionUniform must match POSITION_BAKE_SHADER's WGSL PositionParams layout"
);

/// Checks [`bake_position_map`]'s inputs independently of any GPU call —
/// factored out so these are unit-testable without a device, matching
/// `ao::validate`'s convention.
fn validate(mesh: &MeshData, width: u32, height: u32) -> Result<(), PositionMapError> {
    let tri_count = mesh.triangle_count();
    if tri_count == 0 {
        return Err(PositionMapError::EmptyMesh);
    }
    if tri_count > MAX_TRIS_PER_BAKE as usize {
        return Err(PositionMapError::TooManyTriangles {
            count: tri_count,
            max: MAX_TRIS_PER_BAKE,
        });
    }
    if width == 0 || height == 0 {
        return Err(PositionMapError::EmptyTarget { width, height });
    }
    Ok(())
}

/// Builds the GPU triangle list from `mesh`'s positions/UVs/indices.
///
/// The face normal is recomputed from the triangle's own vertex positions
/// (not read from `mesh.normals`, which the OBJ loader can leave short —
/// see `umber_mesh::load_obj`'s doc comment) — the same choice
/// `ao::build_triangles` makes, for the same reason.
///
/// # Errors
///
/// Returns [`PositionMapError::MalformedMesh`] instead of panicking if an
/// index runs past `mesh.positions` or `mesh.uvs` (a corrupt or
/// hand-built mesh), rather than the slice-index panic a direct `[i]`
/// lookup would produce.
fn build_pos_triangles(mesh: &MeshData) -> Result<Vec<GpuPosTri>, PositionMapError> {
    mesh.indices
        .chunks_exact(3)
        .enumerate()
        .map(|(triangle, idx)| {
            let pos = |i: u32| -> Result<[f32; 3], PositionMapError> {
                mesh.positions
                    .get(i as usize)
                    .copied()
                    .ok_or(PositionMapError::MalformedMesh {
                        triangle,
                        index: i,
                        len: mesh.positions.len(),
                        attr: "positions",
                    })
            };
            let uv = |i: u32| -> Result<[f32; 2], PositionMapError> {
                mesh.uvs
                    .get(i as usize)
                    .copied()
                    .ok_or(PositionMapError::MalformedMesh {
                        triangle,
                        index: i,
                        len: mesh.uvs.len(),
                        attr: "uvs",
                    })
            };
            let p0 = pos(idx[0])?;
            let p1 = pos(idx[1])?;
            let p2 = pos(idx[2])?;
            let uv0 = uv(idx[0])?;
            let uv1 = uv(idx[1])?;
            let uv2 = uv(idx[2])?;

            let a = Vec3::from(p0);
            let b = Vec3::from(p1);
            let c = Vec3::from(p2);
            let normal = (b - a).cross(c - a).normalize_or_zero();

            Ok(GpuPosTri {
                v0: [p0[0], p0[1], p0[2], uv0[0]],
                v1: [p1[0], p1[1], p1[2], uv1[0]],
                v2: [p2[0], p2[1], p2[2], uv2[0]],
                uv_y: [uv0[1], uv1[1], uv2[1], 0.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
            })
        })
        .collect()
}

/// The GPU-resident output of the position-rasterization pass: a
/// world-position texture and a face-normal texture, both `Rgba32Float`,
/// written by `POSITION_BAKE_SHADER::cs_main` and (for the normal
/// texture) never read back to the CPU at all — `ao::bake_ao_mesh` reads
/// both directly on the GPU via `cs_main_from_position`, see
/// `LANDING_NOTES_POSITION.md`'s "no CPU round-trip" section for why this
/// stays crate-internal instead of being built from [`bake_position_map`]'s
/// public `Vec<f32>` output.
pub(crate) struct PositionMapGpu {
    pub(crate) position_texture: wgpu::Texture,
    pub(crate) position_view: wgpu::TextureView,
    pub(crate) normal_texture: wgpu::Texture,
    pub(crate) normal_view: wgpu::TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Runs the position-rasterization pass, returning the GPU-resident
/// position/normal textures without reading either back to the CPU.
/// Shared by [`bake_position_map`] (which reads the position texture back)
/// and `ao::bake_ao_mesh` (which binds both textures straight into the
/// AO pass).
pub(crate) fn bake_position_and_normal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
) -> Result<PositionMapGpu, PositionMapError> {
    validate(mesh, width, height)?;

    let triangles = build_pos_triangles(mesh)?;
    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_position_triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let uniform = PositionUniform {
        dims: [width, height],
        tri_count: triangles.len() as u32,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_position_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Rgba32Float write-only storage is core WebGPU (no device feature
    // needed) — empirically confirmed on this sandbox's adapter before
    // writing this pass; see LANDING_NOTES_POSITION.md. TEXTURE_BINDING is
    // needed on both so `ao::bake_ao_mesh`'s second pass can `textureLoad`
    // them; COPY_SRC only on the position texture, which is the only one
    // `bake_position_map` ever reads back to the CPU.
    let make_target = |label: &str, extra_usage: wgpu::TextureUsages| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | extra_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (position_texture, position_view) =
        make_target("umber_bake_position_target", wgpu::TextureUsages::COPY_SRC);
    let (normal_texture, normal_view) =
        make_target("umber_bake_normal_target", wgpu::TextureUsages::COPY_SRC);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_position_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(POSITION_BAKE_SHADER)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_position_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuPosTri>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<PositionUniform>() as u64),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_bake_position_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_position_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_position_bind_group"),
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
                resource: wgpu::BindingResource::TextureView(&position_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_position_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_position_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(PositionMapGpu {
        position_texture,
        position_view,
        normal_texture,
        normal_view,
        width,
        height,
    })
}

/// Reads a `Rgba32Float` texture back as tightly-packed `f32` RGBA
/// (`width * height * 4` floats, row-major), de-padding the 256-byte-row
/// GPU copy alignment — the `f32` analog of `paint::PaintTarget`'s
/// `read_back_rgba8` (16 bytes/texel instead of 4, otherwise the same
/// shape).
fn read_back_rgba32f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, PositionMapError> {
    let unpadded_row = width * 16;
    let padding = (256 - (unpadded_row % 256)) % 256;
    let padded_row = unpadded_row + padding;
    let size = padded_row as u64 * height as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("umber_bake_position_readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_position_readback_encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| PositionMapError::Readback(format!("device poll: {e}")))?;
    receiver
        .recv()
        .map_err(|_| PositionMapError::Readback("map callback channel closed".into()))?
        .map_err(|e| PositionMapError::Readback(format!("buffer map: {e}")))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| PositionMapError::Readback(format!("mapped range: {e}")))?;
    let bytes: &[u8] = &data;
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height as usize {
        let start = row * padded_row as usize;
        let row_bytes = &bytes[start..start + unpadded_row as usize];
        out.extend_from_slice(bytemuck::cast_slice::<u8, f32>(row_bytes));
    }
    drop(data);
    buffer.unmap();
    Ok(out)
}

/// Bakes `mesh`'s UV→world-position map into a fresh `width`x`height`
/// `Rgba32Float` texture and reads it back as `width * height * 4` `f32`s
/// in row-major RGBA order: `xyz` is the texel's world-space position,
/// `w` is coverage (`1.0` if some mesh triangle's UV footprint covers
/// this texel, `0.0` — with `xyz` left at `0.0` too — otherwise).
///
/// A texel's UV is derived from its texel-center grid coordinate with
/// the `v` axis flipped (`v = 1 - (y + 0.5) / height`), matching
/// `umber_app::paint_state`'s `texel = [u * texels_per_uv, (1 - v) *
/// texels_per_uv]` convention — see `POSITION_BAKE_SHADER`'s doc comment.
///
/// # Errors
///
/// Returns [`PositionMapError::EmptyMesh`] if `mesh` has no triangles,
/// [`PositionMapError::TooManyTriangles`] if `mesh` exceeds
/// [`MAX_TRIS_PER_BAKE`], [`PositionMapError::EmptyTarget`] if
/// `params.width`/`params.height` is zero, [`PositionMapError::MalformedMesh`]
/// if a triangle indexes past `mesh.positions`/`mesh.uvs`, or
/// [`PositionMapError::Readback`] if the GPU readback fails.
pub fn bake_position_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    params: &PositionMapParams,
) -> Result<Vec<f32>, PositionMapError> {
    let gpu = bake_position_and_normal(device, queue, mesh, params.width, params.height)?;
    read_back_rgba32f(device, queue, &gpu.position_texture, gpu.width, gpu.height)
}

/// Bakes the world-space normal map: the position pass's per-texel
/// face normal, read back as `Rgba32Float` (xyz = unit normal, w =
/// coverage) — the `WorldSpaceNormal` slot in the mesh-map naming
/// convention. Tangent-space conversion is the exporter's concern.
///
/// # Errors
///
/// Same surface as [`bake_position_map`] (mesh/target validation,
/// GPU readback).
pub fn bake_world_normal_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    params: &PositionMapParams,
) -> Result<Vec<f32>, PositionMapError> {
    let gpu = bake_position_and_normal(device, queue, mesh, params.width, params.height)?;
    read_back_rgba32f(device, queue, &gpu.normal_texture, gpu.width, gpu.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_mesh() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_names: vec!["m".into()],
        }
    }

    /// Two-triangle quad spanning world `[-1, 1]` with full-`[0, 1]` UVs —
    /// the same shape as `umber_mesh::raycast::tests::quad`, reproduced
    /// here since that fixture is private to its own crate's test module.
    #[cfg(feature = "gpu")]
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

    /// Quad whose UVs fill `[0.5, 1]` instead of `[0, 1]`: the `[0, 0.5)`
    /// UV region is never covered by any triangle, exercising the
    /// coverage-alpha path.
    #[cfg(feature = "gpu")]
    fn half_uv_quad() -> MeshData {
        let mut mesh = full_uv_quad();
        mesh.uvs = vec![[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]];
        mesh
    }

    #[test]
    fn gpu_pos_tri_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<GpuPosTri>(), 80);
    }

    #[test]
    fn position_uniform_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<PositionUniform>(), 16);
    }

    #[test]
    fn build_pos_triangles_computes_face_normal_and_packs_uvs() {
        let tris = build_pos_triangles(&unit_mesh()).expect("valid mesh");
        assert_eq!(tris.len(), 1);
        let tri = tris[0];
        assert!((tri.normal[2] - 1.0).abs() < 1e-6, "{tri:?}");
        assert_eq!(tri.v0, [0.0, 0.0, 0.0, 0.0]); // uv0.x = 0.0
        assert_eq!(tri.v1, [1.0, 0.0, 0.0, 1.0]); // uv1.x = 1.0
        assert_eq!(tri.v2, [0.0, 1.0, 0.0, 0.0]); // uv2.x = 0.0
        assert_eq!(tri.uv_y, [0.0, 0.0, 1.0, 0.0]); // uv0.y, uv1.y, uv2.y
    }

    #[test]
    fn build_pos_triangles_rejects_out_of_range_position_index() {
        let mut mesh = unit_mesh();
        mesh.indices = vec![0, 1, 5];
        let err = build_pos_triangles(&mesh).unwrap_err();
        assert!(matches!(
            err,
            PositionMapError::MalformedMesh {
                index: 5,
                attr: "positions",
                ..
            }
        ));
    }

    #[test]
    fn build_pos_triangles_rejects_out_of_range_uv_index() {
        let mut mesh = unit_mesh();
        mesh.uvs.truncate(2);
        let err = build_pos_triangles(&mesh).unwrap_err();
        assert!(matches!(
            err,
            PositionMapError::MalformedMesh {
                index: 2,
                attr: "uvs",
                ..
            }
        ));
    }

    #[test]
    fn validate_rejects_empty_mesh() {
        let err = validate(&MeshData::default(), 32, 32).unwrap_err();
        assert!(matches!(err, PositionMapError::EmptyMesh));
    }

    #[test]
    fn validate_rejects_too_many_triangles() {
        let mut mesh = unit_mesh();
        mesh.positions = vec![[0.0, 0.0, 0.0]; 3];
        mesh.uvs = vec![[0.0, 0.0]; 3];
        mesh.indices = [0, 1, 2].repeat(MAX_TRIS_PER_BAKE as usize + 1);
        let err = validate(&mesh, 32, 32).unwrap_err();
        assert!(matches!(
            err,
            PositionMapError::TooManyTriangles {
                max: MAX_TRIS_PER_BAKE,
                ..
            }
        ));
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let err = validate(&unit_mesh(), 0, 32).unwrap_err();
        assert!(matches!(
            err,
            PositionMapError::EmptyTarget {
                width: 0,
                height: 32
            }
        ));
    }

    #[test]
    fn validate_accepts_well_formed_inputs() {
        assert!(validate(&unit_mesh(), 32, 32).is_ok());
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::*;

        /// Requests a plain default device — `POSITION_BAKE_SHADER` needs
        /// no special feature (write-only `Rgba32Float` storage is core
        /// WebGPU; probe-verified on this sandbox's adapter before this
        /// shader was written, see `LANDING_NOTES_POSITION.md`). Skips
        /// gracefully if no adapter is available, matching `ao`'s
        /// convention.
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

        /// Reads back texel `(x, y)` of a `width`x`height` RGBA-f32
        /// `Vec<f32>` (as returned by [`bake_position_map`]).
        fn texel(map: &[f32], width: u32, x: u32, y: u32) -> [f32; 4] {
            let i = ((y * width + x) * 4) as usize;
            [map[i], map[i + 1], map[i + 2], map[i + 3]]
        }

        /// A texel's UV under `POSITION_BAKE_SHADER`'s texel-center,
        /// `v`-flipped convention — used to derive each test's expected
        /// value independently of the shader under test.
        fn texel_uv(width: u32, height: u32, x: u32, y: u32) -> (f32, f32) {
            let u = (x as f32 + 0.5) / width as f32;
            let v = 1.0 - (y as f32 + 0.5) / height as f32;
            (u, v)
        }

        /// 33x33 so the exact center texel (16, 16) sits at UV (0.5, 0.5)
        /// — `(33 - 1) / 2 = 16`, and `(16 + 0.5) / 33 = 0.5` exactly —
        /// rather than merely "close to" it, which a 32x32 target (even
        /// texel count, no texel centered on 0.5) can't give. 33 is also
        /// not a multiple of the shader's 8x8 tile, exercising the
        /// partial-tile tail.
        const SIZE: u32 = 33;

        #[test]
        fn full_uv_quad_center_and_corner_match_analytic_position() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let map = bake_position_map(
                &device,
                &queue,
                &full_uv_quad(),
                &PositionMapParams {
                    width: SIZE,
                    height: SIZE,
                },
            )
            .expect("bake should succeed");
            assert_eq!(map.len(), (SIZE * SIZE * 4) as usize);

            // The quad's UV->world map is the exact affine
            // pos = (2u - 1, 2v - 1, 0) (see LANDING_NOTES_POSITION.md).
            let expected = |x: u32, y: u32| {
                let (u, v) = texel_uv(SIZE, SIZE, x, y);
                [2.0 * u - 1.0, 2.0 * v - 1.0, 0.0]
            };

            let center = texel(&map, SIZE, 16, 16);
            assert_eq!(center[3], 1.0, "center texel must be covered");
            let want = expected(16, 16);
            for i in 0..3 {
                assert!(
                    (center[i] - want[i]).abs() < 1e-4,
                    "center[{i}] = {} (expected {})",
                    center[i],
                    want[i]
                );
            }
            // The exact center maps to the quad's own center.
            assert!(center[0].abs() < 1e-4 && center[1].abs() < 1e-4);

            let corner = texel(&map, SIZE, 0, 0);
            assert_eq!(corner[3], 1.0, "corner texel must be covered");
            let want = expected(0, 0);
            for i in 0..3 {
                assert!(
                    (corner[i] - want[i]).abs() < 1e-4,
                    "corner[{i}] = {} (expected {})",
                    corner[i],
                    want[i]
                );
            }
            // Texel (0, 0)'s center is close to, but not exactly at, the
            // mesh's own (-1, 1, 0) corner vertex (see this test's doc
            // comment on why — texel centers never land exactly on a UV
            // edge at finite resolution).
            assert!((corner[0] - -1.0).abs() < 0.05);
            assert!((corner[1] - 1.0).abs() < 0.05);
        }

        #[test]
        fn world_normal_of_flat_quad_is_plus_z_everywhere() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let map = bake_world_normal_map(
                &device,
                &queue,
                &full_uv_quad(),
                &PositionMapParams {
                    width: SIZE,
                    height: SIZE,
                },
            )
            .expect("bake should succeed");
            assert_eq!(map.len(), (SIZE * SIZE * 4) as usize);

            // The quad lies in the z=0 plane facing +z: every covered
            // texel's world normal is (0, 0, 1).
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let n = texel(&map, SIZE, x, y);
                    assert_eq!(n[3], 1.0, "texel ({x}, {y}) must be covered");
                    assert!(
                        (n[0].abs() < 1e-4) && (n[1].abs() < 1e-4) && ((n[2] - 1.0).abs() < 1e-4),
                        "normal at ({x}, {y}) = {n:?}, expected +z"
                    );
                }
            }
        }

        #[test]
        fn half_uv_quad_leaves_the_other_half_uncovered() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let map = bake_position_map(
                &device,
                &queue,
                &half_uv_quad(),
                &PositionMapParams {
                    width: SIZE,
                    height: SIZE,
                },
            )
            .expect("bake should succeed");

            // Covered: both u, v >= 0.5. Texel (28, 4) -> u ~= 0.864,
            // v ~= 0.864 (comfortably inside [0.5, 1]).
            let covered = texel(&map, SIZE, 28, 4);
            assert_eq!(covered[3], 1.0, "texel inside the UV footprint");
            let (u, v) = texel_uv(SIZE, SIZE, 28, 4);
            // half_uv_quad maps UV [0.5, 1] -> world [-1, 1]:
            // pos = (u - 0.5) * 4 - 1.
            let want = [(u - 0.5) * 4.0 - 1.0, (v - 0.5) * 4.0 - 1.0, 0.0];
            for i in 0..3 {
                assert!(
                    (covered[i] - want[i]).abs() < 1e-4,
                    "covered[{i}] = {} (expected {})",
                    covered[i],
                    want[i]
                );
            }

            // Uncovered: both u, v < 0.5. Texel (4, 28) -> u ~= 0.136,
            // v ~= 0.136.
            let uncovered = texel(&map, SIZE, 4, 28);
            assert_eq!(
                uncovered,
                [0.0, 0.0, 0.0, 0.0],
                "texel outside the UV footprint must read fully zero"
            );
        }

        #[test]
        fn empty_mesh_is_rejected_without_touching_the_gpu_pipeline() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let err = bake_position_map(
                &device,
                &queue,
                &MeshData::default(),
                &PositionMapParams {
                    width: 8,
                    height: 8,
                },
            )
            .unwrap_err();
            assert!(matches!(err, PositionMapError::EmptyMesh));
        }

        #[test]
        fn mesh_over_the_triangle_budget_is_rejected() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let mut mesh = unit_mesh();
            mesh.positions = vec![[0.0, 0.0, 0.0]; 3];
            mesh.uvs = vec![[0.0, 0.0]; 3];
            mesh.indices = [0, 1, 2].repeat(MAX_TRIS_PER_BAKE as usize + 1);
            let err = bake_position_map(
                &device,
                &queue,
                &mesh,
                &PositionMapParams {
                    width: 8,
                    height: 8,
                },
            )
            .unwrap_err();
            assert!(matches!(err, PositionMapError::TooManyTriangles { .. }));
        }
    }
}

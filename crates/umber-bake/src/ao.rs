//! Ambient-occlusion baking via compute-shader raycasting.
//!
//! Wave 3, P0 bake path (docs/specs/requirements.md §3: "Compute-shader
//! raycast/bvh bake path (no RT hardware dependency)"). This first slice
//! proves the raycast core — brute-force Möller–Trumbore against a
//! triangle list, no BVH — by baking into a flat parameter plane rather
//! than a mesh's own UV-mapped position map. See `LANDING_NOTES_AO.md`
//! for why, the perf budget, and what the next slice needs to add (a
//! UV→world position map, replacing the plane, plus an acceleration
//! structure for meshes beyond a few thousand triangles).

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use glam::Vec3;
use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::AO_BAKE_SHADER;
use umber_gpu::{PaintError, PaintTarget};
use umber_mesh::MeshData;

/// A plane of texels to raycast ambient occlusion against — the Wave-3
/// first slice's stand-in for a UV-mapped position map (see the module
/// docs).
///
/// `origin` is the plane's world-space center. `u_axis`/`v_axis` must be
/// unit-length and mutually perpendicular; `cross(u_axis, v_axis)` is the
/// hemisphere pole every cast ray is centered on, so the two axes must be
/// ordered to make that cross product point toward the side of the plane
/// being baked (away from the plane, into the scene). `extent` is the
/// plane's full width/height in world units, centered on `origin` — a
/// texel at normalized grid position `(u, v)` sits at `origin + (u - 0.5)
/// * extent.x * u_axis + (v - 0.5) * extent.y * v_axis`.
///
/// Three private `f32` padding fields (not shown in the field list below)
/// round each axis up to 16 bytes, matching
/// `bake_shaders::AO_BAKE_SHADER`'s WGSL `PlaneUniform`'s `vec3<f32>`
/// alignment — the same `#[repr(C)]`-plus-explicit-padding convention
/// `paint::Dab` uses for its WGSL `vec4<f32>` fields. Build via
/// [`PlaneDesc::new`]; the padding fields being private means a bare
/// struct literal can't be constructed any other way.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlaneDesc {
    /// World-space center of the plane.
    pub origin: [f32; 3],
    _pad0: f32,
    /// Unit axis spanning the plane's "u" (width) direction.
    pub u_axis: [f32; 3],
    _pad1: f32,
    /// Unit axis spanning the plane's "v" (height) direction.
    pub v_axis: [f32; 3],
    _pad2: f32,
    /// Full width/height of the plane in world units, centered on `origin`.
    pub extent: [f32; 2],
    _pad3: [f32; 2],
}

const _: () = assert!(
    size_of::<PlaneDesc>() == 64,
    "PlaneDesc must match AO_BAKE_SHADER's WGSL PlaneUniform layout"
);

impl PlaneDesc {
    /// Builds a plane descriptor from its conceptual fields (padding is internal).
    pub fn new(origin: [f32; 3], u_axis: [f32; 3], v_axis: [f32; 3], extent: [f32; 2]) -> Self {
        Self {
            origin,
            _pad0: 0.0,
            u_axis,
            _pad1: 0.0,
            v_axis,
            _pad2: 0.0,
            extent,
            _pad3: [0.0; 2],
        }
    }
}

/// Parameters for one [`run`] call.
#[derive(Debug, Clone, Copy)]
pub struct AoBakeParams {
    /// Hemisphere rays cast per texel.
    pub rays: u32,
    /// Maximum ray travel distance, in world units; geometry beyond this
    /// distance along a ray is treated as if it weren't there (that ray
    /// counts as unoccluded, not as "unknown").
    pub max_distance: f32,
    /// Offset along the plane normal the ray origin is nudged by, to
    /// avoid a ray immediately re-hitting coplanar/near-coplanar geometry
    /// at `t ≈ 0`.
    pub bias: f32,
    /// The parameter plane rays are cast from (see [`PlaneDesc`]).
    pub plane: PlaneDesc,
}

impl AoBakeParams {
    /// The default hemisphere ray count used by [`AoBakeParams::new`].
    pub const DEFAULT_RAYS: u32 = 16;

    /// Builds params with [`AoBakeParams::DEFAULT_RAYS`] rays; override
    /// the `rays` field afterward (struct-update syntax) for a different
    /// sample count.
    pub fn new(max_distance: f32, bias: f32, plane: PlaneDesc) -> Self {
        Self {
            rays: Self::DEFAULT_RAYS,
            max_distance,
            bias,
            plane,
        }
    }
}

/// Errors from [`run`].
#[derive(Debug, thiserror::Error)]
pub enum AoBakeError {
    /// `mesh` has no triangles to raycast against.
    #[error("mesh has no triangles to bake against")]
    EmptyMesh,
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
    /// Reading the baked texture back to CPU memory failed.
    #[error("bake-target readback failed: {0}")]
    Readback(String),
}

/// The GPU AO-bake output surface.
///
/// A thin wrapper over [`umber_gpu::PaintTarget`] — see the module docs:
/// this slice reuses `PaintTarget`'s existing `Rgba8Unorm`
/// texture/view/readback machinery directly rather than re-inventing a
/// second storage-texture type. It is deliberately *not* drawn from
/// [`umber_gpu::TilePool`]'s fixed-512² tile grid: a bake target's
/// resolution is set by the caller (the GPU test below bakes at 32×32),
/// not quantized to [`umber_gpu::TILE_SIZE`] — `PaintTarget` is the
/// right-sized reusable piece here, not the pool built on top of it.
pub struct BakeTarget {
    paint_target: PaintTarget,
}

impl BakeTarget {
    /// Creates a new `width`x`height` bake target, zero-initialized.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self {
            paint_target: PaintTarget::new(device, width, height),
        }
    }

    /// Target dimensions, in texels.
    pub fn dimensions(&self) -> (u32, u32) {
        self.paint_target.dimensions()
    }

    /// The backing texture's default view, for the bake pass's storage
    /// binding.
    pub fn view(&self) -> &wgpu::TextureView {
        self.paint_target.view()
    }

    /// Reads the full target back as tightly-packed RGBA8 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`AoBakeError::Readback`] if the GPU readback fails.
    pub fn read_back_rgba8(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, AoBakeError> {
        self.paint_target
            .read_back_rgba8(device, queue)
            .map_err(|e: PaintError| AoBakeError::Readback(e.to_string()))
    }
}

/// GPU-side triangle layout: three vertex positions plus a face normal,
/// each padded to `vec4<f32>` to match `AO_BAKE_SHADER`'s WGSL `Tri`
/// struct — every field is already 16 bytes so, unlike `PlaneDesc`, no
/// extra padding fields are needed to line the two layouts up.
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
    "GpuTriangle must match AO_BAKE_SHADER's WGSL Tri layout"
);

/// Builds the GPU triangle list from `mesh`'s positions/indices. The face
/// normal is recomputed from the triangle's own vertex positions (not
/// read from `mesh.normals`) so it is always available even for meshes
/// imported without vertex normals; it is carried in the buffer for the
/// next slice's use (e.g. backface culling) — this slice's shader does
/// not yet consume it.
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

/// GPU-side params uniform: byte-identical to `AO_BAKE_SHADER`'s WGSL
/// `AoParams` struct (`PlaneDesc` plus four trailing 4-byte scalars — no
/// extra padding needed at the end since 64 + 16 = 80 is already a
/// multiple of 16, satisfying WGSL's uniform-struct alignment without
/// help).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AoUniform {
    plane: PlaneDesc,
    rays: u32,
    max_distance: f32,
    bias: f32,
    tri_count: u32,
}

const _: () = assert!(
    size_of::<AoUniform>() == 80,
    "AoUniform must match AO_BAKE_SHADER's WGSL AoParams layout"
);

/// Checks `run`'s inputs independently of any GPU call, so the validation
/// itself is unit-testable without a device.
fn validate(
    mesh: &MeshData,
    params: &AoBakeParams,
    width: u32,
    height: u32,
) -> Result<(), AoBakeError> {
    if mesh.triangle_count() == 0 {
        return Err(AoBakeError::EmptyMesh);
    }
    if params.rays == 0 {
        return Err(AoBakeError::InvalidRayCount);
    }
    if width == 0 || height == 0 {
        return Err(AoBakeError::EmptyTarget { width, height });
    }
    Ok(())
}

/// Bakes ambient occlusion for `mesh` into a fresh `width`x`height`
/// [`BakeTarget`], raycasting from `params.plane`'s parameter plane, and
/// returns the result as `width * height` single-channel bytes in
/// row-major order (the target's R channel — see [`BakeTarget`]'s docs
/// for why the GPU texture backing it is `Rgba8Unorm`, not a real R8
/// format). A fully unoccluded texel reads `255`; fully occluded reads
/// `0`.
///
/// Builds a fresh compute pipeline on every call rather than caching one
/// across bakes — reasonable for this slice's one-shot entry point; see
/// `LANDING_NOTES_AO.md` for the caching tradeoff if `run` grows a
/// hot-loop caller.
///
/// # Errors
///
/// Returns [`AoBakeError::EmptyMesh`] if `mesh` has no triangles,
/// [`AoBakeError::InvalidRayCount`] if `params.rays == 0`,
/// [`AoBakeError::EmptyTarget`] if `width`/`height` is zero, or
/// [`AoBakeError::Readback`] if the GPU readback fails.
pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    params: &AoBakeParams,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, AoBakeError> {
    validate(mesh, params, width, height)?;

    let triangles = build_triangles(mesh);
    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_ao_triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let uniform = AoUniform {
        plane: params.plane,
        rays: params.rays,
        max_distance: params.max_distance,
        bias: params.bias,
        tri_count: triangles.len() as u32,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_ao_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_ao_dims"),
        contents: bytemuck::cast_slice(&[width, height]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let target = BakeTarget::new(device, width, height);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_ao_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(AO_BAKE_SHADER)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_ao_bind_group_layout"),
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
                    min_binding_size: NonZeroU64::new(size_of::<AoUniform>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<[u32; 2]>() as u64),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_bake_ao_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_ao_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_ao_bind_group"),
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
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &dims_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_ao_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_ao_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width, height, 1);
    }
    queue.submit(Some(encoder.finish()));

    let rgba = target.read_back_rgba8(device, queue)?;
    let r_channel = rgba.chunks_exact(4).map(|px| px[0]).collect();
    Ok(r_channel)
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

    fn test_plane() -> PlaneDesc {
        PlaneDesc::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0],
        )
    }

    #[test]
    fn plane_desc_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<PlaneDesc>(), 64);
    }

    #[test]
    fn ao_uniform_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<AoUniform>(), 80);
    }

    #[test]
    fn gpu_triangle_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<GpuTriangle>(), 64);
    }

    #[test]
    fn plane_desc_new_round_trips_fields() {
        let plane = PlaneDesc::new(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 5.0],
        );
        assert_eq!(plane.origin, [1.0, 2.0, 3.0]);
        assert_eq!(plane.u_axis, [1.0, 0.0, 0.0]);
        assert_eq!(plane.v_axis, [0.0, 1.0, 0.0]);
        assert_eq!(plane.extent, [4.0, 5.0]);
    }

    #[test]
    fn ao_bake_params_new_defaults_to_sixteen_rays() {
        let params = AoBakeParams::new(10.0, 0.01, test_plane());
        assert_eq!(params.rays, AoBakeParams::DEFAULT_RAYS);
        assert_eq!(params.rays, 16);
        assert_eq!(params.max_distance, 10.0);
        assert_eq!(params.bias, 0.01);
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
    fn validate_rejects_empty_mesh() {
        let params = AoBakeParams::new(10.0, 0.01, test_plane());
        let err = validate(&MeshData::default(), &params, 32, 32).unwrap_err();
        assert!(matches!(err, AoBakeError::EmptyMesh));
    }

    #[test]
    fn validate_rejects_zero_rays() {
        let mut params = AoBakeParams::new(10.0, 0.01, test_plane());
        params.rays = 0;
        let err = validate(&unit_mesh(), &params, 32, 32).unwrap_err();
        assert!(matches!(err, AoBakeError::InvalidRayCount));
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let params = AoBakeParams::new(10.0, 0.01, test_plane());
        let err = validate(&unit_mesh(), &params, 0, 32).unwrap_err();
        assert!(matches!(
            err,
            AoBakeError::EmptyTarget {
                width: 0,
                height: 32
            }
        ));
    }

    #[test]
    fn validate_accepts_well_formed_inputs() {
        let params = AoBakeParams::new(10.0, 0.01, test_plane());
        assert!(validate(&unit_mesh(), &params, 32, 32).is_ok());
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        /// Requests a plain default device — this pass needs no special
        /// feature (see `bake_shaders::AO_BAKE_SHADER`'s doc comment on
        /// why `write`-only storage-texture access avoids the
        /// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` requirement
        /// `paint::PaintCompositor` needs). Skips gracefully (mirroring
        /// `umber_gpu::paint`'s test convention) if no adapter is
        /// available in this environment.
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

        fn triangle_mesh(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> MeshData {
            MeshData {
                positions: vec![v0, v1, v2],
                normals: vec![[0.0, 1.0, 0.0]; 3],
                uvs: vec![[0.0, 0.0]; 3],
                indices: vec![0, 1, 2],
                material_names: vec!["m".into()],
            }
        }

        /// The task's literal scenario: a large triangle floats above a
        /// 60×60 parameter plane; a texel directly beneath the triangle
        /// must read heavily occluded, a texel in the plane's far corner
        /// (well outside the triangle's footprint) must read clear.
        ///
        /// Geometry is hand-derived in `LANDING_NOTES_AO.md` ("Picking the
        /// GPU test's numbers") rather than guessed: the triangle is
        /// equilateral, centered on the plane's origin, with circumradius
        /// 18 and inradius 9 at height 2 above the plane. For the 16
        /// stratified hemisphere rays this shader casts, every ray with
        /// `cos(theta) >= 0.21875` (indices 0..=12, 13 of 16) is
        /// *guaranteed* to land within the triangle's inscribed circle
        /// regardless of azimuth, so the center texel's occlusion is at
        /// least 13/16 -> AO byte <= 48, comfortably under the 64
        /// threshold even if the one azimuth-dependent ray (index 13)
        /// misses. The far corner sits roughly 41 world units from the
        /// triangle's centroid; since `max_distance = 15` bounds every
        /// ray's total travel and the triangle's circumradius is only 18,
        /// no ray fired from that corner can geometrically reach the
        /// triangle at all (`41 - 18 = 23 > 15`), so that texel is
        /// provably fully unoccluded (byte 255).
        #[test]
        fn ao_bake_center_occluded_far_corner_clear() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };

            let plane = PlaneDesc::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [60.0, 60.0],
            );
            let params = AoBakeParams {
                rays: 16,
                ..AoBakeParams::new(15.0, 0.01, plane)
            };

            // Equilateral triangle, circumradius 18, centered at world
            // origin, 2 units above the plane (see the doc comment above).
            let mesh = triangle_mesh(
                [0.0, 2.0, 18.0],
                [-15.588_457, 2.0, -9.0],
                [15.588_457, 2.0, -9.0],
            );

            let bytes = run(&device, &queue, &mesh, &params, 32, 32).expect("bake should succeed");
            assert_eq!(bytes.len(), 32 * 32);

            let idx = |x: usize, y: usize| y * 32 + x;
            let center = bytes[idx(16, 16)];
            let far_corner = bytes[idx(0, 0)];

            assert!(
                center < 64,
                "center texel (directly under the triangle) should be heavily occluded: {center}"
            );
            assert!(
                far_corner >= 200,
                "far-corner texel (outside the triangle's reach) should be unoccluded: {far_corner}"
            );
        }

        #[test]
        fn ao_bake_rejects_empty_mesh_without_touching_the_gpu_pipeline() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let plane = PlaneDesc::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 2.0],
            );
            let params = AoBakeParams::new(10.0, 0.01, plane);
            let err = run(&device, &queue, &MeshData::default(), &params, 8, 8).unwrap_err();
            assert!(matches!(err, AoBakeError::EmptyMesh));
        }
    }
}

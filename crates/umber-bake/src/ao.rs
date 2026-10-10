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

use crate::position::{bake_position_and_normal, PositionMapError};

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
    /// Bent-normal toggle (wave-4 item 5 slice 2): when `true`, the
    /// mesh-fed pass ([`bake_ao_and_bent_mesh`]) also accumulates the
    /// unoccluded ray directions into a second output map (see
    /// [`AoWithBent`]). Defaults to `false` (AO only); set via
    /// [`AoBakeParams::with_bent_normals`]. Ignored by [`run`]'s
    /// parameter-plane path, which has no bent output.
    pub bent_normals: bool,
}

impl AoBakeParams {
    /// The default hemisphere ray count used by [`AoBakeParams::new`].
    pub const DEFAULT_RAYS: u32 = 16;

    /// Builds params with [`AoBakeParams::DEFAULT_RAYS`] rays and the
    /// bent-normal path off; override the `rays` field afterward
    /// (struct-update syntax) for a different sample count, or chain
    /// [`AoBakeParams::with_bent_normals`] for the dual output.
    pub fn new(max_distance: f32, bias: f32, plane: PlaneDesc) -> Self {
        Self {
            rays: Self::DEFAULT_RAYS,
            max_distance,
            bias,
            plane,
            bent_normals: false,
        }
    }

    /// Returns these params with the bent-normal toggle set (see
    /// [`AoBakeParams::bent_normals`]).
    pub fn with_bent_normals(mut self, on: bool) -> Self {
        self.bent_normals = on;
        self
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
    /// [`bake_ao_mesh`]'s position-map pass (see [`crate::position`])
    /// failed before AO raycasting ever started.
    #[error("position map: {0}")]
    PositionMap(#[from] PositionMapError),
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
/// `AoParams` struct (`PlaneDesc` plus the four pre-bent trailing words
/// `rays`/`max_distance`/`bias`/`tri_count`, then the wave-4 bent-normal
/// word `bent_normals` as a `u32` (WGSL uniforms have no bools) plus three
/// `u32` pads — 64 + 24 + 4 + 12 = 96, already a multiple of WGSL's 16-byte
/// uniform-struct alignment. The bent word is additive at the END so the
/// pre-existing prefix layout is stable; field ORDER here must match the
/// WGSL struct order (a reordered Rust field is a silent-corruption bug —
/// the size asserts below would still pass).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AoUniform {
    plane: PlaneDesc,
    rays: u32,
    max_distance: f32,
    bias: f32,
    tri_count: u32,
    bent_normals: u32,
    _pad_end: [u32; 3],
}

const _: () = assert!(
    size_of::<AoUniform>() == 96,
    "AoUniform must match AO_BAKE_SHADER's WGSL AoParams layout"
);

/// Maps [`AoBakeParams`] onto the GPU uniform word-for-word
/// (`bent_normals` becomes `1`/`0` — factored out so the bool-to-word
/// mapping is unit-testable without a device, matching
/// `normal_map::uniform_for`'s convention).
fn uniform_for(params: &AoBakeParams, tri_count: u32) -> AoUniform {
    AoUniform {
        plane: params.plane,
        rays: params.rays,
        max_distance: params.max_distance,
        bias: params.bias,
        tri_count,
        bent_normals: u32::from(params.bent_normals),
        _pad_end: [0; 3],
    }
}

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

    let uniform = uniform_for(params, triangles.len() as u32);
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

/// The dual output of [`bake_ao_and_bent_mesh`]: the AO map plus the
/// bent-normal map baked from the SAME ray set in one dispatch (see that
/// fn's docs for both encodings).
pub struct AoWithBent {
    /// Full `width * height * 4` RGBA8 AO bytes — identical to what
    /// [`bake_ao_mesh`] returns for the same inputs (the bent path never
    /// perturbs the AO accumulation; the off-regression GPU test pins
    /// this).
    pub ao: Vec<u8>,
    /// Full `width * height * 4` RGBA8 bent-normal bytes: `rgb` is the
    /// world-space bent vector encoded `bent * 0.5 + 0.5`, `a` is the
    /// confidence `|sum| / ray_count`. Fully-occluded-but-covered texels
    /// read the geometric normal encoded with `a = 0`; uncovered texels
    /// read `(0, 0, 0, 0)`. When `params.bent_normals` was `false` the
    /// shader skips every bent store and this buffer reads all zeros (the
    /// target starts zero-initialized) — set the flag via
    /// [`AoBakeParams::with_bent_normals`] to get a real bent map.
    pub bent: Vec<u8>,
}

/// Bakes ambient occlusion for `mesh` against *itself*: first rasterizes
/// `mesh`'s own UV layout into a world-position + face-normal map (see
/// [`crate::position::bake_position_and_normal`]), then raycasts a
/// hemisphere from every covered texel against `mesh`'s own triangle list
/// — the real self-occlusion bake [`run`]'s parameter-plane stand-in was
/// always meant to lead to (see `LANDING_NOTES_AO.md`).
///
/// Unlike [`run`] (which returns only the R channel, since its plane has
/// no notion of "uncovered"), this returns the full `width * height * 4`
/// RGBA8 bytes: alpha `0` means the position pass found no UV triangle
/// covering that texel (no raycast was attempted, not "zero occlusion"),
/// alpha `255` with `rgb = (0, 0, 0)` means a texel that *is* covered and
/// fully occluded. Collapsing to an R-only channel like [`run`] does would
/// make those two states indistinguishable on readback — see
/// `LANDING_NOTES_POSITION.md`'s reviewer checklist.
///
/// `params.plane` is ignored by this path (the position map supplies each
/// texel's ray origin and tangent frame instead) — kept as a field anyway
/// so both AO entry points share one uniform layout; pass any value.
///
/// `params.bent_normals` is honored by the shared pass but the bent map is
/// discarded here — set the flag AND call [`bake_ao_and_bent_mesh`] to
/// keep both outputs.
///
/// # Errors
///
/// Returns [`AoBakeError::InvalidRayCount`] if `params.rays == 0`, or
/// [`AoBakeError::PositionMap`] wrapping whatever
/// [`crate::position::bake_position_and_normal`] rejected (empty mesh,
/// over the triangle budget, zero-sized target, malformed indices, or a
/// GPU readback failure).
pub fn bake_ao_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
    params: &AoBakeParams,
) -> Result<Vec<u8>, AoBakeError> {
    bake_ao_mesh_inner(device, queue, mesh, width, height, params).map(|both| both.ao)
}

/// Bakes ambient occlusion AND bent normals for `mesh` against *itself* in
/// a single dispatch: the exact same two-pass composition [`bake_ao_mesh`]
/// uses (position map, then the mesh-fed hemisphere raycast), with the
/// shader's bent-normal accumulator enabled so the unoccluded ray
/// directions are summed alongside the AO visibility count — Substance's
/// shared-sample approach, no second raycast (see
/// `bake_shaders::AO_BAKE_SHADER`'s `cs_main_from_position` doc comment
/// for the world-space frame and the output encoding contract).
///
/// Returns [`AoWithBent`] with both full-RGBA8 maps; the `ao` half is
/// byte-identical to [`bake_ao_mesh`]'s output for the same inputs.
///
/// # Errors
///
/// Same as [`bake_ao_mesh`].
pub fn bake_ao_and_bent_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
    params: &AoBakeParams,
) -> Result<AoWithBent, AoBakeError> {
    bake_ao_mesh_inner(device, queue, mesh, width, height, params)
}

/// The shared single dispatch behind [`bake_ao_mesh`] and
/// [`bake_ao_and_bent_mesh`]: builds one pipeline over
/// `AO_BAKE_SHADER::cs_main_from_position` with BOTH output textures bound
/// (the AO target plus the bent target at binding 6), dispatches once, and
/// reads both back. The shader gates every bent store on the uniform's bent
/// word (mapped from `params.bent_normals` by [`uniform_for`]), so a
/// flag-off bake pays for one extra zero-initialized texture and nothing
/// else — and the AO bytes can't diverge between the two entries.
fn bake_ao_mesh_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshData,
    width: u32,
    height: u32,
    params: &AoBakeParams,
) -> Result<AoWithBent, AoBakeError> {
    if params.rays == 0 {
        return Err(AoBakeError::InvalidRayCount);
    }

    let position_map = bake_position_and_normal(device, queue, mesh, width, height)?;

    let triangles = build_triangles(mesh);
    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_ao_mesh_triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let uniform = uniform_for(params, triangles.len() as u32);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_ao_mesh_params"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let target = BakeTarget::new(device, width, height);
    // Second output of the shared dispatch (binding 6): always allocated
    // and bound — even flag-off, where the shader never stores into it —
    // so both entries share one bind-group layout and one pipeline, and
    // the AO bytes can't diverge between them. Starts zero-initialized, so
    // a flag-off bent readback is deterministically all zeros.
    let bent_target = BakeTarget::new(device, width, height);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_ao_mesh_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(AO_BAKE_SHADER)),
    });

    // `cs_main_from_position`'s binding set, not `cs_main`'s: no `dims`
    // uniform (the texel coordinate comes straight from `workgroup_id`,
    // see that entry point's doc comment in `bake_shaders::AO_BAKE_SHADER`),
    // plus the two read-only textures the position pass produced.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_ao_mesh_bind_group_layout"),
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
            // `bent_tex`: the shared pass's second output (see
            // `AO_BAKE_SHADER::cs_main_from_position`'s doc comment). Bound
            // on every dispatch — even flag-off — so the layout never
            // diverges between the two entries.
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_bake_ao_mesh_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_ao_mesh_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main_from_position"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_bake_ao_mesh_bind_group"),
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
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(bent_target.view()),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_ao_mesh_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("umber_bake_ao_mesh_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per texel, matching `cs_main`'s dispatch
        // convention (`cs_main_from_position` reads `workgroup_id.xy`
        // directly as the texel coordinate, no `dims` uniform involved).
        pass.dispatch_workgroups(width, height, 1);
    }
    queue.submit(Some(encoder.finish()));

    let ao = target.read_back_rgba8(device, queue)?;
    let bent = bent_target.read_back_rgba8(device, queue)?;
    Ok(AoWithBent { ao, bent })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rust mirror of `AO_BAKE_SHADER::hemisphere_sample` (wave-4 item 5
    /// slice 2's mean test): the same stratified
    /// `cos(theta) = 1 - (i + 0.5) / n` and golden-angle
    /// `phi = 2 * PI * fract(i * GOLDEN_CONJ)`, evaluated in `f32` op-for-op
    /// so the bent-mean expectation tracks the shader's fixed sample set.
    /// Lives in the test module (like `id`'s FNV mirror) because only tests
    /// use it. The constants are the shader's own decimals truncated to
    /// what `f32` can hold — verified bit-identical to the shader-side
    /// rounding (see `mirror_constants_match_shader_decimals`), which is
    /// what lets the mean test assert exact bytes.
    fn hemisphere_sample_mirror(i: u32, n: u32) -> [f32; 3] {
        const GOLDEN_CONJ: f32 = 0.618034;
        let nf = (n as f32).max(1.0);
        let cos_theta = 1.0 - ((i as f32) + 0.5) / nf;
        let sin_theta = (0.0f32).max(1.0 - cos_theta * cos_theta).sqrt();
        let scaled = (i as f32) * GOLDEN_CONJ;
        let phi = 2.0 * std::f32::consts::PI * (scaled - scaled.floor());
        [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta]
    }

    /// Rust mirror of `AO_BAKE_SHADER::orthonormal_basis` (Duff et al.):
    /// the same branchless frame from a unit normal, so the mean test
    /// steers the mirrored samples into world space exactly the way the
    /// shader does.
    fn orthonormal_basis_mirror(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let sign_z = if n[2] >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign_z + n[2]);
        let b = n[0] * n[1] * a;
        (
            [1.0 + sign_z * n[0] * n[0] * a, sign_z * b, -sign_z * n[0]],
            [b, sign_z + n[1] * n[1] * a, -n[1]],
        )
    }

    /// Encodes one `0..=1` float channel the way `Rgba8Unorm` storage does
    /// for values clear of a quantization boundary: `(v * 255 + 0.5).floor()`.
    /// The GPU's own rounding (nearest-even vs. half-up) only disagrees
    /// with this formula at exact `.5`-in-byte-units boundaries, which the
    /// irrational trig outputs of [`hemisphere_sample_mirror`] never land
    /// on — pinned by `expected_unoccluded_bent_has_quantization_margin`.
    /// (Deliberately NOT used for exact-`0.5` encodings like the geometric
    /// normal fallback, where the driver band `127..=128` applies — see
    /// `normal_map`'s flat-quad test precedent.)
    fn encode_unorm(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
    }

    /// The exact bent texel a fully-unoccluded surface with `normal` bakes
    /// (wave-4 item 5 slice 2's mean test): ports the shader's whole
    /// per-ray chain — mirror sample, Duff-basis steering, `normalize`,
    /// serial accumulation in ray order (matching the shader's serial slot
    /// reduction when `rays <= 64`, where each workgroup slot holds exactly
    /// one ray) — then the `bent * 0.5 + 0.5` encoding and
    /// `|sum| / rays` confidence, quantized with [`encode_unorm`].
    fn expected_unoccluded_bent(rays: u32, normal: [f32; 3]) -> ([u8; 3], u8) {
        let (b1, b2) = orthonormal_basis_mirror(normal);
        let mut sum = [0.0f32; 3];
        for i in 0..rays {
            let l = hemisphere_sample_mirror(i, rays);
            let d = [
                l[0] * b1[0] + l[1] * b2[0] + l[2] * normal[0],
                l[0] * b1[1] + l[1] * b2[1] + l[2] * normal[1],
                l[0] * b1[2] + l[1] * b2[2] + l[2] * normal[2],
            ];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            sum = [
                sum[0] + d[0] / len,
                sum[1] + d[1] / len,
                sum[2] + d[2] / len,
            ];
        }
        let sum_len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        let bent = [sum[0] / sum_len, sum[1] / sum_len, sum[2] / sum_len];
        let rgb = [
            encode_unorm(bent[0] * 0.5 + 0.5),
            encode_unorm(bent[1] * 0.5 + 0.5),
            encode_unorm(bent[2] * 0.5 + 0.5),
        ];
        let confidence = encode_unorm(sum_len / (rays.max(1) as f32));
        (rgb, confidence)
    }

    /// Pins the mirror's constants to the shader's decimal literals: both
    /// must round to the same `f32` bits, or the mean test's byte-exactness
    /// rests on a lie. (The literals here are deliberately written the
    /// short way clippy's `excessive_precision`/`approx_constant` lints
    /// demand; this test proves the truncation is lossless.)
    #[test]
    fn mirror_constants_match_shader_decimals() {
        let golden_mirror: f32 = 0.618034;
        let golden_shader: f32 = "0.6180339887498949".parse().unwrap();
        assert_eq!(golden_mirror.to_bits(), golden_shader.to_bits());
        assert_eq!(
            std::f32::consts::PI.to_bits(),
            "3.14159265358979".parse::<f32>().unwrap().to_bits()
        );
    }

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
        assert_eq!(size_of::<AoUniform>(), 96);
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
    fn ao_bake_params_new_leaves_bent_normals_off() {
        let params = AoBakeParams::new(10.0, 0.01, test_plane());
        assert!(!params.bent_normals);
    }

    #[test]
    fn with_bent_normals_builder_sets_the_flag() {
        let plane = test_plane();
        assert!(
            AoBakeParams::new(10.0, 0.01, plane)
                .with_bent_normals(true)
                .bent_normals
        );
        assert!(
            !AoBakeParams::new(10.0, 0.01, plane)
                .with_bent_normals(false)
                .bent_normals
        );
        // The builder preserves every other field.
        let params = AoBakeParams {
            rays: 8,
            ..AoBakeParams::new(10.0, 0.01, plane).with_bent_normals(true)
        };
        assert_eq!(params.rays, 8);
        assert!(params.bent_normals);
    }

    #[test]
    fn uniform_for_maps_bent_bool_to_word() {
        let plane = test_plane();
        let off = uniform_for(&AoBakeParams::new(10.0, 0.01, plane), 7);
        assert_eq!(
            (
                off.rays,
                off.max_distance,
                off.bias,
                off.tri_count,
                off.bent_normals
            ),
            (AoBakeParams::DEFAULT_RAYS, 10.0, 0.01, 7, 0)
        );
        let on = uniform_for(
            &AoBakeParams::new(10.0, 0.01, plane).with_bent_normals(true),
            7,
        );
        assert_eq!(on.bent_normals, 1);
        assert_eq!(on.tri_count, 7);
    }

    #[test]
    fn hemisphere_mirror_stratifies_cos_theta() {
        // Stratification is exact rational arithmetic in f32 — pins the
        // mirror's spine independently of any trig implementation.
        assert_eq!(hemisphere_sample_mirror(0, 16)[2], 0.96875);
        assert_eq!(hemisphere_sample_mirror(15, 16)[2], 0.03125);
        assert_eq!(hemisphere_sample_mirror(0, 1)[2], 0.5);
        // Spot-check the golden-angle azimuth of ray 1: phi = 2*PI*fract(G)
        // with G ~= 0.618034 lands in the third quadrant — both lateral
        // components negative.
        let s1 = hemisphere_sample_mirror(1, 16);
        assert!(s1[0] < 0.0, "{s1:?}");
        assert!(s1[1] < 0.0, "{s1:?}");
    }

    #[test]
    fn orthonormal_basis_mirror_is_identity_for_plus_z() {
        let (b1, b2) = orthonormal_basis_mirror([0.0, 0.0, 1.0]);
        for (got, want) in b1.iter().zip([1.0, 0.0, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{b1:?}");
        }
        for (got, want) in b2.iter().zip([0.0, 1.0, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{b2:?}");
        }
    }

    #[test]
    fn encode_unorm_pins_anchor_values() {
        assert_eq!(encode_unorm(0.0), 0);
        assert_eq!(encode_unorm(0.5), 128);
        assert_eq!(encode_unorm(1.0), 255);
    }

    /// The byte-exactness license for the GPU mean test: every channel of
    /// [`expected_unoccluded_bent`] at the test's ray count must sit well
    /// clear of a `Rgba8Unorm` quantization boundary, so last-ulp CPU/GPU
    /// trig differences (libm vs. the driver's builtins, MAD contraction
    /// in the basis steering) cannot flip a byte. Fails loudly — with the
    /// offending values — if anyone changes the ray count into a boundary
    /// collision.
    #[test]
    fn expected_unoccluded_bent_has_quantization_margin() {
        let (rgb, confidence) = expected_unoccluded_bent(16, [0.0, 0.0, 1.0]);
        // Recompute the pre-quantized floats the same way the helper does.
        let (b1, b2) = orthonormal_basis_mirror([0.0, 0.0, 1.0]);
        let normal = [0.0, 0.0, 1.0];
        let mut sum = [0.0f32; 3];
        for i in 0..16 {
            let l = hemisphere_sample_mirror(i, 16);
            let d = [
                l[0] * b1[0] + l[1] * b2[0] + l[2] * normal[0],
                l[0] * b1[1] + l[1] * b2[1] + l[2] * normal[1],
                l[0] * b1[2] + l[1] * b2[2] + l[2] * normal[2],
            ];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            sum = [
                sum[0] + d[0] / len,
                sum[1] + d[1] / len,
                sum[2] + d[2] / len,
            ];
        }
        let sum_len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        let bent = [sum[0] / sum_len, sum[1] / sum_len, sum[2] / sum_len];
        let floats = [
            bent[0] * 0.5 + 0.5,
            bent[1] * 0.5 + 0.5,
            bent[2] * 0.5 + 0.5,
            sum_len / 16.0,
        ];
        let bytes = [rgb[0], rgb[1], rgb[2], confidence];
        for (v, b) in floats.iter().zip(bytes) {
            let scaled = v * 255.0;
            let dist_to_boundary = (scaled - scaled.round()).abs();
            assert!(
                dist_to_boundary > 0.05,
                "channel too close to a quantization boundary for byte-exact asserts: value {v}, byte {b}"
            );
            assert_eq!(encode_unorm(*v), b);
        }
        // Sanity on the physics: a flat unoccluded hemisphere's mean leans
        // hard +z with modest confidence (~|sum|/n ~= 0.5, never 1).
        assert!(bent[2] > 0.9, "{bent:?}");
        assert!(confidence < 255 && confidence > 0);
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

        /// Self-occlusion mesh for `bake_ao_mesh`'s GPU test: a flat quad
        /// (the surface being baked, full `[0, 1]` UVs, scaled to `±30`
        /// world units) plus a floating equilateral triangle (the
        /// occluder, present purely for AO raycasting) with all three UVs
        /// collapsed to the same point — a zero-UV-area triangle that
        /// `POSITION_BAKE_SHADER`'s `abs(denom) >= DET_EPS` check skips
        /// entirely, so it never claims a position-map texel despite
        /// being part of the same `MeshData` (`bake_ao_mesh` takes one
        /// mesh for both the UV target and the occlusion geometry — real
        /// self-occlusion AO needs exactly this).
        ///
        /// Geometry mirrors `ao_bake_center_occluded_far_corner_clear`'s
        /// hand-derived plane scenario, rotated onto this quad's `+Z`
        /// face normal instead of a plane's arbitrary `u_axis`/`v_axis`
        /// (`build_triangles` computes the quad's normal as `(0, 0, 1)`
        /// for this vertex winding — see that test's doc comment for the
        /// inradius/circumradius bound, which is rotation-invariant and
        /// so holds unchanged here): equilateral, circumradius 18,
        /// centered at `(0, 0, 2)` — 2 world units above the quad along
        /// its normal, directly over the quad's own center.
        fn quad_with_floating_occluder() -> MeshData {
            MeshData {
                positions: vec![
                    [-30.0, -30.0, 0.0],
                    [30.0, -30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [-30.0, 30.0, 0.0],
                    [0.0, 18.0, 2.0],
                    [-15.588_457, -9.0, 2.0],
                    [15.588_457, -9.0, 2.0],
                ],
                normals: vec![[0.0, 0.0, 1.0]; 7],
                uvs: vec![
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6],
                material_names: vec![],
            }
        }

        /// 33x33 so the exact center texel (16, 16) sits at UV (0.5, 0.5)
        /// — see `position::tests::gpu`'s `SIZE` constant for why — which
        /// here means it maps to the quad's own world center `(0, 0, 0)`,
        /// directly beneath the occluder's centroid.
        const MESH_AO_SIZE: u32 = 33;

        #[test]
        fn bake_ao_mesh_center_occluded_far_corner_clear() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };

            let dummy_plane = PlaneDesc::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0],
            );
            let params = AoBakeParams {
                rays: 16,
                ..AoBakeParams::new(15.0, 0.01, dummy_plane)
            };

            let mesh = quad_with_floating_occluder();
            let bytes = bake_ao_mesh(&device, &queue, &mesh, MESH_AO_SIZE, MESH_AO_SIZE, &params)
                .expect("bake should succeed");
            assert_eq!(bytes.len(), (MESH_AO_SIZE * MESH_AO_SIZE * 4) as usize);

            let idx = |x: u32, y: u32| ((y * MESH_AO_SIZE + x) * 4) as usize;

            let center = idx(16, 16);
            assert_eq!(bytes[center + 3], 255, "center texel must be covered");
            assert!(
                bytes[center] < 64,
                "center texel (directly under the occluder) should be heavily occluded: {}",
                bytes[center]
            );

            let far_corner = idx(0, 0);
            assert_eq!(
                bytes[far_corner + 3],
                255,
                "far-corner texel must be covered"
            );
            assert!(
                bytes[far_corner] >= 200,
                "far-corner texel (outside the occluder's reach) should be unoccluded: {}",
                bytes[far_corner]
            );
        }

        #[test]
        fn bake_ao_mesh_rejects_zero_rays_without_touching_the_gpu_pipeline() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let plane = PlaneDesc::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0],
            );
            let mut params = AoBakeParams::new(10.0, 0.01, plane);
            params.rays = 0;
            let err = bake_ao_mesh(
                &device,
                &queue,
                &quad_with_floating_occluder(),
                8,
                8,
                &params,
            )
            .unwrap_err();
            assert!(matches!(err, AoBakeError::InvalidRayCount));
        }

        #[test]
        fn bake_ao_mesh_propagates_position_map_errors() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let plane = PlaneDesc::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0],
            );
            let params = AoBakeParams::new(10.0, 0.01, plane);
            let err =
                bake_ao_mesh(&device, &queue, &MeshData::default(), 8, 8, &params).unwrap_err();
            assert!(matches!(
                err,
                AoBakeError::PositionMap(crate::position::PositionMapError::EmptyMesh)
            ));
        }

        /// Reads texel `(x, y)` of a `width`-wide RGBA8 `Vec<u8>` (as
        /// returned by [`bake_ao_mesh`]/[`bake_ao_and_bent_mesh`]).
        fn texel_rgba8(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = ((y * width + x) * 4) as usize;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        fn dummy_plane() -> PlaneDesc {
            PlaneDesc::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0],
            )
        }

        /// Single flat quad (the mean-test fixture): world `[-1, 1]²` at
        /// `z = 0` with full-`[0, 1]` UVs and the `[0, 1, 2, 0, 2, 3]`
        /// winding whose face normal is `(0, 0, 1)` — the same shape as
        /// `normal_map::tests::gpu::full_uv_quad`. Nothing else shares the
        /// mesh, so no ray can hit anything: every covered texel is fully
        /// unoccluded and its bent sum is the whole 16-ray sample set.
        fn unoccluded_quad() -> MeshData {
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

        /// THE MEAN TEST (the design doc's centerpiece): with nothing to
        /// occlude, the bent output of every covered texel equals EXACTLY
        /// the Rust-mirrored hemisphere mean ([`expected_unoccluded_bent`]
        /// — the WGSL `hemisphere_sample` ported op-for-op plus the same
        /// Duff-basis steering the shader uses for this quad's `(0, 0, 1)`
        /// normal), byte-for-byte. Note what this does NOT assert: that
        /// the mean equals the normal (it doesn't — only the z leans that
        /// way; x/y carry the spiral's residual, which is exactly why the
        /// test mirrors the pattern instead of assuming symmetry).
        /// Byte-exactness is licensed by the CPU-side
        /// `expected_unoccluded_bent_has_quantization_margin` test: no
        /// channel sits near a `Rgba8Unorm` boundary, so last-ulp CPU/GPU
        /// trig differences cannot flip a byte.
        #[test]
        fn bent_mean_matches_mirrored_spiral_on_unoccluded_quad() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const SIZE: u32 = 8;
            let params = AoBakeParams {
                rays: 16,
                ..AoBakeParams::new(10.0, 0.01, dummy_plane()).with_bent_normals(true)
            };
            let both =
                bake_ao_and_bent_mesh(&device, &queue, &unoccluded_quad(), SIZE, SIZE, &params)
                    .expect("bake should succeed");
            assert_eq!(both.ao.len(), (SIZE * SIZE * 4) as usize);
            assert_eq!(both.bent.len(), (SIZE * SIZE * 4) as usize);

            let (expected_rgb, expected_confidence) =
                super::expected_unoccluded_bent(16, [0.0, 0.0, 1.0]);
            let expected = [
                expected_rgb[0],
                expected_rgb[1],
                expected_rgb[2],
                expected_confidence,
            ];
            for y in 0..SIZE {
                for x in 0..SIZE {
                    assert_eq!(
                        texel_rgba8(&both.ao, SIZE, x, y),
                        [255, 255, 255, 255],
                        "texel ({x}, {y}) must be fully unoccluded"
                    );
                    assert_eq!(
                        texel_rgba8(&both.bent, SIZE, x, y),
                        expected,
                        "texel ({x}, {y}) bent must equal the mirrored spiral mean"
                    );
                }
            }
        }

        /// Floor quad (full UVs, `±30`) plus one occluder WALL standing on
        /// the `x = 2` plane (`y ∈ [-30, 30]`, `z ∈ [0, 10]`), with
        /// collapsed all-`[0, 0]` UVs so the position pass never covers it
        /// (the same zero-UV-area trick
        /// `quad_with_floating_occluder` uses) while the raycast still
        /// hits it. One-sided on purpose: every blocked ray leans `+x`,
        /// so the survivor mean must tilt `-x` — away from the occluder —
        /// with no symmetry to hide behind.
        fn quad_with_side_wall() -> MeshData {
            MeshData {
                positions: vec![
                    [-30.0, -30.0, 0.0],
                    [30.0, -30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [-30.0, 30.0, 0.0],
                    [2.0, -30.0, 0.0],
                    [2.0, 30.0, 0.0],
                    [2.0, 30.0, 10.0],
                    [2.0, -30.0, 10.0],
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

        /// Occluder-tilt test: the center texel (world `(0, 0, 0)`, wall
        /// 2 units up `+x`) has part of its hemisphere eaten, so its bent
        /// normal tilts AWAY from the wall (`dot(bent, (-1, 0, 0)) > 0`,
        /// i.e. the encoded x byte reads below the `0.5` band) and its
        /// confidence drops below 1 — while the far corner (31 world units
        /// from the wall, past `max_distance = 15`) stays fully
        /// unoccluded and matches the mean test's exact expectation.
        #[test]
        fn bent_tilts_away_from_side_wall_occluder() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const SIZE: u32 = 33;
            let params = AoBakeParams {
                rays: 16,
                ..AoBakeParams::new(15.0, 0.01, dummy_plane()).with_bent_normals(true)
            };
            let mesh = quad_with_side_wall();
            let both = bake_ao_and_bent_mesh(&device, &queue, &mesh, SIZE, SIZE, &params)
                .expect("bake should succeed");

            // Center texel (16, 16) sits at UV (0.5, 0.5) -> world
            // (0, 0, 0) (see MESH_AO_SIZE's doc comment above).
            let center_ao = texel_rgba8(&both.ao, SIZE, 16, 16);
            let center_bent = texel_rgba8(&both.bent, SIZE, 16, 16);
            assert_eq!(center_ao[3], 255, "center texel must be covered");
            assert!(
                center_ao[0] < 255,
                "center texel must be partially occluded by the wall: {}",
                center_ao[0]
            );
            assert!(
                center_bent[3] < 255,
                "confidence must drop below 1: {}",
                center_bent[3]
            );
            assert!(
                center_bent[0] < 128,
                "bent x must tilt away from the +x wall (dot with (-1,0,0) > 0): {:?}",
                center_bent
            );

            // Far corner (0, 0) maps to world (-29.1, +29.1, 0) — over 31
            // units from the wall, unreachable within max_distance, so it
            // must match the unoccluded mean exactly (both maps).
            let corner_ao = texel_rgba8(&both.ao, SIZE, 0, 0);
            let corner_bent = texel_rgba8(&both.bent, SIZE, 0, 0);
            assert_eq!(corner_ao, [255, 255, 255, 255]);
            let (expected_rgb, expected_confidence) =
                super::expected_unoccluded_bent(16, [0.0, 0.0, 1.0]);
            assert_eq!(
                corner_bent,
                [
                    expected_rgb[0],
                    expected_rgb[1],
                    expected_rgb[2],
                    expected_confidence
                ]
            );
            // And the occluded center is strictly less confident than it.
            assert!(
                center_bent[3] < corner_bent[3],
                "occluded confidence {} must be below unoccluded {}",
                center_bent[3],
                corner_bent[3]
            );
        }

        /// Closed-box fixture for the fully-occluded texel: the baked
        /// floor quad plus four walls (`|x| = 4` / `|y| = 4`, `z ∈ [0, 8]`)
        /// and a ceiling (`z = 8`, `±8²`), all with collapsed UVs (never
        /// position-covered, always raycast). Hand-verified in the doc
        /// comment below that every one of the 16 rays from the center
        /// texel hits a face within `max_distance = 15`.
        fn quad_in_closed_box() -> MeshData {
            MeshData {
                positions: vec![
                    [-30.0, -30.0, 0.0],
                    [30.0, -30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [-30.0, 30.0, 0.0],
                    // +x / -x walls.
                    [4.0, -4.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [4.0, 4.0, 8.0],
                    [4.0, -4.0, 8.0],
                    [-4.0, 4.0, 0.0],
                    [-4.0, -4.0, 0.0],
                    [-4.0, -4.0, 8.0],
                    [-4.0, 4.0, 8.0],
                    // +y / -y walls.
                    [-4.0, 4.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [4.0, 4.0, 8.0],
                    [-4.0, 4.0, 8.0],
                    [4.0, -4.0, 0.0],
                    [-4.0, -4.0, 0.0],
                    [-4.0, -4.0, 8.0],
                    [4.0, -4.0, 8.0],
                    // Ceiling.
                    [-8.0, -8.0, 8.0],
                    [8.0, -8.0, 8.0],
                    [8.0, 8.0, 8.0],
                    [-8.0, 8.0, 8.0],
                ],
                normals: vec![[0.0, 0.0, 1.0]; 24],
                uvs: vec![
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                indices: vec![
                    0, 1, 2, 0, 2, 3, // Floor.
                    4, 5, 6, 4, 6, 7, // +x wall.
                    8, 9, 10, 8, 10, 11, // -x wall.
                    12, 13, 14, 12, 14, 15, // +y wall.
                    16, 17, 18, 16, 18, 19, // -y wall.
                    20, 21, 22, 20, 22, 23, // Ceiling.
                ],
                material_names: vec![],
            }
        }

        /// Fully-occluded texel contract: the center texel sees the box on
        /// every ray (proof sketch: any unit ray with `cos(theta) > 0`
        /// either lands on the `±8²` ceiling within `t < 15`, or its
        /// dominant lateral axis exceeds `~0.58`, meeting a wall below
        /// `z = 8` within `t < 7` — the unit-length constraint leaves no
        /// third option; all 16 rays have `cos(theta) >= 0.03125`), so AO
        /// reads exactly 0, the bent sum is exactly zero, and the bent
        /// texel reads the geometric normal `(0, 0, 1)` encoded with
        /// confidence 0. Red/green assert the two-value driver band, not
        /// one exact byte: exact-`0.5` `Rgba8Unorm` quantization rounds to
        /// 127 or 128 depending on the driver (see `normal_map`'s
        /// flat-quad test precedent); blue (`1.0`) and alpha (`0.0`) are
        /// exact everywhere.
        #[test]
        fn fully_occluded_texel_encodes_geometric_normal_with_zero_confidence() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const SIZE: u32 = 33;
            let params = AoBakeParams {
                rays: 16,
                ..AoBakeParams::new(15.0, 0.01, dummy_plane()).with_bent_normals(true)
            };
            let both =
                bake_ao_and_bent_mesh(&device, &queue, &quad_in_closed_box(), SIZE, SIZE, &params)
                    .expect("bake should succeed");

            let center_ao = texel_rgba8(&both.ao, SIZE, 16, 16);
            assert_eq!(center_ao, [0, 0, 0, 255], "center must be fully occluded");

            let center_bent = texel_rgba8(&both.bent, SIZE, 16, 16);
            assert_eq!(center_bent[3], 0, "confidence must be exactly 0");
            assert!(
                (127..=128).contains(&center_bent[0]),
                "bent x must encode the geometric normal's 0: {}",
                center_bent[0]
            );
            assert!(
                (127..=128).contains(&center_bent[1]),
                "bent y must encode the geometric normal's 0: {}",
                center_bent[1]
            );
            assert_eq!(
                center_bent[2], 255,
                "bent z must encode the geometric normal's 1"
            );
        }

        /// OFF-REGRESSION + determinism (ao.rs has no pre-existing
        /// determinism test — the ID baker's `double_bake_is_byte_identical`
        /// is the pattern this follows): on the floating-occluder fixture
        /// (occluded center AND clear corner in one bake),
        /// - baking twice with the flag off is byte-identical,
        /// - the dual entry's AO half equals the plain entry's output
        ///   exactly (the bent accumulator never perturbs AO, flag on or
        ///   off — this is the byte-regression core: flag-off output is
        ///   what the pre-change shader wrote, since the AO path is
        ///   untouched integer hit counting),
        /// - the bent half is byte-identical across the two dual bakes.
        /// The pre-existing AO golden tests above run unmodified alongside.
        #[test]
        fn flag_off_ao_matches_plain_entry_and_double_bake_is_deterministic() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let mesh = quad_with_floating_occluder();
            let off = AoBakeParams {
                rays: 16,
                ..AoBakeParams::new(15.0, 0.01, dummy_plane())
            };
            assert!(!off.bent_normals);
            let on = off.with_bent_normals(true);

            let plain_first =
                bake_ao_mesh(&device, &queue, &mesh, MESH_AO_SIZE, MESH_AO_SIZE, &off)
                    .expect("plain bake should succeed");
            let plain_second =
                bake_ao_mesh(&device, &queue, &mesh, MESH_AO_SIZE, MESH_AO_SIZE, &off)
                    .expect("plain re-bake should succeed");
            assert_eq!(
                plain_first, plain_second,
                "flag-off AO must be deterministic"
            );

            let dual_first =
                bake_ao_and_bent_mesh(&device, &queue, &mesh, MESH_AO_SIZE, MESH_AO_SIZE, &on)
                    .expect("dual bake should succeed");
            let dual_second =
                bake_ao_and_bent_mesh(&device, &queue, &mesh, MESH_AO_SIZE, MESH_AO_SIZE, &on)
                    .expect("dual re-bake should succeed");
            assert_eq!(
                dual_first.ao, plain_first,
                "dual entry's AO half must equal the plain entry byte-for-byte"
            );
            assert_eq!(
                dual_first.bent, dual_second.bent,
                "bent map must be deterministic"
            );
            assert_eq!(
                dual_first.ao, dual_second.ao,
                "dual AO half must be deterministic"
            );
        }
    }
}

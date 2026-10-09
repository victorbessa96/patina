//! The mesh render pass: pipeline, GPU buffers, and the egui paint-callback
//! glue that lets it draw into egui's shared wgpu render pass.
//!
//! Pipeline descriptor shapes are copied from the vendored egui-wgpu 0.36.2
//! renderer (`docs/claw-artifacts/umber-gpu/egui_wgpu_renderer.rs:354-436`),
//! not written from memory, since wgpu 30 changed several field shapes
//! (`depth_write_enabled`/`depth_compare` became `Option`, pipeline layouts
//! take `immediate_size`, vertex/fragment states take `entry_point: Option`
//! and `compilation_options`).

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt as _;

use crate::shaders::MESH_SHADER;

/// Errors raised while preparing GPU-side mesh data.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// The source mesh had zero vertices.
    #[error("mesh has no vertices")]
    EmptyMesh,
    /// The index buffer length wasn't a multiple of 3, so it can't be a
    /// triangle list.
    #[error("mesh index count {0} is not a multiple of 3 (not a triangle list)")]
    InvalidIndexCount(usize),
    /// Index count exceeds u32::MAX — cannot be expressed in a
    /// [`wgpu::IndexFormat::Uint32`] draw range (review #3).
    #[error("index count {0} exceeds u32::MAX")]
    IndexCountOverflow(usize),
}

/// Interleaved vertex layout uploaded to the GPU: position + normal,
/// padded to 32 bytes.
///
/// UVs are intentionally omitted — this pass is normal-shaded only; texture
/// sampling lands with the paint engine. `_pad` keeps the struct at a
/// 16-byte multiple — portable across backends (review #2: some
/// Vulkan/Metal drivers are stricter than the WebGPU floor) and leaves a
/// natural slot for Wave-2 UV attributes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Object-space normal (recomputed on upload if the source mesh lacks
    /// one — see [`MeshBuffers::upload`]).
    pub normal: [f32; 3],
    /// Unused; pads the struct to 32 bytes.
    pub _pad: [f32; 2],
}

/// Per-frame camera uniform, matching `shaders::MESH_SHADER`'s `Camera`
/// struct layout exactly (16-byte alignment: `light_dir` carries an unused
/// `w` so it packs cleanly after the mat4x4).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// Combined projection * view matrix, column-major.
    pub view_proj: [[f32; 4]; 4],
    /// Normalized direction the light travels (surface -> fragment); `w` is
    /// unused padding.
    pub light_dir: [f32; 4],
}

impl CameraUniform {
    /// Builds the uniform from a view-projection matrix and a light
    /// direction (need not be pre-normalized).
    pub fn new(view_proj: glam::Mat4, light_dir: glam::Vec3) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
        }
    }
}

/// The single device context + mesh render pipeline for the viewport.
///
/// Owns no surface or frame-loop state — eframe/egui own the surface and
/// the shared render pass; this only owns the pipeline and the device
/// handles needed to build and update [`MeshBuffers`].
pub struct GpuContext {
    /// The eframe-provided device, reused for every buffer/pipeline this
    /// crate creates.
    pub device: wgpu::Device,
    /// The eframe-provided queue, used to submit buffer writes.
    pub queue: wgpu::Queue,
    /// The eframe-provided adapter (kept for diagnostics via [`crate::Backend`]).
    pub adapter: wgpu::Adapter,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_format: Option<wgpu::TextureFormat>,
}

/// Depth format used by the viewport mesh pipeline when depth is enabled.
///
/// The app requests the matching bit depth from eframe (`DEPTH_FORMAT_BITS`)
/// so the egui renderer's shared render pass carries the attachment; the
/// value is kept here so umber-app never names a wgpu type.
pub fn depth_format() -> Option<wgpu::TextureFormat> {
    Some(wgpu::TextureFormat::Depth32Float)
}

impl GpuContext {
    /// The depth format this context's pipeline depth-tests against.
    ///
    /// Returned as an opaque value so umber-app can pass it to
    /// `GpuContext::new` without naming a wgpu type.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth_format
    }

    /// `adapter`/`device`/`queue` must come from eframe's
    /// `CreationContext::wgpu_render_state` — this never constructs a new
    /// `wgpu::Instance` or requests its own adapter.
    ///
    /// `color_format` should be `RenderState::target_format` so the pipeline
    /// matches the surface egui is already configured for.
    ///
    /// `depth_format` is `None` in the current wiring: the shared render
    /// pass egui builds only gains a depth attachment if eframe threads
    /// `NativeOptions::depth_buffer` through to the wgpu `Renderer`'s
    /// `RendererOptions::depth_stencil_format` — that internal wiring isn't
    /// in the vendored source, so flipping this to `Some(..)` without first
    /// verifying it on real hardware risks a wgpu validation panic on the
    /// first painted frame. Depth correctness is instead provided by the
    /// callback owning its own depth texture (see `MeshPaintCallback`).
    /// See `LANDING_NOTES.md` and review finding #1.
    pub fn new(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber_mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MESH_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("umber_mesh_camera_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<CameraUniform>() as u64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("umber_mesh_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("umber_mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: Some("vs_main"),
                module: &shader,
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                unclipped_depth: false,
                conservative: false,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::default(),
                polygon_mode: wgpu::PolygonMode::default(),
                strip_index_format: None,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: 1,
                mask: !0,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            device,
            queue,
            adapter,
            pipeline,
            bind_group_layout,
            depth_format,
        }
    }
}

/// GPU-resident mesh data for the viewport: vertex/index buffers plus the
/// per-mesh camera uniform buffer and its bind group.
pub struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl MeshBuffers {
    /// Uploads `mesh` to GPU buffers. Recomputes vertex normals when the
    /// source mesh has none (or a mismatched count) rather than failing —
    /// OBJ files without `vn` lines and in-flight glTF/FBX loaders are
    /// expected to hit this path.
    pub fn upload(gpu: &GpuContext, mesh: &umber_mesh::MeshData) -> Result<Self, GpuError> {
        let device = &gpu.device;
        if mesh.positions.is_empty() {
            return Err(GpuError::EmptyMesh);
        }
        if !mesh.indices.len().is_multiple_of(3) {
            return Err(GpuError::InvalidIndexCount(mesh.indices.len()));
        }

        let normals: Cow<'_, [[f32; 3]]> = if mesh.normals.len() == mesh.positions.len() {
            Cow::Borrowed(&mesh.normals)
        } else {
            Cow::Owned(compute_vertex_normals(&mesh.positions, &mesh.indices))
        };

        let vertices: Vec<Vertex> = mesh
            .positions
            .iter()
            .zip(normals.iter())
            .map(|(p, n)| Vertex {
                position: *p,
                normal: *n,
                _pad: [0.0; 2],
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_mesh_index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_mesh_camera_uniform_buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::new(
                glam::Mat4::IDENTITY,
                glam::Vec3::NEG_Y,
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_mesh_camera_bind_group"),
            layout: &gpu.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let index_count = u32::try_from(mesh.indices.len())
            .map_err(|_| GpuError::IndexCountOverflow(mesh.indices.len()))?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            bind_group,
        })
    }

    /// Number of indices in this mesh's index buffer (3x the triangle count).
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Builds this frame's paint callback. `uniform` is computed by the
    /// caller from the current camera + viewport aspect ratio.
    pub fn paint_callback(&self, gpu: &GpuContext, uniform: CameraUniform) -> MeshPaintCallback {
        MeshPaintCallback {
            pipeline: gpu.pipeline.clone(),
            bind_group: self.bind_group.clone(),
            vertex_buffer: self.vertex_buffer.clone(),
            index_buffer: self.index_buffer.clone(),
            index_count: self.index_count,
            uniform_buffer: self.uniform_buffer.clone(),
            uniform,
        }
    }
}

/// Per-frame egui paint callback for the mesh pass.
///
/// Holds cloned wgpu resource handles (cheap: wgpu resource types are
/// internally `Arc`-backed) rather than reaching back into `GpuContext`/
/// `MeshBuffers`, because [`egui_wgpu::CallbackTrait`] callbacks must be
/// `'static` + `Send + Sync`.
pub struct MeshPaintCallback {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform: CameraUniform,
}

impl egui_wgpu::CallbackTrait for MeshPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
        Vec::new()
    }

    fn paint(
        &self,
        _info: epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Wraps `callback` into an `epaint::Shape` ready for `ui.painter().add(..)`.
///
/// Kept in `umber-gpu` (rather than calling `egui_wgpu::Callback` from
/// `umber-app` directly) so the app crate never needs to name an
/// egui-wgpu/wgpu type itself — it only ever holds opaque values handed
/// back by this crate.
pub fn mesh_paint_shape(rect: epaint::emath::Rect, callback: MeshPaintCallback) -> epaint::Shape {
    egui_wgpu::Callback::new_paint_callback(rect, callback).into()
}

/// Smooth per-vertex normals for a shared-vertex (single-indexed) triangle
/// mesh, used when the source data has no normals at all (or a mismatched
/// count — some importers emit per-face data).
fn compute_vertex_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut accum = vec![glam::Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (Some(&pa), Some(&pb), Some(&pc)) =
            (positions.get(a), positions.get(b), positions.get(c))
        else {
            continue;
        };
        let (pa, pb, pc) = (
            glam::Vec3::from(pa),
            glam::Vec3::from(pb),
            glam::Vec3::from(pc),
        );
        // Assumes CCW front faces, matching the pipeline's cull_mode::Back
        // (review #7).
        let face_normal = (pb - pa).cross(pc - pa);
        accum[a] += face_normal;
        accum[b] += face_normal;
        accum[c] += face_normal;
    }
    accum
        .into_iter()
        .map(|n| {
            let n = n.normalize_or_zero();
            if n == glam::Vec3::ZERO {
                glam::Vec3::Y
            } else {
                n
            }
            .to_array()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_vertex_normals_single_triangle_points_up() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let indices = [0u32, 1, 2];
        let normals = compute_vertex_normals(&positions, &indices);
        assert_eq!(normals.len(), 3);
        for n in normals {
            let n = glam::Vec3::from(n);
            assert!((n.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn compute_vertex_normals_handles_degenerate_triangle() {
        let positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let indices = [0u32, 1, 2];
        let normals = compute_vertex_normals(&positions, &indices);
        assert_eq!(normals.len(), 3);
        for n in normals {
            assert!(glam::Vec3::from(n).is_finite());
        }
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        fn try_request_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .ok()?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                    .ok()?;
            Some((adapter, device, queue))
        }

        #[test]
        fn mesh_buffers_upload_roundtrip() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!("skipping mesh_buffers_upload_roundtrip: no wgpu adapter available");
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let mesh = umber_mesh::MeshData {
                positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                normals: vec![],
                uvs: vec![],
                indices: vec![0, 1, 2],
                material_names: vec![],
            };
            let buffers = MeshBuffers::upload(&gpu, &mesh).expect("upload should succeed");
            assert_eq!(buffers.index_count(), 3);

            let uniform = CameraUniform::new(glam::Mat4::IDENTITY, glam::Vec3::NEG_Y);
            let _callback = buffers.paint_callback(&gpu, uniform);
        }

        #[test]
        fn empty_mesh_is_rejected() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!("skipping empty_mesh_is_rejected: no wgpu adapter available");
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let mesh = umber_mesh::MeshData::default();
            assert!(matches!(
                MeshBuffers::upload(&gpu, &mesh),
                Err(GpuError::EmptyMesh)
            ));
        }
    }
}

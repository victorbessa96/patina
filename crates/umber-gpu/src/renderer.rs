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

use crate::ibl::{EnvFlags, EnvIrradiance};
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
    /// World-space eye position; `w` unused. The mesh pass ignores it
    /// (no view-dependent term); `OPENPBR_SHADER` reads it for its
    /// specular/Fresnel view vector.
    pub eye: [f32; 4],
}

impl CameraUniform {
    /// Builds the uniform from a view-projection matrix and a light
    /// direction (need not be pre-normalized). Eye defaults to the
    /// +Z axis — use [`Self::with_eye`] for view-dependent shading.
    pub fn new(view_proj: glam::Mat4, light_dir: glam::Vec3) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            eye: [0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Sets the world-space eye position (view-dependent shading).
    #[must_use]
    pub fn with_eye(mut self, eye: glam::Vec3) -> Self {
        self.eye = [eye.x, eye.y, eye.z, 0.0];
        self
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
    /// 1×1 fallback irradiance texture, bound at group 0 binding 1
    /// whenever no environment is loaded. Never sampled while
    /// `env_flags == 0` (the shader's procedural arm doesn't touch it),
    /// but the bind group must always be fully populated.
    fallback_env_view: wgpu::TextureView,
    /// Shared irradiance sampler (group 0 binding 2) — bilinear,
    /// repeat-U/clamp-V (see `crate::ibl::ibl_sampler`).
    env_sampler: wgpu::Sampler,
    depth_format: Option<wgpu::TextureFormat>,
    /// Textured-quad display pipeline for paint-target presentation
    /// (built once; see `texture_display`).
    texture_display: crate::texture_display::TextureDisplay,
    /// OpenPBR viewport pipeline (binding 0 camera + binding 1 params).
    openpbr_pipeline: wgpu::RenderPipeline,
    /// Bind-group layout shared by OpenPBR draw calls (camera + params).
    openpbr_layout: wgpu::BindGroupLayout,
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

    /// The texture-display pipeline for presenting paint targets.
    pub fn texture_display(&self) -> &crate::texture_display::TextureDisplay {
        &self.texture_display
    }

    /// Creates a uniform buffer holding `params`, ready for an OpenPBR
    /// bind group (binding 1 in [`Self::openpbr_draw`]).
    pub fn openpbr_params_buffer(&self, params: &crate::material::OpenPbrParams) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("umber_openpbr_params"),
                contents: params.as_bytes(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Renders `buffers` with the OpenPBR surface model: returns a paint
    /// callback shape drawing the mesh with `params` through the OpenPBR
    /// pipeline. `camera_uniform_buffer` is the same buffer the plain
    /// mesh pass uses (rebound here at group 0 of the OpenPBR layout).
    ///
    /// The params buffer is written on `prepare` every frame from
    /// `params` so live parameter edits show without re-upload plumbing.
    pub fn openpbr_paint_shape(
        &self,
        rect: epaint::emath::Rect,
        buffers: &MeshBuffers,
        uniform: CameraUniform,
        params: crate::material::OpenPbrParams,
    ) -> epaint::Shape {
        let callback = self.openpbr_callback(buffers, uniform, params);
        egui_wgpu::Callback::new_paint_callback(rect, callback).into()
    }

    /// Builds the OpenPBR callback without the egui shape wrapper —
    /// used by [`Self::openpbr_paint_shape`] and by offscreen tests.
    pub fn openpbr_callback(
        &self,
        buffers: &MeshBuffers,
        uniform: CameraUniform,
        params: crate::material::OpenPbrParams,
    ) -> OpenPbrPaintCallback {
        let params_buffer = self.openpbr_params_buffer(&params);
        // Fresh camera uniform buffer per callback: the OpenPBR layout's
        // binding 0 is byte-compatible with the mesh pass's.
        let camera_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("umber_openpbr_camera"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_openpbr_bind_group"),
            layout: &self.openpbr_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &camera_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        OpenPbrPaintCallback {
            pipeline: self.openpbr_pipeline.clone(),
            bind_group,
            vertex_buffer: buffers.vertex_buffer.clone(),
            index_buffer: buffers.index_buffer.clone(),
            index_count: buffers.index_count,
        }
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<CameraUniform>() as u64),
                    },
                    count: None,
                },
                // IBL set (Wave-4 item 6 — see `shaders::MESH_SHADER`):
                // the convolved irradiance map (always Rgba16Float,
                // filterable), its sampler, and the selector uniform.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<EnvFlags>() as u64),
                    },
                    count: None,
                },
            ],
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
            depth_stencil: depth_stencil.clone(),
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

        let texture_display = crate::texture_display::TextureDisplay::new(&device, color_format);

        // OpenPBR pipeline: camera (binding 0) + material params
        // (binding 1, the 96-byte OpenPbrParams uniform).
        let openpbr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("umber_openpbr_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<CameraUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(96),
                    },
                    count: None,
                },
            ],
        });
        let openpbr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber_openpbr_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::OPENPBR_SHADER)),
        });
        let openpbr_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("umber_openpbr_pipeline_layout"),
                bind_group_layouts: &[Some(&openpbr_layout)],
                immediate_size: 0,
            });
        let openpbr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("umber_openpbr_pipeline"),
            layout: Some(&openpbr_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &openpbr_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
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
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: 1,
                mask: !0,
            },
            fragment: Some(wgpu::FragmentState {
                module: &openpbr_shader,
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

        let (fallback_env_view, env_sampler) = Self::fallback_env_resources(&device);

        Self {
            device,
            queue,
            adapter,
            pipeline,
            bind_group_layout,
            depth_format,
            texture_display,
            openpbr_pipeline,
            openpbr_layout,
            fallback_env_view,
            env_sampler,
        }
    }

    /// Builds the fallback IBL resources for [`Self::new`]: a 1×1
    /// `Rgba16Float` texture view (never sampled while `env_flags == 0`,
    /// but the bind group must stay fully populated) and the shared
    /// irradiance sampler (bilinear, repeat-U/clamp-V).
    fn fallback_env_resources(device: &wgpu::Device) -> (wgpu::TextureView, wgpu::Sampler) {
        let fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("umber_mesh_fallback_irradiance"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        (
            fallback.create_view(&wgpu::TextureViewDescriptor::default()),
            crate::ibl::ibl_sampler(device),
        )
    }

    /// Builds the mesh pass's group-0 bind group: camera + IBL set
    /// (irradiance view, sampler, flags buffer). Used by
    /// [`MeshBuffers::upload`] (fallback view/sampler, procedural
    /// flags) and [`MeshBuffers::set_environment`] (the map's own view
    /// and sampler, or the fallback pair).
    pub(crate) fn mesh_bind_group(
        &self,
        uniform_buffer: &wgpu::Buffer,
        flags_buffer: &wgpu::Buffer,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_mesh_camera_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: flags_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }
}

/// GPU-resident mesh data for the viewport: vertex/index buffers plus the
/// per-mesh camera uniform buffer, the environment-flag uniform buffer,
/// and the combined bind group (camera + IBL set).
pub struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    flags_buffer: wgpu::Buffer,
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
        // Procedural fallback by default: no map bound, flags == 0.
        // `set_environment` flips this when an environment loads.
        let flags_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_mesh_env_flags_buffer"),
            contents: bytemuck::bytes_of(&EnvFlags::procedural()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = gpu.mesh_bind_group(
            &uniform_buffer,
            &flags_buffer,
            &gpu.fallback_env_view,
            &gpu.env_sampler,
        );

        let index_count = u32::try_from(mesh.indices.len())
            .map_err(|_| GpuError::IndexCountOverflow(mesh.indices.len()))?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            flags_buffer,
            bind_group,
        })
    }

    /// Swaps the bound environment: `Some(env)` samples the convolved
    /// map (`env_flags = 1`), `None` restores the procedural fallback
    /// (`env_flags = 0`, 1×1 fallback texture bound but never sampled).
    /// The next [`Self::paint_callback`] picks up the new bind group —
    /// the viewport builds one per frame, so this takes effect
    /// immediately with no re-upload.
    pub fn set_environment(&mut self, gpu: &GpuContext, env: Option<&EnvIrradiance>) {
        let flags = match env {
            Some(_) => EnvFlags::from_map(),
            None => EnvFlags::procedural(),
        };
        gpu.queue
            .write_buffer(&self.flags_buffer, 0, bytemuck::bytes_of(&flags));
        let (view, sampler) = match env {
            Some(env) => (env.view(), env.sampler()),
            None => (&gpu.fallback_env_view, &gpu.env_sampler),
        };
        self.bind_group =
            gpu.mesh_bind_group(&self.uniform_buffer, &self.flags_buffer, view, sampler);
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

/// Per-frame egui paint callback for the OpenPBR mesh pass.
///
/// Structural mirror of [`MeshPaintCallback`]: cloned handles, params
/// baked into the bind group at shape-build time.
pub struct OpenPbrPaintCallback {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl OpenPbrPaintCallback {
    /// Draws the OpenPBR mesh into `render_pass` — the exact state the
    /// egui `CallbackTrait::paint` applies, reusable by offscreen tests.
    pub(crate) fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

impl egui_wgpu::CallbackTrait for OpenPbrPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Buffers were initialized at creation (create_buffer_init maps +
        // writes), nothing to stage per frame.
        Vec::new()
    }

    fn paint(
        &self,
        _info: epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        self.draw(render_pass);
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
        // NOTE: `wgpu::util::DeviceExt` (for `create_buffer_init` below)
        // arrives via this glob: the parent module imports it and glob
        // imports carry a module's `use` names to child modules — so no
        // explicit import here (clippy would flag it unused).
        use super::super::*;
        use crate::ibl::{EnvFormat, EnvIrradiance};

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

        /// Offscreen-renders a triangle through the OpenPBR pipeline and
        /// returns the 4x4 target's bytes.
        fn render_openpbr(gpu: &GpuContext, params: crate::material::OpenPbrParams) -> Vec<u8> {
            let device = &gpu.device;
            let mesh = umber_mesh::MeshData {
                positions: vec![[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]],
                normals: vec![[0.0, 0.0, 1.0]; 3],
                uvs: vec![],
                indices: vec![0, 1, 2],
                material_names: vec![],
            };
            let buffers = MeshBuffers::upload(gpu, &mesh).expect("upload succeeds");
            let uniform = CameraUniform::new(glam::Mat4::IDENTITY, glam::Vec3::NEG_Y);
            let callback = gpu.openpbr_callback(&buffers, uniform, params);

            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("umber_openpbr_test_target"),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_openpbr_test_encoder"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("umber_openpbr_test_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                callback.draw(&mut pass);
            }
            gpu.queue.submit(Some(encoder.finish()));

            // Readback: rows must respect COPY_BYTES_PER_ROW_ALIGNMENT
            // (256); 4x4 RGBA rows are 16 bytes, so pad each row to 256
            // and extract the first 16 bytes of every padded row.
            const PADDED_ROW: u32 = 256;
            let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("umber_openpbr_test_readback"),
                size: PADDED_ROW as u64 * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_openpbr_test_readback_encoder"),
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(PADDED_ROW),
                        rows_per_image: Some(4),
                    },
                },
                wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit(Some(enc.finish()));

            let (sx, rx) = std::sync::mpsc::channel();
            readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sx.send(result);
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("poll succeeds");
            rx.recv().expect("map callback ran").expect("map ok");
            let padded = readback_buffer
                .slice(..)
                .get_mapped_range()
                .expect("mapped range available")
                .to_vec();
            readback_buffer.unmap();
            let mut data = Vec::with_capacity(4 * 4 * 4);
            for row in padded.chunks_exact(PADDED_ROW as usize) {
                data.extend_from_slice(&row[..16]);
            }
            data
        }

        #[test]
        fn openpbr_defaults_render_nonzero_and_bounded() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!("skipping openpbr_defaults_render_nonzero_and_bounded: no wgpu adapter available");
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let bytes = render_openpbr(&gpu, crate::material::OpenPbrParams::default());
            // Non-zero: the triangle covers every pixel of the 4x4.
            let nonzero = bytes
                .chunks_exact(4)
                .any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
            assert!(nonzero, "OpenPBR default render must be non-zero");
            // Energy sanity note: channels are u8 and in-range by
            // construction; a NaN leak surfaces as saturation, which
            // the nonzero check plus the live-render tests would
            // catch as a pixel-level anomaly. No vacuous range
            // assertion here (clippy: absurd_extreme_comparisons).
        }

        #[test]
        fn openpbr_metalness_changes_response() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!("skipping openpbr_metalness_changes_response: no wgpu adapter available");
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let dielectric = render_openpbr(&gpu, crate::material::OpenPbrParams::default());
            let mut metal = crate::material::OpenPbrParams::default();
            metal.surface[0] = 1.0; // base_metalness = 1
            let metallic = render_openpbr(&gpu, metal);
            assert_ne!(
                dielectric, metallic,
                "metalness=1 must shade differently from metalness=0"
            );
        }

        // --- Wave-4 item 6 (IBL) mesh-pass tests. `golden::RenderTarget`
        // can't serve here — its `callback_paint` is a documented no-op
        // stub — so these tests draw for real: same pipeline, same bind
        // group, same buffers as `MeshPaintCallback::paint`, into a
        // linear (non-sRGB) Rgba8Unorm target so the expected bytes are
        // closed-form shader math, not tone-mapped guesses.

        /// Full-screen triangle (the big-triangle trick: covers every
        /// pixel) with a constant normal, CCW so backface culling keeps
        /// it (see `compute_vertex_normals`' CCW note).
        fn fullscreen_mesh(normal: [f32; 3]) -> umber_mesh::MeshData {
            umber_mesh::MeshData {
                positions: vec![[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]],
                normals: vec![normal; 3],
                uvs: vec![],
                indices: vec![0, 1, 2],
                material_names: vec![],
            }
        }

        /// Renders `buffers` (with its CURRENT bind group — whatever
        /// `set_environment` last installed) into an 8×8 linear target
        /// and returns tightly-packed RGBA8 bytes.
        fn render_mesh_offscreen(gpu: &GpuContext, buffers: &MeshBuffers) -> Vec<u8> {
            const W: u32 = 8;
            const H: u32 = 8;
            let device = &gpu.device;
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("umber_ibl_test_target"),
                size: wgpu::Extent3d {
                    width: W,
                    height: H,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let uniform = CameraUniform::new(glam::Mat4::IDENTITY, glam::Vec3::NEG_Y);
            gpu.queue
                .write_buffer(&buffers.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_ibl_test_encoder"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("umber_ibl_test_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                // The exact state `MeshPaintCallback::paint` applies.
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &buffers.bind_group, &[]);
                pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..buffers.index_count, 0, 0..1);
            }
            gpu.queue.submit(Some(encoder.finish()));

            const PADDED_ROW: u32 = 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("umber_ibl_test_readback"),
                size: PADDED_ROW as u64 * H as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_ibl_test_readback_encoder"),
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(PADDED_ROW),
                        rows_per_image: Some(H),
                    },
                },
                wgpu::Extent3d {
                    width: W,
                    height: H,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit(Some(enc.finish()));

            let (sx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sx.send(result);
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("poll succeeds");
            rx.recv().expect("map callback ran").expect("map ok");
            let padded = readback
                .slice(..)
                .get_mapped_range()
                .expect("mapped range available")
                .to_vec();
            readback.unmap();
            let mut data = Vec::with_capacity((W * H * 4) as usize);
            for row in padded.chunks_exact(PADDED_ROW as usize) {
                data.extend_from_slice(&row[..(W * 4) as usize]);
            }
            data
        }

        /// Closed-form procedural byte for a constant normal under the
        /// test light (`NEG_Y`, so `to_light = +Y`): the shader's
        /// `BASE_COLOR * (mix(GROUND, SKY, t) + diffuse * SUN * 0.55)`,
        /// rounded half-up. The GPU's own unorm rounding (nearest-even
        /// vs. half-up) can only disagree at an exact .5-in-byte-units
        /// boundary — the margin note below shows the test normal sits
        /// ≥0.03 byte-units clear of every boundary (the ao.rs
        /// `encode_unorm` precedent).
        fn expected_procedural_byte(normal: [f32; 3]) -> [u8; 4] {
            const GROUND: [f32; 3] = [0.30, 0.27, 0.24];
            const SKY: [f32; 3] = [0.45, 0.55, 0.78];
            const BASE: [f32; 3] = [0.72, 0.72, 0.75];
            const SUN: [f32; 3] = [1.0, 0.96, 0.90];
            let t = (normal[1] * 0.5 + 0.5).clamp(0.0, 1.0);
            let env = [
                GROUND[0] + (SKY[0] - GROUND[0]) * t,
                GROUND[1] + (SKY[1] - GROUND[1]) * t,
                GROUND[2] + (SKY[2] - GROUND[2]) * t,
            ];
            // to_light = +Y (light_dir = NEG_Y).
            let diffuse = normal[1].max(0.0);
            let px = [
                BASE[0] * (env[0] + diffuse * SUN[0] * 0.55),
                BASE[1] * (env[1] + diffuse * SUN[1] * 0.55),
                BASE[2] * (env[2] + diffuse * SUN[2] * 0.55),
            ];
            [
                (px[0] * 255.0 + 0.5).floor() as u8,
                (px[1] * 255.0 + 0.5).floor() as u8,
                (px[2] * 255.0 + 0.5).floor() as u8,
                255,
            ]
        }

        /// FALLBACK REGRESSION (the design's test 3): with `env_flags =
        /// 0` the mesh pass renders the procedural path byte-identical
        /// to the closed-form expectation above — the shader change must
        /// not reshape the fallback by a single LSB beyond unorm
        /// rounding. Normal +Z, light +Y: diffuse = 0, t = 0.5, expected
        /// (69, 75, 98, 255); margins to the nearest .5 boundary are
        /// 0.35/0.22/0.04 byte-units — f32 arithmetic noise is ~1e-5, so
        /// a ≤1-LSB tolerance only admits genuine rounding, never drift.
        #[test]
        fn fallback_procedural_matches_closed_form() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!(
                    "skipping fallback_procedural_matches_closed_form: no wgpu adapter available"
                );
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let mesh = fullscreen_mesh([0.0, 0.0, 1.0]);
            let buffers = MeshBuffers::upload(&gpu, &mesh).expect("upload succeeds");
            let bytes = render_mesh_offscreen(&gpu, &buffers);
            let want = expected_procedural_byte([0.0, 0.0, 1.0]);
            assert_eq!(bytes.len(), 8 * 8 * 4);
            for (i, px) in bytes.chunks_exact(4).enumerate() {
                for c in 0..4 {
                    let diff = px[c].abs_diff(want[c]);
                    assert!(
                        diff <= 1,
                        "pixel {i} ch{c}: got {px:?}, want {want:?} (≤1 LSB unorm rounding)"
                    );
                }
            }
        }

        /// The flag-0 path must be *texture-independent*: the same scene
        /// with a bright-white map bound but flags forced to 0 renders
        /// byte-IDENTICAL (not within tolerance — identical) to the
        /// fallback bind group. An `if` on a uniform is uniform control
        /// flow, so the unsampled texture cannot leak into the output;
        /// any difference is a real wiring bug.
        #[test]
        fn fallback_ignores_bound_map() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!("skipping fallback_ignores_bound_map: no wgpu adapter available");
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let mesh = fullscreen_mesh([0.0, 1.0, 0.0]);
            let buffers = MeshBuffers::upload(&gpu, &mesh).expect("upload succeeds");
            let plain = render_mesh_offscreen(&gpu, &buffers);

            // Bright uniform map, but flags forced procedural: build the
            // bind group by hand (the public `set_environment` couples
            // map+flag, which is exactly what this test bypasses).
            let white = vec![4.0f32; 16 * 8 * 4];
            let env = EnvIrradiance::from_equirect(
                &gpu.device,
                &gpu.queue,
                bytemuck::cast_slice(&white),
                16,
                8,
                EnvFormat::Rgba32Float,
            )
            .expect("convolve succeeds");
            let flags = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("umber_ibl_test_flags0"),
                    contents: bytemuck::bytes_of(&crate::ibl::EnvFlags::procedural()),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let mut masked = MeshBuffers::upload(&gpu, &mesh).expect("upload succeeds");
            masked.bind_group =
                gpu.mesh_bind_group(&masked.uniform_buffer, &flags, env.view(), env.sampler());
            let masked_bytes = render_mesh_offscreen(&gpu, &masked);

            assert_eq!(
                plain, masked_bytes,
                "flags=0 must ignore the bound map byte-for-byte"
            );
        }

        /// THE CAN-FAIL END-TO-END (the design's test 4): the same scene
        /// with the map enabled vs. procedural must DIFFER — an IBL that
        /// silently no-ops fails this. Synthetic two-tone equirect (red
        /// +Y cap, green -Y) so the delta is unmistakable at a +Y-facing
        /// normal; threshold is >50% of pixels differing by >8 LSB in
        /// any channel — robust across adapters (convolution + unorm
        /// noise is ~1 LSB, the expected delta is ~100).
        #[test]
        fn env_enabled_differs_from_procedural() {
            let Some((adapter, device, queue)) = try_request_device() else {
                eprintln!(
                    "skipping env_enabled_differs_from_procedural: no wgpu adapter available"
                );
                return;
            };
            let gpu = GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            );
            let mesh = fullscreen_mesh([0.0, 1.0, 0.0]);
            let mut buffers = MeshBuffers::upload(&gpu, &mesh).expect("upload succeeds");
            let plain = render_mesh_offscreen(&gpu, &buffers);

            // Two-tone equirect in GPU row order (row 0 = -Y = green,
            // top rows = +Y = red).
            const W: u32 = 64;
            const H: u32 = 32;
            let mut pixels = vec![0.0f32; (W * H * 4) as usize];
            for y in 0..H {
                let top = y >= H / 2;
                for x in 0..W {
                    let base = ((y * W + x) * 4) as usize;
                    if top {
                        pixels[base..base + 4].copy_from_slice(&[3.0, 0.25, 0.2, 1.0]);
                    } else {
                        pixels[base..base + 4].copy_from_slice(&[0.2, 1.5, 0.25, 1.0]);
                    }
                }
            }
            let env = EnvIrradiance::from_equirect(
                &gpu.device,
                &gpu.queue,
                bytemuck::cast_slice(&pixels),
                W,
                H,
                EnvFormat::Rgba32Float,
            )
            .expect("convolve succeeds");
            buffers.set_environment(&gpu, Some(&env));
            let mapped = render_mesh_offscreen(&gpu, &buffers);

            assert_eq!(plain.len(), mapped.len());
            let mut differ = 0usize;
            let (mut sum_r, mut sum_b) = (0u64, 0u64);
            for (a, b) in plain.chunks_exact(4).zip(mapped.chunks_exact(4)) {
                if a[0].abs_diff(b[0]) > 8 || a[1].abs_diff(b[1]) > 8 || a[2].abs_diff(b[2]) > 8 {
                    differ += 1;
                }
                sum_r += b[0] as u64;
                sum_b += b[2] as u64;
            }
            let total = plain.len() / 4;
            assert!(
                differ > total / 2,
                "env render must differ from procedural on most pixels (got {differ}/{total})"
            );
            // The map actually drove the output: red cap above means
            // the +Y-facing render skews red over blue.
            assert!(
                sum_r > sum_b,
                "mapped render should skew red (R sum {sum_r} vs B sum {sum_b})"
            );

            // And unsetting restores the fallback exactly.
            buffers.set_environment(&gpu, None);
            let restored = render_mesh_offscreen(&gpu, &buffers);
            assert_eq!(
                plain, restored,
                "unsetting the env must restore fallback bytes"
            );
        }
    }
}

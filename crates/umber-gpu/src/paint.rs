//! The paint-target compute pass: an RGBA8 storage texture plus a compute
//! pipeline that splats dab batches into it with premultiplied alpha-over
//! compositing.
//!
//! Wave 2 scope (docs/specs/architecture.md "Paint data flow"): one tile is
//! one [`PaintTarget`] texture; the tile-pool allocator that packs many
//! tiles into a shared atlas is a later slice. The dab batch this pass
//! consumes is the GPU-side mirror of `umber_brush::spacing::DabPlan` —
//! `DabPlan` carries stroke-space position + pressure; by the time a batch
//! reaches here it has been resolved into paint-space [`Dab`]s (position,
//! radius, premultiplied color, alpha, hardness) ready to rasterize.
//!
//! **Overlap contract:** a single [`PaintCompositor::splat_dabs`] call is
//! one compute dispatch, one workgroup per dab, with no atomics on the
//! shared storage texture. Dabs within one call must not have overlapping
//! bounding circles, or the two workgroups race on the shared texels.
//! Overlapping dabs (e.g. consecutive dabs along a stroke) must be split
//! across separate `splat_dabs` calls — wgpu's resource hazard tracking
//! orders successive compute passes on the same texture, so a second call
//! always sees the first call's writes. See `shaders::PAINT_COMPUTE_SHADER`
//! for the shader-side half of this contract.

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt as _;

use crate::shaders::PAINT_COMPUTE_SHADER;

/// A single stamp to composite into a [`PaintTarget`].
///
/// `color` is premultiplied (`color.rgb` already scaled by `color.a`);
/// `alpha` is a separate, independent flow/opacity multiplier applied on
/// top (so a stroke can fade in/out without re-baking it into `color`).
/// `hardness` is the radial falloff knob: `1.0` is a hard-edged disc, `0.0`
/// falls off linearly-ish from the center.
///
/// Field *declaration* order here is `pos, radius, alpha, color, hardness`
/// rather than the conceptual `pos, radius, color, alpha, hardness` order,
/// plus a trailing `_pad` — purely to make the `#[repr(C)]` byte layout
/// match `shaders::PAINT_COMPUTE_SHADER`'s WGSL `Dab` struct, where
/// `vec4<f32>` forces 16-byte alignment that Rust's `[f32; 4]` (4-byte
/// aligned) wouldn't otherwise produce. [`Dab::new`] takes the conceptual
/// order; the padding field being private means a struct-literal can't be
/// built any other way.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Dab {
    /// Dab center, in target texel space.
    pub pos: [f32; 2],
    /// Dab radius, in texels.
    pub radius: f32,
    /// Flow/opacity multiplier applied on top of `color`'s own alpha.
    pub alpha: f32,
    /// Premultiplied RGBA.
    pub color: [f32; 4],
    /// Radial falloff: `1.0` hard edge, `0.0` soft falloff from the center.
    pub hardness: f32,
    _pad: [f32; 3],
}

const _: () = assert!(size_of::<Dab>() == 48, "Dab must match the WGSL Dab layout");

impl Dab {
    /// Builds a dab from its conceptual fields (padding is internal).
    pub fn new(pos: [f32; 2], radius: f32, color: [f32; 4], alpha: f32, hardness: f32) -> Self {
        Self {
            pos,
            radius,
            alpha,
            color,
            hardness,
            _pad: [0.0; 3],
        }
    }
}

/// Errors from staging or splatting a dab batch.
#[derive(Debug, thiserror::Error)]
pub enum PaintError {
    /// `splat_dabs`/`DabBuffer::stage` was called with no dabs.
    #[error("dab batch is empty")]
    EmptyBatch,
    /// The batch exceeds the compositor's/buffer's configured capacity.
    #[error("dab batch of {requested} exceeds capacity {capacity}")]
    CapacityOverflow {
        /// Dabs the caller tried to stage.
        requested: usize,
        /// The configured maximum.
        capacity: usize,
    },
    /// `device` lacks `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`,
    /// which this pass's `read_write` storage-texture binding requires on
    /// `Rgba8Unorm` (that access mode is only free on the core r32
    /// formats; everything else is a non-portable adapter capability
    /// gated behind this feature). See `LANDING_NOTES_PAINT.md`.
    #[error(
        "device is missing wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, \
         required for read_write storage-texture access on Rgba8Unorm"
    )]
    MissingDeviceFeature,
    /// `PaintThread::publish` tried to send on its command channel after the
    /// receiving end was dropped. `PaintThread` owns both ends for its whole
    /// lifetime, so in practice this is unreachable — it exists because
    /// `mpsc::Sender::send` returns a `Result` that must be handled.
    #[error("paint-thread command channel is closed")]
    ChannelClosed,
    /// [`PaintTarget::read_back_rgba8`](PaintTarget::read_back_rgba8) failed:
    /// device poll, buffer-map callback loss, or the map itself.
    #[error("paint-target readback failed: {0}")]
    Readback(String),
}

/// CPU-side staging for a dab batch, uploaded to a GPU storage buffer just
/// before a dispatch.
///
/// `capacity` bounds how many dabs one batch may hold — in practice this
/// should never exceed `wgpu::Limits::max_compute_workgroups_per_dimension`
/// (one workgroup per dab), which is exactly what
/// [`PaintCompositor::splat_dabs`] enforces when it stages internally.
pub struct DabBuffer {
    capacity: usize,
    staged: Vec<Dab>,
}

impl DabBuffer {
    /// Builds an empty buffer that rejects batches larger than `capacity`.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            staged: Vec::new(),
        }
    }

    /// The configured maximum batch size.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Replaces the staged batch with `dabs`.
    ///
    /// # Errors
    ///
    /// Returns [`PaintError::EmptyBatch`] if `dabs` is empty, or
    /// [`PaintError::CapacityOverflow`] if it exceeds `capacity`. On error
    /// the previously staged batch is left untouched.
    pub fn stage(&mut self, dabs: &[Dab]) -> Result<(), PaintError> {
        if dabs.is_empty() {
            return Err(PaintError::EmptyBatch);
        }
        if dabs.len() > self.capacity {
            return Err(PaintError::CapacityOverflow {
                requested: dabs.len(),
                capacity: self.capacity,
            });
        }
        self.staged.clear();
        self.staged.extend_from_slice(dabs);
        Ok(())
    }

    /// The currently staged batch.
    pub fn staged(&self) -> &[Dab] {
        &self.staged
    }

    /// Uploads the staged batch to a fresh GPU storage buffer.
    ///
    /// Uses `create_buffer_init` (not `queue.write_buffer`) deliberately:
    /// it writes the bytes synchronously at buffer creation, so there is no
    /// deferred-write ordering hazard if a caller stages and uploads two
    /// different batches onto the same encoder before submitting.
    ///
    /// # Errors
    ///
    /// Returns [`PaintError::EmptyBatch`] if nothing has been staged yet,
    /// rather than handing wgpu a zero-size buffer.
    pub fn upload(&self, device: &wgpu::Device) -> Result<wgpu::Buffer, PaintError> {
        if self.staged.is_empty() {
            return Err(PaintError::EmptyBatch);
        }
        Ok(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("umber_paint_dab_buffer"),
                contents: bytemuck::cast_slice(&self.staged),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        )
    }
}

/// An RGBA8 (linear, `Rgba8Unorm`) GPU paint surface.
///
/// Wave 2 scope: one tile is one `PaintTarget`; the tile-pool atlas that
/// packs many tiles into shared physical pages is a later slice (see
/// docs/specs/architecture.md "Virtual texturing").
pub struct PaintTarget {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    dims_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl PaintTarget {
    /// Creates a new `width`x`height` paint target.
    ///
    /// Newly created wgpu textures are zero-initialized per the WebGPU
    /// spec, so a fresh target already starts fully transparent; call
    /// [`PaintTarget::clear`] to reset an existing target instead.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("umber_paint_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_paint_target_dims"),
            contents: bytemuck::cast_slice(&[width, height]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Self {
            texture,
            texture_view,
            dims_buffer,
            width,
            height,
        }
    }

    /// Clears the whole target to transparent black.
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("umber_paint_target_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
    }

    /// Target dimensions in texels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The backing GPU texture — for readback (e.g. in golden-image tests)
    /// or future tile-pool wiring.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The texture's default view — for display callbacks sampling the
    /// paint surface.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.texture_view
    }

    /// Reads the full target back as tightly-packed RGBA8 bytes
    /// (width×height×4, row-major), blocking until the copy completes.
    ///
    /// Rows are padded to 256 bytes for the GPU copy and de-padded on
    /// extraction, so callers get exactly `width * height * 4` bytes —
    /// the format `umber_export::png::write_png` consumes.
    pub fn read_back_rgba8(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, PaintError> {
        let (width, height) = (self.width, self.height);
        let unpadded_row = width * 4;
        let padding = (256 - (unpadded_row % 256)) % 256;
        let padded_row = unpadded_row + padding;
        let size = padded_row as u64 * height as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("umber_paint_readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("umber_paint_readback_encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
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
            .map_err(|e| PaintError::Readback(format!("device poll: {e}")))?;
        receiver
            .recv()
            .map_err(|_| PaintError::Readback("map callback channel closed".into()))?
            .map_err(|e| PaintError::Readback(format!("buffer map: {e}")))?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| PaintError::Readback(format!("mapped range: {e}")))?;
        let bytes: &[u8] = &data;
        let mut out = Vec::with_capacity(unpadded_row as usize * height as usize);
        for row in 0..height as usize {
            let start = row * padded_row as usize;
            out.extend_from_slice(&bytes[start..start + unpadded_row as usize]);
        }
        drop(data);
        buffer.unmap();
        Ok(out)
    }
}

/// Owns the dab-splat compute pipeline and bind-group layout shared by
/// every [`PaintTarget`] it composites into.
///
/// A fresh dab storage buffer and bind group are built on every
/// [`PaintCompositor::splat_dabs`] call rather than cached per-target: the
/// dab buffer's *contents* change every call (and `wgpu::BindGroup`s bind
/// to a buffer's identity, not a snapshot of its contents), so a cached
/// bind group would either need to be rebuilt anyway or reuse a persistent
/// buffer via `queue.write_buffer` — whose write lands at submit time, not
/// encoder-record time, breaking the ordering two `splat_dabs` calls on one
/// encoder depend on. `splat_dabs` takes `&self` accordingly: there's no
/// cached, mutated state to protect.
pub struct PaintCompositor {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    max_dabs_per_batch: usize,
}

impl PaintCompositor {
    /// Builds the compute pipeline against `device`.
    ///
    /// `device` is cloned (wgpu resource handles are internally
    /// `Arc`-backed, so this is cheap) and kept so `splat_dabs` can create
    /// the per-call dab buffer and bind group without an extra parameter.
    ///
    /// # Errors
    ///
    /// Returns [`PaintError::MissingDeviceFeature`] if `device` wasn't
    /// created with `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` —
    /// checked up front so this fails with a typed error instead of a wgpu
    /// validation panic partway through building the bind-group layout.
    pub fn new(device: wgpu::Device) -> Result<Self, PaintError> {
        if !device
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            return Err(PaintError::MissingDeviceFeature);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber_paint_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PAINT_COMPUTE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("umber_paint_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<Dab>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
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
                        min_binding_size: NonZeroU64::new(size_of::<[u32; 2]>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("umber_paint_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("umber_paint_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let max_dabs_per_batch = device.limits().max_compute_workgroups_per_dimension as usize;

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            max_dabs_per_batch,
        })
    }

    /// The maximum dabs accepted in one [`PaintCompositor::splat_dabs`]
    /// call — one compute workgroup per dab, bounded by
    /// `wgpu::Limits::max_compute_workgroups_per_dimension`.
    pub fn max_dabs_per_batch(&self) -> usize {
        self.max_dabs_per_batch
    }

    /// Records one compute dispatch that splats `dabs` into `target`: one
    /// workgroup per dab, premultiplied alpha-over compositing.
    ///
    /// See the module-level docs for the overlap contract: dabs within one
    /// call must not spatially overlap.
    ///
    /// # Errors
    ///
    /// Returns [`PaintError::EmptyBatch`] if `dabs` is empty, or
    /// [`PaintError::CapacityOverflow`] if `dabs.len()` exceeds
    /// [`PaintCompositor::max_dabs_per_batch`].
    pub fn splat_dabs(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &PaintTarget,
        dabs: &[Dab],
    ) -> Result<(), PaintError> {
        let mut staging = DabBuffer::new(self.max_dabs_per_batch);
        staging.stage(dabs)?;
        let dab_buffer = staging.upload(&self.device)?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_paint_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dab_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &target.dims_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let dab_count = dabs.len() as u32;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("umber_paint_splat_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dab_count, 1, 1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dab_buffer_rejects_empty_batch() {
        let mut buf = DabBuffer::new(16);
        assert!(matches!(buf.stage(&[]), Err(PaintError::EmptyBatch)));
    }

    #[test]
    fn dab_buffer_rejects_overflow() {
        let mut buf = DabBuffer::new(1);
        let dabs = [
            Dab::new([0.0, 0.0], 1.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1.0),
            Dab::new([1.0, 1.0], 1.0, [0.0, 1.0, 0.0, 1.0], 1.0, 1.0),
        ];
        assert!(matches!(
            buf.stage(&dabs),
            Err(PaintError::CapacityOverflow {
                requested: 2,
                capacity: 1
            })
        ));
    }

    #[test]
    fn dab_buffer_stage_replaces_previous_batch() {
        let mut buf = DabBuffer::new(4);
        let first = [Dab::new([0.0, 0.0], 1.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1.0)];
        buf.stage(&first).expect("first batch fits");
        assert_eq!(buf.staged().len(), 1);

        let second = [
            Dab::new([0.0, 0.0], 1.0, [0.0, 1.0, 0.0, 1.0], 1.0, 1.0),
            Dab::new([1.0, 1.0], 1.0, [0.0, 0.0, 1.0, 1.0], 1.0, 1.0),
        ];
        buf.stage(&second).expect("second batch fits");
        assert_eq!(buf.staged().len(), 2);
        assert_eq!(buf.staged()[0].color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn dab_layout_matches_wgsl_struct_size() {
        assert_eq!(size_of::<Dab>(), 48);
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        /// Requests a device with `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
        /// enabled.
        ///
        /// `Rgba8Unorm` reports `STORAGE_READ_WRITE` in
        /// `Adapter::get_texture_format_features` on this sandbox's
        /// adapter, but wgpu only grants non-portable per-format
        /// capabilities (read-write storage access on anything other than
        /// the r32 formats) when this feature is requested on the device —
        /// without it, `PaintCompositor::new`'s bind-group-layout creation
        /// hits a validation error. See `LANDING_NOTES_PAINT.md` for the
        /// production-wiring consequence (eframe's device needs this too).
        fn try_request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
            let instance = wgpu::Instance::default();
            let Ok(adapter) = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            ) else {
                eprintln!("skipping: no wgpu adapter available");
                return None;
            };
            if !adapter
                .features()
                .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
            {
                eprintln!(
                    "skipping: adapter {:?} lacks wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES",
                    adapter.get_info().name
                );
                return None;
            }
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }))
            .ok()
        }

        fn read_back_rgba8(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            target: &PaintTarget,
        ) -> Vec<u8> {
            let (width, height) = target.dimensions();
            let unpadded_row = width * 4;
            let padding = (256 - (unpadded_row % 256)) % 256;
            let padded_row = unpadded_row + padding;
            let size = padded_row as u64 * height as u64;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("umber_paint_test_readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_paint_test_readback_encoder"),
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: target.texture(),
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
                .expect("poll should succeed in tests");
            receiver
                .recv()
                .expect("map_async callback should fire")
                .expect("buffer map should succeed");

            let data = slice.get_mapped_range().expect("mapped range");
            let bytes: &[u8] = &data;
            let mut out = Vec::with_capacity(unpadded_row as usize * height as usize);
            for row in 0..height as usize {
                let start = row * padded_row as usize;
                out.extend_from_slice(&bytes[start..start + unpadded_row as usize]);
            }
            drop(data);
            buffer.unmap();
            out
        }

        fn pixel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = (y * width + x) as usize * 4;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        #[test]
        fn splat_one_dab_paints_center_leaves_edges_untouched() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let compositor =
                PaintCompositor::new(device.clone()).expect("device has the required feature");
            let target = PaintTarget::new(&device, 64, 64);

            let red = Dab::new([32.0, 32.0], 16.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1.0);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_paint_test_encoder"),
            });
            compositor
                .splat_dabs(&mut encoder, &target, &[red])
                .expect("single dab batch should splat");
            queue.submit(Some(encoder.finish()));

            let bytes = read_back_rgba8(&device, &queue, &target);
            let center = pixel(&bytes, 64, 32, 32);
            assert!(center[0] > 200, "center should be strongly red: {center:?}");
            assert!(
                center[1] < 20 && center[2] < 20,
                "center should have no g/b: {center:?}"
            );
            assert!(center[3] > 200, "center should be opaque: {center:?}");

            // Just outside the dab's radius (16 + 2 margin) along +x.
            let edge = pixel(&bytes, 64, 50, 32);
            assert_eq!(
                edge,
                [0, 0, 0, 0],
                "outside the dab radius should be untouched: {edge:?}"
            );

            let corner = pixel(&bytes, 64, 0, 0);
            assert_eq!(
                corner,
                [0, 0, 0, 0],
                "corner should be untouched: {corner:?}"
            );
        }

        #[test]
        fn splat_two_dabs_composites_overlap_alpha_over() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let compositor =
                PaintCompositor::new(device.clone()).expect("device has the required feature");
            let target = PaintTarget::new(&device, 64, 64);

            // Two half-alpha dabs, far enough apart that neither call's
            // single dab overlaps itself, but their footprints overlap each
            // other — hence two separate splat_dabs calls (see the module
            // contract) rather than one batch of two.
            let red = Dab::new([28.0, 32.0], 16.0, [1.0, 0.0, 0.0, 1.0], 0.5, 1.0);
            let blue = Dab::new([36.0, 32.0], 16.0, [0.0, 0.0, 1.0, 1.0], 0.5, 1.0);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_paint_test_encoder_overlap"),
            });
            compositor
                .splat_dabs(&mut encoder, &target, &[red])
                .expect("red batch should splat");
            compositor
                .splat_dabs(&mut encoder, &target, &[blue])
                .expect("blue batch should splat");
            queue.submit(Some(encoder.finish()));

            let bytes = read_back_rgba8(&device, &queue, &target);

            // Red-only region: inside red's radius (center 28, r=16 ->
            // [12, 44]) but outside blue's (center 36, r=16 -> [20, 52]).
            let red_only = pixel(&bytes, 64, 14, 32);
            assert!(
                red_only[0] > 100,
                "red-only region should be red-ish: {red_only:?}"
            );
            assert!(
                red_only[2] < 20,
                "red-only region should have no blue: {red_only:?}"
            );

            // Blue-only region: inside blue's radius but outside red's.
            let blue_only = pixel(&bytes, 64, 50, 32);
            assert!(
                blue_only[2] > 100,
                "blue-only region should be blue-ish: {blue_only:?}"
            );
            assert!(
                blue_only[0] < 20,
                "blue-only region should have no red: {blue_only:?}"
            );

            // Overlap region at the midpoint: blue was drawn second (over
            // red) at half alpha, so it should carry both some red
            // (showing through) and the dominant blue contribution.
            let overlap = pixel(&bytes, 64, 32, 32);
            assert!(
                overlap[2] > overlap[0],
                "blue drawn over red should dominate: {overlap:?}"
            );
            assert!(
                overlap[0] > 20,
                "red should still show through underneath: {overlap:?}"
            );
            // Two 0.5-alpha overs compose to 1 - (1 - 0.5)^2 = 0.75 coverage
            // (~191/255), not full opacity.
            assert!(
                (170..210).contains(&overlap[3]),
                "two half-alpha overs should land near 0.75 coverage: {overlap:?}"
            );
        }

        /// Exercises, in one batch, everything the two tests above don't:
        /// `workgroup_id.x` beyond 0 (three dabs → the shader indexes
        /// `dabs[1]`/`dabs[2]`, proving the 48-byte stride is right, not
        /// just that one `dabs[0]` read works), bounding-box clamping
        /// against both the min and max edges of the target, and a
        /// non-hard (`hardness = 0.0`) falloff. Finishes by clearing the
        /// target and confirming it goes back to fully transparent.
        #[test]
        fn splat_three_dabs_one_batch_clamps_edges_and_falls_off_then_clears() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let compositor =
                PaintCompositor::new(device.clone()).expect("device has the required feature");
            let target = PaintTarget::new(&device, 64, 64);

            // dab0: hard disc clipped against the min (0,0) corner.
            let corner_min = Dab::new([2.0, 2.0], 6.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1.0);
            // dab1: soft (hardness 0) falloff, fully interior — no clamping.
            let soft = Dab::new([32.0, 32.0], 12.0, [0.0, 1.0, 0.0, 1.0], 1.0, 0.0);
            // dab2: hard disc clipped against the max (63,63) corner.
            let corner_max = Dab::new([62.0, 62.0], 6.0, [0.0, 0.0, 1.0, 1.0], 1.0, 1.0);
            // None of the three bounding circles overlap each other.
            let dabs = [corner_min, soft, corner_max];

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_paint_test_encoder_three"),
            });
            compositor
                .splat_dabs(&mut encoder, &target, &dabs)
                .expect("three-dab batch should splat");
            queue.submit(Some(encoder.finish()));

            let bytes = read_back_rgba8(&device, &queue, &target);

            // dab0 clamped against the min corner: (0,0) is still inside
            // its radius (distance ~2.1 from center (2,2)) even though the
            // dab's unclamped bounding box would start at x,y = -4.
            let min_corner = pixel(&bytes, 64, 0, 0);
            assert!(
                min_corner[0] > 200 && min_corner[1] < 20 && min_corner[2] < 20,
                "min-corner dab should paint (0,0) red: {min_corner:?}"
            );

            // dab2 clamped against the max corner: (63,63) is inside its
            // radius even though the unclamped bounding box would end at
            // x,y = 68 (target is only 64 wide/tall, max index 63).
            let max_corner = pixel(&bytes, 64, 63, 63);
            assert!(
                max_corner[2] > 200 && max_corner[0] < 20 && max_corner[1] < 20,
                "max-corner dab should paint (63,63) blue: {max_corner:?}"
            );

            // dab1's soft falloff: alpha strictly decreases from the
            // center outward toward the radius-12 edge.
            let soft_center = pixel(&bytes, 64, 32, 32);
            let soft_mid = pixel(&bytes, 64, 38, 32);
            let soft_edge = pixel(&bytes, 64, 42, 32);
            assert!(
                soft_center[3] > 240,
                "soft dab center should be nearly opaque: {soft_center:?}"
            );
            assert!(
                soft_center[3] > soft_mid[3] && soft_mid[3] > soft_edge[3],
                "soft dab alpha should strictly decrease outward: center={soft_center:?} mid={soft_mid:?} edge={soft_edge:?}"
            );

            // Points inside none of the three dabs' radii stay untouched.
            let untouched_a = pixel(&bytes, 64, 15, 50);
            let untouched_b = pixel(&bytes, 64, 63, 0);
            assert_eq!(
                untouched_a,
                [0, 0, 0, 0],
                "point outside all three dabs should be untouched: {untouched_a:?}"
            );
            assert_eq!(
                untouched_b,
                [0, 0, 0, 0],
                "point outside all three dabs should be untouched: {untouched_b:?}"
            );

            // clear() resets the whole target back to transparent black.
            let mut clear_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("umber_paint_test_clear_encoder"),
                });
            target.clear(&mut clear_encoder);
            queue.submit(Some(clear_encoder.finish()));

            let cleared = read_back_rgba8(&device, &queue, &target);
            assert!(
                cleared.iter().all(|&b| b == 0),
                "clear() should reset every byte to 0"
            );
        }
    }
}

//! Textured-quad display callback: presents the paint target inside an
//! egui panel (the 2D UV view's live canvas).
//!
//! Structural mirror of `renderer::MeshPaintCallback` — same
//! pipeline-in-callback, cloned-handles, `CallbackTrait` shape — but for
//! a full-target textured quad instead of the 3D mesh pass. The quad
//! covers the whole `PaintTarget`; the clip rect maps it onto the panel
//! region (same approach the mesh callback uses).

use wgpu::util::DeviceExt as _;

/// Full-target textured quad, four vertices, triangle-strip topology.
/// Positions are 0..1 corner coords.
///
/// Texture-coord contract: paint-texture row 0 is the TOP of the painted
/// image. The pointer→UV→texel path (`umber_app::paint_state::push_event`)
/// flips V exactly once, so texel (x, 0) is what the pointer drew at the
/// top of the UV square. The display quad therefore samples row 0 at the
/// top of the screen: pos.y == 0 maps to uv.y == 0. No second flip here —
/// a V-flip in this data would mirror strokes vertically.
const QUAD_VERTS: [TexturedVertex; 4] = [
    TexturedVertex {
        pos: [0.0, 0.0],
        uv: [0.0, 0.0],
    },
    TexturedVertex {
        pos: [1.0, 0.0],
        uv: [1.0, 0.0],
    },
    TexturedVertex {
        pos: [0.0, 1.0],
        uv: [0.0, 1.0],
    },
    TexturedVertex {
        pos: [1.0, 1.0],
        uv: [1.0, 1.0],
    },
];

/// Vertex layout for the display quad: corner position + UV.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TexturedVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

/// WGSL for the textured-quad display pass. Group 1 is the display LUT
/// (`crate::display_lut` — the snippet below is that module's
/// `DISPLAY_LUT_WGSL`, verbatim).
pub(crate) const TEXTURE_DISPLAY_SHADER: &str = r#"
// Full-target textured quad: present the paint surface inside a panel.

// ---- display LUT (crate::display_lut — keep verbatim) ----
@group(1) @binding(0) var display_lut: texture_2d<f32>;

fn display_lut_index(c: f32) -> i32 {
    return i32(floor(clamp(c, 0.0, 1.0) * 255.0 + 0.5));
}

fn apply_display_lut(color: vec3<f32>) -> vec3<f32> {
    let r = textureLoad(display_lut, vec2<i32>(display_lut_index(color.r), 0), 0).r;
    let g = textureLoad(display_lut, vec2<i32>(display_lut_index(color.g), 0), 0).g;
    let b = textureLoad(display_lut, vec2<i32>(display_lut_index(color.b), 0), 0).b;
    return vec3<f32>(r, g, b);
}
// ---- end display LUT ----

struct ScreenUniform {
    resolution: vec2<f32>,  // target surface size (pixels)
    quad_origin: vec2<f32>, // surface-space top-left of the quad
    quad_size: vec2<f32>,   // surface-space quad extent
}

@group(0) @binding(0) var<uniform> screen: ScreenUniform;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv_in: vec2<f32>) -> VertexOutput {
    // Quad corner 0..1 -> surface pixels, then to NDC (Y down in
    // surface space, up in NDC).
    let px = screen.quad_origin + pos * screen.quad_size;
    let ndc = vec2<f32>(
        px.x / screen.resolution.x * 2.0 - 1.0,
        1.0 - (px.y / screen.resolution.y) * 2.0,
    );
    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = uv_in;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(tex, samp, in.uv);
    // The paint buffer is linear rgba8unorm; the pipeline's fragment
    // target format matches the egui surface (non-sRGB). The display
    // LUT is the last step before the target: nearest sampling hands it
    // one of the 256 stored levels, which addresses its entry exactly
    // (the identity LUT returns the texel unchanged). Alpha forced to 1
    // (opaque presentation).
    return vec4<f32>(apply_display_lut(color.rgb), 1.0);
}
"#;

/// CPU-side mirror of the WGSL `ScreenUniform` (repr(C), Pod).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScreenUniformData {
    resolution: [f32; 2],
    quad_origin: [f32; 2],
    quad_size: [f32; 2],
}

/// Static display resources: quad pipeline, bind-group layout, sampler,
/// and the identity display LUT bound when a caller passes no LUT.
/// Built once per `GpuContext`; per-frame work is only the callback.
pub struct TextureDisplay {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    identity_lut: crate::display_lut::DisplayLut,
}

impl TextureDisplay {
    /// Creates the pipeline + shared resources on `device`. `lut_layout`
    /// is the context's shared display-LUT layout (group 1);
    /// `identity_lut` is the context's identity fallback.
    pub(crate) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        lut_layout: &wgpu::BindGroupLayout,
        identity_lut: crate::display_lut::DisplayLut,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber_texture_display_shader"),
            source: wgpu::ShaderSource::Wgsl(TEXTURE_DISPLAY_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("umber_texture_display_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ScreenUniformData>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("umber_texture_display_pipeline_layout"),
            // Group 1: the display LUT (crate::display_lut).
            bind_group_layouts: &[Some(&layout), Some(lut_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("umber_texture_display_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TexturedVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("umber_texture_display_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            layout,
            sampler,
            identity_lut,
        }
    }

    /// Builds the per-frame callback presenting `target` at panel-space
    /// `rect` (egui points; converted to pixels per-frame in `prepare`
    /// using the frame's actual scale factor), through `lut` — the app's
    /// display LUT, or `None` for the identity table (renders exactly
    /// the pre-LUT bytes).
    pub fn callback(
        &self,
        device: &wgpu::Device,
        target: &crate::paint::PaintTarget,
        rect: epaint::emath::Rect,
        lut: Option<&crate::display_lut::DisplayLut>,
    ) -> TextureDisplayCallback {
        self.callback_for_view(device, target.view(), rect, lut)
    }

    /// [`Self::callback`] over any filterable 2D view — the golden test
    /// presents a plain `Rgba8Unorm` texture (a `PaintTarget` needs the
    /// storage-texture feature some test adapters lack).
    pub(crate) fn callback_for_view(
        &self,
        device: &wgpu::Device,
        source: &wgpu::TextureView,
        rect: epaint::emath::Rect,
        lut: Option<&crate::display_lut::DisplayLut>,
    ) -> TextureDisplayCallback {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_texture_display_uniform"),
            // Placeholder contents; `prepare` overwrites this every frame
            // with the pixel-space uniform built from `rect_points` and
            // the frame's `ScreenDescriptor`.
            contents: bytemuck::bytes_of(&ScreenUniformData {
                resolution: [0.0, 0.0],
                quad_origin: [0.0, 0.0],
                quad_size: [0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_texture_display_bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(source),
                },
            ],
        });
        let lut_bind_group = lut.unwrap_or(&self.identity_lut).bind_group().clone();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_texture_display_quad"),
            contents: bytemuck::bytes_of(&QUAD_VERTS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        TextureDisplayCallback {
            pipeline: self.pipeline.clone(),
            bind_group,
            lut_bind_group,
            vertex_buffer,
            uniform_buffer,
            rect_points: rect,
        }
    }
}

/// Per-frame egui paint callback presenting the paint texture.
///
/// Cloned handles, same rationale as `renderer::MeshPaintCallback`
/// (`'static + Send + Sync`).
pub struct TextureDisplayCallback {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// Group 1: the display LUT's bind group (the app's, or the identity
    /// fallback).
    lut_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    /// The quad rect in egui points, as of this frame. Converted to
    /// pixel space in `prepare` using the frame's actual scale factor
    /// (from `ScreenDescriptor`), so the uniform and the surface size
    /// are always in one coordinate system.
    rect_points: epaint::emath::Rect,
}

impl egui_wgpu::CallbackTrait for TextureDisplayCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // egui works in points; the render surface is physical pixels.
        // Same conversion the egui_wgpu renderer applies to clip rects
        // (`ScissorRect::new`): scale by pixels_per_point, and divide by
        // the descriptor's size_in_pixels — the actual surface size —
        // not a caller-reconstructed viewport approximation.
        self.write_uniform(queue, screen_descriptor);
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

impl TextureDisplayCallback {
    /// `prepare`'s uniform write, callable without egui's callback
    /// plumbing (the golden test's offscreen pass).
    pub(crate) fn write_uniform(
        &self,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
    ) {
        let uniform = screen_uniform(&self.rect_points, screen_descriptor);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// The draw `paint` records, onto any render pass.
    pub(crate) fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, &self.lut_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}

/// Wraps the callback into an `epaint::Shape` for `ui.painter().add(..)`.
pub fn texture_display_shape(
    rect: epaint::emath::Rect,
    callback: TextureDisplayCallback,
) -> epaint::Shape {
    egui_wgpu::Callback::new_paint_callback(rect, callback).into()
}

/// Builds the screen uniform: the quad rect (egui points) converted to
/// pixel space with the frame's scale factor, against the surface's
/// actual pixel size. Both quantities in one coordinate system —
/// mixing points with pixels misplaces the quad whenever
/// `pixels_per_point != 1.0` (HiDPI) or the outer viewport is larger
/// than the client surface.
fn screen_uniform(
    rect: &epaint::emath::Rect,
    screen_descriptor: &egui_wgpu::ScreenDescriptor,
) -> ScreenUniformData {
    let ppp = screen_descriptor.pixels_per_point;
    ScreenUniformData {
        resolution: [
            screen_descriptor.size_in_pixels[0] as f32,
            screen_descriptor.size_in_pixels[1] as f32,
        ],
        quad_origin: [rect.left() * ppp, rect.top() * ppp],
        quad_size: [rect.width() * ppp, rect.height() * ppp],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_vertices_are_triangle_strip_order() {
        // Strip order: (0,0) (1,0) (0,1) (1,1) forms two CCW triangles
        // with the standard strip winding. Pins the layout so accidental
        // reorderings break this test, not the render.
        let positions: [[f32; 2]; 4] = QUAD_VERTS.map(|v| v.pos);
        assert_eq!(positions[0], [0.0, 0.0]);
        assert_eq!(positions[1], [1.0, 0.0]);
        assert_eq!(positions[2], [0.0, 1.0]);
        assert_eq!(positions[3], [1.0, 1.0]);
    }

    #[test]
    fn quad_uv_matches_row0_top_contract() {
        // The paint texture's row 0 is the top of the painted image
        // (paint_state::push_event flips V once at input). The display
        // must sample row 0 at the top of the screen — pos.y and uv.y
        // agree — so strokes appear where the pointer drew them. A
        // V-flip here mirrors strokes vertically.
        for v in QUAD_VERTS {
            assert_eq!(v.pos[1], v.uv[1]);
        }
    }

    #[test]
    fn uniform_layout_is_pod() {
        // ScreenUniformData must stay exactly 24 bytes (2+2+2 f32s) for
        // the WGSL struct alignment; repr(C) + Pod is checked at the
        // derive, this pins the size.
        assert_eq!(std::mem::size_of::<ScreenUniformData>(), 24);
    }

    #[test]
    fn screen_uniform_scales_rect_and_surface_consistently() {
        // HiDPI: 2560x1440 physical, 2.0 px/pt. A quad at points
        // (100, 50) sized (300, 300) must land at pixels (200, 100) and
        // cover (600, 600) against the 2560x1440 surface. The old shape
        // (points quad_origin ÷ pixels resolution) failed this: the quad
        // shrank by ppp and misaligned with the clip rect.
        let rect = epaint::emath::Rect::from_min_size(
            epaint::emath::Pos2::new(100.0, 50.0),
            epaint::emath::vec2(300.0, 300.0),
        );
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [2560, 1440],
            pixels_per_point: 2.0,
        };
        let u = screen_uniform(&rect, &sd);
        assert_eq!(u.resolution, [2560.0, 1440.0]);
        assert_eq!(u.quad_origin, [200.0, 100.0]);
        assert_eq!(u.quad_size, [600.0, 600.0]);

        // Scale invariance: the same points-rect against a surface whose
        // pixel dimensions scale with pixels_per_point lands at the same
        // NDC — the quad tracks the window, not the pixel grid.
        let sd1 = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [1280, 720],
            pixels_per_point: 1.0,
        };
        let u1 = screen_uniform(&rect, &sd1);
        let ndc = |u: &ScreenUniformData| {
            (
                (u.quad_origin[0] + u.quad_size[0]) / u.resolution[0] * 2.0 - 1.0,
                1.0 - (u.quad_origin[1] + u.quad_size[1]) / u.resolution[1] * 2.0,
            )
        };
        let (x2, y2) = ndc(&u);
        let (x1, y1) = ndc(&u1);
        assert!((x2 - x1).abs() < 1e-6 && (y2 - y1).abs() < 1e-6);
    }

    // ---- the display-LUT golden (GPU-gated: the harness's adapter
    // rule — any adapter incl. lavapipe/WARP, skip with a note when none;
    // bit-equality is within one adapter, which these comparisons are).

    #[cfg(feature = "gpu")]
    mod gpu {
        use crate::display_lut::{identity_lut_bytes, DisplayLut, DISPLAY_LUT_BYTES};
        use crate::golden::{compare_rgba8, RenderTarget};
        use crate::renderer::GpuContext;

        /// 16×16 = 256 texels: the source holds every byte level per
        /// channel, so every LUT entry is exercised.
        const SIDE: u32 = 16;

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

        /// The context on the eframe surface's format class (non-sRGB
        /// `Rgba8Unorm`), no depth (the display quad has none).
        fn context() -> Option<GpuContext> {
            let (adapter, device, queue) = try_request_device()?;
            Some(GpuContext::new(
                adapter,
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
            ))
        }

        /// Source texel k (row-major): r = k, g = 255 − k, b = 97k mod
        /// 256 (97 is odd, so b also walks all 256 levels), a = k (the
        /// display forces alpha to 1 regardless).
        fn source_bytes() -> Vec<u8> {
            (0..256u32)
                .flat_map(|k| [k as u8, (255 - k) as u8, (k * 97 % 256) as u8, k as u8])
                .collect()
        }

        /// What the pre-LUT shader drew: the texel's rgb, alpha 255.
        fn pre_lut_expected(source: &[u8]) -> Vec<u8> {
            source
                .chunks_exact(4)
                .flat_map(|t| [t[0], t[1], t[2], 255])
                .collect()
        }

        fn source_view(gpu: &GpuContext, bytes: &[u8]) -> wgpu::TextureView {
            let extent = wgpu::Extent3d {
                width: SIDE,
                height: SIDE,
                depth_or_array_layers: 1,
            };
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("umber_lut_golden_source"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // The paint target's format.
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIDE * 4),
                    rows_per_image: Some(SIDE),
                },
                extent,
            );
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        }

        /// Presents `source` 1:1 (quad = the whole 16×16 target at 1
        /// px/pt, so pixel (x, y) samples texel (x, y) — nearest, row 0
        /// at the top) through `lut`, and reads the target back.
        fn present(
            gpu: &GpuContext,
            source: &wgpu::TextureView,
            lut: Option<&DisplayLut>,
        ) -> Vec<u8> {
            let target =
                RenderTarget::with_format(&gpu.device, SIDE, SIDE, wgpu::TextureFormat::Rgba8Unorm);
            let rect = epaint::emath::Rect::from_min_size(
                epaint::emath::Pos2::ZERO,
                epaint::emath::vec2(SIDE as f32, SIDE as f32),
            );
            let callback = gpu
                .texture_display()
                .callback_for_view(&gpu.device, source, rect, lut);
            callback.write_uniform(
                &gpu.queue,
                &egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [SIDE, SIDE],
                    pixels_per_point: 1.0,
                },
            );
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("umber_lut_golden_encoder"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("umber_lut_golden_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view(),
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
                callback.draw(&mut pass);
            }
            gpu.queue.submit(Some(encoder.finish()));
            target
                .read_back(&gpu.device, &gpu.queue)
                .expect("golden readback")
        }

        /// THE GOLDEN (the design's test 3): the UV view's display is
        /// byte-identical to the pre-LUT render with the identity LUT —
        /// both the context's fallback (`None`) and an app-style LUT
        /// built from the identity bytes and bound explicitly.
        #[test]
        fn identity_lut_render_is_byte_identical() {
            let Some(gpu) = context() else {
                eprintln!("skipping identity_lut_render_is_byte_identical: no wgpu adapter");
                return;
            };
            let source = source_bytes();
            let view = source_view(&gpu, &source);
            let expected = pre_lut_expected(&source);

            let fallback = present(&gpu, &view, None);
            compare_rgba8(&fallback, &expected, 0).expect("fallback identity LUT");

            let bound = DisplayLut::new(&gpu, &identity_lut_bytes());
            let with_lut = present(&gpu, &view, Some(&bound));
            compare_rgba8(&with_lut, &expected, 0).expect("bound identity LUT");
            assert_eq!(with_lut, fallback);
        }

        /// THE CAN-FAIL COMPANION: an identity-only golden also passes
        /// if the shader ignores the LUT. An inverting table (entry i =
        /// 255 − i), installed through the app's in-place `upload` path,
        /// must invert every channel — proving the LUT is read, each
        /// channel picks its own entry, and a rebuild needs no rebind.
        #[test]
        fn uploaded_lut_reaches_the_pixels() {
            let Some(gpu) = context() else {
                eprintln!("skipping uploaded_lut_reaches_the_pixels: no wgpu adapter");
                return;
            };
            let source = source_bytes();
            let view = source_view(&gpu, &source);

            let lut = DisplayLut::new(&gpu, &identity_lut_bytes());
            let mut inverted = [0u8; DISPLAY_LUT_BYTES];
            for (i, entry) in inverted.chunks_exact_mut(4).enumerate() {
                let b = 255 - i as u8;
                entry.copy_from_slice(&[b, b, b, 255]);
            }
            lut.upload(&gpu, &inverted);

            let out = present(&gpu, &view, Some(&lut));
            let expected: Vec<u8> = source
                .chunks_exact(4)
                .flat_map(|t| [255 - t[0], 255 - t[1], 255 - t[2], 255])
                .collect();
            compare_rgba8(&out, &expected, 0).expect("inverted LUT");
            assert_ne!(out, pre_lut_expected(&source));
        }
    }
}

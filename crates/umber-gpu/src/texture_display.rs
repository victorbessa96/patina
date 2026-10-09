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
/// Positions are 0..1 corner coords; UVs sample with V flipped (paint
/// texture row 0 = top, UV origin = bottom-left).
const QUAD_VERTS: [TexturedVertex; 4] = [
    TexturedVertex {
        pos: [0.0, 0.0],
        uv: [0.0, 1.0],
    },
    TexturedVertex {
        pos: [1.0, 0.0],
        uv: [1.0, 1.0],
    },
    TexturedVertex {
        pos: [0.0, 1.0],
        uv: [0.0, 0.0],
    },
    TexturedVertex {
        pos: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
];

/// Vertex layout for the display quad: corner position + UV.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TexturedVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

/// WGSL for the textured-quad display pass.
const TEXTURE_DISPLAY_SHADER: &str = r#"
// Full-target textured quad: present the paint surface inside a panel.

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
    // target format matches the egui surface, whose transfer does the
    // sRGB encode. Alpha forced to 1 (opaque presentation).
    return vec4<f32>(color.rgb, 1.0);
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

/// Static display resources: quad pipeline, bind-group layout, sampler.
/// Built once per `GpuContext`; per-frame work is only the callback.
pub struct TextureDisplay {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl TextureDisplay {
    /// Creates the pipeline + shared resources on `device`.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
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
            bind_group_layouts: &[Some(&layout)],
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
                            offset: 16,
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
        }
    }

    /// Builds the per-frame callback presenting `target` at panel-space
    /// `rect` on a surface of `resolution` pixels.
    pub fn callback(
        &self,
        device: &wgpu::Device,
        target: &crate::paint::PaintTarget,
        resolution: [f32; 2],
        rect: epaint::emath::Rect,
    ) -> TextureDisplayCallback {
        let uniform = ScreenUniformData {
            resolution,
            quad_origin: [rect.left(), rect.top()],
            quad_size: [rect.width(), rect.height()],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_texture_display_uniform"),
            contents: bytemuck::bytes_of(&uniform),
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
                    resource: wgpu::BindingResource::TextureView(target.view()),
                },
            ],
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_texture_display_quad"),
            contents: bytemuck::bytes_of(&QUAD_VERTS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        TextureDisplayCallback {
            pipeline: self.pipeline.clone(),
            bind_group,
            vertex_buffer,
            uniform_buffer,
            uniform,
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
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform: ScreenUniformData,
}

impl egui_wgpu::CallbackTrait for TextureDisplayCallback {
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
    fn uniform_layout_is_pod() {
        // ScreenUniformData must stay exactly 24 bytes (2+2+2 f32s) for
        // the WGSL struct alignment; repr(C) + Pod is checked at the
        // derive, this pins the size.
        assert_eq!(std::mem::size_of::<ScreenUniformData>(), 24);
    }
}

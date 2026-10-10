//! The reference-scene benchmark (`perf` feature only) — the measuring
//! harness behind the §12 viewport row of `docs/perf/protocol.md`
//! ("60 fps sustained, P95 frame-time < 16.7ms, 1M-tri + 4K set").
//!
//! The scene:
//! - **Mesh**: [`reference_mesh`] — a cube whose six faces are each an
//!   n×n quad grid ([`REFERENCE_SUBDIVISIONS`] = 289 → 1,002,252 tris).
//!   Pure arithmetic, no RNG: identical bytes every run, layout pinned
//!   by a spot-value test. "Checked in" means the generator is checked in,
//!   not a 30 MB asset.
//! - **4K set**: [`TextureSet`] — four 4096² [`PaintTarget`]s (base
//!   color, normal, roughness/metal, height) cleared to solid values.
//!   RESIDENT, UNSAMPLED: the viewport mesh pass has no texture binding
//!   yet (`Vertex` drops UVs; sampling lands with the paint engine), so
//!   the set contributes VRAM pressure only. When the mesh pass gains
//!   material sampling the bench must bind it — until then the baseline
//!   does not exercise the "+ 4K set" half of the budget, and says so.
//!
//! The frame is the production viewport frame minus egui: the camera
//! uniform write + [`crate::renderer::MeshPaintCallback`]'s own `draw`
//! (not a copy of its state) into a [`TARGET_WIDTH`]×[`TARGET_HEIGHT`]
//! offscreen color + Depth32Float target — the app's depth format. Grid
//! and wireframe overlays are off by default in the app, so they are off
//! here. Each timed frame is `Instant` deltas around update + encode +
//! submit + a blocking `device.poll` on that submission.
//!
//! Configuration is by environment, not flags: libtest parses everything
//! after `--` and rejects unknown options before any test runs, so a
//! `--json` flag cannot reach a `#[test]`. `UMBER_REF_SCENE_FRAMES`
//! (default [`DEFAULT_FRAMES`]) sets the measured frame count;
//! `UMBER_REF_SCENE_JSON=<path>` writes [`RefSceneReport::to_json`] there.
//! The JSON is also always printed as one stdout line.
//!
//! The test asserts harness correctness only — frame count, non-zero
//! frame times, ordered percentiles, and real pixel coverage — and PRINTS
//! the budget verdict without failing on it. Absolute numbers vary by
//! hardware class (lavapipe vs the dev GPU); the budget gate is the
//! nightly job's comparison against `perf/baseline.json` (the protocol's
//! >10% regression rule), not this test.
//!
//! Run: `cargo test -p umber-gpu --features perf --release ref_scene -- --nocapture`

use std::time::{Duration, Instant};

use crate::camera::OrbitCamera;
use crate::paint::PaintTarget;
use crate::renderer::{depth_format, CameraUniform, GpuContext, GpuError, MeshBuffers};

/// Per-face grid subdivisions of the reference cube: 12·289² = 1,002,252 tris.
pub const REFERENCE_SUBDIVISIONS: u32 = 289;
/// Edge length of each map in the 4K texture-set stand-in.
pub const TEXTURE_SET_SIZE: u32 = 4096;
/// Offscreen target width (a 1080p viewport).
pub const TARGET_WIDTH: u32 = 1920;
/// Offscreen target height (a 1080p viewport).
pub const TARGET_HEIGHT: u32 = 1080;
/// Measured frames when `UMBER_REF_SCENE_FRAMES` is unset. The protocol's
/// recorded baselines want ≥ 1000 — set the env var for those.
pub const DEFAULT_FRAMES: usize = 120;
/// Unmeasured frames rendered first (driver/shader JIT on first use).
pub const WARMUP_FRAMES: usize = 3;
/// The §12 viewport budget: P95 frame time under one 60Hz frame.
pub const BUDGET_P95_MS: f64 = 16.7;

/// Color format of the offscreen target (an sRGB surface, like the app's).
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Clear color; any covered pixel differs from it.
const CLEAR: wgpu::Color = wgpu::Color::BLACK;
/// Per-frame orbit step (one degree): the deterministic "update".
const YAW_STEP: f32 = std::f32::consts::PI / 180.0;

/// Builds the cube grid: the [-1,1]³ cube, each face an
/// `subdivisions`×`subdivisions` quad grid with its own (flat-normal,
/// per-face-UV) vertices. Every triangle winds CCW seen from outside, so
/// the viewport pipeline's back-face culling keeps the outer surface.
///
/// Vertices: `6·(n+1)²`; triangles: `12·n²`.
pub fn cube_grid(subdivisions: u32) -> umber_mesh::MeshData {
    assert!(subdivisions > 0, "cube grid needs at least one subdivision");
    // (normal, u, v) with u × v = normal: a→b→c below is then CCW seen
    // from +normal.
    const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    let n = subdivisions as usize;
    let row = n + 1;
    let verts_per_face = row * row;
    let mut mesh = umber_mesh::MeshData {
        positions: Vec::with_capacity(6 * verts_per_face),
        normals: Vec::with_capacity(6 * verts_per_face),
        uvs: Vec::with_capacity(6 * verts_per_face),
        indices: Vec::with_capacity(6 * n * n * 6),
        material_names: vec!["ReferenceMaterial".to_string()],
    };
    for (face, (normal, u, v)) in FACES.iter().enumerate() {
        for j in 0..row {
            let t = j as f32 / n as f32;
            for i in 0..row {
                let s = i as f32 / n as f32;
                let (a, b) = (2.0 * s - 1.0, 2.0 * t - 1.0);
                mesh.positions.push(std::array::from_fn(|k| normal[k] + a * u[k] + b * v[k]));
                mesh.normals.push(*normal);
                mesh.uvs.push([s, t]);
            }
        }
        let base = (face * verts_per_face) as u32;
        for j in 0..n {
            for i in 0..n {
                let a = base + (j * row + i) as u32;
                let b = a + 1;
                let c = a + 1 + row as u32;
                let d = a + row as u32;
                mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
    mesh
}

/// The reference scene's mesh: [`cube_grid`] at [`REFERENCE_SUBDIVISIONS`].
pub fn reference_mesh() -> umber_mesh::MeshData {
    cube_grid(REFERENCE_SUBDIVISIONS)
}

/// The 4K texture-set stand-in: four solid [`TEXTURE_SET_SIZE`]² maps,
/// resident on the device (see the module docs: unsampled until the mesh
/// pass binds material textures).
pub struct TextureSet {
    maps: [PaintTarget; 4],
}

impl TextureSet {
    /// Solid fill per map: base color, tangent-space normal (+Z),
    /// roughness/metal, height.
    const FILLS: [wgpu::Color; 4] = [
        wgpu::Color {
            r: 0.5,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        },
        wgpu::Color {
            r: 0.5,
            g: 0.5,
            b: 1.0,
            a: 1.0,
        },
        wgpu::Color {
            r: 0.5,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        wgpu::Color {
            r: 0.5,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        },
    ];

    /// Allocates the four maps and clears each to its solid fill
    /// (submitted and waited on — outside any timed region).
    pub fn new(gpu: &GpuContext) -> Self {
        let size = TEXTURE_SET_SIZE;
        let maps: [PaintTarget; 4] = std::array::from_fn(|_| PaintTarget::new(&gpu.device, size, size));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_ref_scene_texture_set_fill"),
            });
        for (map, fill) in maps.iter().zip(Self::FILLS) {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("umber_ref_scene_texture_set_fill_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: map.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(fill),
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
        let index = gpu.queue.submit(Some(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("texture-set fill completes");
        Self { maps }
    }

    /// Total resident bytes across the four maps.
    pub fn byte_len(&self) -> usize {
        self.maps.iter().map(PaintTarget::byte_len).sum()
    }
}

/// One benchmark run's numbers. Percentiles are nearest-rank over the
/// measured (post-warmup) frames.
#[derive(Debug, Clone)]
pub struct RefSceneReport {
    /// Raw per-frame times, in render order (warmup excluded).
    pub frame_times: Vec<Duration>,
    /// Unmeasured warmup frames rendered before `frame_times`.
    pub warmup: usize,
    /// Triangles drawn per frame.
    pub tris: usize,
    /// Resident bytes of the texture-set stand-in.
    pub texture_set_bytes: usize,
    /// Offscreen target size in pixels.
    pub target: (u32, u32),
    /// Adapter name (`wgpu::AdapterInfo::name`): the hardware class.
    pub adapter: String,
    /// Adapter device type (`Cpu` = lavapipe-class, `DiscreteGpu`, ...).
    pub device_type: String,
    /// Backend the adapter ran on.
    pub backend: String,
    /// Pixels of the final frame differing from the clear color — proof
    /// the bench timed real rasterization, not an empty pass.
    pub covered_pixels: u64,
    /// Mean frame time, ms.
    pub mean_ms: f64,
    /// P50 frame time, ms.
    pub p50_ms: f64,
    /// P95 frame time, ms.
    pub p95_ms: f64,
    /// P99 frame time, ms.
    pub p99_ms: f64,
}

/// Nearest-rank percentile over a sorted slice (caller sorts ascending);
/// same rule as `umber-app`'s perf soak, over `Duration`s — a 1M-tri
/// frame on a dev GPU is ~1-2ms, so whole-ms buckets would erase it.
fn percentile_sorted(sorted: &[Duration], p: f64) -> Duration {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&p));
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Minimal JSON string escaping (no serde in this crate).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl RefSceneReport {
    fn from_frames(frame_times: Vec<Duration>, scene: SceneInfo) -> Self {
        let mut sorted = frame_times.clone();
        sorted.sort_unstable();
        let total: Duration = sorted.iter().sum();
        Self {
            mean_ms: ms(total) / sorted.len() as f64,
            p50_ms: ms(percentile_sorted(&sorted, 0.50)),
            p95_ms: ms(percentile_sorted(&sorted, 0.95)),
            p99_ms: ms(percentile_sorted(&sorted, 0.99)),
            frame_times,
            warmup: WARMUP_FRAMES,
            tris: scene.tris,
            texture_set_bytes: scene.texture_set_bytes,
            target: (TARGET_WIDTH, TARGET_HEIGHT),
            adapter: scene.adapter,
            device_type: scene.device_type,
            backend: scene.backend,
            covered_pixels: scene.covered_pixels,
        }
    }

    /// Measured frame count.
    pub fn frames(&self) -> usize {
        self.frame_times.len()
    }

    /// Whether P95 meets [`BUDGET_P95_MS`] (reported, never asserted here).
    pub fn within_budget(&self) -> bool {
        self.p95_ms < BUDGET_P95_MS
    }

    /// The human baseline line, in the perf-soak style.
    pub fn baseline_line(&self) -> String {
        let verdict = if self.within_budget() {
            "WITHIN"
        } else {
            "OVER"
        };
        format!(
            "ref-scene baseline: {tris} tris + 4K set (resident, unsampled) | {w}x{h} | \
             adapter={adapter} ({device_type}, {backend}) | frames={frames} warmup={warmup} | \
             mean={mean:.2}ms p50={p50:.2}ms p95={p95:.2}ms p99={p99:.2}ms | \
             budget: P95<{BUDGET_P95_MS}ms -> {verdict}",
            tris = self.tris,
            w = self.target.0,
            h = self.target.1,
            adapter = self.adapter,
            device_type = self.device_type,
            backend = self.backend,
            frames = self.frames(),
            warmup = self.warmup,
            mean = self.mean_ms,
            p50 = self.p50_ms,
            p95 = self.p95_ms,
            p99 = self.p99_ms,
        )
    }

    /// Machine-readable numbers for the baseline table: the protocol keys
    /// (`frames`, `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `tris`) plus
    /// the hardware-class context. One line, all floats finite.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"frames\":{},\"mean_ms\":{:.4},\"p50_ms\":{:.4},\"p95_ms\":{:.4},\"p99_ms\":{:.4},\
             \"tris\":{},\"warmup\":{},\"width\":{},\"height\":{},\"texture_set_bytes\":{},\
             \"texture_set_sampled\":false,\"adapter\":{},\"device_type\":{},\"backend\":{},\
             \"budget_p95_ms\":{},\"within_budget\":{}}}",
            self.frames(),
            self.mean_ms,
            self.p50_ms,
            self.p95_ms,
            self.p99_ms,
            self.tris,
            self.warmup,
            self.target.0,
            self.target.1,
            self.texture_set_bytes,
            json_string(&self.adapter),
            json_string(&self.device_type),
            json_string(&self.backend),
            BUDGET_P95_MS,
            self.within_budget(),
        )
    }
}

/// Non-timing facts gathered around the frame loop.
struct SceneInfo {
    tris: usize,
    texture_set_bytes: usize,
    adapter: String,
    device_type: String,
    backend: String,
    covered_pixels: u64,
}

/// Runs the benchmark: builds the viewport [`GpuContext`] (the app's
/// depth format, an sRGB color target), uploads [`reference_mesh`] and
/// the [`TextureSet`], renders [`WARMUP_FRAMES`] unmeasured frames, then
/// times `frames` frames and reads the last one back for coverage.
///
/// Panics if `frames` is zero.
pub fn run(
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    frames: usize,
) -> Result<RefSceneReport, GpuError> {
    assert!(frames > 0, "the bench must measure at least one frame");
    let info = adapter.get_info();
    let gpu = GpuContext::new(adapter, device, queue, COLOR_FORMAT, depth_format());

    let mesh = reference_mesh();
    let (min, max) = mesh.bounds().expect("reference mesh is non-empty");
    let buffers = MeshBuffers::upload(&gpu, &mesh)?;
    let tris = mesh.triangle_count();
    drop(mesh);
    let texture_set = TextureSet::new(&gpu);

    let extent = wgpu::Extent3d {
        width: TARGET_WIDTH,
        height: TARGET_HEIGHT,
        depth_or_array_layers: 1,
    };
    let color = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("umber_ref_scene_color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("umber_ref_scene_depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.depth_format().expect("viewport context has depth"),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let aspect = TARGET_WIDTH as f32 / TARGET_HEIGHT as f32;
    // The viewport's light (viewport.rs display pass).
    let light_dir = glam::Vec3::new(-0.4, -1.0, -0.3).normalize();
    let (base_yaw, pitch) = (0.6_f32, 0.4_f32);

    let mut frame_times = Vec::with_capacity(frames);
    for i in 0..WARMUP_FRAMES + frames {
        let start = Instant::now();
        // Update: orbit one step, rebuild the uniform + this frame's
        // callback exactly as the viewport does each frame.
        let camera = OrbitCamera::framing(min, max, base_yaw + i as f32 * YAW_STEP, pitch);
        let uniform = CameraUniform::new(camera.view_proj(aspect), light_dir);
        let callback = buffers.paint_callback(&gpu, uniform);
        callback.write_uniform(&gpu.queue);
        // Encode + submit + wait.
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_ref_scene_frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("umber_ref_scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // The mesh pipeline compares `Less`: clear to far.
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            callback.draw(&mut pass);
        }
        let index = gpu.queue.submit(Some(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("frame completes");
        let elapsed = start.elapsed();
        if i >= WARMUP_FRAMES {
            frame_times.push(elapsed);
        }
    }

    let covered_pixels = count_covered_pixels(&gpu, &color);
    Ok(RefSceneReport::from_frames(
        frame_times,
        SceneInfo {
            tris,
            texture_set_bytes: texture_set.byte_len(),
            adapter: info.name,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            covered_pixels,
        },
    ))
}

/// Reads `color` back (untimed) and counts pixels whose RGB differs from
/// the black clear.
fn count_covered_pixels(gpu: &GpuContext, color: &wgpu::Texture) -> u64 {
    // 1920·4 = 7680 = 30·256: rows are already copy-aligned.
    let row_bytes = TARGET_WIDTH * 4;
    debug_assert!(row_bytes.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("umber_ref_scene_readback"),
        size: row_bytes as u64 * TARGET_HEIGHT as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("umber_ref_scene_readback_encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(TARGET_HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: TARGET_WIDTH,
            height: TARGET_HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll succeeds");
    rx.recv().expect("map callback ran").expect("map ok");
    let covered = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped range available")
        .chunks_exact(4)
        .filter(|px| px[..3] != [0, 0, 0])
        .count() as u64;
    readback.unmap();
    covered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_mesh_counts_and_bounds() {
        let mesh = reference_mesh();
        assert_eq!(mesh.triangle_count(), 1_002_252, "12·289² tris");
        assert_eq!(mesh.vertex_count(), 504_600, "6·290² verts");
        assert_eq!(mesh.normals.len(), mesh.vertex_count());
        assert_eq!(mesh.uvs.len(), mesh.vertex_count());
        assert!(mesh
            .indices
            .iter()
            .all(|&i| (i as usize) < mesh.vertex_count()));
        let (min, max) = mesh.bounds().unwrap();
        assert_eq!(min, glam::Vec3::splat(-1.0));
        assert_eq!(max, glam::Vec3::splat(1.0));
    }

    /// Deterministic: two generations are identical, and the layout is
    /// pinned at hand-derived spots — first/last vertex of the first/last
    /// face, first/last quad's indices — so generator drift (which would
    /// silently re-base the benchmark) fails here.
    #[test]
    fn reference_mesh_is_deterministic_and_pinned() {
        let a = reference_mesh();
        let b = reference_mesh();
        assert!(a == b, "two generations must be identical");

        // Face 0 (+X; u=Y, v=Z) at s=t=0 and face 5 (-Z; u=Y, v=X) at s=t=1.
        assert_eq!(a.positions[0], [1.0, -1.0, -1.0]);
        assert_eq!(a.normals[0], [1.0, 0.0, 0.0]);
        assert_eq!(a.uvs[0], [0.0, 0.0]);
        let last = a.vertex_count() - 1;
        assert_eq!(a.positions[last], [1.0, 1.0, -1.0]);
        assert_eq!(a.normals[last], [0.0, 0.0, -1.0]);
        assert_eq!(a.uvs[last], [1.0, 1.0]);
        // Rows are 290 verts wide; face 5 starts at 5·290² = 420,500.
        assert_eq!(a.indices[..6], [0, 1, 291, 0, 291, 290]);
        assert_eq!(
            a.indices[a.indices.len() - 6..],
            [504_308, 504_309, 504_599, 504_308, 504_599, 504_598]
        );
    }

    /// Every triangle winds CCW seen from outside (geometric normal agrees
    /// with the face normal): the culled pipeline keeps the outer surface,
    /// so the bench rasterizes the near faces, not the inside of the far
    /// ones.
    #[test]
    fn cube_grid_winds_ccw_from_outside() {
        let mesh = cube_grid(3);
        assert_eq!(mesh.triangle_count(), 12 * 9);
        for tri in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| glam::Vec3::from(mesh.positions[tri[k] as usize]));
            let geometric = (b - a).cross(c - a);
            let face = glam::Vec3::from(mesh.normals[tri[0] as usize]);
            assert!(
                geometric.dot(face) > 0.0,
                "triangle {tri:?} winds against its face normal {face}"
            );
            // Area check: no degenerate triangles.
            assert!(geometric.length() > 1e-6);
        }
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let sorted: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        assert_eq!(percentile_sorted(&sorted, 0.50), Duration::from_millis(50));
        assert_eq!(percentile_sorted(&sorted, 0.95), Duration::from_millis(95));
        assert_eq!(percentile_sorted(&sorted, 0.99), Duration::from_millis(99));
        assert_eq!(percentile_sorted(&sorted[..1], 0.99), Duration::from_millis(1));
    }

    #[test]
    fn json_carries_the_protocol_keys_and_escapes() {
        let report = RefSceneReport::from_frames(
            vec![Duration::from_micros(1500), Duration::from_micros(2500)],
            SceneInfo {
                tris: 12,
                texture_set_bytes: 64,
                adapter: "llvmpipe \"x\"\\y".into(),
                device_type: "Cpu".into(),
                backend: "Vulkan".into(),
                covered_pixels: 1,
            },
        );
        let json = report.to_json();
        for key in [
            "\"frames\":2,",
            "\"mean_ms\":2.0000,",
            "\"p50_ms\":1.5000,",
            "\"p95_ms\":2.5000,",
            "\"p99_ms\":2.5000,",
            "\"tris\":12,",
            "\"within_budget\":true",
            "\"adapter\":\"llvmpipe \\\"x\\\"\\\\y\"",
        ] {
            assert!(json.contains(key), "{key} missing from {json}");
        }
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(!json.contains('\n'));
    }

    /// `UMBER_REF_SCENE_FRAMES`, defaulting to [`DEFAULT_FRAMES`]. A set
    /// but invalid value is a misconfiguration: fail loudly, never fall
    /// back silently to a different frame count.
    fn frames_from_env() -> usize {
        match std::env::var("UMBER_REF_SCENE_FRAMES") {
            Ok(v) => match v.trim().parse::<usize>() {
                Ok(n) if n > 0 => n,
                _ => panic!("UMBER_REF_SCENE_FRAMES must be a positive integer, got {v:?}"),
            },
            Err(_) => DEFAULT_FRAMES,
        }
    }

    /// Graceful skip where no wgpu adapter exists (the adapter rule —
    /// same as `umber-app`'s perf soak; lavapipe covers CI).
    fn try_request_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("skipping: no wgpu adapter available");
            return None;
        };
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
            Ok((device, queue)) => Some((adapter, device, queue)),
            Err(e) => {
                eprintln!("skipping: device request failed: {e}");
                None
            }
        }
    }

    /// The reference-scene bench. Asserts the HARNESS (frame count,
    /// non-zero times, ordered percentiles, real coverage); prints the
    /// budget verdict without failing on it — see the module docs.
    #[test]
    fn ref_scene_bench() {
        let frames = frames_from_env();
        let Some((adapter, device, queue)) = try_request_device() else {
            return;
        };
        let report = run(adapter, device, queue, frames).expect("reference scene uploads");

        assert_eq!(report.frames(), frames, "every requested frame was timed");
        assert!(
            report.frame_times.iter().all(|d| !d.is_zero()),
            "every frame time must be non-zero"
        );
        assert!(report.mean_ms > 0.0 && report.mean_ms.is_finite());
        assert!(
            report.p50_ms <= report.p95_ms && report.p95_ms <= report.p99_ms,
            "percentiles out of order: p50={} p95={} p99={}",
            report.p50_ms,
            report.p95_ms,
            report.p99_ms
        );
        assert_eq!(report.tris, 1_002_252);
        assert_eq!(
            report.texture_set_bytes,
            4 * (TEXTURE_SET_SIZE as usize).pow(2) * 4,
            "four RGBA8 4K maps resident"
        );
        // The framed cube covers ~12-17% of the 1080p target (yaw
        // dependent); an empty or culled-away render leaves ~0%. The 2%
        // floor catches the latter without flaking on the former.
        let total = (TARGET_WIDTH * TARGET_HEIGHT) as u64;
        assert!(
            report.covered_pixels * 50 > total,
            "the mesh must cover >2% of the target (covered {} of {total})",
            report.covered_pixels
        );

        println!("{}", report.baseline_line());
        let json = report.to_json();
        println!("ref-scene json: {json}");
        if let Ok(path) = std::env::var("UMBER_REF_SCENE_JSON") {
            std::fs::write(&path, format!("{json}\n"))
                .unwrap_or_else(|e| panic!("writing UMBER_REF_SCENE_JSON={path}: {e}"));
            println!("ref-scene json written to {path}");
        }
    }
}

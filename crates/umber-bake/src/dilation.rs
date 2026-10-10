//! UV-padding dilation (post-bake island-margin expansion).
//!
//! Wave 3 bake path (docs/specs/requirements.md §3): every mesh-map baker
//! (`ao`, `curvature`, `thickness`) leaves texels outside any UV island
//! transparent (`a = 0`). Sampling those maps with bilinear filtering (or
//! mipmapping) bleeds the transparent black into island edges — the classic
//! export seam. This module spreads island-edge colors outward into the
//! uncovered margin, one texel-ring per GPU pass, through
//! `umber_gpu::bake_shaders::DILATE_BAKE_SHADER`. See
//! `LANDING_NOTES_DILATION.md` for the algorithm, the iteration cost, and
//! the diagonal-streak artifact this nearest-donor scheme accepts.
//!
//! Composition is deliberately *not* mesh-fed like the other bakers: the
//! input is a CPU-side RGBA8 map (typically a bake readback), uploaded once
//! into a `TEXTURE_BINDING` staging texture, then ping-ponged between two
//! [`crate::ao::BakeTarget`]s for `N` iterations — each pass reads the
//! previous pass's output, so the covered front advances exactly one texel
//! per dispatch.

use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt as _;

use umber_gpu::bake_shaders::DILATE_BAKE_SHADER;

use crate::ao::{AoBakeError, BakeTarget};

/// Parameters for [`dilate_map`].
#[derive(Debug, Clone, Copy)]
pub struct DilateParams {
    /// Dilation steps to run; each step claims one more ring of uncovered
    /// texels around every island, so `iterations` is the padding width in
    /// texels. `0` is a no-op returning the input unchanged.
    pub iterations: u32,
}

impl DilateParams {
    /// The default padding width used by [`DilateParams::new`]: 16 texels,
    /// enough margin for a full mip chain on a 1k–2k map without touching
    /// interior pixels.
    pub const DEFAULT_ITERATIONS: u32 = 16;

    /// Builds params with an explicit iteration count.
    pub fn new(iterations: u32) -> Self {
        Self { iterations }
    }
}

impl Default for DilateParams {
    fn default() -> Self {
        Self::new(Self::DEFAULT_ITERATIONS)
    }
}

/// Errors from [`dilate_map`].
#[derive(Debug, thiserror::Error)]
pub enum DilateError {
    /// The requested bake-target width or height was zero.
    #[error("bake target dimensions must be non-zero (got {width}x{height})")]
    EmptyTarget {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// `map`'s byte length does not match `width * height * 4`.
    #[error(
        "map length {len} does not match dimensions {width}x{height} (expected {expected} bytes)"
    )]
    SizeMismatch {
        /// The input slice's actual length in bytes.
        len: usize,
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
        /// The expected `width * height * 4` byte count.
        expected: usize,
    },
    /// Reading the dilated texture back to CPU memory failed.
    #[error("bake-target readback failed: {0}")]
    Readback(String),
}

/// Checks [`dilate_map`]'s dimensions and input length independently of any
/// GPU call, so the validation itself is unit-testable without a device —
/// matching `curvature::validate`'s convention.
fn validate(map: &[u8], width: u32, height: u32) -> Result<(), DilateError> {
    if width == 0 || height == 0 {
        return Err(DilateError::EmptyTarget { width, height });
    }
    let expected = width as usize * height as usize * 4;
    if map.len() != expected {
        return Err(DilateError::SizeMismatch {
            len: map.len(),
            width,
            height,
            expected,
        });
    }
    Ok(())
}

/// Spreads island-edge colors outward into uncovered texels of a CPU-side
/// RGBA8 map (`width * height * 4` bytes, row-major — the format every
/// mesh-map baker's readback returns).
///
/// Uploads `map` once, runs `params.iterations` ping-pong dilation passes
/// between two [`BakeTarget`]s (one texel-ring claimed per pass — see
/// `umber_gpu::bake_shaders::DILATE_BAKE_SHADER`'s doc comment for the
/// per-texel rule and why dilated texels normalize to full coverage), and
/// reads the final pass's target back. Covered input texels (`a > 0`) are
/// copied through every pass untouched, so interior pixels survive any
/// iteration count byte-identical; uncovered texels within `iterations`
/// texels (Chebyshev distance) of an island take the nearest donor's rgb
/// with `a = 255`, and texels farther out stay `(0, 0, 0, 0)`.
///
/// `iterations == 0` skips the GPU entirely and returns the input cloned.
///
/// Builds a fresh compute pipeline on every call rather than caching one
/// across runs — reasonable for this slice's one-shot entry point, the
/// same tradeoff `ao::run` documents.
///
/// # Errors
///
/// Returns [`DilateError::EmptyTarget`] if `width`/`height` is zero,
/// [`DilateError::SizeMismatch`] if `map.len() != width * height * 4`, or
/// [`DilateError::Readback`] if the GPU readback fails.
pub fn dilate_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    map: &[u8],
    width: u32,
    height: u32,
    params: &DilateParams,
) -> Result<Vec<u8>, DilateError> {
    profiling::scope!("bake_pass");
    validate(map, width, height)?;

    if params.iterations == 0 {
        return Ok(map.to_vec());
    }

    // Upload staging: the input map as a read-only shader input. A fresh
    // `BakeTarget` cannot serve here — its texture lacks `COPY_DST`, so
    // neither `queue.write_texture` nor a staging-buffer copy could land
    // the bytes in it. This texture is only ever *read* (binding 0), so it
    // needs `TEXTURE_BINDING` for the shader plus `COPY_DST` for the
    // upload, and nothing else.
    let upload = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("umber_bake_dilate_upload"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &upload,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        map,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let upload_view = upload.create_view(&wgpu::TextureViewDescriptor::default());

    // Reuses `ao::BakeTarget`'s `Rgba8Unorm` texture/view/readback
    // machinery (including the 256-byte-row-pitch de-padding) rather than
    // re-inventing a second storage-texture type — the same reuse
    // `curvature` makes over `ao`. Both targets already carry
    // `TEXTURE_BINDING` (via `PaintTarget`), so either can serve as the
    // read-only input of a later pass.
    let target_a = BakeTarget::new(device, width, height);
    let target_b = BakeTarget::new(device, width, height);

    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_bake_dilate_dims"),
        contents: bytemuck::cast_slice(&[width, height]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("umber_bake_dilate_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DILATE_BAKE_SHADER)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("umber_bake_dilate_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
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
                    min_binding_size: NonZeroU64::new(size_of::<[u32; 2]>() as u64),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_bake_dilate_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_bake_dilate_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Pass `i` writes to B when its parity matches the last pass's, else
    // to A — so the final pass (`i = iterations - 1`) always lands in B,
    // the target read back below, regardless of whether `iterations` is
    // odd or even. Pass 0 reads the upload texture; every later pass reads
    // the target the previous pass wrote.
    let writes_to_b = |i: u32| i % 2 == (params.iterations - 1) % 2;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_bake_dilate_encoder"),
    });
    for i in 0..params.iterations {
        let src_view: &wgpu::TextureView = if i == 0 {
            &upload_view
        } else if writes_to_b(i - 1) {
            target_b.view()
        } else {
            target_a.view()
        };
        let dst = if writes_to_b(i) { &target_b } else { &target_a };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_bake_dilate_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dims_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("umber_bake_dilate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per texel, matching `cs_main_from_position`'s
            // dispatch convention (`DILATE_BAKE_SHADER` runs
            // `@workgroup_size(1)`, so `workgroup_id.xy` is directly the
            // texel coordinate).
            pass.dispatch_workgroups(width, height, 1);
        }
    }
    queue.submit(Some(encoder.finish()));

    // The final pass always lands in B by the `writes_to_b` construction
    // above (`i = iterations - 1` matches its own parity).
    target_b
        .read_back_rgba8(device, queue)
        .map_err(|e| match e {
            AoBakeError::Readback(msg) => DilateError::Readback(msg),
            other => DilateError::Readback(other.to_string()),
        })
}

/// The §6 "infinite dilation" entry point: fills every UNCOVERAGED
/// texel **connected** to an island — no transparent holes anywhere a
/// seed can reach. A dilation front claims one texel ring per pass,
/// so `width + height` passes bound the worst-case diagonal crossing
/// of the whole map: after that many rings, any still-uncovered texel
/// is unreachable from every island (a fully enclosed UV hole — the
/// map genuinely has no donor) and passes beyond it change nothing.
///
/// Equivalent to [`dilate_map`] with `iterations = width + height`,
/// exposed so callers say what they mean (the CLI's `--dilate inf`
/// maps here; Substance's "infinite dilation" is the same bound).
///
/// # Errors
///
/// Same surface as [`dilate_map`].
pub fn dilate_map_filled(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    map: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, DilateError> {
    profiling::scope!("bake_pass");
    let params = DilateParams {
        iterations: width.saturating_add(height),
    };
    dilate_map(device, queue, map, width, height, &params)
}

/// The §6 "default-color fill" padding mode: paints every UNCOVERED
/// texel (`a == 0`) with a solid caller-chosen color, in place on a
/// CPU-side RGBA8 map. The companion "transparent fill" mode is the
/// no-op — uncovered texels already stay `(0, 0, 0, 0)`.
///
/// Runs AFTER dilation when both are wanted (dilate first so real
/// island colors win the seams; the fill then only paints the
/// genuinely unreachable interior holes).
///
/// # Errors
///
/// Returns [`DilateError::SizeMismatch`] when `map.len()` doesn't
/// match `width * height * 4`.
pub fn fill_uncovered(
    map: &mut [u8],
    width: u32,
    height: u32,
    color: [u8; 4],
) -> Result<(), DilateError> {
    profiling::scope!("bake_pass");
    let expected = (width as usize) * (height as usize) * 4;
    if map.len() != expected {
        return Err(DilateError::SizeMismatch {
            len: map.len(),
            width,
            height,
            expected,
        });
    }
    for px in map.chunks_exact_mut(4) {
        if px[3] == 0 {
            px.copy_from_slice(&color);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilate_params_new_uses_default_iterations() {
        let params = DilateParams::new(4);
        assert_eq!(params.iterations, 4);
        assert_eq!(DilateParams::DEFAULT_ITERATIONS, 16);
        assert_eq!(DilateParams::default().iterations, 16);
    }

    #[test]
    fn validate_rejects_zero_sized_target() {
        let map = vec![0u8; 4];
        let err = validate(&map, 0, 32).unwrap_err();
        assert!(matches!(
            err,
            DilateError::EmptyTarget {
                width: 0,
                height: 32
            }
        ));
    }

    #[test]
    fn validate_rejects_size_mismatch() {
        let map = vec![0u8; 8];
        let err = validate(&map, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            DilateError::SizeMismatch {
                len: 8,
                width: 2,
                height: 2,
                expected: 16
            }
        ));
    }

    #[test]
    fn validate_accepts_well_formed_inputs() {
        let map = vec![0u8; 2 * 3 * 4];
        assert!(validate(&map, 2, 3).is_ok());
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        /// Requests a plain default device — this pass needs no special
        /// feature (write-only `Rgba8Unorm` storage is core WebGPU, and the
        /// `Rgba8Unorm` input is sampled read-only via `textureLoad`, not
        /// read-written — see `bake_shaders::DILATE_BAKE_SHADER`'s doc
        /// comment). Skips gracefully (mirroring `umber_gpu::paint`'s test
        /// convention) if no adapter is available in this environment.
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

        /// Reads texel `(x, y)` of a `width`-wide RGBA8 `Vec<u8>` (as
        /// returned by [`dilate_map`]).
        fn texel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = ((y * width + x) * 4) as usize;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        /// Writes texel `(x, y)` of a `width`-wide RGBA8 map under
        /// construction.
        fn set_texel(map: &mut [u8], width: u32, x: u32, y: u32, px: [u8; 4]) {
            let i = ((y * width + x) * 4) as usize;
            map[i..i + 4].copy_from_slice(&px);
        }

        /// A single covered texel at the center of an otherwise transparent
        /// map, dilated 4 steps. The 8-neighborhood claims one Chebyshev
        /// ring per pass, so the covered front after 4 iterations is the
        /// *square* of Chebyshev radius 4 — which contains the Manhattan
        /// diamond of radius 4 (every diamond texel is covered with the
        /// center's color) while the diamond's outside corners (e.g.
        /// `(±4, ±4)`, Manhattan 8 but Chebyshev 4) are covered too, by the
        /// same king-move expansion. The brief's "diamond" wording describes
        /// the Manhattan core; the suite asserts the exact square the shader
        /// contract guarantees (see `LANDING_NOTES_DILATION.md`).
        #[test]
        fn single_texel_dilates_to_chebyshev_square_containing_diamond() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const SIZE: u32 = 15;
            const C: u32 = 7;
            const COLOR: [u8; 4] = [200, 100, 50, 255];
            let mut map = vec![0u8; (SIZE * SIZE * 4) as usize];
            set_texel(&mut map, SIZE, C, C, COLOR);

            let params = DilateParams::new(4);
            let out = dilate_map(&device, &queue, &map, SIZE, SIZE, &params)
                .expect("dilate should succeed");
            assert_eq!(out.len(), (SIZE * SIZE * 4) as usize);

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let dx = x.abs_diff(C);
                    let dy = y.abs_diff(C);
                    let chebyshev = dx.max(dy);
                    let manhattan = dx + dy;
                    let px = texel(&out, SIZE, x, y);
                    if chebyshev <= 4 {
                        assert_eq!(
                            px, COLOR,
                            "texel ({x}, {y}) within 4 rings must carry the center color"
                        );
                    } else {
                        assert_eq!(
                            px,
                            [0, 0, 0, 0],
                            "texel ({x}, {y}) beyond 4 rings must stay uncovered"
                        );
                    }
                    // The brief's diamond half, stated directly: every
                    // Manhattan-close texel is covered with the donor color.
                    if manhattan <= 4 {
                        assert_eq!(px[3], 255, "diamond texel ({x}, {y}) must be covered");
                    }
                }
            }
        }

        /// `fill_uncovered` (the §6 default-color fill mode): paints
        /// only the uncovered texels, leaves covered ones untouched,
        /// and validates the buffer shape.
        #[test]
        fn fill_uncovered_paints_only_holes() {
            let mut map = vec![0u8; 2 * 2 * 4];
            // One covered texel, distinct color.
            map[0..4].copy_from_slice(&[10, 20, 30, 255]);
            let fill = [200, 150, 100, 255];

            fill_uncovered(&mut map, 2, 2, fill).expect("shape is valid");

            assert_eq!(&map[0..4], &[10, 20, 30, 255], "covered texel untouched");
            for px in map[4..].chunks_exact(4) {
                assert_eq!(px, fill, "uncovered texel takes the fill color");
            }
        }

        #[test]
        fn fill_uncovered_rejects_mismatched_shape() {
            let mut map = vec![0u8; 7]; // 2x2 needs 16
            assert!(matches!(
                fill_uncovered(&mut map, 2, 2, [255; 4]),
                Err(DilateError::SizeMismatch { .. })
            ));
        }

        #[test]
        fn fill_uncovered_preserves_transparent_fill_semantics_by_noop() {
            // The transparent mode IS the default: without the fill,
            // uncovered texels stay (0,0,0,0) through dilation —
            // documented as the companion no-op mode.
            let map = [0u8; 2 * 2 * 4];
            assert!(map.chunks_exact(4).all(|px| px == [0, 0, 0, 0]));
        }

        /// `dilate_map_filled` (the §6 "infinite dilation" entry point)
        /// claims the WHOLE map from a single seed: every texel is
        /// connected to the center on a 15x15 grid, so all of them
        /// carry the donor color with full alpha.
        #[test]
        fn filled_dilation_covers_the_entire_map_from_one_seed() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const SIZE: u32 = 15;
            const C: u32 = 7;
            const COLOR: [u8; 4] = [200, 100, 50, 255];
            let mut map = vec![0u8; (SIZE * SIZE * 4) as usize];
            set_texel(&mut map, SIZE, C, C, COLOR);

            let out = dilate_map_filled(&device, &queue, &map, SIZE, SIZE)
                .expect("filled dilate should succeed");

            for y in 0..SIZE {
                for x in 0..SIZE {
                    let px = texel(&out, SIZE, x, y);
                    assert_eq!(
                        px, COLOR,
                        "every texel ({x}, {y}) is connected — filled dilation covers all"
                    );
                }
            }
        }

        /// `iterations == 0` must return the input byte-identical without
        /// touching the GPU pipeline (mixed covered/uncovered content, so a
        /// copy-through bug or a dropped alpha would show).
        #[test]
        fn zero_iterations_returns_input_unchanged() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const W: u32 = 8;
            const H: u32 = 6;
            let mut map = vec![0u8; (W * H * 4) as usize];
            set_texel(&mut map, W, 0, 0, [10, 20, 30, 255]);
            set_texel(&mut map, W, 7, 5, [200, 100, 50, 255]);
            set_texel(&mut map, W, 3, 2, [0, 0, 0, 0]);

            let params = DilateParams::new(0);
            let out =
                dilate_map(&device, &queue, &map, W, H, &params).expect("dilate should succeed");
            assert_eq!(out, map);
        }

        /// An 8x8 covered square next to an uncovered region, dilated 5
        /// steps: the front advances exactly one texel per iteration, so a
        /// texel 5 out from the island edge is covered while texels 6+ out
        /// are not — pinning the "N steps = N rings" cost model.
        #[test]
        fn square_front_advances_exactly_one_texel_per_iteration() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const SIZE: u32 = 32;
            const COLOR: [u8; 4] = [11, 22, 33, 255];
            let mut map = vec![0u8; (SIZE * SIZE * 4) as usize];
            for y in 8..16 {
                for x in 8..16 {
                    set_texel(&mut map, SIZE, x, y, COLOR);
                }
            }

            let params = DilateParams::new(5);
            let out = dilate_map(&device, &queue, &map, SIZE, SIZE, &params)
                .expect("dilate should succeed");
            assert_eq!(out.len(), (SIZE * SIZE * 4) as usize);

            // Middle row of the square; the island's +x edge sits at x = 15.
            // Chebyshev distance from the edge decides coverage.
            let edge_px = texel(&out, SIZE, 15, 12);
            assert_eq!(edge_px, COLOR, "island interior must copy through");
            let five_out = texel(&out, SIZE, 20, 12);
            assert_eq!(
                five_out, COLOR,
                "texel 5 out from the edge must be covered with the island color"
            );
            let six_out = texel(&out, SIZE, 21, 12);
            assert_eq!(
                six_out,
                [0, 0, 0, 0],
                "texel 6 out from the edge must stay uncovered"
            );
            // Same check along -y (edge at y = 8) to pin both axes.
            let five_up = texel(&out, SIZE, 12, 3);
            assert_eq!(
                five_up, COLOR,
                "texel 5 up from the edge must be covered with the island color"
            );
            let six_up = texel(&out, SIZE, 12, 2);
            assert_eq!(
                six_up,
                [0, 0, 0, 0],
                "texel 6 up from the edge must stay uncovered"
            );
        }

        /// A fully-covered map must round-trip byte-identical: the
        /// copy-through path (`a > 0` → store unchanged) may not alter
        /// covered texels, no matter the iteration count (odd and even both
        /// exercised, since the ping-pong parity differs).
        #[test]
        fn fully_covered_map_round_trips_byte_identical() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const W: u32 = 10;
            const H: u32 = 7;
            let mut map = Vec::with_capacity((W * H * 4) as usize);
            for y in 0..H {
                for x in 0..W {
                    // Varied channels incl. edge values, alpha always
                    // covered (including mid-alpha, which must also copy
                    // through verbatim).
                    let px = [
                        ((x * 37 + y * 11) % 256) as u8,
                        ((x * 91 + y * 53) % 256) as u8,
                        ((x * 17 + y * 101) % 256) as u8,
                        if (x + y) % 3 == 0 { 128 } else { 255 },
                    ];
                    map.extend_from_slice(&px);
                }
            }

            for iterations in [3, 4] {
                let params = DilateParams::new(iterations);
                let out = dilate_map(&device, &queue, &map, W, H, &params)
                    .expect("dilate should succeed");
                assert_eq!(
                    out, map,
                    "fully-covered map must round-trip at {iterations} iterations"
                );
            }
        }

        #[test]
        fn dilate_map_rejects_zero_sized_target() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = DilateParams::default();
            let err = dilate_map(&device, &queue, &[], 0, 32, &params).unwrap_err();
            assert!(matches!(
                err,
                DilateError::EmptyTarget {
                    width: 0,
                    height: 32
                }
            ));
        }

        #[test]
        fn dilate_map_rejects_size_mismatch_without_touching_the_gpu_pipeline() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            let params = DilateParams::default();
            let err = dilate_map(&device, &queue, &[0u8; 8], 2, 2, &params).unwrap_err();
            assert!(matches!(
                err,
                DilateError::SizeMismatch {
                    len: 8,
                    width: 2,
                    height: 2,
                    expected: 16
                }
            ));
        }
    }
}

//! Image-based lighting: the load-time convolved irradiance map.
//!
//! Wave-4 item 6 (docs/specs/env-map-ibl-design.md). The viewport's
//! `environment_irradiance` was a procedural sky/ground mix; this module
//! is the image-based version: an equirectangular environment (HDR EXR
//! or LDR PNG, decoded by `umber-export`'s new read APIs) is
//! hemisphere-integrated ONCE at load time by [`IBL_CONVOLVE_SHADER`]
//! into a small [`EnvIrradiance`] texture the mesh pass samples
//! per-fragment, plus the v1 specular tier: three cone-scaled
//! prefiltered levels ([`PREFILTER_ROUGHNESS`]) with the analytic
//! integrated-BRDF fit ([`brdf_approx_cpu`], no LUT texture v1) — the
//! design's named wave-5 follow-up, scoped to the honest cone-scaling
//! simplification (see [`IBL_PREFILTER_SHADER`]).
//!
//! # Equirect convention (read this before touching any of the three sites)
//!
//! All three sites — the convolve shader below, `shaders::MESH_SHADER`'s
//! `equirect_uv`, and the Rust mirror [`equirect_uv_cpu`] — share one
//! formula:
//!
//! ```text
//! u = atan2(n.z, n.x) / 2π + 0.5
//! v = asin(n.y) / π + 0.5
//! ```
//!
//! so the convolve→render round trip is self-consistent by construction:
//! the convolve shader maps each output texel's normal through the
//! *inverse* equirect map and samples input directions through the
//! *forward* map, and the mesh pass samples the convolved map through
//! the same forward map. +Y (zenith) is v=1, i.e. the LAST data row;
//! -Y (nadir) is v=0, the first data row. `umber-export`'s PNG/EXR
//! readers both return rows in this GPU order (their doc comments pin
//! the flips), so viewer-conventional assets (sky at the displayed top)
//! land upright.
//!
//! Texel lookup is nearest (`textureLoad` at `floor(uv * dims)`,
//! clamped) on the convolve side — exact texel fetches keep the Rust
//! mirror bit-comparable — and bilinear (`textureSample`) on the render
//! side, where the 32×16 map is smooth.
//!
//! # Sampling
//!
//! The hemisphere set is [`AO_BAKE_SHADER`](crate::bake_shaders)'s
//! `hemisphere_sample` verbatim (stratified `cos(theta)`, golden-angle
//! `phi`, deterministic, no per-texel seed) steered by the same
//! Duff-et-al. tangent frame. The estimator is the standard uniform-pdf
//! one: `E(n) = (2π/N) Σ L(ωᵢ)cosθᵢ`. A uniform-white environment
//! therefore convolves to exactly π per channel (pinned by
//! `uniform_white_convolves_to_pi`), which doubles as the absolute-scale
//! sanity check.

use std::mem::size_of;

use wgpu::util::DeviceExt as _;

/// Irradiance-map width: 32×16 is plenty for diffuse (the convolved
/// result is low-frequency — the design's §"irradiance integration").
pub const IRRADIANCE_WIDTH: u32 = 32;
/// Irradiance-map height (see [`IRRADIANCE_WIDTH`]).
pub const IRRADIANCE_HEIGHT: u32 = 16;
/// Hemisphere samples per output texel (the task brief's ~1024).
pub const IBL_SAMPLES: u32 = 1024;

/// Load-time equirect→irradiance convolve: one invocation per output
/// texel (`@workgroup_size(1)`, dispatched `(32, 16, 1)`), each running
/// the full serial hemisphere loop.
///
/// # Layout contracts
///
/// - Binding 0 is the uploaded equirect as an *unfilterable* float
///   texture: [`EnvIrradiance::from_equirect`] always uploads
///   `Rgba32Float` (LDR inputs are widened on the CPU), and the Rust
///   side binds it with `Float { filterable: false }`. The shader only
///   ever `textureLoad`s it (no sampler), so filterability never
///   matters — but declaring it filterable would reject the
///   non-filterable `Rgba32Float` format at bind-group creation.
/// - Binding 1 is the `Rgba16Float` irradiance target, `write`-only
///   (core WebGPU, no device feature — the same reason
///   [`AO_BAKE_SHADER`](crate::bake_shaders) uses `write`).
/// - `ConvolveParams` must match [`ConvolveParams`] byte-for-byte
///   (three `u32`s + one pad = 16 bytes).
///
/// # Determinism
///
/// The per-texel loop is serial in sample index order with a fixed
/// sample set, so output is deterministic texel-to-texel — which is
/// what the `convolve_mirror_bright_band` GPU test relies on when it
/// ports this loop to Rust op-for-op.
pub const IBL_CONVOLVE_SHADER: &str = r#"
struct ConvolveParams {
    equirect_width: u32,
    equirect_height: u32,
    samples: u32,
    _pad: u32,
};

@group(0) @binding(0) var equirect_tex: texture_2d<f32>;
@group(0) @binding(1) var irradiance_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: ConvolveParams;

const IBL_PI: f32 = 3.14159265358979;
const IBL_TWO_PI: f32 = 6.283185307179586;
const IBL_GOLDEN_CONJ: f32 = 0.6180339887498949;

struct IblBasis {
    b1: vec3<f32>,
    b2: vec3<f32>,
};

/// Verbatim `AO_BAKE_SHADER::hemisphere_sample` (see this module's doc
/// comment): stratified cos(theta), golden-angle phi, no RNG.
fn ibl_hemisphere_sample(i: u32, n: u32) -> vec3<f32> {
    let nf = max(f32(n), 1.0);
    let cos_theta = 1.0 - (f32(i) + 0.5) / nf;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * IBL_PI * fract(f32(i) * IBL_GOLDEN_CONJ);
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

/// Verbatim `AO_BAKE_SHADER::orthonormal_basis` (Duff et al., "Building
/// an Orthonormal Basis, Revisited", JCGT 2017).
fn ibl_orthonormal_basis(n: vec3<f32>) -> IblBasis {
    let sign_z = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign_z + n.z);
    let b = n.x * n.y * a;
    return IblBasis(
        vec3<f32>(1.0 + sign_z * n.x * n.x * a, sign_z * b, -sign_z * n.x),
        vec3<f32>(b, sign_z + n.y * n.y * a, -n.y),
    );
}

/// Forward equirect map (see this module's convention block): +Y is
/// v=1 (last data row), -Y is v=0.
fn ibl_equirect_uv(n: vec3<f32>) -> vec2<f32> {
    let u = atan2(n.z, n.x) / IBL_TWO_PI + 0.5;
    let v = asin(clamp(n.y, -1.0, 1.0)) / IBL_PI + 0.5;
    return vec2<f32>(u, v);
}

/// Inverse equirect map: output-texel UV center back to the normal that
/// texel integrates over. Exact inverse of `ibl_equirect_uv` for all
/// non-polar texels (the poles are singular in any equirect — the two
/// pole rows integrate over near-polar caps, which is correct).
fn ibl_output_normal(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * IBL_TWO_PI;
    let lat = (uv.y - 0.5) * IBL_PI;
    let cos_lat = cos(lat);
    return vec3<f32>(cos_lat * cos(phi), sin(lat), cos_lat * sin(phi));
}

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(irradiance_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(dims);
    let n = normalize(ibl_output_normal(uv));
    let basis = ibl_orthonormal_basis(n);

    let eq_dims = vec2<f32>(f32(params.equirect_width), f32(params.equirect_height));
    let count = max(params.samples, 1u);
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let local_dir = ibl_hemisphere_sample(i, count);
        let dir = normalize(
            local_dir.x * basis.b1 + local_dir.y * basis.b2 + local_dir.z * n
        );
        let sample_uv = ibl_equirect_uv(dir);
        let coord = vec2<i32>(vec2<u32>(
            min(u32(sample_uv.x * eq_dims.x), params.equirect_width - 1u),
            min(u32(sample_uv.y * eq_dims.y), params.equirect_height - 1u),
        ));
        let radiance = textureLoad(equirect_tex, coord, 0).xyz;
        sum = sum + radiance * local_dir.z;
    }

    // Uniform-pdf hemisphere estimator: pdf = 1/2π, and the samples are
    // stratified uniform in cos(theta), so E = (2π/N) Σ L·cosθ.
    let irradiance = sum * (IBL_TWO_PI / f32(count));
    textureStore(irradiance_tex, vec2<i32>(gid.xy), vec4<f32>(irradiance, 1.0));
}
"#;

/// Prefiltered-level roughness values (Split-Sum factor one, v1): three
/// fixed levels the mesh pass blends between. Three small 32×16 f16
/// textures (not a mip chain or array — three bindings is the simpler
/// bind, documented here so the next reader doesn't "optimize" it into
/// drift).
pub const PREFILTER_ROUGHNESS: [f32; 3] = [0.0, 0.5, 1.0];

/// Load-time equirect→prefiltered-radiance convolve, one instance per
/// roughness level (see [`PREFILTER_ROUGHNESS`]): one invocation per
/// output texel, same dispatch shape as [`IBL_CONVOLVE_SHADER`].
///
/// # Layout contracts
///
/// - Binding 0 is the uploaded equirect, same unfilterable-float
///   contract as the irradiance convolve.
/// - Binding 1 is the `Rgba16Float` prefilter target, `write`-only.
/// - `PrefilterParams` must match its Rust namesake byte-for-byte
///   (three `u32`s + one `f32` = 16 bytes).
///
/// # The v1 cone-scaling simplification (read before "improving" this)
///
/// Full Split-Sum importance-samples the GGX normal distribution
/// (D(roughness)-weighted) around the reflection vector. v1 instead
/// steers the SAME uniform golden-spiral hemisphere set toward r by
/// lerping each leg (`mix(r, hemi, roughness)`): roughness 0 collapses
/// every sample onto r (mirror), 0.5 halves the spread, 1.0 keeps the
/// full hemisphere. The lobe narrowing — the entire point of the
/// prefilter — is preserved; full GGX importance sampling is the named
/// follow-up. The estimator is a uniform average with NO cosθ weight
/// (deliberate): the specular prefilter averages incident radiance
/// over the lobe, unlike the diffuse irradiance estimator. A
/// uniform-white environment therefore prefilters to exactly 1.0 at
/// every level (not π).
///
/// # Determinism
///
/// Same contract as the irradiance convolve: serial per-texel loop in
/// sample-index order over a fixed set.
pub const IBL_PREFILTER_SHADER: &str = r#"
struct PrefilterParams {
    equirect_width: u32,
    equirect_height: u32,
    samples: u32,
    roughness: f32,
};

@group(0) @binding(0) var equirect_tex: texture_2d<f32>;
@group(0) @binding(1) var prefilter_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: PrefilterParams;

const IBL_PI: f32 = 3.14159265358979;
const IBL_TWO_PI: f32 = 6.283185307179586;
const IBL_GOLDEN_CONJ: f32 = 0.6180339887498949;

struct IblBasis {
    b1: vec3<f32>,
    b2: vec3<f32>,
};

/// Verbatim `IBL_CONVOLVE_SHADER::ibl_hemisphere_sample` (see this
/// module's doc comment).
fn ibl_hemisphere_sample(i: u32, n: u32) -> vec3<f32> {
    let nf = max(f32(n), 1.0);
    let cos_theta = 1.0 - (f32(i) + 0.5) / nf;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * IBL_PI * fract(f32(i) * IBL_GOLDEN_CONJ);
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

/// Verbatim `IBL_CONVOLVE_SHADER::ibl_orthonormal_basis` (Duff et al.).
fn ibl_orthonormal_basis(n: vec3<f32>) -> IblBasis {
    let sign_z = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign_z + n.z);
    let b = n.x * n.y * a;
    return IblBasis(
        vec3<f32>(1.0 + sign_z * n.x * n.x * a, sign_z * b, -sign_z * n.x),
        vec3<f32>(b, sign_z + n.y * n.y * a, -n.y),
    );
}

/// Forward equirect map (see this module's convention block).
fn ibl_equirect_uv(n: vec3<f32>) -> vec2<f32> {
    let u = atan2(n.z, n.x) / IBL_TWO_PI + 0.5;
    let v = asin(clamp(n.y, -1.0, 1.0)) / IBL_PI + 0.5;
    return vec2<f32>(u, v);
}

/// Inverse equirect map (see `IBL_CONVOLVE_SHADER::ibl_output_normal`).
fn ibl_output_normal(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * IBL_TWO_PI;
    let lat = (uv.y - 0.5) * IBL_PI;
    let cos_lat = cos(lat);
    return vec3<f32>(cos_lat * cos(phi), sin(lat), cos_lat * sin(phi));
}

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(prefilter_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(dims);
    // The output texel's normal doubles as the reflection vector: the
    // prefiltered map is indexed by r, exactly like the irradiance map
    // is indexed by n.
    let r = normalize(ibl_output_normal(uv));
    let basis = ibl_orthonormal_basis(r);

    let eq_dims = vec2<f32>(f32(params.equirect_width), f32(params.equirect_height));
    let count = max(params.samples, 1u);
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let local_dir = ibl_hemisphere_sample(i, count);
        let hemi = normalize(
            local_dir.x * basis.b1 + local_dir.y * basis.b2 + local_dir.z * r
        );
        let dir = normalize(mix(r, hemi, params.roughness));
        let sample_uv = ibl_equirect_uv(dir);
        let coord = vec2<i32>(vec2<u32>(
            min(u32(sample_uv.x * eq_dims.x), params.equirect_width - 1u),
            min(u32(sample_uv.y * eq_dims.y), params.equirect_height - 1u),
        ));
        let radiance = textureLoad(equirect_tex, coord, 0).xyz;
        sum = sum + radiance;
    }

    let prefiltered = sum / f32(count);
    textureStore(prefilter_tex, vec2<i32>(gid.xy), vec4<f32>(prefiltered, 1.0));
}
"#;

/// Which pixel encoding [`EnvIrradiance::from_equirect`]'s `pixels`
/// slice holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvFormat {
    /// Width×height×4 bytes, RGBA8 in GPU row order (row 0 = -Y —
    /// exactly what `umber_export::png::read_png_rgba8` returns).
    /// Widened to float (`v / 255`) on the CPU before upload.
    Rgba8Unorm,
    /// Width×height×4 `f32`s, RGBA float in GPU row order (exactly
    /// what `umber_export::exr::read_exr_rgba_f32` returns).
    Rgba32Float,
}

impl EnvFormat {
    /// Expected `pixels` byte length for a `width`×`height` image.
    pub fn expected_bytes(self, width: u32, height: u32) -> usize {
        let texels = width as usize * height as usize;
        match self {
            Self::Rgba8Unorm => texels * 4,
            Self::Rgba32Float => texels * 16,
        }
    }
}

/// Errors from building an [`EnvIrradiance`].
#[derive(Debug, thiserror::Error)]
pub enum IblError {
    /// Either dimension was zero.
    #[error("environment image has zero dimensions ({width}x{height})")]
    ZeroDimensions {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The pixel slice doesn't match the dimensions/format.
    #[error("buffer size {actual} != {expected} ({width}x{height} as {format:?})")]
    SizeMismatch {
        /// Actual byte count.
        actual: usize,
        /// Expected byte count.
        expected: usize,
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
        /// Pixel encoding.
        format: EnvFormat,
    },
}

/// Uniform block for binding 2 of [`IBL_CONVOLVE_SHADER`]: three
/// `u32`s + explicit pad = 16 bytes (WGSL uniform alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvolveParams {
    /// Equirect width in texels.
    equirect_width: u32,
    /// Equirect height in texels.
    equirect_height: u32,
    /// Hemisphere samples per output texel ([`IBL_SAMPLES`]).
    samples: u32,
    /// Explicit padding to 16 bytes.
    _pad: u32,
}

/// Uniform block for binding 2 of [`IBL_PREFILTER_SHADER`]: three
/// `u32`s + one `f32` = 16 bytes (WGSL uniform alignment — the `f32`
/// sits at offset 12, no padding needed).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefilterParams {
    /// Equirect width in texels.
    equirect_width: u32,
    /// Equirect height in texels.
    equirect_height: u32,
    /// Hemisphere samples per output texel ([`IBL_SAMPLES`]).
    samples: u32,
    /// Cone scale: one of [`PREFILTER_ROUGHNESS`].
    roughness: f32,
}
/// The mesh pass's environment-flag uniform (`MESH_SHADER` binding 3):
/// `flags == 1` samples the irradiance map, anything else runs the
/// byte-identical procedural path. 16 bytes (WGSL uniform alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvFlags {
    /// 1 = sample the bound irradiance map, 0 = procedural fallback.
    pub flags: u32,
    /// Explicit padding (see struct doc).
    pub _pad: [u32; 3],
}

impl EnvFlags {
    /// Procedural fallback: the mesh pass ignores any bound map.
    pub fn procedural() -> Self {
        Self {
            flags: 0,
            _pad: [0; 3],
        }
    }

    /// Sample the bound irradiance map.
    pub fn from_map() -> Self {
        Self {
            flags: 1,
            _pad: [0; 3],
        }
    }
}

/// The mesh pass's specular-tier uniform (`MESH_SHADER` binding 7,
/// appended AFTER the IBL set — additive binding, existing layout
/// untouched): roughness + metallic floats. v1 stand-ins (defaults
/// 0.5/0.0) until the OpenPBR wiring exposes the material's own
/// rough/metal params. 16 bytes (WGSL uniform alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpecParams {
    /// Perceptual roughness in [0, 1] (selects/blends the prefilter
    /// levels, drives the analytic BRDF fit).
    pub roughness: f32,
    /// Metalness in [0, 1] (`F0 = mix(0.04, base_color, metallic)`).
    pub metallic: f32,
    /// Explicit padding (see struct doc).
    pub _pad: [f32; 2],
}

impl SpecParams {
    /// Builds the uniform from explicit values (clamping happens in
    /// the shader — out-of-range inputs are legal here).
    pub fn new(roughness: f32, metallic: f32) -> Self {
        Self {
            roughness,
            metallic,
            _pad: [0.0; 2],
        }
    }
}

impl Default for SpecParams {
    /// v1 stand-in defaults: mid roughness, dielectric.
    fn default() -> Self {
        Self::new(0.5, 0.0)
    }
}

/// Validates equirect dimensions + buffer length without touching the
/// GPU, so the error paths are unit-testable headless (the `From` side
/// still needs a device for the upload + convolve dispatch).
pub(crate) fn validate_equirect_input(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: EnvFormat,
) -> Result<(), IblError> {
    if width == 0 || height == 0 {
        return Err(IblError::ZeroDimensions { width, height });
    }
    let expected = format.expected_bytes(width, height);
    if pixels.len() != expected {
        return Err(IblError::SizeMismatch {
            actual: pixels.len(),
            expected,
            width,
            height,
            format,
        });
    }
    Ok(())
}

/// Builds one prefiltered level's output texture + view + compute
/// pipeline + bind group for [`IBL_PREFILTER_SHADER`]: `level` indexes
/// [`PREFILTER_ROUGHNESS`]. The caller dispatches
/// (`IRRADIANCE_WIDTH`×`IRRADIANCE_HEIGHT`×1) and keeps the texture
/// alive (see [`EnvIrradiance`]).
fn prefilter_level_resources(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    equirect_view: &wgpu::TextureView,
    width: u32,
    height: u32,
    level: usize,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::ComputePipeline,
    wgpu::BindGroup,
) {
    let roughness = PREFILTER_ROUGHNESS[level];
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("umber_ibl_prefilter"),
        size: wgpu::Extent3d {
            width: IRRADIANCE_WIDTH,
            height: IRRADIANCE_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("umber_ibl_prefilter_bind_group_layout"),
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
                        format: wgpu::TextureFormat::Rgba16Float,
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
                        min_binding_size: std::num::NonZeroU64::new(
                            size_of::<PrefilterParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
    let params = PrefilterParams {
        equirect_width: width,
        equirect_height: height,
        samples: IBL_SAMPLES,
        roughness,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("umber_ibl_prefilter_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("umber_ibl_prefilter_bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(equirect_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&target_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("umber_ibl_prefilter_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("umber_ibl_prefilter_pipeline"),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (target, target_view, pipeline, bind_group)
}

/// The convolved diffuse irradiance map + the v1 specular tier: owns
/// the 32×16 `Rgba16Float` irradiance texture + its view + the sampling
/// sampler, plus three 32×16 `Rgba16Float` cone-scaled prefiltered
/// levels ([`PREFILTER_ROUGHNESS`]) with their views. Created once per
/// environment load; cheap to clone (wgpu handles are `Arc`-backed).
/// Additive fields only — `from_equirect`'s signature and every
/// pre-existing accessor are stable.
///
/// When no `EnvIrradiance` is bound anywhere (`None` at the
/// `MeshBuffers::set_environment` call site), the mesh pass renders
/// the procedural path — that `None`-able fallback is the design's
/// fallback contract, so there is deliberately no
/// `procedural_fallback()` constructor: not binding a map IS the
/// procedural mode.
#[derive(Debug, Clone)]
pub struct EnvIrradiance {
    // Kept alive explicitly: the view/sampler reference the texture on
    // the GPU timeline, and owned-handle clarity beats relying on
    // wgpu's internal Arc retention. (Only read under
    // `all(test, feature = "gpu")`, hence the allow.)
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// The three prefiltered levels, index ↔ [`PREFILTER_ROUGHNESS`].
    /// Same keep-alive contract as `texture` (only read under
    /// `all(test, feature = "gpu")`, hence the allow).
    #[allow(dead_code)]
    prefilter_textures: [wgpu::Texture; 3],
    /// Views bound at mesh-pass bindings 4–6 (see
    /// [`EnvIrradiance::prefilter_view`]).
    prefilter_views: [wgpu::TextureView; 3],
}

impl EnvIrradiance {
    /// Convolves an equirect (GPU row order — row 0 = -Y; see this
    /// module's convention block) into the 32×16 irradiance map plus
    /// the three 32×16 prefiltered levels (same load-time pass, one
    /// queue submission for all four dispatches).
    ///
    /// LDR inputs are widened to linear float on the CPU (`v / 255`,
    /// no sRGB decode — PNG equirects are interpreted as linear-light
    /// values; author accordingly) and everything uploads as
    /// `Rgba32Float`, so the convolve shaders see one format.
    pub fn from_equirect(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: EnvFormat,
    ) -> Result<Self, IblError> {
        validate_equirect_input(pixels, width, height, format)?;

        // Widen to float up front (see this method's doc comment). The
        // f32 path reinterprets bytes: `try_cast_slice` because a
        // caller-handed &[u8] is not guaranteed 4-byte aligned — the
        // chunk fallback (native-endian) covers that case.
        let floats: Vec<f32> = match format {
            EnvFormat::Rgba32Float => match bytemuck::try_cast_slice(pixels) {
                Ok(v) => v.to_vec(),
                Err(_) => pixels
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            },
            EnvFormat::Rgba8Unorm => pixels.iter().map(|v| *v as f32 / 255.0).collect(),
        };

        let equirect = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("umber_ibl_equirect"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &equirect,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&floats),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let equirect_view = equirect.create_view(&wgpu::TextureViewDescriptor::default());

        let irradiance = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("umber_ibl_irradiance"),
            size: wgpu::Extent3d {
                width: IRRADIANCE_WIDTH,
                height: IRRADIANCE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let irradiance_view = irradiance.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber_ibl_convolve_shader"),
            source: wgpu::ShaderSource::Wgsl(IBL_CONVOLVE_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("umber_ibl_convolve_bind_group_layout"),
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
                        format: wgpu::TextureFormat::Rgba16Float,
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
                        min_binding_size: std::num::NonZeroU64::new(
                            size_of::<ConvolveParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let params = ConvolveParams {
            equirect_width: width,
            equirect_height: height,
            samples: IBL_SAMPLES,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("umber_ibl_convolve_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("umber_ibl_convolve_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&equirect_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("umber_ibl_convolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("umber_ibl_convolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("umber_ibl_convolve_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("umber_ibl_convolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT, 1);
        }
        // The three prefilter levels ride the same encoder: one queue
        // submission covers all four dispatches (same ordering guarantee
        // as the irradiance pass — see below).
        let prefilter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber_ibl_prefilter_shader"),
            source: wgpu::ShaderSource::Wgsl(IBL_PREFILTER_SHADER.into()),
        });
        let l0 =
            prefilter_level_resources(device, &prefilter_module, &equirect_view, width, height, 0);
        let l1 =
            prefilter_level_resources(device, &prefilter_module, &equirect_view, width, height, 1);
        let l2 =
            prefilter_level_resources(device, &prefilter_module, &equirect_view, width, height, 2);
        for (pipeline, bind_group) in [(&l0.2, &l0.3), (&l1.2, &l1.3), (&l2.2, &l2.3)] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("umber_ibl_prefilter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT, 1);
        }
        // Same-queue submission order guarantees the convolve finishes
        // before any later render pass samples the map — no explicit
        // sync needed (and test readbacks submitted after this see the
        // finished writes for the same reason).
        queue.submit(Some(encoder.finish()));

        Ok(Self {
            texture: irradiance,
            view: irradiance_view,
            sampler: ibl_sampler(device),
            prefilter_textures: [l0.0, l1.0, l2.0],
            prefilter_views: [l0.1, l1.1, l2.1],
        })
    }

    /// One prefiltered level's texture view (mesh-pass bindings 4–6):
    /// `level` indexes [`PREFILTER_ROUGHNESS`] — 0 is the mirror
    /// (roughness 0.0) level, 2 the full-hemisphere (roughness 1.0) one.
    pub(crate) fn prefilter_view(&self, level: usize) -> &wgpu::TextureView {
        &self.prefilter_views[level]
    }

    /// The irradiance texture view (mesh-pass binding 1).
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The irradiance sampler (mesh-pass binding 2).
    pub(crate) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Irradiance-map dimensions (`32×16` — [`IRRADIANCE_WIDTH`]×
    /// [`IRRADIANCE_HEIGHT`]).
    pub fn dimensions(&self) -> (u32, u32) {
        (IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT)
    }

    /// Reads the irradiance map back as f16 bits (width×height×4
    /// `u16`s, row-major) — the test-only inspection path (the
    /// convolve-mirror test compares these bits against the
    /// f16-quantized Rust mirror).
    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn read_back_f16(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u16> {
        read_texture_f16(
            device,
            queue,
            &self.texture,
            IRRADIANCE_WIDTH,
            IRRADIANCE_HEIGHT,
        )
    }

    /// Reads one prefiltered level back as f16 bits (same layout as
    /// [`Self::read_back_f16`]) — the tightness test's inspection path.
    /// `level` indexes [`PREFILTER_ROUGHNESS`].
    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn read_back_prefilter_f16(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        level: usize,
    ) -> Vec<u16> {
        read_texture_f16(
            device,
            queue,
            &self.prefilter_textures[level],
            IRRADIANCE_WIDTH,
            IRRADIANCE_HEIGHT,
        )
    }
}

/// Test-only texture readback shared by [`EnvIrradiance::read_back_f16`]
/// and [`EnvIrradiance::read_back_prefilter_f16`].
#[cfg(all(test, feature = "gpu"))]
fn read_texture_f16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u16> {
    // Row pitch must respect COPY_BYTES_PER_ROW_ALIGNMENT (256):
    // a 32-texel Rgba16Float row is 256 bytes — exactly aligned,
    // so no per-row padding is needed.
    let row_bytes = width * 8;
    debug_assert_eq!(row_bytes % 256, 0);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("umber_ibl_test_readback"),
        size: (row_bytes * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("umber_ibl_test_readback_encoder"),
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
                bytes_per_row: Some(row_bytes),
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

    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll succeeds");
    receiver.recv().expect("map callback ran").expect("map ok");
    let bytes = buffer
        .slice(..)
        .get_mapped_range()
        .expect("mapped range available")
        .to_vec();
    buffer.unmap();
    // Manual LE decode (not bytemuck::cast_slice): the mapped copy
    // is a Vec<u8> with no alignment guarantee. x86-64/aarch64 —
    // umber's targets — are little-endian, matching the texture
    // copy's byte order.
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// The shared irradiance sampler: bilinear, repeat in U (seamless
/// longitude wrap — u=0 and u=1 are the same meridian), clamp in V.
/// Used for real maps ([`EnvIrradiance`]) and the context's 1×1
/// procedural-fallback texture alike.
pub(crate) fn ibl_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("umber_ibl_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

/// Rust mirror of the WGSL `ibl_equirect_uv` / mesh-pass `equirect_uv`
/// (see this module's convention block): the convolve-mirror test and
/// the six-axis exactness test both build on this.
pub fn equirect_uv_cpu(n: [f32; 3]) -> [f32; 2] {
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
    let u = n[2].atan2(n[0]) / TWO_PI + 0.5;
    let v = n[1].clamp(-1.0, 1.0).asin() / std::f32::consts::PI + 0.5;
    [u, v]
}

/// Inverse equirect map (mirror of `ibl_output_normal`): texel-center
/// UV back to the normal that texel integrates over.
pub fn equirect_normal_cpu(u: f32, v: f32) -> [f32; 3] {
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
    let phi = (u - 0.5) * TWO_PI;
    let lat = (v - 0.5) * std::f32::consts::PI;
    let cos_lat = lat.cos();
    [cos_lat * phi.cos(), lat.sin(), cos_lat * phi.sin()]
}

/// Split-Sum factor two without the LUT: the analytic fit of the
/// integrated GGX BRDF (Karis, "Real Shading in Unreal Engine 4",
/// SIGGRAPH 2013 — the `EnvBRDFApprox` closed form mobile pipelines
/// use in place of the 2D DFGLUT). Returns `(scale, bias)` with
/// `specular = F0 * scale + bias`. Op-for-op mirror of
/// `MESH_SHADER::env_brdf_approx` (same literals, same order).
///
/// Precision note: WGSL permits approximate `exp2`, so the GPU and
/// this mirror may differ ~1 ULP in the fit — no GPU test depends on
/// the fit's exact bits (the energy-delta tests assert with wide
/// margins); the exact-bit asserts live in headless Rust only.
pub fn brdf_approx_cpu(n_dot_v: f32, roughness: f32) -> [f32; 2] {
    let c0 = [-1.0f32, -0.0275, -0.572, 0.022];
    let c1 = [1.0f32, 0.0425, 1.04, -0.04];
    let r = [
        roughness * c0[0] + c1[0],
        roughness * c0[1] + c1[1],
        roughness * c0[2] + c1[2],
        roughness * c0[3] + c1[3],
    ];
    let a004 = (r[0] * r[0]).min((-9.28 * n_dot_v).exp2()) * r[0] + r[1];
    [-1.04 * a004 + r[2], 1.04 * a004 + r[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolve_params_is_16_bytes() {
        assert_eq!(size_of::<ConvolveParams>(), 16);
    }

    #[test]
    fn prefilter_params_is_16_bytes() {
        assert_eq!(size_of::<PrefilterParams>(), 16);
    }

    #[test]
    fn spec_params_layout_and_defaults() {
        assert_eq!(size_of::<SpecParams>(), 16);
        let d = SpecParams::default();
        assert_eq!([d.roughness, d.metallic], [0.5, 0.0]);
        let m = SpecParams::new(0.0, 1.0);
        assert_eq!([m.roughness, m.metallic], [0.0, 1.0]);
    }

    /// The Karis analytic fit, headless (the mirrored-math pattern):
    /// (nv=1, rough=0) is the mirror at normal incidence → (scale~1,
    /// bias~0); (nv=0, rough=1) is the grazing/rough corner → the
    /// fit's small-scale, slightly-negative-bias values. Exact `f32`
    /// asserts pin the fit constants (any literal drift fails loudly);
    /// the looser shape asserts document what the numbers mean.
    #[test]
    fn brdf_approx_matches_karis_fit_endpoints() {
        let [s, b] = brdf_approx_cpu(1.0, 0.0);
        assert!((s - 1.0).abs() < 0.01, "mirror scale ~1: {s}");
        assert!(b.abs() < 0.02, "mirror bias ~0: {b}");
        // Exact f32 bits (measured from this fn — the WGSL port must
        // evaluate the same literals to the same arithmetic).
        assert_eq!(s.to_bits(), 0x3f7e_7f1c, "mirror scale bits");
        assert_eq!(b.to_bits(), 0x3bc0_71a0, "mirror bias bits");

        let [s0, b0] = brdf_approx_cpu(0.0, 1.0);
        assert!((s0 - 0.45).abs() < 0.02, "grazing-rough scale: {s0}");
        assert!(b0 < 0.0 && b0 > -0.02, "grazing-rough bias: {b0}");
        assert_eq!(s0.to_bits(), 0x3ee7_a0f7, "grazing scale bits");
        assert_eq!(b0.to_bits(), 0xbb1d_494c, "grazing bias bits");
    }

    #[test]
    fn env_flags_layout_and_values() {
        assert_eq!(size_of::<EnvFlags>(), 16);
        assert_eq!(EnvFlags::procedural().flags, 0);
        assert_eq!(EnvFlags::from_map().flags, 1);
    }

    #[test]
    fn env_format_expected_bytes() {
        assert_eq!(EnvFormat::Rgba8Unorm.expected_bytes(4, 2), 32);
        assert_eq!(EnvFormat::Rgba32Float.expected_bytes(4, 2), 128);
    }

    #[test]
    fn validate_rejects_zero_dimensions() {
        let err = validate_equirect_input(&[], 0, 16, EnvFormat::Rgba8Unorm).unwrap_err();
        assert!(matches!(err, IblError::ZeroDimensions { .. }));
        let err = validate_equirect_input(&[], 32, 0, EnvFormat::Rgba32Float).unwrap_err();
        assert!(matches!(err, IblError::ZeroDimensions { .. }));
    }

    #[test]
    fn validate_rejects_size_mismatch() {
        let err = validate_equirect_input(&[0u8; 10], 4, 2, EnvFormat::Rgba8Unorm).unwrap_err();
        assert!(matches!(
            err,
            IblError::SizeMismatch {
                actual: 10,
                expected: 32,
                ..
            }
        ));
        // Rgba32Float counts bytes (4x the texel count).
        let err = validate_equirect_input(&[0u8; 32], 4, 2, EnvFormat::Rgba32Float).unwrap_err();
        assert!(matches!(err, IblError::SizeMismatch { expected: 128, .. }));
        // Exact sizes pass validation (no device needed — the GPU
        // upload is what `from_equirect` adds on top).
        assert!(validate_equirect_input(&[0u8; 32], 4, 2, EnvFormat::Rgba8Unorm).is_ok());
        assert!(validate_equirect_input(&[0u8; 128], 4, 2, EnvFormat::Rgba32Float).is_ok());
    }

    /// Six-axis exactness (the design's sampler-math test): +X/-X/±Y/±Z
    /// map to the exact expected UVs. Exact `assert_eq!`, not epsilon:
    /// every component here is a ratio of the form `(k·π/2)/(m·π)`,
    /// which is exact in IEEE-754 (halving/doubling needs no rounding),
    /// and `atan2`/`asin` return the exact special values (±0, ±π,
    /// ±π/2) for axis inputs — so any deviation is a real formula bug,
    /// not rounding.
    #[test]
    fn equirect_uv_six_axis_exact() {
        assert_eq!(equirect_uv_cpu([1.0, 0.0, 0.0]), [0.5, 0.5]);
        assert_eq!(equirect_uv_cpu([-1.0, 0.0, 0.0]), [1.0, 0.5]);
        assert_eq!(equirect_uv_cpu([0.0, 1.0, 0.0]), [0.5, 1.0]);
        assert_eq!(equirect_uv_cpu([0.0, -1.0, 0.0]), [0.5, 0.0]);
        assert_eq!(equirect_uv_cpu([0.0, 0.0, 1.0]), [0.75, 0.5]);
        assert_eq!(equirect_uv_cpu([0.0, 0.0, -1.0]), [0.25, 0.5]);
    }

    /// The forward/inverse maps round-trip away from the poles
    /// (self-consistency: the convolve→render path depends on it).
    /// The u comparison is seam-aware: u=0 and u=1 are the same
    /// meridian, so a round trip across the seam (e.g. u=0.0 → 1.0)
    /// is correct, not a bug.
    #[test]
    fn equirect_forward_inverse_roundtrip() {
        for (u, v) in [
            (0.1, 0.2),
            (0.9, 0.8),
            (0.5, 0.5),
            (0.0, 0.99),
            (0.33, 0.66),
        ] {
            let n = equirect_normal_cpu(u, v);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal must be unit: {n:?}");
            let [u2, v2] = equirect_uv_cpu(n);
            let du = (u2 - u).abs().min(1.0 - (u2 - u).abs());
            assert!(du < 1e-5, "u round trip: {u} vs {u2}");
            assert!((v2 - v).abs() < 1e-6, "v round trip: {v} vs {v2}");
        }
    }

    // --- f16 bit-exact conversion (no `half` crate in the tree: `half`
    // is only a transitive dep of `exr`, not addressable without adding
    // a dependency — so the convolve-mirror test converts bits by hand
    // and asserts in f16 ULP space instead of fudging with epsilons).

    /// Round-to-nearest-even f32→f16 bits (handles Inf/NaN/overflow/
    /// subnormals; negative zero preserved).
    fn f32_to_f16_bits(v: f32) -> u16 {
        const F32_EXP_BIAS: i32 = 127;
        const F16_EXP_BIAS: i32 = 15;
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 0xff {
            // Inf/NaN: preserve payload class, quiet the NaN.
            return sign
                | 0x7c00
                | if mant == 0 {
                    0
                } else {
                    (mant >> 13) as u16 | 0x0200
                };
        }
        let unbiased = exp - F32_EXP_BIAS;
        let f16_exp = unbiased + F16_EXP_BIAS;
        if f16_exp >= 31 {
            // Overflow → max finite (test inputs never approach it;
            // kept finite rather than Inf so ordering asserts stay sane).
            return sign | 0x7bff;
        }
        if f16_exp <= 0 {
            // Subnormal (or underflow to zero): shift the hidden leading 1 in.
            if f16_exp < -10 {
                return sign; // underflows to signed zero
            }
            let m = mant | 0x80_0000;
            let shift = (1 - f16_exp) + 13;
            let mut half = (m >> shift) as u16;
            // Round to nearest even on the dropped bits.
            let dropped = m & ((1 << shift) - 1);
            let halfway = 1 << (shift - 1);
            if dropped > halfway || (dropped == halfway && (half & 1) == 1) {
                half += 1;
            }
            return sign | half;
        }
        // Normal: drop 13 bits with round-to-nearest-even.
        let mut half = ((f16_exp as u16) << 10) | (mant >> 13) as u16;
        let dropped = mant & 0x1fff;
        if dropped > 0x1000 || (dropped == 0x1000 && (half & 1) == 1) {
            half += 1;
            // Rounding can carry into the exponent (e.g. 0x3FFF → 0x4000):
            // the increment handles it naturally.
        }
        sign | half
    }

    /// Exact f16-bits→f32 (all 65536 values map without rounding).
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // Subnormal: normalize.
                let mut m = mant;
                let mut e = 127 - 15 + 1 - 10 + 1;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                sign | (e << 23) | (m << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (mant << 13)
        } else {
            sign | ((exp + 127 - 15) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    #[test]
    fn f16_conversion_matches_known_vectors() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(100000.0), 0x7bff, "overflow clamps to max");
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        // π: 3.140625 in f16 (0x4248) — the convolve scale factor.
        assert_eq!(f32_to_f16_bits(std::f32::consts::PI), 0x4248);
        // Round trip across magnitudes (relative error < 2^-10).
        for v in [0.05f32, 0.375, 1.5, 8.0, 100.0, 1e-4, 3.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!((back - v).abs() / v < 0.001, "{v} -> {back}");
        }
        // Subnormal range converts without panicking.
        assert!(f16_bits_to_f32(0x0001) > 0.0);
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;
        use super::{f16_bits_to_f32, f32_to_f16_bits};

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

        /// Rust mirror of `IBL_CONVOLVE_SHADER::ibl_hemisphere_sample`:
        /// same stratified cos(theta) + golden-angle phi in f32
        /// op-for-op (the ao.rs `hemisphere_sample_mirror` precedent —
        /// including its truncated-decimal constants, pinned by
        /// `mirror_constants_match_shader_decimals`).
        fn hemisphere_sample_mirror(i: u32, n: u32) -> [f32; 3] {
            const GOLDEN_CONJ: f32 = 0.618034;
            let nf = (n as f32).max(1.0);
            let cos_theta = 1.0 - ((i as f32) + 0.5) / nf;
            let sin_theta = (0.0f32).max(1.0 - cos_theta * cos_theta).sqrt();
            let scaled = (i as f32) * GOLDEN_CONJ;
            let phi = 2.0 * std::f32::consts::PI * (scaled - scaled.floor());
            [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta]
        }

        /// Mirror of `ibl_orthonormal_basis` (Duff et al. — the ao.rs
        /// `orthonormal_basis_mirror` precedent).
        fn orthonormal_basis_mirror(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
            let sign_z = if n[2] >= 0.0 { 1.0 } else { -1.0 };
            let a = -1.0 / (sign_z + n[2]);
            let b = n[0] * n[1] * a;
            (
                [1.0 + sign_z * n[0] * n[0] * a, sign_z * b, -sign_z * n[0]],
                [b, sign_z + n[1] * n[1] * a, -n[1]],
            )
        }

        /// Full mirror of one convolve output texel: output-UV → normal
        /// → Duff frame → 1024-sample serial accumulation → 2π/N scale.
        /// Nearest-texel fetch replicates the shader's
        /// `floor(uv * dims)` clamp exactly.
        fn expected_irradiance(
            pixels: &[f32],
            width: u32,
            height: u32,
            out_x: u32,
            out_y: u32,
        ) -> [f32; 3] {
            let u = (out_x as f32 + 0.5) / IRRADIANCE_WIDTH as f32;
            let v = (out_y as f32 + 0.5) / IRRADIANCE_HEIGHT as f32;
            let n = equirect_normal_cpu(u, v);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let n = [n[0] / len, n[1] / len, n[2] / len];
            let (b1, b2) = orthonormal_basis_mirror(n);
            let mut sum = [0.0f32; 3];
            for i in 0..IBL_SAMPLES {
                let l = hemisphere_sample_mirror(i, IBL_SAMPLES);
                let d = [
                    l[0] * b1[0] + l[1] * b2[0] + l[2] * n[0],
                    l[0] * b1[1] + l[1] * b2[1] + l[2] * n[1],
                    l[0] * b1[2] + l[1] * b2[2] + l[2] * n[2],
                ];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let d = [d[0] / len, d[1] / len, d[2] / len];
                let [su, sv] = equirect_uv_cpu(d);
                let ix = ((su * width as f32) as u32).min(width - 1);
                let iy = ((sv * height as f32) as u32).min(height - 1);
                let base = ((iy * width + ix) * 4) as usize;
                let cos_theta = l[2];
                sum[0] += pixels[base] * cos_theta;
                sum[1] += pixels[base + 1] * cos_theta;
                sum[2] += pixels[base + 2] * cos_theta;
            }
            let scale = 2.0 * std::f32::consts::PI / IBL_SAMPLES as f32;
            [sum[0] * scale, sum[1] * scale, sum[2] * scale]
        }

        /// ULP distance between two f16 bit patterns (same-sign finite
        /// values — everything this test convolves is non-negative).
        fn ulp_distance(a: u16, b: u16) -> u16 {
            a.abs_diff(b)
        }

        #[test]
        fn mirror_constants_match_shader_decimals() {
            // The ao.rs precedent: truncated Rust decimals must round to
            // the same f32 bits as the shader's long decimals, or the
            // mirror's byte-exactness rests on a lie.
            let golden_mirror: f32 = 0.618034;
            let golden_shader: f32 = "0.6180339887498949".parse().unwrap();
            assert_eq!(golden_mirror.to_bits(), golden_shader.to_bits());
            assert_eq!(
                std::f32::consts::PI.to_bits(),
                "3.14159265358979".parse::<f32>().unwrap().to_bits()
            );
            assert_eq!(
                (2.0 * std::f32::consts::PI).to_bits(),
                "6.283185307179586".parse::<f32>().unwrap().to_bits()
            );
        }

        /// Builds the synthetic bright-band equirect: dim gray
        /// everywhere, HDR-bright inside a small cap around +X. The
        /// threshold (cos > 0.995, ~5.7° half-angle) is deliberately
        /// coarser than the 64-wide texel spacing (2π/64 ≈ 5.6° per
        /// texel, texel centers offset a half step from phi=0): a
        /// tighter threshold would select NO texel at all and the
        /// "bright band" would be silently empty. At 0.995 the cap is
        /// the 2×2 block around (u, v) = (0.5, 0.5).
        fn bright_band_image(width: u32, height: u32) -> Vec<f32> {
            let mut pixels = vec![0.0f32; (width * height * 4) as usize];
            let mut bright_count = 0u32;
            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;
                    let d = equirect_normal_cpu(u, v);
                    let base = ((y * width + x) * 4) as usize;
                    if d[0] > 0.995 {
                        bright_count += 1;
                        pixels[base..base + 4].copy_from_slice(&[64.0, 32.0, 16.0, 1.0]);
                    } else {
                        pixels[base..base + 4].copy_from_slice(&[0.05, 0.05, 0.05, 1.0]);
                    }
                }
            }
            // Guard against the empty-cap trap documented above.
            assert!(
                bright_count > 0,
                "bright cap must select at least one texel"
            );
            pixels
        }

        /// THE CONVOLVE MIRROR TEST (the design's test 1): a synthetic
        /// 64×32 equirect with a bright cap at +X, convolved on the GPU;
        /// three probe texels (+X cap center, -X antipode, +Y pole) must
        /// match the Rust mirror's f16-quantized expectation within 4
        /// f16 ULPs.
        ///
        /// # The 4-ULP bound (not a fudged epsilon)
        ///
        /// Both sides accumulate the same 1024 terms serially in the
        /// same order in IEEE-754 f32, so the only divergence source is
        /// FMA contraction (the shader compiler may fuse `sum + s·cosθ`
        /// where Rust/x86 without FMA does not — or vice versa on other
        /// adapters). One FMA-vs-separate rounding differs by ≤1 ULP of
        /// f32 per accumulation step; 1024 such steps random-walk to
        /// ≪1 ULP of f16 (f16's ULP is 2¹¹× f32's at the same
        /// magnitude... precisely: f16 ULP = 2^(e-10), f32 ULP =
        /// 2^(e-23), ratio 8192 — even a 1024-step worst-case linear
        /// drift of 1024 f32 ULPs is 1/8 of an f16 ULP). 4 ULPs is
        /// therefore ~32× headroom over the analytic worst case while
        /// still catching any real bug (wrong sample set, flipped axis,
        /// dropped cosθ — all of which move the answer by whole f16
        /// codes, hundreds of ULPs).
        #[test]
        fn convolve_mirror_bright_band() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const W: u32 = 64;
            const H: u32 = 32;
            let pixels = bright_band_image(W, H);
            let env = EnvIrradiance::from_equirect(
                &device,
                &queue,
                bytemuck::cast_slice(&pixels),
                W,
                H,
                EnvFormat::Rgba32Float,
            )
            .expect("convolve succeeds");
            assert_eq!(env.dimensions(), (IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT));
            let bits = env.read_back_f16(&device, &queue);
            assert_eq!(
                bits.len(),
                (IRRADIANCE_WIDTH * IRRADIANCE_HEIGHT * 4) as usize
            );

            // Probe texels: the output texel whose normal is closest to
            // +X (cap center), its antipode -X, and the +Y pole row.
            // (u=0.5 → phi=0 → +X; u=0 → phi=-π → -X.)
            let probes = [
                (16u32, 8u32, "+X cap center"),
                (0u32, 8u32, "-X antipode"),
                (8u32, 15u32, "+Y pole"),
            ];
            for (ox, oy, name) in probes {
                let expected = expected_irradiance(&pixels, W, H, ox, oy);
                let base = ((oy * IRRADIANCE_WIDTH + ox) * 4) as usize;
                for (c, channel) in ["r", "g", "b"].iter().enumerate() {
                    let want = f32_to_f16_bits(expected[c]);
                    let got = bits[base + c];
                    let dist = ulp_distance(got, want);
                    assert!(
                        dist <= 4,
                        "{name} {channel}: got f16 {got:#06x} ({}) vs mirrored {want:#06x} ({}): {dist} ULPs",
                        f16_bits_to_f32(got),
                        f16_bits_to_f32(want),
                    );
                }
                // Alpha is always 1.
                assert_eq!(bits[base + 3], 0x3c00, "{name} alpha");
            }

            // Directionality: the +X-facing texel must read much
            // brighter than its antipode (the map actually varies with
            // direction — a flat copy would fail here).
            let px = ((8 * IRRADIANCE_WIDTH + 16) * 4) as usize;
            let nx = (8 * IRRADIANCE_WIDTH * 4) as usize;
            assert!(
                f16_bits_to_f32(bits[px]) > 4.0 * f16_bits_to_f32(bits[nx]),
                "+X must dominate -X"
            );
        }

        /// Absolute-scale check, independent of the sampling pattern: a
        /// uniform-white environment integrates to exactly π per channel
        /// (E = (2π/N)·Σ1·cosθᵢ with stratified cosθᵢ averaging exactly
        /// 1/2). Assert within 1% — sampling-pattern-agnostic.
        #[test]
        fn uniform_white_convolves_to_pi() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const W: u32 = 16;
            const H: u32 = 8;
            let pixels = vec![1.0f32; (W * H * 4) as usize];
            let env = EnvIrradiance::from_equirect(
                &device,
                &queue,
                bytemuck::cast_slice(&pixels),
                W,
                H,
                EnvFormat::Rgba32Float,
            )
            .expect("convolve succeeds");
            let bits = env.read_back_f16(&device, &queue);
            let pi = std::f32::consts::PI;
            // Spot-check five texels spread over the map.
            for (ox, oy) in [(0, 0), (31, 15), (16, 8), (7, 3), (24, 12)] {
                let base = ((oy * IRRADIANCE_WIDTH + ox) * 4) as usize;
                for c in 0..3 {
                    let got = f16_bits_to_f32(bits[base + c]);
                    let rel = ((got - pi) / pi).abs();
                    assert!(rel < 0.01, "texel ({ox},{oy}) ch{c}: {got} vs π");
                }
            }
        }

        /// The LDR upload path (what the app's PNG default exercises):
        /// uniform mid-gray must come back at gray·π — covers the CPU
        /// `v/255` widening plus the u8 quantization step.
        #[test]
        fn rgba8_upload_path_matches_gray_times_pi() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const W: u32 = 8;
            const H: u32 = 4;
            let pixels = vec![128u8; (W * H * 4) as usize];
            let env =
                EnvIrradiance::from_equirect(&device, &queue, &pixels, W, H, EnvFormat::Rgba8Unorm)
                    .expect("convolve succeeds");
            let bits = env.read_back_f16(&device, &queue);
            let want = 128.0 / 255.0 * std::f32::consts::PI;
            let base = ((8 * IRRADIANCE_WIDTH + 16) * 4) as usize;
            for c in 0..3 {
                let got = f16_bits_to_f32(bits[base + c]);
                let rel = ((got - want) / want).abs();
                assert!(rel < 0.015, "ch{c}: {got} vs {want}");
            }
        }

        /// Broad-cap sibling of `bright_band_image` for the prefilter
        /// tests: dim gray everywhere, HDR-bright inside a ~20°
        /// half-angle cap around +X (`d[0] > 0.94`). The narrow 0.995
        /// cap can't serve here: the roughness-0 level samples ONE
        /// direction per output texel, and texel-quantization (±1 input
        /// texel from f32/adapter transcendental wobble) would flip a
        /// 2×2 cap to dim — the 20° cap swallows that wobble with ~9°
        /// of headroom on every side while staying fully inside the
        /// probe's hemisphere.
        fn broad_band_image(width: u32, height: u32) -> Vec<f32> {
            let mut pixels = vec![0.0f32; (width * height * 4) as usize];
            let mut bright_count = 0u32;
            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;
                    let d = equirect_normal_cpu(u, v);
                    let base = ((y * width + x) * 4) as usize;
                    if d[0] > 0.94 {
                        bright_count += 1;
                        pixels[base..base + 4].copy_from_slice(&[64.0, 32.0, 16.0, 1.0]);
                    } else {
                        pixels[base..base + 4].copy_from_slice(&[0.05, 0.05, 0.05, 1.0]);
                    }
                }
            }
            assert!(
                bright_count > 0,
                "bright cap must select at least one texel"
            );
            pixels
        }

        /// PREFILTER TIGHTNESS (the specular tier's entire point): at
        /// the +X-facing probe texel the roughness-0 level must read
        /// the bare cap radiance while the roughness-1 level reads the
        /// dim hemisphere average — the lobe narrowing made visible.
        ///
        /// # The margins (derived, not fudged)
        ///
        /// Level 0 samples ONE direction (≈+X, inside the 20° cap with
        /// ~9° headroom either side — see `broad_band_image`), so all
        /// 1024 samples fetch the cap texel: exactly 64.0 in f16, with
        /// serial-accumulation noise ≪0.01 — the 0.25 bound is ~25×
        /// headroom. Level 2 averages the hemisphere: the cap covers
        /// 2π(1−0.94)≈0.38 sr ≈ 6% of the hemisphere, so ≈3.9 against
        /// the 0.05 floor; the 8× relation needs <12.5% bright — the
        /// 1024-sample golden spiral is equidistributed to ~1%, so 8×
        /// is ~2× headroom over the ~16× structural ratio. The
        /// monotonic chain (L0 > L1 > L2) is structural too: widening
        /// the cone can only dilute the cap with dim floor.
        #[test]
        fn prefilter_tightness_broad_band() {
            let Some((device, queue)) = try_request_device() else {
                return;
            };
            const W: u32 = 64;
            const H: u32 = 32;
            let pixels = broad_band_image(W, H);
            let env = EnvIrradiance::from_equirect(
                &device,
                &queue,
                bytemuck::cast_slice(&pixels),
                W,
                H,
                EnvFormat::Rgba32Float,
            )
            .expect("convolve succeeds");
            let l0 = env.read_back_prefilter_f16(&device, &queue, 0);
            let l1 = env.read_back_prefilter_f16(&device, &queue, 1);
            let l2 = env.read_back_prefilter_f16(&device, &queue, 2);
            assert_eq!(
                l0.len(),
                (IRRADIANCE_WIDTH * IRRADIANCE_HEIGHT * 4) as usize
            );

            // The output texel whose normal is closest to +X (same
            // probe as the irradiance mirror test).
            let base = ((8 * IRRADIANCE_WIDTH + 16) * 4) as usize;
            let r0 = f16_bits_to_f32(l0[base]);
            let r1 = f16_bits_to_f32(l1[base]);
            let r2 = f16_bits_to_f32(l2[base]);
            assert!(
                (r0 - 64.0).abs() < 0.25,
                "L0 must read the bare cap radiance: {r0}"
            );
            assert!(
                r0 > 8.0 * r2,
                "tight level must dominate the full-hemisphere average: L0={r0} L2={r2}"
            );
            assert!(
                r0 > r1 && r1 > r2,
                "widening the cone must dilute monotonically: {r0} > {r1} > {r2}"
            );
            // Alpha is always 1, at every level.
            assert_eq!(l0[base + 3], 0x3c00, "L0 alpha");
            assert_eq!(l2[base + 3], 0x3c00, "L2 alpha");
        }
    }
}

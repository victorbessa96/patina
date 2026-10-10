//! WGSL shader sources for the viewport mesh pass.

/// Normal-shaded mesh pass with a directional light and a hemispherical
/// environment light: the procedural sky/ground irradiance by default,
/// or a convolved environment map when one is bound (Wave-4 item 6 —
/// see `crate::ibl`'s convention block for the equirect math shared
/// with the convolve shader). Driven entirely by vertex normals (no
/// textures yet — that lands with the paint engine). Matches the
/// `Vertex` and `CameraUniform` layouts in `renderer.rs`; bindings 1–3
/// are the IBL set (see `GpuContext`'s bind-group layout).
pub const MESH_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    // xyz = normalized direction the light travels (surface -> fragment),
    // w unused (padding to satisfy uniform alignment).
    light_dir: vec4<f32>,
    // World-space eye position; unused here (this pass has no
    // view-dependent term) but kept so this struct's layout matches
    // `CameraUniform` exactly, as `renderer.rs` documents — `OPENPBR_SHADER`
    // reads this same trailing field for its specular/Fresnel terms.
    eye: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
// IBL set (Wave-4 item 6): the convolved 32x16 irradiance map, its
// sampler, and the procedural/map selector. All three are always bound
// (the context supplies a 1x1 fallback texture when no environment is
// loaded) so the pipeline layout never changes at runtime.
@group(0) @binding(1) var irradiance_tex: texture_2d<f32>;
@group(0) @binding(2) var irradiance_sampler: sampler;
struct EnvFlags {
    flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(3) var<uniform> env_flags: EnvFlags;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    return out;
}

const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.96, 0.90);
const SKY_COLOR: vec3<f32> = vec3<f32>(0.45, 0.55, 0.78);
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.30, 0.27, 0.24);
const BASE_COLOR: vec3<f32> = vec3<f32>(0.72, 0.72, 0.75);

/// Hemispherical environment irradiance: the bound convolved map when
/// `env_flags.flags == 1`, else the procedural stand-in — sky above the
/// horizon, ground below, blended by the normal's vertical component.
///
/// The procedural arm is byte-identical to the pre-IBL shader (the
/// fallback-regression tests pin this): the flag only *selects*, it
/// never reshapes the fallback math.
///
/// `equirect_uv` shares its formula with the convolve shader's
/// `ibl_equirect_uv` and the Rust `crate::ibl::equirect_uv_cpu` (see
/// that module's convention block): +Y is v=1 (last data row), -Y is
/// v=0. The `OPENPBR_SHADER` copy below is deliberately untouched —
/// its bind-group layout has no IBL set, and its env integration rides
/// the wave-5 specular tier, not this slice.
fn equirect_uv(n: vec3<f32>) -> vec2<f32> {
    let u = atan2(n.z, n.x) / 6.283185307179586 + 0.5;
    let v = asin(clamp(n.y, -1.0, 1.0)) / 3.14159265358979 + 0.5;
    return vec2<f32>(u, v);
}

fn environment_irradiance(n: vec3<f32>) -> vec3<f32> {
    if (env_flags.flags == 1u) {
        return textureSample(irradiance_tex, irradiance_sampler, equirect_uv(n)).rgb;
    }
    let t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(GROUND_COLOR, SKY_COLOR, t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let to_light = normalize(-camera.light_dir.xyz);
    let diffuse = max(dot(n, to_light), 0.0);
    let env = environment_irradiance(n);
    // Environment acts as the ambient term; the sun adds directional
    // energy on top, weighted so lit surfaces approach full brightness.
    let color = BASE_COLOR * (env + diffuse * SUN_COLOR * 0.55);
    return vec4<f32>(color, 1.0);
}
"#;

/// Dab-splat compute pass: one workgroup per dab, rasterizing a filled,
/// radially-falloff'd circle into a storage texture with premultiplied
/// alpha-over compositing.
///
/// `Dab`'s field order/padding here must match `paint::Dab` byte-for-byte
/// (see that module's doc comment) — WGSL's `vec4<f32>` forces 16-byte
/// alignment, which Rust's `[f32; 4]` (4-byte aligned) does not provide for
/// free.
///
/// Workgroup-local contract (see `paint::PaintCompositor::splat_dabs`):
/// every invocation in a workgroup paints only pixels inside *its own*
/// dab's clamped bounding box, strided by `local_invocation_index` so no
/// two invocations in the same workgroup touch the same texel. There is no
/// such guarantee *across* workgroups in one dispatch — two dabs in the
/// same batch whose bounding boxes overlap would race on the shared
/// `read_write` storage texture (no atomics are used). The caller must
/// therefore keep spatially-overlapping dabs in separate `splat_dabs`
/// calls; wgpu's automatic hazard tracking orders successive compute
/// passes on the same texture correctly.
pub const PAINT_COMPUTE_SHADER: &str = r#"
struct Dab {
    pos: vec2<f32>,
    radius: f32,
    alpha: f32,
    color: vec4<f32>,
    hardness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> dabs: array<Dab>;
@group(0) @binding(1) var paint_tex: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;

const WORKGROUP_SIZE: u32 = 64u;

/// Radial falloff: 1.0 inside `hardness * radius`, smoothly down to 0.0 at
/// `radius`. `t` is the normalized distance from center (0 at center, 1 at
/// the edge). Written without WGSL's built-in `smoothstep` so the
/// `hardness == 1.0` (hard disc) case can't divide by zero.
fn radial_falloff(t: f32, hardness: f32) -> f32 {
    let denom = max(1.0 - hardness, 1e-4);
    let edge_t = clamp((t - hardness) / denom, 0.0, 1.0);
    return 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let dab = dabs[workgroup_id.x];
    let safe_radius = max(dab.radius, 1e-5);

    let min_xy = vec2<i32>(floor(dab.pos - safe_radius));
    let max_xy = vec2<i32>(ceil(dab.pos + safe_radius));
    let clamped_min = max(min_xy, vec2<i32>(0, 0));
    let clamped_max = min(max_xy, vec2<i32>(dims) - vec2<i32>(1, 1));
    if (clamped_max.x < clamped_min.x || clamped_max.y < clamped_min.y) {
        return;
    }

    let box_w = u32(clamped_max.x - clamped_min.x + 1);
    let box_h = u32(clamped_max.y - clamped_min.y + 1);
    let box_count = box_w * box_h;

    for (var i: u32 = local_index; i < box_count; i = i + WORKGROUP_SIZE) {
        let local_x = i32(i % box_w);
        let local_y = i32(i / box_w);
        let texel = clamped_min + vec2<i32>(local_x, local_y);

        let center = vec2<f32>(texel) + vec2<f32>(0.5, 0.5);
        let dist = distance(center, dab.pos);
        if (dist > safe_radius) {
            continue;
        }

        let coverage = dab.alpha * radial_falloff(dist / safe_radius, dab.hardness);
        let src = dab.color * coverage;
        let dst = textureLoad(paint_tex, texel);
        let out = src + dst * (1.0 - src.a);
        textureStore(paint_tex, texel, out);
    }
}
"#;

/// OpenPBR Surface viewport shader — real-time subset (spec v1.1.1).
///
/// Implemented: base (weight/color/metalness), specular GGX (weight/
/// color/roughness/IOR), coat GGX lobe (weight/color/roughness/IOR)
/// with base roughening, emission, geometry opacity. Energy-conserving
/// lobe mixture per spec §3.10.
///
/// Deferred (documented deviations from the full spec): subsurface,
/// translucent base, thin-film, fuzz, anisotropy, thin-walled mode.
/// Schlick Fresnel replaces the exact dielectric Fresnel (spec §3.1
/// gives the exact form; the real-time deviation is standard and
/// visually indistinguishable in-viewport).
///
/// The `OpenPbrParams` uniform mirrors `material::OpenPbrParams`'s six
/// vec4 slots byte-for-byte (see that module's layout contract).
pub const OPENPBR_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    // World-space eye (CameraUniform::eye) — the view vector source.
    eye: vec4<f32>,
};

// Six vec4 slots — byte-identical to material::OpenPbrParams (96 bytes).
struct OpenPbrParams {
    base: vec4<f32>,     // xyz = base_color, w = base_weight
    surface: vec4<f32>,  // x = metalness, y = specular_roughness, z = coat_roughness, w = opacity
    specular: vec4<f32>, // xyz = specular_color, w = specular_weight
    coat: vec4<f32>,     // xyz = coat_color, w = coat_weight
    emission: vec4<f32>, // xyz = emission, w unused
    iors: vec4<f32>,     // x = specular_ior, y = coat_ior, zw unused
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: OpenPbrParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.world_pos = in.position;
    return out;
}

const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.96, 0.90);
const SKY_COLOR: vec3<f32> = vec3<f32>(0.45, 0.55, 0.78);
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.30, 0.27, 0.24);

fn environment_irradiance(n: vec3<f32>) -> vec3<f32> {
    let t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(GROUND_COLOR, SKY_COLOR, t);
}

/// Schlick Fresnel (spec §3.1 deviation: exact dielectric form replaced
/// by Schlick — standard real-time approximation).
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

/// GGX/Trowbridge-Reitz normal distribution, alpha = roughness^2
/// (spec §3.1 microfacet model).
fn ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d + 1e-7);
}

/// Smith separable masking-shadowing, Schlick-GGX form (spec §3.1).
fn smith_g1(n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    return 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v) + 1e-7);
}

/// Energy-conserving coat: the coat lobe reflects first (Fresnel-weighted
/// at the coat IOR), and the base sees reduced, roughened energy
/// (spec §3.4: roughening + darkening-by-TIR is approximated by the
/// Fresnel split alone in this real-time subset).
fn coat_attenuation(n_dot_v: f32, coat_weight: f32, coat_ior: f32) -> f32 {
    // Approximate coat Fresnel F0 from IOR: ((ior-1)/(ior+1))^2.
    let f0 = (coat_ior - 1.0) / (coat_ior + 1.0);
    let f0sq = f0 * f0;
    let fc = f0sq + (1.0 - f0sq) * pow(1.0 - n_dot_v, 5.0);
    return (1.0 - coat_weight) + coat_weight * fc;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let to_light = normalize(-camera.light_dir.xyz);
    // True view vector from the camera eye (the uniform's trailing
    // field; see CameraUniform::with_eye).
    let to_view = normalize(camera.eye.xyz - in.world_pos);
    let n_dot_v = max(dot(n, to_view), 1e-4);
    let n_dot_l = max(dot(n, to_light), 0.0);
    let h = normalize(to_light + to_view);
    let n_dot_h = max(dot(n, h), 0.0);

    let metalness = clamp(params.surface.x, 0.0, 1.0);
    let spec_rough = clamp(params.surface.y, 0.0, 1.0);
    let coat_rough = clamp(params.surface.z, 0.0, 1.0);
    let opacity = clamp(params.surface.w, 0.0, 1.0);

    // Specular alpha (spec §3.1: alpha = roughness^2).
    let spec_alpha = spec_rough * spec_rough;
    let coat_alpha = coat_rough * coat_rough;

    // Specular F0: dielectric from IOR, tinted by specular_color;
    // metals use base_color as F0 (spec §3.2.1).
    let ior_f0 = (params.iors.x - 1.0) / (params.iors.x + 1.0);
    let dielectric_f0 = vec3<f32>(ior_f0 * ior_f0) * params.specular.xyz;
    let metal_f0 = params.base.xyz;
    let f0 = mix(dielectric_f0, metal_f0, metalness) * params.specular.w;

    // GGX specular lobe (Smith masking-shadowing).
    let spec_ndf = ggx_ndf(n_dot_h, spec_alpha);
    let spec_g = smith_g1(n_dot_v, spec_alpha) * smith_g1(n_dot_l, spec_alpha);
    let spec_f = fresnel_schlick(n_dot_h, f0);
    let specular_lobe = spec_ndf * spec_g * spec_f * n_dot_l;

    // Lambert diffuse (spec §3.2.2 gives Oren-Nayar; Lambert is the
    // accepted real-time simplification — documented deviation).
    let diffuse_albedo = params.base.xyz * params.base.w * (1.0 - metalness);
    let diffuse_lobe = diffuse_albedo * n_dot_l / 3.14159265;

    // Coat: second GGX lobe over the base, attenuating what reaches it.
    let coat_f0v = (params.iors.y - 1.0) / (params.iors.y + 1.0);
    let coat_f0sq = coat_f0v * coat_f0v;
    let coat_ndf = ggx_ndf(n_dot_h, coat_alpha);
    let coat_g = smith_g1(n_dot_v, coat_alpha) * smith_g1(n_dot_l, coat_alpha);
    let coat_f = fresnel_schlick(n_dot_h, vec3<f32>(coat_f0sq));
    let coat_lobe = coat_ndf * coat_g * coat_f * n_dot_l * params.coat.w;

    // Base energy after coat attenuation (approximate darkening).
    let base_atten = coat_attenuation(n_dot_v, params.coat.w, params.iors.y);

    // Emission is additive (spec §3.6).
    let emission = params.emission.xyz;

    // Direct-light mixture.
    let sun = SUN_COLOR;
    let direct = (diffuse_lobe + specular_lobe) * base_atten * sun + coat_lobe * sun;

    // Ambient: hemispherical env on the diffuse base (spec's full IBL
    // is deferred; the procedural hemisphere stands in as in MESH_SHADER).
    let env = environment_irradiance(n);
    let ambient = diffuse_albedo * env * base_atten;

    let color = direct + ambient + emission;
    return vec4<f32>(color, opacity);
}
"#;

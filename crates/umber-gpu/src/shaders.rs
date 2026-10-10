//! WGSL shader sources for the viewport mesh pass.

/// Normal-shaded mesh pass with a directional light and a hemispherical
/// environment light: the procedural sky/ground irradiance by default,
/// or a convolved environment map when one is bound (Wave-4 item 6 —
/// see `crate::ibl`'s convention block for the equirect math shared
/// with the convolve shader) plus the v1 specular IBL tier (three
/// cone-scaled prefiltered levels × the analytic BRDF fit — see
/// `crate::ibl::IBL_PREFILTER_SHADER`). Driven entirely by vertex
/// normals (no textures yet — that lands with the paint engine).
/// Matches the `Vertex` and `CameraUniform` layouts in `renderer.rs`;
/// bindings 1–3 are the diffuse IBL set, 4–7 the specular tier (see
/// `GpuContext`'s bind-group layout). Group 1 is the display LUT
/// (`crate::display_lut`; the snippet is that module's
/// `DISPLAY_LUT_WGSL`, verbatim), applied to the final shaded color as
/// the last step before the target.
pub const MESH_SHADER: &str = r#"
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
// Specular IBL tier (wave-5 v1): the three cone-scaled prefiltered
// levels (`crate::ibl::PREFILTER_ROUGHNESS`) plus the roughness/
// metallic uniform. The irradiance sampler (binding 2) is reused —
// the prefiltered maps share its repeat-U/clamp-V filtering. All four
// bindings are always populated (1x1 fallbacks when no environment is
// loaded) so the pipeline layout never changes at runtime.
@group(0) @binding(4) var prefilter0_tex: texture_2d<f32>;
@group(0) @binding(5) var prefilter1_tex: texture_2d<f32>;
@group(0) @binding(6) var prefilter2_tex: texture_2d<f32>;
struct SpecParams {
    roughness: f32,
    metallic: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(7) var<uniform> spec_params: SpecParams;

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

/// Split-Sum factor two without the LUT: the analytic fit of the
/// integrated GGX BRDF (Karis, "Real Shading in Unreal Engine 4",
/// SIGGRAPH 2013 — the `EnvBRDFApprox` closed form mobile pipelines
/// use in place of the 2D DFGLUT). Returns (scale, bias) with
/// specular = F0 * scale + bias. Mirrored op-for-op by
/// `crate::ibl::brdf_approx_cpu` (same literals, same order).
fn env_brdf_approx(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

/// Split-Sum factor one: cone-scaled prefiltered radiance at the
/// reflection vector, linearly blending the three fixed roughness
/// levels (0.0/0.5/1.0).
fn prefiltered_env(r: vec3<f32>, roughness: f32) -> vec3<f32> {
    let uv = equirect_uv(r);
    let p0 = textureSample(prefilter0_tex, irradiance_sampler, uv).rgb;
    let p1 = textureSample(prefilter1_tex, irradiance_sampler, uv).rgb;
    let p2 = textureSample(prefilter2_tex, irradiance_sampler, uv).rgb;
    let clo = clamp(roughness, 0.0, 1.0);
    if (clo < 0.5) {
        return mix(p0, p1, clo * 2.0);
    }
    return mix(p1, p2, (clo - 0.5) * 2.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let to_light = normalize(-camera.light_dir.xyz);
    let diffuse = max(dot(n, to_light), 0.0);
    let env = environment_irradiance(n);
    // Environment acts as the ambient term; the sun adds directional
    // energy on top, weighted so lit surfaces approach full brightness.
    var color = BASE_COLOR * (env + diffuse * SUN_COLOR * 0.55);
    // Specular IBL tier (wave-5 v1): Split-Sum factor one x factor two,
    // gated on the map flag — the diffuse term above is byte-untouched.
    // v1 stand-ins: a uniform view (this pass has no world-pos varying,
    // so camera.eye is the view source) and the SpecParams uniform
    // (roughness/metallic until the OpenPBR wiring exposes them).
    if (env_flags.flags == 1u) {
        let to_view = normalize(camera.eye.xyz);
        let reflect_dir = reflect(-to_view, n);
        let n_dot_v = clamp(dot(n, to_view), 0.0, 1.0);
        let rough = clamp(spec_params.roughness, 0.0, 1.0);
        let metal = clamp(spec_params.metallic, 0.0, 1.0);
        let f0 = mix(vec3<f32>(0.04), BASE_COLOR, metal);
        let ab = env_brdf_approx(n_dot_v, rough);
        color = color + prefiltered_env(reflect_dir, rough) * (f0 * ab.x + ab.y);
    }
    // Display LUT: the viewer chain, the last step before the target
    // (after diffuse + specular — the chain sees the finished color).
    // The identity table returns round(clamp(c)·255)/255, the byte the
    // unorm target would store for c anyway.
    return vec4<f32>(apply_display_lut(color), 1.0);
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

/// Barycentric-edge wireframe overlay (Wave-4 item 7).
///
/// NOT a line list: the pass draws the same triangles (from the
/// duplicated-vertex buffer `umber_mesh::wire_vertices_from_indices`
/// builds — positions plus one unit-simplex corner each) and shades
/// edges in the fragment stage via the standard `fwidth` technique, so
/// lines stay ~1px at any zoom. Drawn AFTER the mesh in the same render
/// pass with `LessEqual` depth and a clip-space z bias (see the vertex
/// stage) so coplanar edges win the z-fight against the mesh surface.
///
/// Bindings: group 0 binding 0 is the SAME `Camera` uniform the mesh
/// pass uses — the pipeline reuses the mesh's bind-group layout object,
/// so the app hands the already-built camera bind group straight
/// through. Group 1 binding 0 is the wire color (user-set, default 40%
/// white — see `renderer::WireColor`).
pub const WIREFRAME_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    eye: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct WireColor {
    color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> wire_color: WireColor;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) bary: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) bary: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var clip = camera.view_proj * vec4<f32>(in.position, 1.0);
    // Z-fight mitigation (the design's clip-space bias arm): nudge the
    // edge toward the camera so it wins against the coplanar mesh
    // surface under LessEqual depth. 1e-3 of clip.w sits far above f32
    // depth noise (~1e-7 relative) yet far below any visible parallax.
    clip.z -= 0.001 * clip.w;
    out.clip_position = clip;
    out.bary = in.bary;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let edge = min(min(in.bary.x, in.bary.y), in.bary.z);
    let w = fwidth(edge);
    let alpha = 1.0 - smoothstep(0.0, w * 1.0, edge);
    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(wire_color.color.rgb, wire_color.color.a * alpha);
}
"#;

/// Procedural infinite ground grid (Wave-4 item 7).
///
/// NOT a mesh: a fullscreen big-triangle pass whose fragment
/// reconstructs the world position per pixel by unprojecting the NDC
/// near/far points through the inverse view-proj and intersecting the
/// resulting ray with the y=0 plane. `world_from_ndc` below is the
/// shader-side copy of `crate::camera::world_from_ndc` (same formula —
/// the language boundary forces the duplication, documented here so
/// the next reader doesn't "dedupe" one side into drift); the Rust side
/// is shared by viewport picking and the offscreen-test math.
///
/// Pattern: minor lines every 1 world unit (25% white), major every 10
/// (thicker, 45% white), x-axis red-ish / z-axis blue-ish (the
/// Maya/Blender convention), infinite-grid fade to zero ~30 units from
/// the origin. Fragments below the alpha threshold discard so the clear
/// color shows through untouched.
///
/// Drawn BEFORE the mesh with depth-write OFF and depth-compare Always:
/// it is a reference plane, not geometry — the mesh overwrites it
/// wherever geometry exists.
pub const GRID_SHADER: &str = r#"
struct GridUniform {
    inv_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> grid: GridUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // NDC passed through explicitly: `@builtin(position)` arrives in
    // the FRAGMENT stage in window (pixel) space, not clip space, so
    // the fragment cannot recover NDC from it — the vertex hands the
    // big-triangle XY down directly (w=1, so clip.xy IS ndc.xy).
    @location(0) ndc_xy: vec2<f32>,
};

/// Big-triangle fullscreen pass: no vertex buffer, positions from the
/// vertex index. z=0 (near plane) — the fragment ignores the depth and
/// re-derives world pos from NDC instead.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.ndc_xy = positions[vertex_index];
    return out;
}

/// Shader-side copy of `crate::camera::world_from_ndc` — see that
/// function's doc comment for the sharing contract.
fn world_from_ndc(inv: mat4x4<f32>, ndc: vec4<f32>) -> vec3<f32> {
    let world = inv * ndc;
    return world.xyz / world.w;
}

/// Line coverage for a 1-unit grid along one axis: 0 at cell centers,
/// 1 on the line, ~1px wide via `fwidth` (screen-constant width).
fn grid_factor(coord: f32, pixel_world: f32) -> f32 {
    let dist = abs(fract(coord - 0.5) - 0.5);
    return 1.0 - smoothstep(0.0, pixel_world * 1.0, dist);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc_xy = in.ndc_xy;
    let ray_origin = world_from_ndc(grid.inv_view_proj, vec4<f32>(ndc_xy, 0.0, 1.0));
    let ray_far = world_from_ndc(grid.inv_view_proj, vec4<f32>(ndc_xy, 1.0, 1.0));
    let ray_dir = ray_far - ray_origin;
    // Looking parallel to (or away from) the plane: no grid here.
    if (abs(ray_dir.y) < 1e-6) {
        discard;
    }
    let t = -ray_origin.y / ray_dir.y;
    if (t < 0.0) {
        discard;
    }
    let world = ray_origin + ray_dir * t;

    let pixel_world = length(vec2<f32>(fwidth(world.x), fwidth(world.z)));
    let minor = max(grid_factor(world.x, pixel_world), grid_factor(world.z, pixel_world));
    // Major lines every 10 units: distance rescaled to world units so
    // the same pixel width reads thicker and brighter.
    let major_dist = abs(fract(world.x / 10.0 - 0.5) - 0.5) * 10.0;
    let major_dist_z = abs(fract(world.z / 10.0 - 0.5) - 0.5) * 10.0;
    let major = max(
        1.0 - smoothstep(0.0, pixel_world * 1.5, major_dist),
        1.0 - smoothstep(0.0, pixel_world * 1.5, major_dist_z)
    );
    // Axis highlight: the z=0 line runs along x (red-ish), the x=0
    // line along z (blue-ish).
    let axis_x = 1.0 - smoothstep(0.0, pixel_world * 1.5, abs(world.z));
    let axis_z = 1.0 - smoothstep(0.0, pixel_world * 1.5, abs(world.x));

    // Infinite-grid fade: full strength at the origin, gone by 30u.
    let fade = 1.0 - smoothstep(0.0, 30.0, length(world.xz));

    var color = vec3<f32>(0.25) * minor + vec3<f32>(0.45) * major;
    color = mix(color, vec3<f32>(0.80, 0.25, 0.25), axis_x);
    color = mix(color, vec3<f32>(0.25, 0.40, 0.90), axis_z);
    let alpha = max(max(minor, major), max(axis_x, axis_z)) * fade;
    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(color, alpha);
}
"#;

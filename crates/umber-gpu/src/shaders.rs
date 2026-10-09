//! WGSL shader sources for the viewport mesh pass.

/// Normal-shaded mesh pass with a directional light and a hemispherical
/// environment light (procedural sky/ground irradiance — the Wave-1
/// "bundled env map"; full HDR-texture IBL lands with Wave-2 image import).
/// Driven entirely by vertex normals (no textures yet — that lands with
/// the paint engine). Matches the `Vertex` and `CameraUniform` layouts in
/// `renderer.rs`.
pub const MESH_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    // xyz = normalized direction the light travels (surface -> fragment),
    // w unused (padding to satisfy uniform alignment).
    light_dir: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

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

/// Hemispherical environment irradiance: the procedural stand-in for an
/// environment map — sky above the horizon, ground below, blended by the
/// normal's vertical component.
fn environment_irradiance(n: vec3<f32>) -> vec3<f32> {
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

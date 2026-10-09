//! WGSL shader sources for the viewport mesh pass.

/// Normal-shaded mesh pass: a directional light plus flat ambient, driven
/// entirely by vertex normals (no textures yet — that lands with the paint
/// engine). Matches the `Vertex` and `CameraUniform` layouts in `renderer.rs`.
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

const AMBIENT: f32 = 0.18;
const BASE_COLOR: vec3<f32> = vec3<f32>(0.72, 0.72, 0.75);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let to_light = normalize(-camera.light_dir.xyz);
    let diffuse = max(dot(n, to_light), 0.0);
    let intensity = AMBIENT + diffuse * (1.0 - AMBIENT);
    return vec4<f32>(BASE_COLOR * intensity, 1.0);
}
"#;

//! umber-gpu — the single wgpu device context and all GPU work submission.
//!
//! ARCHITECTURE RULE (docs/specs/architecture.md): GPU work is submitted
//! only through this crate; UI never touches wgpu objects directly.
//!
//! The viewport render pass lands here: an orbit camera ([`camera`]), the
//! WGSL mesh shader ([`shaders`]), and the device context + egui paint
//! callback that records the mesh draw ([`renderer`]). All wgpu objects
//! (adapter/device/queue) are supplied by the caller from eframe's
//! `CreationContext::wgpu_render_state` — this crate never constructs a
//! `wgpu::Instance` or requests its own adapter outside of tests.
//!
//! Version policy (docs/specs/tech-stack.md): wgpu+naga+naga_oil pin as a
//! trio; no experimental wgpu features in the core paint path.

#![warn(missing_docs)]

pub mod camera;
pub mod golden;
pub mod paint;
pub mod paint_thread;
pub mod renderer;
pub mod shaders;

pub use camera::OrbitCamera;
pub use paint::{Dab, DabBuffer, PaintCompositor, PaintError, PaintTarget};
pub use paint_thread::{FrameStats, PaintThread, PaintThreadCommand};
/// The depth format the app must request from eframe
/// (`NativeOptions::depth_buffer = 32`) so the egui renderer's render pass
/// carries a depth attachment matching [`renderer::GpuContext`]'s pipeline.
pub const DEPTH_FORMAT_BITS: u8 = 32;
pub use renderer::{
    mesh_paint_shape, CameraUniform, GpuContext, GpuError, MeshBuffers, MeshPaintCallback, Vertex,
};

/// Identifies a backend device at runtime (for diagnostics + the future
/// lavapipe/WARP test path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Vulkan.
    Vulkan,
    /// Direct3D 12.
    Dx12,
    /// Apple Metal.
    Metal,
    /// OpenGL / OpenGL ES (includes the llvmpipe/lavapipe software path).
    Gl,
    /// Any backend not listed above (e.g. WebGPU, or an empty/unknown backend).
    Other,
}

impl Backend {
    /// Maps a `wgpu::Backend` to our own diagnostics-facing enum.
    pub fn from_wgpu(b: wgpu::Backend) -> Self {
        match b {
            wgpu::Backend::Vulkan => Backend::Vulkan,
            wgpu::Backend::Dx12 => Backend::Dx12,
            wgpu::Backend::Metal => Backend::Metal,
            wgpu::Backend::Gl => Backend::Gl,
            _ => Backend::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_mapping_covers_the_primary_backends() {
        assert_eq!(Backend::from_wgpu(wgpu::Backend::Vulkan), Backend::Vulkan);
        assert_eq!(Backend::from_wgpu(wgpu::Backend::Dx12), Backend::Dx12);
        assert_eq!(Backend::from_wgpu(wgpu::Backend::Metal), Backend::Metal);
        assert_eq!(Backend::from_wgpu(wgpu::Backend::Gl), Backend::Gl);
        assert_eq!(
            Backend::from_wgpu(wgpu::Backend::BrowserWebGpu),
            Backend::Other
        );
    }
}

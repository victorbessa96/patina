//! umber-gpu — the single wgpu device context and all GPU work submission.
//!
//! ARCHITECTURE RULE (docs/specs/architecture.md): GPU work is submitted
//! only through this crate; UI never touches wgpu objects directly.
//!
//! Wave 1 scope: type skeletons only. The claw pass lands the device
//! context, the frame loop integration (drain-events-then-acquire,
//! wgpu#2269), and the viewport render passes.
//!
//! Version policy (docs/specs/tech-stack.md): wgpu+naga+naga_oil pin as a
//! trio; no experimental wgpu features in the core paint path.

/// Identifies a backend device at runtime (for diagnostics + the future
/// lavapipe/WARP test path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    Dx12,
    Metal,
    Gl,
    Other,
}

impl Backend {
    pub fn from_wgpu(b: WgpuBackendId) -> Self {
        match b {
            0 => Backend::Vulkan,
            1 => Backend::Dx12,
            2 => Backend::Metal,
            3 => Backend::Gl,
            _ => Backend::Other,
        }
    }
}

/// Placeholder until the claw pass wires real wgpu instance inspection —
/// the enum above is ours; the mapping lands with the device context.
type WgpuBackendId = u8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_mapping_skeleton() {
        assert_eq!(Backend::from_wgpu(0), Backend::Vulkan);
        assert_eq!(Backend::from_wgpu(9), Backend::Other);
    }
}

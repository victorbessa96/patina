//! Shared bake-sourced map construction: the raw AO bake and the
//! flat-normal placeholder, used by both the Bakes panel and the Export
//! dialog so neither reimplements the other's GPU call.
//!
//! `bake_ao` is the AO arm of `BakesPanel::run_all`, lifted out so the
//! Export dialog can build an `umber_export::MapSet` without re-deriving
//! the bake params; the panel still applies its own dilation afterward
//! (this module returns the raw bake, undilated — see
//! `LANDING_NOTES_EXPORT_DIALOG.md` for why the export path also stays
//! undilated, matching `umber-cli export`).

use anyhow::Context as _;

/// Bakes ambient occlusion over `mesh` at `size²` with `rays` hemisphere
/// samples. Returns the raw RGBA8 bake (coverage alpha, no dilation) —
/// callers that want seam-filled output (the Bakes panel) dilate it
/// themselves; callers that mirror `umber-cli export` (the Export
/// dialog) use it as-is.
pub fn bake_ao(
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
    mesh: &umber_mesh::MeshData,
    size: u32,
    rays: u32,
) -> anyhow::Result<Vec<u8>> {
    let mut params = umber_bake::AoBakeParams::new(
        // max_distance/bias per the bake tests' convention; the plane is
        // unused by the mesh-fed path (the mesh's position map supplies
        // ray origins).
        10.0,
        0.01,
        umber_bake::PlaneDesc::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0],
        ),
    );
    params.rays = rays;
    umber_bake::ao::bake_ao_mesh(device, queue, mesh, size, size, &params).context("ao bake")
}

/// A flat tangent-space-up normal map (`[128, 128, 255, 255]` at every
/// texel) — the placeholder `umber-cli export` writes until the paint
/// pipeline grows a real normal source (painted or baked) to export
/// instead.
pub fn flat_normal_rgba8(size: u32) -> Vec<u8> {
    let texels = (size as usize) * (size as usize);
    let mut out = Vec::with_capacity(texels * 4);
    for _ in 0..texels {
        out.extend_from_slice(&[128, 128, 255, 255]);
    }
    out
}

/// Builds the export-ready [`umber_export::MapSet`] from an already-baked
/// AO buffer plus the flat-normal placeholder — the pure half of the
/// composition (no GPU access), so it is directly testable.
pub fn map_set_from(ao: Vec<u8>, size: u32) -> umber_export::MapSet {
    let mut set = umber_export::MapSet::new(size);
    set.set(umber_export::MapKind::AmbientOcclusion, ao);
    set.set(umber_export::MapKind::Normal, flat_normal_rgba8(size));
    set
}

/// Bakes AO and composes it with the flat-normal placeholder into one
/// [`umber_export::MapSet`] — the GPU-touching entry point the Export
/// dialog calls.
pub fn bake_export_map_set(
    device: &umber_gpu::WgpuDevice,
    queue: &umber_gpu::WgpuQueue,
    mesh: &umber_mesh::MeshData,
    size: u32,
    rays: u32,
) -> anyhow::Result<umber_export::MapSet> {
    let ao = bake_ao(device, queue, mesh, size, rays)?;
    Ok(map_set_from(ao, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_normal_is_up_everywhere() {
        let bytes = flat_normal_rgba8(2);
        assert_eq!(bytes.len(), 2 * 2 * 4);
        for texel in bytes.chunks_exact(4) {
            assert_eq!(texel, [128, 128, 255, 255]);
        }
    }

    #[test]
    fn flat_normal_zero_size_is_empty() {
        assert!(flat_normal_rgba8(0).is_empty());
    }

    #[test]
    fn map_set_from_carries_ao_and_normal_only() {
        let size = 2;
        let ao = vec![9u8; (size * size * 4) as usize];
        let set = map_set_from(ao, size);
        let kinds: Vec<umber_export::MapKind> = set.maps_iter().collect();
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&umber_export::MapKind::AmbientOcclusion));
        assert!(kinds.contains(&umber_export::MapKind::Normal));
        assert!(!kinds.contains(&umber_export::MapKind::BaseColor));
    }
}

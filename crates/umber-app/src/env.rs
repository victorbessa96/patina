//! Environment-map loading (Wave-4 item 6): File > Load Environment…
//! plus the committed studio default.
//!
//! GPU access follows the `viewport`/`paint_state`/`bakes_panel`
//! pattern: this module never names a `wgpu` type — the device and
//! queue travel inside `&GpuContext`, and image decoding goes through
//! `umber-export`'s read APIs (no new dependencies: PNG via the `png`
//! crate umber-export already uses, EXR via the workspace `exr` crate).

use std::path::{Path, PathBuf};

use umber_gpu::{EnvFormat, EnvIrradiance, GpuContext};

/// The committed neutral-studio default (see
/// `assets/gen/make_studio_env.py`).
pub const DEFAULT_ENV_ASSET: &str = "assets/env/studio-neutral.png";

/// Picks the pixel encoding from the file extension (case-insensitive).
/// Returns `None` for anything that isn't an LDR PNG or HDR EXR equirect.
pub fn format_for_path(path: &Path) -> Option<EnvFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some(EnvFormat::Rgba8Unorm),
        "exr" => Some(EnvFormat::Rgba32Float),
        _ => None,
    }
}

/// Loads an environment file (`.png` or `.exr`) and convolves it into an
/// [`EnvIrradiance`] on `gpu` — the File > Load Environment… driver.
/// Both readers return GPU row order (row 0 = -Y; see their doc
/// comments), which is what `EnvIrradiance::from_equirect` expects.
pub fn load_environment_file(gpu: &GpuContext, path: &Path) -> anyhow::Result<EnvIrradiance> {
    let format = format_for_path(path).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported environment format: {} (want .png or .exr)",
            path.display()
        )
    })?;
    let (width, height, bytes) = match format {
        EnvFormat::Rgba8Unorm => {
            let (w, h, rgba) = umber_export::png::read_png_rgba8(path)
                .map_err(|e| anyhow::anyhow!("reading {}: {e:#}", path.display()))?;
            (w, h, rgba)
        }
        EnvFormat::Rgba32Float => {
            let (w, h, floats) = umber_export::exr::read_exr_rgba_f32(path)
                .map_err(|e| anyhow::anyhow!("reading {}: {e:#}", path.display()))?;
            // Native-endian float bytes for `from_equirect`'s
            // reinterpretation (x86-64/aarch64 — umber's targets — are
            // little-endian). Manual loop, not bytemuck: umber-app
            // doesn't depend on it, and this keeps Cargo.toml untouched.
            let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
            (w, h, bytes)
        }
    };
    EnvIrradiance::from_equirect(&gpu.device, &gpu.queue, &bytes, width, height, format)
        .map_err(|e| anyhow::anyhow!("convolving {}: {e:#}", path.display()))
}

/// Candidate locations for the studio default, in order: the
/// repository-relative path (correct when the binary runs with the repo
/// root as CWD — `cargo run`, the usual dev loop), then the
/// compile-time manifest path (correct when CWD is elsewhere but the
/// binary was built from this source tree).
pub fn default_env_candidates() -> Vec<PathBuf> {
    vec![
        PathBuf::from(DEFAULT_ENV_ASSET),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/env/studio-neutral.png"),
    ]
}

/// Loads the committed studio default if present, else `None`
/// (procedural fallback — never a startup failure). Logs which path
/// won so a missing asset is diagnosable, not silent.
pub fn load_default_environment(gpu: &GpuContext) -> Option<EnvIrradiance> {
    for candidate in default_env_candidates() {
        if !candidate.is_file() {
            continue;
        }
        match load_environment_file(gpu, &candidate) {
            Ok(env) => {
                log::info!("default environment: {}", candidate.display());
                return Some(env);
            }
            Err(err) => {
                log::warn!(
                    "default environment {} failed: {err:#}",
                    candidate.display()
                );
                return None;
            }
        }
    }
    log::info!("no default environment found; procedural fallback");
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_dispatch_is_case_insensitive() {
        assert_eq!(
            format_for_path(Path::new("studio.PNG")),
            Some(EnvFormat::Rgba8Unorm)
        );
        assert_eq!(
            format_for_path(Path::new("probe.ExR")),
            Some(EnvFormat::Rgba32Float)
        );
        assert_eq!(format_for_path(Path::new("albedo.jpg")), None);
        assert_eq!(format_for_path(Path::new("no-extension")), None);
    }

    #[test]
    fn default_candidates_end_at_the_committed_asset() {
        let candidates = default_env_candidates();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0], PathBuf::from(DEFAULT_ENV_ASSET));
        assert!(
            candidates[1].ends_with("assets/env/studio-neutral.png"),
            "manifest fallback must land on the asset: {}",
            candidates[1].display()
        );
    }

    #[test]
    fn loading_an_unsupported_format_is_an_error_not_a_panic() {
        // No GPU needed: the format gate runs before any device touch.
        assert!(format_for_path(Path::new("x.tiff")).is_none());
    }

    /// Requests the default device (plain, no features — the convolve
    /// pass needs none). Returns `None` where no wgpu adapter exists,
    /// mirroring `paint_state`'s test convention. NOTE: umber-app has no
    /// `gpu` feature gate (unlike umber-gpu/umber-bake) — GPU tests here
    /// run under plain `cargo test -p umber-app` and skip without an
    /// adapter.
    fn try_request_adapter() -> Option<wgpu::Adapter> {
        let instance = wgpu::Instance::default();
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
            Ok(adapter) => Some(adapter),
            Err(_) => {
                eprintln!("skipping: no wgpu adapter available");
                None
            }
        }
    }

    #[test]
    fn committed_studio_default_loads_and_convolves() {
        let Some(adapter) = try_request_adapter() else {
            return;
        };
        // The committed asset must exist from the crate dir (tests run
        // with CWD = crates/umber-app, so the manifest fallback wins).
        let manifest_asset =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/env/studio-neutral.png");
        assert!(
            manifest_asset.is_file(),
            "committed default missing: {}",
            manifest_asset.display()
        );
        let (w, h, rgba) =
            umber_export::png::read_png_rgba8(&manifest_asset).expect("studio PNG must decode");
        assert_eq!((w, h), (1024, 512), "generator output dimensions");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
        let gpu = umber_gpu::GpuContext::new(
            adapter,
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            None,
        );
        let env = EnvIrradiance::from_equirect(
            &gpu.device,
            &gpu.queue,
            &rgba,
            w,
            h,
            EnvFormat::Rgba8Unorm,
        )
        .expect("studio default must convolve");
        assert_eq!(env.dimensions(), (32, 16));
    }
}

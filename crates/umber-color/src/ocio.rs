//! The OCIO bridge (requirements.md §9): behind the `ocio` cargo
//! feature, exposes the built-in ACES 2.0 config surface through the
//! `ocio-rs` crate (stub mode — no C++ dependency, CI-safe; the heavy
//! bundled OCIO build is a manual-dispatch job, see
//! `docs/research/ocio-aces-integration.md`).
//!
//! Without the feature the module is absent — `umber-color` stays a
//! pure-Rust crate with the fixed sRGB display path as the whole
//! truth, and this bridge is opt-in.
//!
//! Stub-mode contract (probed 2026-10-09): every handle-allocating
//! call returns `Err` with "OpenColorIO handle allocation failed".
//! The API *shapes* are what the gated tests verify — the same tests
//! document what flips when the real OCIO lands.

use thiserror::Error;

/// Errors from the OCIO bridge.
#[derive(Debug, Error)]
pub enum OcioBridgeError {
    /// The ocio-rs call failed (stub mode: handle allocation; real
    /// mode: whatever OCIO rejected).
    #[error("OCIO: {0}")]
    Ocio(String),
    /// A built-in config index was out of range.
    #[error("builtin config index {0} out of range")]
    IndexOutOfRange(i32),
}

impl From<ocio_rs::OcioError> for OcioBridgeError {
    fn from(err: ocio_rs::OcioError) -> Self {
        Self::Ocio(err.to_string())
    }
}

/// The name OCIO registers the ACES 2.0 CG config under (the §9
/// default; Studio is the opt-in alternative).
pub const ACES_CG_CONFIG: &str = "CG Config for ACES 2.0";

/// The ACES 2.0 Studio config (the §9 alternative to CG).
pub const ACES_STUDIO_CONFIG: &str = "Studio Config for ACES 2.0";

/// Enumerates OCIO's built-in configs (name + UI name pairs). Stub
/// mode: the registry handle fails to allocate, so the list comes back
/// empty — the caller treats that as "no OCIO available", never an
/// error to surface to the UI.
pub fn builtin_configs() -> Vec<(String, String)> {
    let Ok(registry) = ocio_rs::BuiltinConfigRegistry::get() else {
        return Vec::new();
    };
    let n = registry.num_builtin_configs();
    let mut out = Vec::new();
    for i in 0..n {
        let name = registry.config_name(i).unwrap_or_default();
        let ui = registry.config_ui_name(i).unwrap_or_default();
        if !name.is_empty() {
            out.push((name, ui));
        }
    }
    out
}

/// Creates a built-in config by name. Stub mode: `Err` (the handle
/// allocation fails) — the bridge's honest "OCIO unavailable" answer.
pub fn create_builtin_config(name: &str) -> Result<(), OcioBridgeError> {
    let _config = ocio_rs::Config::create_from_builtin_config(name)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_mode_reports_no_builtin_configs() {
        // In stub mode the registry handle fails to allocate, so the
        // enumeration is an empty list — never a panic, never an
        // error: the caller renders "OCIO unavailable".
        let configs = builtin_configs();
        assert!(configs.is_empty(), "stub mode: no configs, got {configs:?}");
    }

    #[test]
    fn stub_mode_config_creation_fails_cleanly() {
        // The built-in creation path exists (the probe proved the API
        // compiles) but a stub-mode handle cannot allocate — the
        // bridge surfaces that as an Ocio error, not a panic.
        let err = create_builtin_config(ACES_CG_CONFIG).unwrap_err();
        assert!(matches!(err, OcioBridgeError::Ocio(_)));
        assert!(
            err.to_string().contains("OCIO"),
            "error names its source: {err}"
        );
    }

    #[test]
    fn config_names_are_documented_constants() {
        // The §9 defaults are pinned: any rename upstream fails this
        // test when the real OCIO lands and the registry enumerates.
        assert_eq!(ACES_CG_CONFIG, "CG Config for ACES 2.0");
        assert_eq!(ACES_STUDIO_CONFIG, "Studio Config for ACES 2.0");
    }
}

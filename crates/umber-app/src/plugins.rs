//! WASM plugin loading for the Graph panel — the plugins design's
//! slice 3 (`docs/specs/wasm-plugins-design.md`, "Registry integration").
//!
//! # Where plugins come from ([`plugin_paths`])
//!
//! 1. The per-user plugins dir: the sibling of the brush presets dir
//!    (`$XDG_DATA_HOME/umber/plugins`, `$HOME/.local/share/umber/plugins`,
//!    `%APPDATA%\umber\plugins`).
//! 2. Dev builds only (`debug_assertions`): the repo's `examples/`, where
//!    the three example `.wasm` modules are checked in.
//!
//! Each dir's direct-child `*.wasm` files load in file-name byte order.
//! A missing/unreadable dir is a silent skip (the brush library-loader
//! contract); once a file path is in hand, every failure is a per-file
//! WARNING in the [`PluginLoadReport`] — a broken plugin never takes the
//! graph panel down.
//!
//! # The collision rule
//!
//! [`umber_graph::NodeRegistry::register`] is a plain map insert: a later
//! registration silently REPLACES an earlier one. The loader does not
//! inherit that. A plugin whose name is already registered — a built-in
//! (`blur`, `flood_fill`, …) or a plugin loaded earlier in the scan — is
//! REJECTED as a load failure, so a plugin can never change what an
//! existing document's built-in node means, and the `.mtlx` writer can
//! never declare the same nodedef twice. First registration wins; the
//! user dir scans before `examples/`, so a user's copy of an example
//! plugin is the one that loads.

use std::path::{Path, PathBuf};

use umber_graph::mtlx::NodedefDecl;
use umber_graph::NodeRegistry;
use umber_wasm::PluginRuntime;

/// The outcome of one [`load_plugins`] scan.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PluginLoadReport {
    /// Registered plugin node names, in load order.
    pub loaded: Vec<String>,
    /// Files that did not register: `(path, reason)`, in scan order.
    pub failed: Vec<(PathBuf, String)>,
}

impl PluginLoadReport {
    /// The panel's status line: `None` when the scan found no `.wasm`
    /// files at all (nothing worth saying), else
    /// `N plugin(s) loaded, M failed[: a.wasm, b.wasm]`.
    pub fn status_line(&self) -> Option<String> {
        if self.loaded.is_empty() && self.failed.is_empty() {
            return None;
        }
        let mut line = format!(
            "{} plugin(s) loaded, {} failed",
            self.loaded.len(),
            self.failed.len()
        );
        if !self.failed.is_empty() {
            let names: Vec<String> = self
                .failed
                .iter()
                .map(|(path, _)| {
                    path.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.display().to_string())
                })
                .collect();
            line.push_str(": ");
            line.push_str(&names.join(", "));
        }
        Some(line)
    }
}

/// The plugin dirs in scan order, from an already-resolved user brushes
/// dir (its sibling `plugins/` is the user plugins dir) and the dev
/// `examples/` dir. Pure — [`plugin_paths`] is the env-reading wrapper.
pub fn plugin_paths_from(user_brush_dir: Option<&Path>, dev_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(base) = user_brush_dir.and_then(Path::parent) {
        dirs.push(base.join("plugins"));
    }
    if let Some(dev) = dev_dir {
        dirs.push(dev.to_path_buf());
    }
    dirs
}

/// The repo's `examples/` dir (where the example `.wasm` modules live).
pub fn dev_plugin_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

/// The plugin dirs for this process: the user plugins dir (when a home
/// is discoverable) then, in dev builds, the repo's `examples/`. Dirs
/// need not exist — the loader skips missing ones silently.
pub fn plugin_paths() -> Vec<PathBuf> {
    let user = umber_brush::preset_library::user_preset_dir();
    let dev = cfg!(debug_assertions).then(dev_plugin_dir);
    plugin_paths_from(user.as_deref(), dev.as_deref())
}

/// Scans `dirs` in order and registers every loadable plugin into
/// `registry` (see the module docs for the tolerance + collision rules).
/// The registry is untouched by a failed file.
pub fn load_plugins(registry: &mut NodeRegistry, dirs: &[PathBuf]) -> PluginLoadReport {
    let mut report = PluginLoadReport::default();
    for dir in dirs {
        for path in sorted_wasm_files(dir) {
            match load_one(registry, &path) {
                Ok(name) => report.loaded.push(name),
                Err(reason) => {
                    log::warn!("plugin {} not loaded: {reason}", path.display());
                    report.failed.push((path, reason));
                }
            }
        }
    }
    report
}

/// The `.mtlx` declaration for a loaded plugin node. The v1 wire's
/// node-def record carries only the name and the input/param COUNTS — no
/// param names or types — so the decl declares the one image input
/// (`in: color3`, the filter-node convention) and no params. Param VALUES
/// still round-trip: they ride each `<node>`'s own `<input>` elements,
/// which the mtlx layer preserves regardless of the decl. The decl's job
/// is the interchange contract — the type is declared in-document, so it
/// loads without an unknown-type warning.
pub fn plugin_nodedef(name: &str) -> NodedefDecl {
    NodedefDecl {
        name: name.to_string(),
        category: name.to_string(),
        inputs: vec![("in".into(), "color3".into())],
    }
}

/// Read -> compile + contract-check -> collision check -> register.
fn load_one(registry: &mut NodeRegistry, path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read: {e}"))?;
    let module = PluginRuntime::new()
        .load(&bytes)
        .map_err(|e| e.to_string())?;
    let name = module.def().name.clone();
    if registry.get(&name).is_some() {
        return Err(format!(
            "node name {name:?} is already registered (plugins never shadow)"
        ));
    }
    Ok(umber_wasm::register_module(registry, module))
}

/// Direct-child `*.wasm` files of `dir`, sorted by file name in byte
/// order; empty on any directory-level problem (silent skip).
fn sorted_wasm_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(Vec<u8>, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wasm"))
        .map(|e| (e.file_name().as_encoded_bytes().to_vec(), e.path()))
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files.into_iter().map(|(_, path)| path).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique scratch dir under `temp_dir` (no test-dep crates).
    fn fresh_tmpdir(tag: &str) -> PathBuf {
        let id = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "umber-app-plugins-test-{}-{}-{}",
            std::process::id(),
            id,
            tag
        ));
        std::fs::create_dir_all(&dir).expect("test tmpdir must be creatable");
        dir
    }

    fn blur5_bytes() -> Vec<u8> {
        let path = dev_plugin_dir().join("plugin_blur5.wasm");
        std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{} is git-tracked and must exist: {e}", path.display()))
    }

    #[test]
    fn plugin_paths_put_the_user_sibling_dir_before_examples() {
        let brushes = umber_brush::preset_library::user_preset_dir_from("/home/x", None);
        let dev = Path::new("/repo/examples");
        assert_eq!(
            plugin_paths_from(Some(&brushes), Some(dev)),
            vec![
                PathBuf::from("/home/x/.local/share/umber/plugins"),
                PathBuf::from("/repo/examples"),
            ]
        );
        let xdg = umber_brush::preset_library::user_preset_dir_from("/home/x", Some("/xdg"));
        assert_eq!(
            plugin_paths_from(Some(&xdg), None),
            vec![PathBuf::from("/xdg/umber/plugins")]
        );
        assert!(plugin_paths_from(None, None).is_empty(), "no home, no dev");

        // The env wrapper: every entry is a plugins/ or the examples dir,
        // and dev builds (the test profile) include examples/.
        let live = plugin_paths();
        assert!(live
            .iter()
            .all(|p| p.ends_with("plugins") || p.ends_with("examples")));
        if cfg!(debug_assertions) {
            assert_eq!(live.last(), Some(&dev_plugin_dir()));
        }
    }

    #[test]
    fn loader_registers_the_good_plugin_and_warns_on_the_rest() {
        let dir = fresh_tmpdir("tolerance");
        std::fs::write(dir.join("a_blur5.wasm"), blur5_bytes()).unwrap();
        std::fs::write(dir.join("b_garbage.wasm"), b"not wasm at all").unwrap();
        // A second blur5: the collision rule rejects it (first wins).
        std::fs::write(dir.join("c_blur5_again.wasm"), blur5_bytes()).unwrap();
        // Ignored: not *.wasm.
        std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();
        let missing = dir.join("does-not-exist");

        let mut registry = NodeRegistry::seeded();
        let report = load_plugins(&mut registry, &[missing, dir.clone()]);
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(report.loaded, vec!["blur5".to_string()]);
        let failed: Vec<&str> = report
            .failed
            .iter()
            .map(|(p, _)| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(failed, vec!["b_garbage.wasm", "c_blur5_again.wasm"]);
        assert!(report.failed[1].1.contains("already registered"));
        let defs = registry.node_defs();
        assert!(defs.contains(&"blur5"), "good plugin registers: {defs:?}");
        assert!(defs.contains(&"uniform"), "built-ins survive: {defs:?}");
        assert_eq!(
            report.status_line().as_deref(),
            Some("1 plugin(s) loaded, 2 failed: b_garbage.wasm, c_blur5_again.wasm")
        );
    }

    #[test]
    fn a_plugin_never_shadows_a_built_in() {
        // Pre-register a fake built-in under the plugin's name.
        let mut registry = NodeRegistry::seeded();
        let passthrough = registry.get("passthrough").unwrap();
        registry.register("blur5", passthrough);
        let dir = fresh_tmpdir("shadow");
        std::fs::write(dir.join("blur5.wasm"), blur5_bytes()).unwrap();
        let report = load_plugins(&mut registry, std::slice::from_ref(&dir));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(report.loaded.is_empty());
        assert_eq!(report.failed.len(), 1);
    }

    #[test]
    fn empty_scan_says_nothing() {
        let report = load_plugins(&mut NodeRegistry::new(), &[]);
        assert_eq!(report.status_line(), None);
    }
}

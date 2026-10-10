//! The brush-preset shelf: priority-ordered library loading.
//!
//! Wave-4 slice 3 (docs/specs/brush-presets-design.md: the "Library search
//! path" paragraph + the properties-panel data model section). A [`Library`]
//! is built by [`Library::load_from_dirs`] from a priority-ordered list of
//! directories — earlier directories win — or by [`Library::load_default_dirs`]
//! from the packaged-app search path (`[user_dir, install_dir/brushes]`).
//!
//! The search-path contract, in full:
//! - Nonexistent directories are skipped silently: they are not errors.
//! - Only `*.umberbrush` files are read (extension match, case-sensitive).
//! - Subdirectories are NOT recursed in v1; only direct children load.
//! - Each file runs the `read bytes -> BrushPreset::from_json ->
//!   validate()` pipeline. Any per-file failure (IO, UTF-8, parse, format,
//!   validate) lands in [`Library::errors`] as a `(path, err)` tuple and the
//!   library CONTINUES: a broken user preset must not kill the shelf.
//! - Name collisions resolve user-dir-wins (the Substance behavior): if two
//!   files produce presets with the same `name`, the earlier directory's
//!   entry wins; within one directory, the first file in byte order of file
//!   name wins. Losers are silently overridden — not errors — and counted in
//!   [`Library::overridden`] for observability and tests.
//! - Final entries are sorted by preset name (byte order) for stable shelf
//!   order. Loading is deterministic: same directories + same files produce
//!   a byte-identical [`Library`].
//!
//! No function in this module panics on user data: every filesystem and
//! parse failure is either a silent skip (missing dirs, unreadable dir
//! listings) or an [`Library::errors`] tuple (per-file failures).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::preset::{BrushPreset, PresetError};

/// One successfully loaded preset plus the file it came from.
///
/// `source_path` is the full path of the winning `*.umberbrush` file, kept
/// so the properties panel's explicit-save path can write back to it.
#[derive(Debug, Clone, PartialEq)]
pub struct LibraryEntry {
    /// The validated preset.
    pub preset: BrushPreset,
    /// The `*.umberbrush` file this preset was loaded from.
    pub source_path: PathBuf,
}

/// A loaded preset shelf: the winning entries plus every per-file failure.
///
/// Built by [`Self::load_from_dirs`] or [`Self::load_default_dirs`].
/// `entries` is sorted by preset name (byte order); `errors` is in load
/// order (directories in priority order, files in byte order of file name
/// within each directory), so repeated loads compare equal.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Library {
    /// Successfully loaded presets, sorted by preset name (byte order).
    pub entries: Vec<LibraryEntry>,
    /// Per-file failures as `(path, err)` tuples, in deterministic load
    /// order. A broken file never aborts the load; it lands here.
    pub errors: Vec<(PathBuf, PresetError)>,
    /// Count of files whose presets loaded and validated fine but lost a
    /// name collision (earlier directory, or earlier file name in the same
    /// directory, won). Overridden files are NOT errors.
    pub overridden: usize,
}

impl Library {
    /// Load presets from priority-ordered directories: earlier dirs win.
    ///
    /// For each directory (nonexistent ones skipped silently — they are not
    /// errors), every direct-child `*.umberbrush` file (extension match,
    /// case-sensitive; subdirectories NOT recursed in v1) runs the
    /// `read bytes -> BrushPreset::from_json -> validate()` pipeline.
    /// Failures land in [`Self::errors`] and the load continues.
    ///
    /// Name collisions: the earlier directory's entry wins (user dir first
    /// = user overrides shipped); within one directory the first file in
    /// byte order of file name wins. Losers bump [`Self::overridden`].
    /// The returned [`Self::entries`] is sorted by preset name (byte order).
    pub fn load_from_dirs(dirs: &[PathBuf]) -> Library {
        let mut library = Library::default();
        let mut seen_names: HashSet<String> = HashSet::new();
        for dir in dirs {
            for path in sorted_preset_files(dir) {
                match load_one_file(&path) {
                    Ok(preset) => {
                        if seen_names.insert(preset.name.clone()) {
                            library.entries.push(LibraryEntry {
                                preset,
                                source_path: path,
                            });
                        } else {
                            library.overridden += 1;
                        }
                    }
                    Err(err) => library.errors.push((path, err)),
                }
            }
        }
        library
            .entries
            .sort_by(|a, b| a.preset.name.cmp(&b.preset.name));
        library
    }

    /// Load the packaged-app search path: `[user_dir, install_dir/brushes]`.
    ///
    /// `user_dir` honors `$XDG_DATA_HOME/umber/brushes`, falling back to
    /// `$HOME/.local/share/umber/brushes` on Linux, and uses
    /// `%APPDATA%\umber\brushes` on Windows. `install_dir` is the
    /// executable's own directory (`std::env::current_exe` -> parent) plus
    /// `"brushes"`. Either side may be absent (missing env vars, missing
    /// exe path): missing entries are simply omitted, and missing
    /// directories skip silently per [`Self::load_from_dirs`].
    ///
    /// This is the packaged path only. A dev-mode caller that also wants
    /// the repo's `assets/brushes` passes extra directories to
    /// [`Self::load_from_dirs`] itself.
    pub fn load_default_dirs() -> Library {
        let mut dirs = Vec::new();
        if let Some(user) = user_brushes_dir() {
            dirs.push(user);
        }
        if let Some(install) = install_brushes_dir() {
            dirs.push(install);
        }
        Self::load_from_dirs(&dirs)
    }
}

/// The per-user brushes directory: `$XDG_DATA_HOME/umber/brushes`, or
/// `$HOME/.local/share/umber/brushes` when `XDG_DATA_HOME` is unset or
/// empty (Linux); `%APPDATA%\umber\brushes` on Windows. `None` when no
/// usable home is discoverable (missing/empty env vars).
fn user_brushes_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let appdata = std::env::var_os("APPDATA")?;
        if appdata.is_empty() {
            return None;
        }
        Some(Path::new(&appdata).join("umber").join("brushes"))
    }
    #[cfg(not(windows))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            if !xdg.is_empty() {
                return Some(Path::new(&xdg).join("umber").join("brushes"));
            }
        }
        let home = std::env::var_os("HOME")?;
        if home.is_empty() {
            return None;
        }
        Some(
            Path::new(&home)
                .join(".local")
                .join("share")
                .join("umber")
                .join("brushes"),
        )
    }
}

/// The shipped brushes directory next to the running executable:
/// `std::env::current_exe` -> parent + `"brushes"`. `None` when the exe
/// path is undiscoverable or has no parent.
fn install_brushes_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let parent = exe.parent()?;
    Some(parent.join("brushes"))
}

/// Direct-child `*.umberbrush` files of `dir`, sorted by file name in byte
/// order. Returns empty on anything unexpected (nonexistent dir, unreadable
/// listing, unreadable entries): directory-level problems are silent skips,
/// never errors — per-file tolerance in [`Library::load_from_dirs`] only
/// starts once a file path is in hand.
fn sorted_preset_files(dir: &Path) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut files: Vec<(Vec<u8>, PathBuf)> = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if !is_file {
            continue;
        }
        let is_preset = path.extension().is_some_and(|ext| ext == "umberbrush");
        if !is_preset {
            continue;
        }
        let name_key = entry.file_name().as_encoded_bytes().to_vec();
        files.push((name_key, path));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files.into_iter().map(|(_, path)| path).collect()
}

/// The per-file pipeline: read bytes -> UTF-8 -> `from_json` -> `validate`.
///
/// IO and UTF-8 failures have no dedicated [`PresetError`] variant, so they
/// fold into `PresetError::Json` with line/column 0 (the same convention
/// [`BrushPreset::to_json`] uses for errors that carry no source position),
/// keeping the message human-readable.
fn load_one_file(path: &Path) -> Result<BrushPreset, PresetError> {
    let bytes = std::fs::read(path).map_err(|e| PresetError::Json {
        source_line: 0,
        source_column: 0,
        message: format!("cannot read {}: {e}", path.display()),
    })?;
    let text = String::from_utf8(bytes).map_err(|e| PresetError::Json {
        source_line: 0,
        source_column: 0,
        message: format!("{} is not valid UTF-8: {e}", path.display()),
    })?;
    let preset = BrushPreset::from_json(&text)?;
    preset.validate()?;
    Ok(preset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{LazyMouseConfig, SpacingConfig};
    use crate::{BrushParams, OneEuroParams};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique scratch directory under `std::env::temp_dir` (no test-dep
    /// crates). Returns the path; the caller owns cleanup.
    fn fresh_tmpdir(tag: &str) -> PathBuf {
        let id = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "umber-brush-lib-test-{}-{}-{}",
            std::process::id(),
            id,
            tag
        ));
        std::fs::create_dir_all(&dir).expect("test tmpdir must be creatable");
        dir
    }

    fn remove_tmpdir(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn test_preset(name: &str, alpha: f32) -> BrushPreset {
        BrushPreset {
            name: name.to_owned(),
            params: BrushParams {
                color: [0.8, 0.2, 0.1, 1.0],
                alpha,
                hardness: 0.25,
                pressure_gamma: 1.0,
            },
            one_euro: OneEuroParams {
                min_cutoff: 1.2,
                beta: 0.02,
                d_cutoff: 1.0,
            },
            lazy_mouse: LazyMouseConfig::default(),
            spacing: SpacingConfig::default(),
            alpha_curve: crate::preset::ControlCurve::identity(),
            radius_curve: crate::preset::ControlCurve::identity(),
        }
    }

    fn write_preset(dir: &Path, file_name: &str, preset: &BrushPreset) -> PathBuf {
        let path = dir.join(file_name);
        let json = preset.to_json().expect("test preset must serialize");
        std::fs::write(&path, json).expect("test preset must be writable");
        path
    }

    fn entry_names(library: &Library) -> Vec<&str> {
        library
            .entries
            .iter()
            .map(|e| e.preset.name.as_str())
            .collect()
    }

    #[test]
    fn happy_path_loads_two_presets_sorted() {
        let dir = fresh_tmpdir("happy");
        // Write out of name order on purpose: shelf order must still sort.
        write_preset(&dir, "b_soft.umberbrush", &test_preset("Soft Round", 0.7));
        write_preset(&dir, "a_hard.umberbrush", &test_preset("Hard Round", 0.95));
        // A non-preset file must be ignored, not error.
        std::fs::write(dir.join("notes.txt"), "not a preset").expect("writable");

        let library = Library::load_from_dirs(std::slice::from_ref(&dir));

        assert_eq!(library.entries.len(), 2, "both presets must load");
        assert!(
            library.errors.is_empty(),
            "no errors expected, got {:?}",
            library.errors
        );
        assert_eq!(library.overridden, 0);
        assert_eq!(entry_names(&library), vec!["Hard Round", "Soft Round"]);
        assert!(
            library
                .entries
                .iter()
                .all(|e| e.source_path.starts_with(&dir)),
            "entries must remember their source files"
        );

        remove_tmpdir(&dir);
    }

    #[test]
    fn broken_files_land_in_errors_and_shelf_survives() {
        let dir = fresh_tmpdir("tolerance");
        let good = write_preset(&dir, "good.umberbrush", &test_preset("Good", 0.5));
        let broken_path = dir.join("broken.umberbrush");
        std::fs::write(&broken_path, "{ this is not json,,").expect("writable");
        let wrong_format_path = dir.join("wrong.umberbrush");
        let mut wrong: serde_json::Value = serde_json::from_str(
            &test_preset("Wrong", 0.5)
                .to_json()
                .expect("test preset must serialize"),
        )
        .expect("valid JSON");
        wrong.as_object_mut().expect("top-level object").insert(
            "format".to_owned(),
            serde_json::Value::String("some-other-format".to_owned()),
        );
        std::fs::write(
            &wrong_format_path,
            serde_json::to_string_pretty(&wrong).expect("re-serialize"),
        )
        .expect("writable");
        // A file that parses but fails validate (empty name) is an error too.
        let invalid_path = write_preset(&dir, "invalid.umberbrush", &test_preset("", 0.5));

        let library = Library::load_from_dirs(std::slice::from_ref(&dir));

        assert_eq!(library.entries.len(), 1);
        assert_eq!(library.entries[0].preset.name, "Good");
        assert_eq!(library.entries[0].source_path, good);
        assert_eq!(
            library.errors.len(),
            3,
            "broken JSON + wrong format + invalid params must each error, got {:?}",
            library.errors
        );
        let mut error_paths: Vec<PathBuf> = library.errors.iter().map(|(p, _)| p.clone()).collect();
        error_paths.sort();
        let mut expected = vec![broken_path, wrong_format_path, invalid_path];
        expected.sort();
        assert_eq!(
            error_paths, expected,
            "error tuples must carry the right paths"
        );
        assert!(
            library
                .errors
                .iter()
                .any(|(_, e)| matches!(e, PresetError::Json { .. })),
            "malformed JSON must surface as PresetError::Json"
        );
        assert!(
            library
                .errors
                .iter()
                .any(|(_, e)| matches!(e, PresetError::WrongFormat { .. })),
            "wrong envelope must surface as PresetError::WrongFormat"
        );
        assert!(
            library
                .errors
                .iter()
                .any(|(_, e)| matches!(e, PresetError::EmptyName)),
            "empty name must surface as PresetError::EmptyName from validate()"
        );

        remove_tmpdir(&dir);
    }

    #[test]
    fn earlier_dir_wins_name_collision() {
        let dir1 = fresh_tmpdir("priority-1");
        let dir2 = fresh_tmpdir("priority-2");
        write_preset(&dir1, "mine.umberbrush", &test_preset("My Brush", 0.2));
        write_preset(&dir2, "mine.umberbrush", &test_preset("My Brush", 0.8));

        let library = Library::load_from_dirs(&[dir1.clone(), dir2.clone()]);

        assert_eq!(library.entries.len(), 1);
        assert_eq!(library.entries[0].preset.name, "My Brush");
        assert_eq!(
            library.entries[0].preset.params.alpha, 0.2,
            "dir1 (earlier = higher priority) must win"
        );
        assert_eq!(library.entries[0].source_path, dir1.join("mine.umberbrush"));
        assert_eq!(library.overridden, 1, "the loser must be counted");
        assert!(
            library.errors.is_empty(),
            "an overridden file is NOT an error, got {:?}",
            library.errors
        );

        remove_tmpdir(&dir1);
        remove_tmpdir(&dir2);
    }

    #[test]
    fn in_dir_duplicate_first_filename_wins() {
        let dir = fresh_tmpdir("indir-dup");
        write_preset(&dir, "z_second.umberbrush", &test_preset("Dup", 0.9));
        write_preset(&dir, "a_first.umberbrush", &test_preset("Dup", 0.1));

        let library = Library::load_from_dirs(std::slice::from_ref(&dir));

        assert_eq!(library.entries.len(), 1);
        assert_eq!(
            library.entries[0].preset.params.alpha, 0.1,
            "first file in byte order of file name must win"
        );
        assert_eq!(
            library.entries[0].source_path,
            dir.join("a_first.umberbrush")
        );
        assert_eq!(library.overridden, 1);
        assert!(library.errors.is_empty());

        remove_tmpdir(&dir);
    }

    #[test]
    fn nonexistent_dir_skips_silently() {
        let missing = std::env::temp_dir().join(format!(
            "umber-brush-lib-test-{}-does-not-exist",
            std::process::id()
        ));
        assert!(
            !missing.exists(),
            "precondition: the probe path must not exist"
        );

        let library = Library::load_from_dirs(std::slice::from_ref(&missing));

        assert!(library.entries.is_empty(), "no entries from a missing dir");
        assert!(
            library.errors.is_empty(),
            "silent skip is CONTRACT: a missing dir is not an error"
        );
        assert_eq!(library.overridden, 0);
    }

    #[test]
    fn repeated_loads_are_equal() {
        let dir = fresh_tmpdir("determinism");
        write_preset(&dir, "b.umberbrush", &test_preset("Beta", 0.5));
        write_preset(&dir, "a.umberbrush", &test_preset("Alpha", 0.5));
        let broken = dir.join("zzz_broken.umberbrush");
        std::fs::write(&broken, "nope{").expect("writable");

        let first = Library::load_from_dirs(std::slice::from_ref(&dir));
        let second = Library::load_from_dirs(std::slice::from_ref(&dir));

        assert_eq!(
            first, second,
            "same dirs + same files must give an equal Library (entries, errors, overridden)"
        );
        assert_eq!(entry_names(&first), vec!["Alpha", "Beta"]);

        remove_tmpdir(&dir);
    }
}

//! The `.umber` project format (requirements.md §8, architecture.md).
//!
//! A project is a directory:
//!
//! ```text
//! myproject.umber/
//!   project.json            # version, texture sets, per-set layer order, settings
//!   layers/<set>/<id>.json  # one JSON file per layer — diffable, mergeable
//! ```
//!
//! [`ProjectModel`] is the in-memory document: full [`Layer`] bodies, grouped
//! per texture set. On disk, layer bodies live in their own files and
//! `project.json` only records *which* layers belong to which set and in
//! what order (the private `ProjectFile`/`LayerSetOrder` schema) — that
//! split is exactly what makes the format diff- and merge-friendly.
//!
//! Save/load round-trip determinism is a hard acceptance criterion
//! (requirements.md §8: "save→load→export identical bytes"); see the
//! round-trip tests at the bottom of this file.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::layers::{Layer, LayerStack};
use crate::TextureSet;

/// The only `.umber` project format version this build understands.
pub const CURRENT_PROJECT_VERSION: u32 = 1;

/// Everything that can go wrong saving or loading a `.umber` project.
#[derive(Debug, Error)]
pub enum ProjectError {
    /// Reading or writing a file under the project directory failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// `project.json` or a layer file was not valid JSON for its expected shape.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// `project.json`'s `version` isn't one this build understands.
    #[error("unsupported project version {0} (this build understands version 1)")]
    UnsupportedVersion(u32),

    /// A texture-set name is empty, reserved, or unsafe as a directory name.
    #[error("invalid texture set name {0:?}")]
    InvalidSetName(String),

    /// Two texture sets share a name (case-insensitively — see
    /// `validate_set_name`).
    #[error("duplicate texture set name {0:?}")]
    DuplicateSetName(String),

    /// A per-set layer order names a texture set that doesn't exist.
    #[error("layer order references unknown texture set {0:?}")]
    UnknownSetInLayerOrder(String),

    /// The same layer id appears twice in one texture set's order.
    #[error("duplicate layer id {id} in texture set {set:?}")]
    DuplicateLayerId {
        /// The texture set the duplicate was found in.
        set: String,
        /// The id that appeared more than once.
        id: u64,
    },

    /// A layer id listed in a set's order has no `<id>.json` file on disk.
    #[error("missing layer file for id {id} in texture set {set:?}")]
    MissingLayerFile {
        /// The texture set the missing file belongs to.
        set: String,
        /// The id that has no corresponding file.
        id: u64,
    },

    /// A layer file's own `id` field doesn't match the order entry that
    /// pointed at it (e.g. the file was renamed or corrupted).
    #[error("layer file for id {expected} in texture set {set:?} actually contains id {found}")]
    LayerIdMismatch {
        /// The texture set the mismatch was found in.
        set: String,
        /// The id the order entry expected.
        expected: u64,
        /// The id the file actually contained.
        found: u64,
    },

    /// A layer id or a set's `next_layer_id` counter is `u64::MAX`, one
    /// increment away from overflowing.
    #[error("layer id overflow in texture set {0:?}")]
    LayerIdOverflow(String),
}

/// Project-wide settings that aren't scoped to a single texture set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectSettings {
    /// Name of the texture set shown/edited when the project was last
    /// saved, if any. Purely a UI convenience — the document is valid
    /// without it.
    pub active_texture_set: Option<String>,
}

/// One texture set's layer stack, as stored in a [`ProjectModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct TextureSetLayers {
    /// Must match the `name` of one entry in [`ProjectModel::texture_sets`].
    pub texture_set: String,
    /// The set's layer stack.
    pub stack: LayerStack,
}

/// The full in-memory `.umber` document.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectModel {
    /// The `.umber` format version this document was built at or loaded from.
    pub version: u32,
    /// Every texture set in the project.
    pub texture_sets: Vec<TextureSet>,
    /// Each texture set's layer stack.
    pub layers: Vec<TextureSetLayers>,
    /// Project-wide settings.
    pub settings: ProjectSettings,
}

impl ProjectModel {
    /// Builds a project at [`CURRENT_PROJECT_VERSION`].
    pub fn new(
        texture_sets: Vec<TextureSet>,
        layers: Vec<TextureSetLayers>,
        settings: ProjectSettings,
    ) -> Self {
        Self {
            version: CURRENT_PROJECT_VERSION,
            texture_sets,
            layers,
            settings,
        }
    }
}

/// On-disk shape of `project.json`. Deliberately distinct from
/// [`ProjectModel`]: layer *bodies* never appear here, only the per-set
/// ordering needed to reassemble them from `layers/<set>/<id>.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ProjectFile {
    version: u32,
    texture_sets: Vec<TextureSet>,
    layer_sets: Vec<LayerSetOrder>,
    settings: ProjectSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct LayerSetOrder {
    texture_set: String,
    /// Stack order, bottom to top — matches `LayerStack::layers` order.
    layer_ids: Vec<u64>,
    /// Persisted rather than re-derived as `max(layer_ids) + 1` on load, so
    /// ids stay monotonic even if the highest-id layer was deleted before
    /// the save that produced this file.
    next_layer_id: u64,
}

/// Just enough of `project.json` to check the version before attempting to
/// parse the rest of the schema — so a future-version file fails with
/// [`ProjectError::UnsupportedVersion`] instead of an opaque JSON error.
#[derive(Debug, Deserialize)]
struct VersionProbe {
    version: u32,
}

/// Windows device names reserved regardless of extension (`CON`, `CON.txt`,
/// ... are all invalid). Compared case-insensitively.
const WINDOWS_RESERVED_STEMS: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Rejects anything unsafe as a texture-set *directory name* on Windows or
/// POSIX: path-traversal segments, reserved characters, trailing dot/space
/// (both silently stripped by Windows, which makes "Body" and "Body." the
/// same directory), and the Windows reserved device names.
///
/// Does not check for cross-name collisions (e.g. "Body" vs "body") — that
/// is a property of a *set* of names, not one name in isolation, and is
/// handled by [`validate_project_structure`].
fn validate_set_name(name: &str) -> Result<(), ProjectError> {
    const RESERVED_CHARS: &[char] = &['<', '>', ':', '"', '|', '?', '*', '/', '\\'];
    // Windows device names are reserved by their stem alone: "CON.txt" and
    // "nul.metal" are just as invalid as "CON" and "NUL".
    let stem = name.split_once('.').map_or(name, |(before, _)| before);
    let invalid = name.is_empty()
        || name == "."
        || name == ".."
        || name.contains(RESERVED_CHARS)
        || name.ends_with('.')
        || name.ends_with(' ')
        || WINDOWS_RESERVED_STEMS.contains(&stem.to_ascii_lowercase().as_str());
    if invalid {
        return Err(ProjectError::InvalidSetName(name.to_string()));
    }
    Ok(())
}

/// Validates the structural invariants shared by [`save_to_dir`] and
/// [`load_from_dir`], so the two can never disagree about what counts as a
/// well-formed project: a version save/load both recognize, set names that
/// are each individually valid and pairwise unique (case-insensitively —
/// Windows directories collide on case, and this format may be cloned
/// cross-platform via git), every layer order naming a set that actually
/// exists, no duplicate layer ids within a set, and no id or counter one
/// increment away from overflowing.
///
/// Called *before* any path is joined or any file is touched, so a crafted
/// `texture_set` name (e.g. `"../../escape"`) is rejected on load rather
/// than used to read or write outside the project directory.
fn validate_project_structure(
    version: u32,
    texture_set_names: &[String],
    layer_sets: &[(String, Vec<u64>, u64)],
) -> Result<(), ProjectError> {
    if version != CURRENT_PROJECT_VERSION {
        return Err(ProjectError::UnsupportedVersion(version));
    }

    let mut seen_set_names = BTreeSet::new();
    for name in texture_set_names {
        validate_set_name(name)?;
        // Full Unicode case folding, not just ASCII: NTFS compares names
        // case-insensitively across the whole alphabet, not only A-Z.
        if !seen_set_names.insert(name.to_lowercase()) {
            return Err(ProjectError::DuplicateSetName(name.clone()));
        }
    }
    let known_sets: BTreeSet<&str> = texture_set_names.iter().map(String::as_str).collect();

    let mut seen_layer_set_names = BTreeSet::new();
    for (set_name, layer_ids, next_layer_id) in layer_sets {
        validate_set_name(set_name)?;
        if !seen_layer_set_names.insert(set_name.to_lowercase()) {
            return Err(ProjectError::DuplicateSetName(set_name.clone()));
        }
        if !known_sets.contains(set_name.as_str()) {
            return Err(ProjectError::UnknownSetInLayerOrder(set_name.clone()));
        }
        if *next_layer_id == u64::MAX {
            return Err(ProjectError::LayerIdOverflow(set_name.clone()));
        }

        let mut seen_ids = BTreeSet::new();
        for &id in layer_ids {
            if id == u64::MAX {
                return Err(ProjectError::LayerIdOverflow(set_name.clone()));
            }
            if !seen_ids.insert(id) {
                return Err(ProjectError::DuplicateLayerId {
                    set: set_name.clone(),
                    id,
                });
            }
        }
    }
    Ok(())
}

fn to_pretty_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ProjectError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Writes `model` to `dir`, creating it if needed.
///
/// `dir`'s `layers/` directory is entirely removed and rewritten from
/// scratch, so a save into an existing project never leaves stale files
/// behind — neither for individual layers deleted from a set, nor for a
/// texture set deleted outright.
pub fn save_to_dir(model: &ProjectModel, dir: &Path) -> Result<(), ProjectError> {
    let texture_set_names: Vec<String> = model
        .texture_sets
        .iter()
        .map(|ts| ts.name.clone())
        .collect();
    let layer_sets_for_validation: Vec<(String, Vec<u64>, u64)> = model
        .layers
        .iter()
        .map(|entry| {
            (
                entry.texture_set.clone(),
                entry.stack.layers.iter().map(|l| l.id).collect(),
                entry.stack.next_layer_id(),
            )
        })
        .collect();
    validate_project_structure(
        model.version,
        &texture_set_names,
        &layer_sets_for_validation,
    )?;

    fs::create_dir_all(dir)?;
    let layers_root = dir.join("layers");
    if layers_root.exists() {
        fs::remove_dir_all(&layers_root)?;
    }

    let mut layer_sets = Vec::with_capacity(model.layers.len());
    for entry in &model.layers {
        let set_dir = layers_root.join(&entry.texture_set);
        fs::create_dir_all(&set_dir)?;

        let mut layer_ids = Vec::with_capacity(entry.stack.layers.len());
        for layer in &entry.stack.layers {
            let bytes = to_pretty_json_bytes(layer)?;
            fs::write(set_dir.join(format!("{}.json", layer.id)), bytes)?;
            layer_ids.push(layer.id);
        }

        layer_sets.push(LayerSetOrder {
            texture_set: entry.texture_set.clone(),
            layer_ids,
            next_layer_id: entry.stack.next_layer_id(),
        });
    }

    let file = ProjectFile {
        version: model.version,
        texture_sets: model.texture_sets.clone(),
        layer_sets,
        settings: model.settings.clone(),
    };
    fs::write(dir.join("project.json"), to_pretty_json_bytes(&file)?)?;
    Ok(())
}

/// Reads a project previously written by [`save_to_dir`].
pub fn load_from_dir(dir: &Path) -> Result<ProjectModel, ProjectError> {
    let bytes = fs::read(dir.join("project.json"))?;

    // Check the version against a minimal schema before attempting to parse
    // the rest, so a future-version project.json fails with
    // ProjectError::UnsupportedVersion rather than an opaque JSON error.
    let probe: VersionProbe = serde_json::from_slice(&bytes)?;
    if probe.version != CURRENT_PROJECT_VERSION {
        return Err(ProjectError::UnsupportedVersion(probe.version));
    }

    let file: ProjectFile = serde_json::from_slice(&bytes)?;

    let texture_set_names: Vec<String> =
        file.texture_sets.iter().map(|ts| ts.name.clone()).collect();
    let layer_sets_for_validation: Vec<(String, Vec<u64>, u64)> = file
        .layer_sets
        .iter()
        .map(|ls| {
            (
                ls.texture_set.clone(),
                ls.layer_ids.clone(),
                ls.next_layer_id,
            )
        })
        .collect();
    validate_project_structure(file.version, &texture_set_names, &layer_sets_for_validation)?;

    let mut layers = Vec::with_capacity(file.layer_sets.len());
    for layer_set in &file.layer_sets {
        let set_dir = dir.join("layers").join(&layer_set.texture_set);
        let mut stack_layers = Vec::with_capacity(layer_set.layer_ids.len());
        for &id in &layer_set.layer_ids {
            let layer_path = set_dir.join(format!("{id}.json"));
            let layer_bytes =
                fs::read(&layer_path).map_err(|_| ProjectError::MissingLayerFile {
                    set: layer_set.texture_set.clone(),
                    id,
                })?;
            let layer: Layer = serde_json::from_slice(&layer_bytes)?;
            if layer.id != id {
                return Err(ProjectError::LayerIdMismatch {
                    set: layer_set.texture_set.clone(),
                    expected: id,
                    found: layer.id,
                });
            }
            stack_layers.push(layer);
        }

        layers.push(TextureSetLayers {
            texture_set: layer_set.texture_set.clone(),
            stack: LayerStack::from_parts(stack_layers, layer_set.next_layer_id),
        });
    }

    Ok(ProjectModel {
        version: file.version,
        texture_sets: file.texture_sets,
        layers,
        settings: file.settings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{BlendMode, LayerKind, LayerMask};
    use crate::{Channel, ChannelKind};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "umber-core-test-{label}-{}-{n}",
            std::process::id()
        ))
    }

    /// Exercises every variant the task calls out: a folder with
    /// passthrough, a populated mask, fractional and edge-case opacities,
    /// multiple sets, several blend modes, and a non-default setting.
    fn fixture_model() -> ProjectModel {
        let set_a = TextureSet::new_default("Body");
        let set_b = TextureSet {
            name: "Helmet".to_string(),
            resolution: 4096,
            channels: vec![Channel {
                name: "baseColor".into(),
                kind: ChannelKind::Color,
            }],
        };

        let mut stack_a = LayerStack::new();
        let base = stack_a.add_layer("Base", LayerKind::Paint);
        stack_a.set_opacity(base, 0.1);
        stack_a.set_blend_mode(base, BlendMode::Multiply);

        let group = stack_a.add_layer("Details", LayerKind::Folder { passthrough: true });
        stack_a.set_opacity(group, 1.0 / 3.0);
        stack_a.set_blend_mode(group, BlendMode::Overlay);
        stack_a.layer_mut(group).unwrap().mask = Some(LayerMask {
            name: "Details Mask".into(),
            enabled: true,
        });

        let fill = stack_a.add_layer("Tint", LayerKind::Fill);
        stack_a.set_blend_mode(fill, BlendMode::SoftLight);
        stack_a.set_visible(fill, false);
        stack_a.remove_layer(fill).unwrap(); // bump next_id past a deleted layer

        let mut stack_b = LayerStack::new();
        stack_b.add_layer("Metal", LayerKind::Paint);

        ProjectModel::new(
            vec![set_a, set_b],
            vec![
                TextureSetLayers {
                    texture_set: "Body".into(),
                    stack: stack_a,
                },
                TextureSetLayers {
                    texture_set: "Helmet".into(),
                    stack: stack_b,
                },
            ],
            ProjectSettings {
                active_texture_set: Some("Body".into()),
            },
        )
    }

    #[test]
    fn project_file_json_bytes_are_deterministic_across_serializations() {
        let model = fixture_model();
        let file = ProjectFile {
            version: model.version,
            texture_sets: model.texture_sets.clone(),
            layer_sets: model
                .layers
                .iter()
                .map(|l| LayerSetOrder {
                    texture_set: l.texture_set.clone(),
                    layer_ids: l.stack.layers.iter().map(|x| x.id).collect(),
                    next_layer_id: l.stack.next_layer_id(),
                })
                .collect(),
            settings: model.settings.clone(),
        };

        let bytes_1 = to_pretty_json_bytes(&file).unwrap();
        let bytes_2 = to_pretty_json_bytes(&file).unwrap();
        assert_eq!(bytes_1, bytes_2);
    }

    #[test]
    fn save_load_save_round_trip_is_byte_identical() {
        let model = fixture_model();
        let dir_a = unique_temp_dir("a");
        let dir_b = unique_temp_dir("b");

        save_to_dir(&model, &dir_a).expect("first save");
        let loaded = load_from_dir(&dir_a).expect("load");
        assert_eq!(loaded, model, "loaded model must equal the original");

        save_to_dir(&loaded, &dir_b).expect("second save");
        assert_dirs_byte_identical(&dir_a, &dir_b);

        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn save_overwrite_prunes_deleted_layer_files_and_removed_sets() {
        let mut model = fixture_model();
        let dir = unique_temp_dir("prune");

        save_to_dir(&model, &dir).expect("first save");
        let body_dir = dir.join("layers").join("Body");
        let helmet_dir = dir.join("layers").join("Helmet");
        let before = fs::read_dir(&body_dir).unwrap().count();
        assert_eq!(before, 2); // base + group survive; the removed fill layer never wrote a file
        assert!(helmet_dir.exists());

        // Remove the "Details" group, and drop the "Helmet" set entirely.
        let group_id = model.layers[0].stack.layers[1].id;
        model.layers[0].stack.remove_layer(group_id);
        model.layers.remove(1);
        model.texture_sets.remove(1);
        save_to_dir(&model, &dir).expect("second save");

        let after = fs::read_dir(&body_dir).unwrap().count();
        assert_eq!(after, 1); // stale file for the removed group must be gone
        assert!(!helmet_dir.exists()); // the removed set's whole directory must be gone

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_rejects_case_insensitive_duplicate_set_names() {
        let dir = unique_temp_dir("case-dup");
        let model = ProjectModel::new(
            vec![TextureSet::new_default("Body")],
            vec![
                TextureSetLayers {
                    texture_set: "Body".into(),
                    stack: LayerStack::new(),
                },
                TextureSetLayers {
                    texture_set: "body".into(),
                    stack: LayerStack::new(),
                },
            ],
            ProjectSettings::default(),
        );

        assert!(matches!(
            save_to_dir(&model, &dir),
            Err(ProjectError::DuplicateSetName(_))
        ));
    }

    #[test]
    fn save_rejects_windows_reserved_device_name() {
        let dir = unique_temp_dir("reserved-name");
        let model = ProjectModel::new(
            vec![],
            vec![TextureSetLayers {
                texture_set: "COM1".into(),
                stack: LayerStack::new(),
            }],
            ProjectSettings::default(),
        );

        assert!(matches!(
            save_to_dir(&model, &dir),
            Err(ProjectError::InvalidSetName(_))
        ));
    }

    #[test]
    fn save_rejects_windows_reserved_device_name_with_extension() {
        let dir = unique_temp_dir("reserved-name-ext");
        let model = ProjectModel::new(
            vec![],
            vec![TextureSetLayers {
                texture_set: "nul.metal".into(),
                stack: LayerStack::new(),
            }],
            ProjectSettings::default(),
        );

        assert!(matches!(
            save_to_dir(&model, &dir),
            Err(ProjectError::InvalidSetName(_))
        ));
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let dir = unique_temp_dir("version");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("project.json"),
            br#"{"version":2,"texture_sets":[],"layer_sets":[],"settings":{"active_texture_set":null}}"#,
        )
        .unwrap();

        let result = load_from_dir(&dir);
        assert!(matches!(result, Err(ProjectError::UnsupportedVersion(2))));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_rejects_duplicate_set_names() {
        let dir = unique_temp_dir("dup-name");
        let model = ProjectModel::new(
            vec![TextureSet::new_default("Body")],
            vec![
                TextureSetLayers {
                    texture_set: "Body".into(),
                    stack: LayerStack::new(),
                },
                TextureSetLayers {
                    texture_set: "Body".into(),
                    stack: LayerStack::new(),
                },
            ],
            ProjectSettings::default(),
        );

        assert!(matches!(
            save_to_dir(&model, &dir),
            Err(ProjectError::DuplicateSetName(_))
        ));
    }

    #[test]
    fn load_rejects_path_traversal_set_name_before_touching_disk() {
        let dir = unique_temp_dir("traversal");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("project.json"),
            br#"{"version":1,"texture_sets":[{"name":"../escape","resolution":1,"channels":[]}],"layer_sets":[{"texture_set":"../escape","layer_ids":[],"next_layer_id":0}],"settings":{"active_texture_set":null}}"#,
        )
        .unwrap();

        assert!(matches!(
            load_from_dir(&dir),
            Err(ProjectError::InvalidSetName(_))
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_layer_id_at_overflow_boundary() {
        let dir = unique_temp_dir("overflow");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("project.json"),
            format!(
                r#"{{"version":1,"texture_sets":[{{"name":"Body","resolution":1,"channels":[]}}],"layer_sets":[{{"texture_set":"Body","layer_ids":[{}],"next_layer_id":0}}],"settings":{{"active_texture_set":null}}}}"#,
                u64::MAX
            ),
        )
        .unwrap();

        assert!(matches!(
            load_from_dir(&dir),
            Err(ProjectError::LayerIdOverflow(_))
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_rejects_invalid_set_name() {
        let dir = unique_temp_dir("invalid-name");
        let model = ProjectModel::new(
            vec![],
            vec![TextureSetLayers {
                texture_set: "../escape".into(),
                stack: LayerStack::new(),
            }],
            ProjectSettings::default(),
        );

        assert!(matches!(
            save_to_dir(&model, &dir),
            Err(ProjectError::InvalidSetName(_))
        ));
    }

    fn assert_dirs_byte_identical(a: &Path, b: &Path) {
        let files_a = collect_relative_files(a);
        let files_b = collect_relative_files(b);
        assert_eq!(
            files_a, files_b,
            "file listings differ between {a:?} and {b:?}"
        );

        for rel in files_a {
            let bytes_a = fs::read(a.join(&rel)).unwrap();
            let bytes_b = fs::read(b.join(&rel)).unwrap();
            assert_eq!(bytes_a, bytes_b, "byte mismatch in {rel:?}");
        }
    }

    fn collect_relative_files(root: &Path) -> Vec<PathBuf> {
        fn walk(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, root, out);
                } else {
                    out.push(path.strip_prefix(root).unwrap().to_path_buf());
                }
            }
        }
        let mut out = Vec::new();
        walk(root, root, &mut out);
        out.sort();
        out
    }
}

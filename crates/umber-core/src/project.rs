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
//! what order (the [`ProjectFile`]/[`LayerSetOrder`] schema) — that split is
//! exactly what makes the format diff- and merge-friendly.
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
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("unsupported project version {0} (this build understands version 1)")]
    UnsupportedVersion(u32),

    #[error("invalid texture set name {0:?}")]
    InvalidSetName(String),

    #[error("duplicate texture set name {0:?}")]
    DuplicateSetName(String),

    #[error("layer order references unknown texture set {0:?}")]
    UnknownSetInLayerOrder(String),

    #[error("duplicate layer id {id} in texture set {set:?}")]
    DuplicateLayerId { set: String, id: u64 },

    #[error("missing layer file for id {id} in texture set {set:?}")]
    MissingLayerFile { set: String, id: u64 },

    #[error("layer file for id {expected} in texture set {set:?} actually contains id {found}")]
    LayerIdMismatch {
        set: String,
        expected: u64,
        found: u64,
    },
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
    pub stack: LayerStack,
}

/// The full in-memory `.umber` document.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectModel {
    pub version: u32,
    pub texture_sets: Vec<TextureSet>,
    pub layers: Vec<TextureSetLayers>,
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

fn validate_set_name(name: &str) -> Result<(), ProjectError> {
    const RESERVED: &[char] = &['<', '>', ':', '"', '|', '?', '*', '/', '\\'];
    if name.is_empty() || name == "." || name == ".." || name.contains(RESERVED) {
        return Err(ProjectError::InvalidSetName(name.to_string()));
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
/// Each texture set's `layers/<set>/` directory is fully replaced (removed
/// then rewritten) so a save into an existing project never leaves stale
/// files behind for layers that were since deleted.
pub fn save_to_dir(model: &ProjectModel, dir: &Path) -> Result<(), ProjectError> {
    let mut seen_names = BTreeSet::new();
    for entry in &model.layers {
        validate_set_name(&entry.texture_set)?;
        if !seen_names.insert(entry.texture_set.as_str()) {
            return Err(ProjectError::DuplicateSetName(entry.texture_set.clone()));
        }
    }

    fs::create_dir_all(dir)?;
    let layers_root = dir.join("layers");

    let mut layer_sets = Vec::with_capacity(model.layers.len());
    for entry in &model.layers {
        let set_dir = layers_root.join(&entry.texture_set);
        if set_dir.exists() {
            fs::remove_dir_all(&set_dir)?;
        }
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

    let probe: VersionProbe = serde_json::from_slice(&bytes)?;
    if probe.version != CURRENT_PROJECT_VERSION {
        return Err(ProjectError::UnsupportedVersion(probe.version));
    }

    let file: ProjectFile = serde_json::from_slice(&bytes)?;
    let known_sets: BTreeSet<&str> = file
        .texture_sets
        .iter()
        .map(|ts| ts.name.as_str())
        .collect();

    let mut layers = Vec::with_capacity(file.layer_sets.len());
    for layer_set in &file.layer_sets {
        if !known_sets.contains(layer_set.texture_set.as_str()) {
            return Err(ProjectError::UnknownSetInLayerOrder(
                layer_set.texture_set.clone(),
            ));
        }

        let set_dir = dir.join("layers").join(&layer_set.texture_set);
        let mut seen_ids = BTreeSet::new();
        let mut stack_layers = Vec::with_capacity(layer_set.layer_ids.len());
        for &id in &layer_set.layer_ids {
            if !seen_ids.insert(id) {
                return Err(ProjectError::DuplicateLayerId {
                    set: layer_set.texture_set.clone(),
                    id,
                });
            }

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
    fn save_overwrite_prunes_deleted_layer_files() {
        let mut model = fixture_model();
        let dir = unique_temp_dir("prune");

        save_to_dir(&model, &dir).expect("first save");
        let body_dir = dir.join("layers").join("Body");
        let before = fs::read_dir(&body_dir).unwrap().count();
        assert_eq!(before, 2); // base + group survive; the removed fill layer never wrote a file

        // Remove the "Details" group too, then save into the same directory.
        let group_id = model.layers[0].stack.layers[1].id;
        model.layers[0].stack.remove_layer(group_id);
        save_to_dir(&model, &dir).expect("second save");

        let after = fs::read_dir(&body_dir).unwrap().count();
        assert_eq!(after, 1); // stale file for the removed group must be gone

        let _ = fs::remove_dir_all(&dir);
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
            vec![],
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

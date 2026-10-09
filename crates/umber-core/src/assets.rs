//! Content-addressed asset store (requirements.md §8: "Textures as
//! content-hash-named PNG/EXR tiles — assets referenced by hash, never
//! embedded blobs").
//!
//! The store lives under `<project>/assets/`: one file per unique
//! content hash, extension preserved. [`AssetRef`] is the serializable
//! handle stored in `project.json` / layer JSON — it names content, so
//! identical pixels dedupe globally, and the on-disk layout is
//! deterministic (the same project data always hashes to the same
//! paths — the round-trip criterion's foundation).
//!
//! Hash: blake3 (fast, no transitive deps, 256-bit). The hex string is
//! the filename; the extension is metadata needed to reopen the file
//! with the right decoder, stored in the ref alongside the hash.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Everything that can go wrong in the asset store.
#[derive(Debug, Error)]
pub enum AssetError {
    /// Reading or writing under the asset directory failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The referenced blob is not present (moved project, pruned
    /// store, or a ref from a different project).
    #[error("asset {hash} ({ext}) not found in store")]
    Missing {
        /// The content hash that was looked up.
        hash: String,
        /// The expected file extension.
        ext: String,
    },
    /// A hash string that doesn't fit the format this build writes.
    #[error("malformed asset hash {0:?} (expected 64 hex chars)")]
    MalformedHash(String),
}

/// A reference to a stored asset: content hash + extension. Serializes
/// into project/layer JSON; never carries pixel bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    /// blake3 of the content, lowercase hex (64 chars).
    pub hash: String,
    /// File extension without the dot (`"png"`, `"exr"`, …) — kept so
    /// the decoder picks correctly without sniffing magic bytes.
    pub ext: String,
    /// Content length in bytes, recorded for sanity checks on load.
    pub size: u64,
}

impl AssetRef {
    /// Validates the hash's shape (64 lowercase hex chars).
    pub fn validate(&self) -> Result<(), AssetError> {
        let ok = self.hash.len() == 64
            && self
                .hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if ok {
            Ok(())
        } else {
            Err(AssetError::MalformedHash(self.hash.clone()))
        }
    }
}

/// Content-addressed blob store rooted at `<project>/assets/`.
#[derive(Debug, Clone)]
pub struct AssetStore {
    root: PathBuf,
}

impl AssetStore {
    /// Opens (or lazily creates) the store under `root`. The directory
    /// is not touched until the first [`AssetStore::store`] call —
    /// loading a project with no assets never creates directories.
    pub fn new(project_root: &Path) -> Self {
        Self {
            root: project_root.join("assets"),
        }
    }

    /// The on-disk path for a ref. Private-sharded (first 2 hex chars
    /// as a subdirectory) so directories stay listable at scale.
    fn path_for(&self, hash: &str, ext: &str) -> PathBuf {
        self.root.join(&hash[..2]).join(format!("{hash}.{ext}"))
    }

    /// Stores `bytes` under its content hash, deduping: if the blob
    /// already exists the write is skipped and the same [`AssetRef`]
    /// is returned. Returns the ref for later [`AssetStore::load`]s.
    pub fn store(&self, bytes: &[u8], ext: &str) -> Result<AssetRef, AssetError> {
        let hash = blake3::hash(bytes).to_hex().to_string();
        let path = self.path_for(&hash, ext);
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            // Write-then-rename for atomicity: a crash mid-save leaves
            // either the old state or the new blob, never a torn file.
            let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
            fs::write(&tmp, bytes)?;
            fs::rename(&tmp, &path)?;
        }
        Ok(AssetRef {
            hash,
            ext: ext.to_string(),
            size: bytes.len() as u64,
        })
    }

    /// Loads the blob a ref points at. Errors with
    /// [`AssetError::Missing`] when the store doesn't hold it.
    pub fn load(&self, reference: &AssetRef) -> Result<Vec<u8>, AssetError> {
        reference.validate()?;
        let path = self.path_for(&reference.hash, &reference.ext);
        if !path.exists() {
            return Err(AssetError::Missing {
                hash: reference.hash.clone(),
                ext: reference.ext.clone(),
            });
        }
        Ok(fs::read(&path)?)
    }

    /// Whether the store holds the blob a ref points at (no I/O on the
    /// body — a stat, for cheap existence checks on project load).
    pub fn contains(&self, reference: &AssetRef) -> bool {
        reference.validate().is_ok() && self.path_for(&reference.hash, &reference.ext).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store(tag: &str) -> (AssetStore, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("umber-assets-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        (AssetStore::new(&dir), dir)
    }

    #[test]
    fn store_and_load_roundtrip() {
        let (store, dir) = tmp_store("roundtrip");
        let content = b"fake png bytes".to_vec();
        let reference = store.store(&content, "png").unwrap();
        assert!(reference.ext == "png");
        assert_eq!(reference.size, content.len() as u64);
        let loaded = store.load(&reference).unwrap();
        assert_eq!(loaded, content);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn identical_content_dedupes() {
        let (store, dir) = tmp_store("dedupe");
        let a = store.store(b"same bytes", "png").unwrap();
        let b = store.store(b"same bytes", "png").unwrap();
        assert_eq!(a, b, "same content must produce the same ref");
        // Exactly one file on disk.
        let files: Vec<_> = fs::read_dir(dir.join("assets")).unwrap().collect();
        assert_eq!(files.len(), 1);
        let shard = fs::read_dir(files[0].as_ref().unwrap().path()).unwrap();
        assert_eq!(shard.count(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn different_content_gets_different_refs() {
        let (store, dir) = tmp_store("distinct");
        let a = store.store(b"one", "png").unwrap();
        let b = store.store(b"two", "png").unwrap();
        assert_ne!(a.hash, b.hash);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hash_is_deterministic_across_stores() {
        // The round-trip criterion: same content in two different
        // project dirs → the same hash → the same relative path.
        let (store_a, dir_a) = tmp_store("deta");
        let (store_b, dir_b) = tmp_store("detb");
        let ra = store_a.store(b"determinism", "exr").unwrap();
        let rb = store_b.store(b"determinism", "exr").unwrap();
        assert_eq!(ra, rb);
        fs::remove_dir_all(&dir_a).unwrap();
        fs::remove_dir_all(&dir_b).unwrap();
    }

    #[test]
    fn missing_asset_errors_cleanly() {
        let (store, dir) = tmp_store("missing");
        let reference = store.store(b"present", "png").unwrap();
        let ghost = AssetRef {
            hash: "0".repeat(64),
            ext: "png".into(),
            size: 1,
        };
        assert!(matches!(
            store.load(&ghost),
            Err(AssetError::Missing { .. })
        ));
        assert!(store.load(&reference).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn malformed_hash_rejected() {
        let (store, dir) = tmp_store("malformed");
        let bad = AssetRef {
            hash: "not-hex!".into(),
            ext: "png".into(),
            size: 1,
        };
        assert!(matches!(
            store.load(&bad),
            Err(AssetError::MalformedHash(_))
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn refs_serialize_into_project_json() {
        // The ref must round-trip through serde as project/layer JSON.
        let reference = AssetRef {
            hash: "a".repeat(64),
            ext: "png".into(),
            size: 42,
        };
        let json = serde_json::to_string(&reference).unwrap();
        let back: AssetRef = serde_json::from_str(&json).unwrap();
        assert_eq!(back, reference);
    }

    #[test]
    fn sharded_path_layout() {
        let (store, dir) = tmp_store("shard");
        let reference = store.store(b"shard test", "png").unwrap();
        // First 2 hex chars form the shard directory.
        let expected = dir
            .join("assets")
            .join(&reference.hash[..2])
            .join(format!("{}.png", reference.hash));
        assert!(expected.exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}

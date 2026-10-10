//! UDIM tile-aware texture-set model (Wave-5 slice 2: the data model,
//! `docs/specs/udim-design.md` "The model").
//!
//! [`TileSet`] is the tile-addressed view of a texture set: each UDIM
//! tile ([`umber_mesh::FIRST_TILE`]..=1100) carries its own
//! [`TileChannels`] — the existing per-set shape (resolution plus the
//! [`Channel`] list, reused verbatim, no invented channel types) — so a
//! tile's address is the canonical `(channel, tile)` pair from the
//! design doc.
//!
//! [`crate::TextureSet`] stays the serialized single-tile-compatible
//! shape: its `tiles` map is empty for legacy/1001-only sets (an empty
//! map *means* "tile 1001 implied by `resolution`/`channels`"), and the
//! [`From`] conversions below normalize a 1001-only [`TileSet`] back to
//! that empty form — which is what makes both conversion directions
//! lossless (see the compat tests).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use umber_mesh::FIRST_TILE;

use crate::{Channel, TextureSet};

/// One UDIM tile's channels: the existing per-set payload (resolution
/// plus channel list) addressed per tile, so each tile keeps its own
/// resolution (the design doc's umber-core row: "each tile's own w/h").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileChannels {
    /// This tile's square resolution, mirroring
    /// [`TextureSet::resolution`](crate::TextureSet).
    pub resolution: u32,
    /// This tile's channels, reusing [`Channel`] verbatim.
    pub channels: Vec<Channel>,
}

impl TileChannels {
    /// Builds one tile's payload from its resolution and channels.
    pub fn new(resolution: u32, channels: Vec<Channel>) -> Self {
        Self {
            resolution,
            channels,
        }
    }

    /// Appends a channel (mirrors the `LayerStack::add_layer` push pattern).
    pub fn add_channel(&mut self, channel: Channel) {
        self.channels.push(channel);
    }
}

impl Default for TileChannels {
    /// Empty channel list at the default resolution (2048, matching
    /// [`TextureSet::new_default`](crate::TextureSet::new_default)).
    fn default() -> Self {
        Self {
            resolution: 2048,
            channels: Vec::new(),
        }
    }
}

/// A texture set addressed per UDIM tile: tile id -> that tile's
/// channels. Tile [`FIRST_TILE`] (1001) always exists on values built
/// through [`Self::new`]; deserialized values may lack it (hand-crafted
/// JSON), in which case [`Self::tile`] simply returns `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileSet {
    /// Texture-set name (mesh material or file stem, as in [`TextureSet`]).
    pub name: String,
    /// Per-tile channels, keyed by UDIM tile id (1001..=1100).
    /// `BTreeMap` keeps [`Self::tiles_present`] sorted for free.
    pub tiles: BTreeMap<u16, TileChannels>,
}

impl TileSet {
    /// Builds a single-tile set: tile [`FIRST_TILE`] present with
    /// default (empty) channels, nothing else.
    pub fn new(name: impl Into<String>) -> Self {
        let mut tiles = BTreeMap::new();
        tiles.insert(FIRST_TILE, TileChannels::default());
        Self {
            name: name.into(),
            tiles,
        }
    }

    /// Borrows one tile's channels, or `None` when the tile is absent.
    pub fn tile(&self, id: u16) -> Option<&TileChannels> {
        self.tiles.get(&id)
    }

    /// Mutably borrows one tile's channels, or `None` when absent.
    pub fn tile_mut(&mut self, id: u16) -> Option<&mut TileChannels> {
        self.tiles.get_mut(&id)
    }

    /// Every tile present, ascending (`BTreeMap` iteration order).
    pub fn tiles_present(&self) -> Vec<u16> {
        self.tiles.keys().copied().collect()
    }

    /// Inserts or replaces one tile's channels.
    pub fn set_tile(&mut self, id: u16, tile: TileChannels) {
        self.tiles.insert(id, tile);
    }

    /// Appends a channel to one tile, creating the tile with default
    /// channels when absent (mirrors the `MapSet::set` upsert pattern).
    pub fn add_channel(&mut self, tile_id: u16, channel: Channel) {
        self.tiles.entry(tile_id).or_default().add_channel(channel);
    }
}

impl From<&TextureSet> for TileSet {
    /// A legacy (empty-`tiles`) set converts to the implied 1001-only
    /// set; a tiled set clones its tile map verbatim.
    fn from(set: &TextureSet) -> Self {
        let tiles = if set.tiles.is_empty() {
            let mut tiles = BTreeMap::new();
            tiles.insert(
                FIRST_TILE,
                TileChannels::new(set.resolution, set.channels.clone()),
            );
            tiles
        } else {
            set.tiles.clone()
        };
        Self {
            name: set.name.clone(),
            tiles,
        }
    }
}

impl From<&TileSet> for TextureSet {
    /// The 1001 tile becomes the legacy `resolution`/`channels`; the
    /// full tile map is carried. A 1001-*only* map normalizes back to
    /// the empty (implied) form, so legacy -> tiled -> legacy and
    /// 1001-only -> legacy -> tiled both round-trip losslessly. A set
    /// with no 1001 tile falls back to default legacy fields (its tile
    /// map is still carried verbatim).
    fn from(set: &TileSet) -> Self {
        let first = set.tiles.get(&FIRST_TILE);
        let (resolution, channels) = first.map_or((2048, Vec::new()), |tile| {
            (tile.resolution, tile.channels.clone())
        });
        let single_first = set.tiles.len() == 1 && first.is_some();
        Self {
            name: set.name.clone(),
            resolution,
            channels,
            tiles: if single_first {
                BTreeMap::new()
            } else {
                set.tiles.clone()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelKind;

    fn two_tile_set() -> TileSet {
        let mut set = TileSet::new("Body");
        set.set_tile(
            1002,
            TileChannels::new(
                1024,
                vec![Channel {
                    name: "baseColor".into(),
                    kind: ChannelKind::Color,
                }],
            ),
        );
        set
    }

    #[test]
    fn two_tile_set_addresses_tiles_sorted() {
        let set = two_tile_set();
        // BTreeMap iteration is ascending: 1001 before 1002.
        assert_eq!(set.tiles_present(), vec![1001, 1002]);
        let tile = set.tile(1002).expect("1002 was just inserted");
        assert_eq!(tile.resolution, 1024);
        assert_eq!(tile.channels.len(), 1);
        assert_eq!(tile.channels[0].name, "baseColor");
        // 999 is below the UDIM grid (1001..=1100): never present.
        assert_eq!(set.tile(999), None);
    }

    #[test]
    fn new_guarantees_first_tile_and_nothing_else() {
        let set = TileSet::new("Body");
        assert_eq!(set.tiles_present(), vec![FIRST_TILE]);
        assert_eq!(
            set.tile(FIRST_TILE),
            Some(&TileChannels::default()),
            "1001 present but otherwise empty"
        );
    }

    #[test]
    fn legacy_converts_to_1001_only_and_back_losslessly() {
        let legacy = TextureSet::new_default("Body");
        let tiled = TileSet::from(&legacy);
        assert_eq!(tiled.tiles_present(), vec![FIRST_TILE]);
        assert_eq!(tiled.tile(FIRST_TILE).unwrap().resolution, 2048);
        assert_eq!(tiled.tile(FIRST_TILE).unwrap().channels, legacy.channels);
        assert_eq!(
            TextureSet::from(&tiled),
            legacy,
            "legacy -> tiled -> legacy must be identical"
        );
    }

    #[test]
    fn tile_set_converts_to_legacy_and_back_losslessly() {
        let tiled = two_tile_set();
        let legacy = TextureSet::from(&tiled);
        assert_eq!(legacy.name, "Body");
        assert_eq!(legacy.resolution, 2048, "legacy fields come from 1001");
        assert_eq!(legacy.tiles, tiled.tiles, "full tile map is carried");
        assert_eq!(
            TileSet::from(&legacy),
            tiled,
            "tiled -> legacy -> tiled must be identical"
        );
    }

    #[test]
    fn tile_of_triangle_output_feeds_the_tile_api() {
        // No from_mesh helper: no caller needs one yet (the bakers and
        // paint router of later slices will consume tile_of_triangle
        // directly), so the model stays dependency-shaped, not
        // mesh-shaped. This test pins the integration point instead: a
        // two-tile MeshData's tile list is accepted by the TileSet API.
        let mesh = umber_mesh::MeshData {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0; 3]; 4],
            // Triangle 0's first vertex UV (0.5, 0.5) -> 1001;
            // triangle 1's first vertex UV (1.5, 0.5) -> 1002.
            uvs: vec![[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [1.5, 0.5]],
            indices: vec![0, 1, 2, 3, 1, 2],
            material_names: vec!["m".into()],
        };
        let per_triangle = umber_mesh::tile_of_triangle(&mesh);
        assert_eq!(per_triangle, vec![1001, 1002]);

        let mut set = TileSet::new("Body");
        let base = TileChannels::default();
        for tile_id in per_triangle {
            // set_tile is idempotent per tile: the two-triangle list
            // collapses to the two distinct tiles.
            set.set_tile(tile_id, base.clone());
        }
        assert_eq!(set.tiles_present(), vec![1001, 1002]);
    }
}

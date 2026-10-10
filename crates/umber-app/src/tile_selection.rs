//! The UDIM tile selector shared by the Bakes panel and the Export
//! dialog (wave-5 UDIM slice 6 — the "Bakes/Export panels tile lists"
//! row of `docs/specs/udim-design.md`).
//!
//! V1 is a tile SELECTOR (which tiles an action covers); the per-tile ×
//! per-map checklist matrix is wave-6 work.
//!
//! Selection semantics ([`TileSelection::sync`]): every tile is selected
//! the first time it appears in the present list (the default is "all
//! present"), tiles that vanish are dropped, and the user's deselections
//! of still-present tiles survive a list change — painting into a new
//! tile never wipes what was unchecked.
//!
//! Single-tile compatibility: with one present tile (or none) the list
//! an action receives is the present list verbatim, whatever the
//! selection holds — the lone tile can't be unchecked, so single-tile
//! meshes behave exactly as before the selector existed.

use std::collections::BTreeSet;

use umber_mesh::FIRST_TILE;

/// Which present UDIM tiles a panel action covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileSelection {
    /// The present tiles as of the last [`sync`](Self::sync), ascending.
    known: Vec<u16>,
    /// The checked tiles (always a subset of `known`).
    selected: BTreeSet<u16>,
}

impl TileSelection {
    /// The pre-mesh state: the single-tile `[1001]` list, selected —
    /// what a single-tile mesh syncs to, so a fresh panel already reads
    /// as it will once a single-tile mesh loads.
    pub fn new() -> Self {
        Self {
            known: vec![FIRST_TILE],
            selected: BTreeSet::from([FIRST_TILE]),
        }
    }

    /// Reconciles the selection with the current `present` tiles
    /// (ascending, unique): newly present tiles are selected, vanished
    /// ones dropped, existing choices kept.
    pub fn sync(&mut self, present: &[u16]) {
        if self.known == present {
            return;
        }
        for tile in present {
            if !self.known.contains(tile) {
                self.selected.insert(*tile);
            }
        }
        self.selected.retain(|tile| present.contains(tile));
        self.known = present.to_vec();
    }

    /// How many tiles are present (as of the last sync).
    pub fn known_len(&self) -> usize {
        self.known.len()
    }

    /// The present tiles as of the last sync, ascending (test hook).
    #[cfg(test)]
    pub(crate) fn known(&self) -> &[u16] {
        &self.known
    }

    /// The checked tiles (test hook).
    #[cfg(test)]
    pub(crate) fn selected(&self) -> &BTreeSet<u16> {
        &self.selected
    }

    /// Checks or unchecks `tile`; a tile not in the present list is
    /// ignored (the selection never names a tile without geometry).
    pub fn set_selected(&mut self, tile: u16, on: bool) {
        if !self.known.contains(&tile) {
            return;
        }
        if on {
            self.selected.insert(tile);
        } else {
            self.selected.remove(&tile);
        }
    }

    /// The tile list an action receives: see [`selected_tiles`].
    pub fn tiles(&self) -> Vec<u16> {
        selected_tiles(&self.known, &self.selected)
    }

    /// [`tiles`](Self::tiles) against `present` without mutating the
    /// panel state: a synced copy's list. The drivers call this so a
    /// tile that appeared since the last frame's sync is covered by the
    /// same default (selected) the UI would have shown.
    pub fn resolve(&self, present: &[u16]) -> Vec<u16> {
        let mut synced = self.clone();
        synced.sync(present);
        synced.tiles()
    }

    /// Whether the user has unchecked every tile of a multi-tile list —
    /// the action buttons gate on this (a single-tile list can't be
    /// emptied, so it never gates).
    pub fn nothing_selected(&self) -> bool {
        self.known.len() > 1 && self.selected.is_empty()
    }

    /// Draws the selector row: one checkbox per present tile, or — for a
    /// single-tile list — a plain `Tile 1001` label (nothing to choose).
    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Tiles:");
            if self.known.len() <= 1 {
                match self.known.first() {
                    Some(tile) => ui.label(format!("Tile {tile}")),
                    None => ui.label("none"),
                };
                return;
            }
            for tile in self.known.clone() {
                let mut on = self.selected.contains(&tile);
                if ui.checkbox(&mut on, format!("Tile {tile}")).changed() {
                    self.set_selected(tile, on);
                }
            }
        });
    }
}

impl Default for TileSelection {
    fn default() -> Self {
        Self::new()
    }
}

/// The selection -> tile list rule (pure): with at most one `present`
/// tile, `present` verbatim (single-tile behavior unchanged — the lone
/// tile can't be deselected); otherwise the present tiles that are
/// `selected`, ascending.
pub fn selected_tiles(present: &[u16], selected: &BTreeSet<u16>) -> Vec<u16> {
    if present.len() <= 1 {
        return present.to_vec();
    }
    present
        .iter()
        .copied()
        .filter(|tile| selected.contains(tile))
        .collect()
}

/// Caches [`umber_mesh::present_tiles`] per loaded mesh so the panels
/// don't walk every triangle each frame. Keyed on the mesh's index/UV
/// buffer identity (address + length): a newly loaded mesh is a new
/// allocation, so the key changes and the tiles recompute.
#[derive(Debug, Default)]
pub struct MeshTilesCache {
    key: Option<(usize, usize, usize, usize)>,
    tiles: Vec<u16>,
}

impl MeshTilesCache {
    /// The present tiles of `mesh` (recomputed only when the mesh changed).
    pub fn get(&mut self, mesh: &umber_mesh::MeshData) -> &[u16] {
        let key = (
            mesh.indices.as_ptr() as usize,
            mesh.indices.len(),
            mesh.uvs.as_ptr() as usize,
            mesh.uvs.len(),
        );
        if self.key != Some(key) {
            self.tiles = umber_mesh::present_tiles(mesh);
            self.key = Some(key);
        }
        &self.tiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_selection_is_the_single_tile_default() {
        let s = TileSelection::new();
        assert_eq!(s.known(), &[1001]);
        assert_eq!(s.selected(), &BTreeSet::from([1001]));
        assert_eq!(s.tiles(), vec![1001]);
        assert!(!s.nothing_selected());
    }

    #[test]
    fn sync_selects_all_present_by_default() {
        let mut s = TileSelection::new();
        s.sync(&[1001, 1002, 1011]);
        let present: BTreeSet<u16> = [1001, 1002, 1011].into();
        assert_eq!(s.selected(), &present);
        assert_eq!(s.tiles(), vec![1001, 1002, 1011]);
    }

    #[test]
    fn deselecting_shrinks_the_list() {
        let mut s = TileSelection::new();
        s.sync(&[1001, 1002]);
        s.set_selected(1002, false);
        assert_eq!(s.tiles(), vec![1001]);
        s.set_selected(1001, false);
        assert_eq!(s.tiles(), Vec::<u16>::new());
        assert!(s.nothing_selected());
        s.set_selected(1002, true);
        assert_eq!(s.tiles(), vec![1002]);
    }

    #[test]
    fn sync_keeps_user_choices_and_selects_new_tiles() {
        let mut s = TileSelection::new();
        s.sync(&[1001, 1002]);
        s.set_selected(1002, false);
        // A new tile appears (e.g. painted into): it defaults on, the
        // user's uncheck of 1002 survives.
        s.sync(&[1001, 1002, 1003]);
        assert_eq!(s.tiles(), vec![1001, 1003]);
        // 1001 vanishes: dropped from the selection too.
        s.sync(&[1002, 1003]);
        assert_eq!(s.selected(), &BTreeSet::from([1003]));
        assert_eq!(s.tiles(), vec![1003]);
    }

    #[test]
    fn unknown_tiles_cannot_be_selected() {
        let mut s = TileSelection::new();
        s.sync(&[1001, 1002]);
        s.set_selected(1099, true);
        assert_eq!(s.selected(), &BTreeSet::from([1001, 1002]));
    }

    #[test]
    fn resolve_covers_tiles_the_ui_has_not_synced_yet() {
        let mut s = TileSelection::new();
        s.sync(&[1001, 1002]);
        s.set_selected(1001, false);
        assert_eq!(s.resolve(&[1001, 1002, 1005]), vec![1002, 1005]);
        // `resolve` never mutates the panel's state.
        assert_eq!(s.known(), &[1001, 1002]);
    }

    #[test]
    fn single_tile_list_ignores_the_selection() {
        // The lone tile can't be unchecked: the list is [1001] even
        // with an empty selection — single-tile behavior unchanged.
        assert_eq!(selected_tiles(&[1001], &BTreeSet::new()), vec![1001]);
        assert_eq!(selected_tiles(&[], &BTreeSet::new()), Vec::<u16>::new());
        let mut s = TileSelection::new();
        s.set_selected(1001, false);
        assert_eq!(s.tiles(), vec![1001]);
        assert!(!s.nothing_selected());
    }

    #[test]
    fn mesh_tiles_cache_tracks_the_loaded_mesh() {
        let single = umber_mesh::MeshData {
            positions: vec![[0.0; 3]; 3],
            normals: vec![[0.0; 3]; 3],
            uvs: vec![[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]],
            indices: vec![0, 1, 2],
            material_names: vec!["m".into()],
        };
        let two = umber_mesh::MeshData {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0; 3]; 4],
            uvs: vec![[0.5, 0.5], [0.7, 0.5], [0.5, 0.7], [1.5, 0.5]],
            indices: vec![0, 1, 2, 3, 1, 2],
            material_names: vec!["m".into()],
        };
        let mut cache = MeshTilesCache::default();
        assert_eq!(cache.get(&single), &[1001]);
        assert_eq!(cache.get(&two), &[1001, 1002]);
        assert_eq!(cache.get(&single), &[1001]);
    }
}

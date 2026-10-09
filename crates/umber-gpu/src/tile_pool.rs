//! The tile pool: lazy, per-tile GPU allocation over a virtual texture.
//!
//! Wave-2 scope (docs/specs/architecture.md "Virtual texturing"): this is
//! the *foundation* slice — a `HashMap<TileId, PaintTarget>` keyed cache,
//! each entry a full-size [`PaintTarget`] ([`crate::paint`]'s module docs
//! call this out as the deferred "later slice"). The physical-page atlas +
//! page-table indirection + background upload thread that the architecture
//! doc describes for full virtual texturing is **not** implemented here;
//! this module only gets the addressing (UV → tile) and lazy
//! create/evict lifecycle right, which the atlas allocator will build on
//! in Wave 3.
//!
//! # Eviction semantics (read before calling [`TilePool::evict`])
//!
//! [`TilePool::evict`] drops the tile's GPU texture outright. **Its
//! painted contents are lost** — there is no readback-before-evict, no
//! dirty tracking, and no swap-to-disk. If the tile is touched again via
//! [`TilePool::get_or_create`], [`PaintTarget::new`]'s zero-initialization
//! means it comes back fully transparent, as if it had never been
//! painted. This is a known Wave-2 limitation, not an oversight: Wave 3's
//! bake integration is where read-back-before-evict (or a disk-backed
//! swap, mirroring the undo journal's "~100 steps then disk swap" budget
//! in docs/specs/architecture.md) lands. Until then, evicting a painted
//! tile is a real data-loss operation and callers must treat it as such.

use std::collections::HashMap;

use crate::paint::PaintTarget;

/// Fixed edge length of every pool tile, in texels.
///
/// Every tile is exactly `TILE_SIZE`×`TILE_SIZE`, including tiles at the
/// edge of a virtual texture whose dimensions aren't an exact multiple of
/// `TILE_SIZE` — the tile simply extends past the virtual texture's own
/// extent in that case (the usual fixed-page-size tradeoff in virtual
/// texturing, traded for simple addressing).
pub const TILE_SIZE: u32 = 512;

/// Identifies one tile in a virtual texture's tile grid.
///
/// `column`/`row` index directly into the grid implied by the virtual
/// texture's width/height and [`TILE_SIZE`] — there is no wraparound, so
/// an id with `column >= ceil(virt_w / TILE_SIZE)` (or the `row`
/// equivalent) addresses a tile outside the virtual texture. IDs produced
/// by [`tile_for`] are always in range by construction; IDs built
/// directly via [`TileId::new`] are the caller's responsibility to keep in
/// range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    /// Tile column (x-axis grid index), zero-based.
    pub column: u32,
    /// Tile row (y-axis grid index), zero-based.
    pub row: u32,
}

impl TileId {
    /// Builds a tile id from its column/row grid indices.
    pub fn new(column: u32, row: u32) -> Self {
        Self { column, row }
    }
}

/// Maps a normalized UV coordinate to the tile that contains it.
///
/// `uv` must land in the half-open `[0.0, 1.0)` range on *both* axes;
/// anything else — negative, `>= 1.0`, or `NaN` — returns `None` (`NaN`
/// is caught because `Range::contains` uses `PartialOrd`, under which
/// every comparison with `NaN` is false). A zero virtual width or height
/// also returns `None`, since no tile grid exists to map into.
///
/// The mapping walks through *texel* space — `floor(u * virt_w) /
/// TILE_SIZE` — rather than `floor(u * tiles_per_axis)`. The two
/// formulas agree when `virt_w`/`virt_h` are exact multiples of
/// `TILE_SIZE`, but only the texel-space version is correct otherwise: on
/// a 1000-wide virtual texture (`tiles_per_axis = ceil(1000/512) = 2`),
/// `uv.x = 0.5` lands at texel 500, which is tile 0 (texels `0..512`) —
/// but `floor(0.5 * 2) = 1` would wrongly say tile 1.
pub fn tile_for(uv: [f32; 2], virt_w: u32, virt_h: u32) -> Option<TileId> {
    if virt_w == 0 || virt_h == 0 {
        return None;
    }
    let [u, v] = uv;
    if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
        return None;
    }
    let texel_x = (u * virt_w as f32) as u32;
    let texel_y = (v * virt_h as f32) as u32;
    Some(TileId::new(texel_x / TILE_SIZE, texel_y / TILE_SIZE))
}

/// Maps a UV coordinate to its texel-center position inside its tile,
/// alongside the tile id itself.
///
/// The returned position is *quantized*: it floors the continuous `uv *
/// (virt_w, virt_h)` position down to an integer texel index (relative to
/// the tile's own origin, i.e. `texel_index % TILE_SIZE`) and adds `0.5`,
/// landing on that texel's center — the convention
/// `shaders::PAINT_COMPUTE_SHADER` samples dab footprints against.
/// Callers that need the unsnapped continuous position should compute
/// `uv * (virt_w, virt_h)` themselves instead of calling this.
///
/// Returns `None` under the same conditions as [`tile_for`].
pub fn texel_within_tile(uv: [f32; 2], virt_w: u32, virt_h: u32) -> Option<([f32; 2], TileId)> {
    let tile = tile_for(uv, virt_w, virt_h)?;
    let [u, v] = uv;
    let texel_x = (u * virt_w as f32) as u32;
    let texel_y = (v * virt_h as f32) as u32;
    let local_x = texel_x % TILE_SIZE;
    let local_y = texel_y % TILE_SIZE;
    Some(([local_x as f32 + 0.5, local_y as f32 + 0.5], tile))
}

/// A lazily-populated grid of [`PaintTarget`] tiles covering one virtual
/// texture.
///
/// See the module docs for the eviction contract. Construction allocates
/// nothing: tiles come into existence one at a time, on first touch, via
/// [`TilePool::get_or_create`].
pub struct TilePool {
    device: wgpu::Device,
    virt_width: u32,
    virt_height: u32,
    tiles: HashMap<TileId, PaintTarget>,
}

impl TilePool {
    /// Creates an empty pool over a `width`×`height` virtual texture.
    ///
    /// No GPU memory is allocated by this call — every tile is created
    /// lazily by [`TilePool::get_or_create`] on its first touch.
    pub fn new(device: wgpu::Device, width: u32, height: u32) -> Self {
        Self {
            device,
            virt_width: width,
            virt_height: height,
            tiles: HashMap::new(),
        }
    }

    /// The virtual texture's full extent, in texels.
    pub fn virtual_dimensions(&self) -> (u32, u32) {
        (self.virt_width, self.virt_height)
    }

    /// Returns the tile at `id`, creating a fresh, zero-initialized one
    /// first if this is its first touch.
    ///
    /// Does not validate `id` against [`TilePool::virtual_dimensions`] —
    /// ids produced by [`tile_for`] are always in range; a caller that
    /// builds a [`TileId`] directly is responsible for keeping it in
    /// range. An out-of-range id simply allocates a tile with no
    /// corresponding region of the virtual texture, which is harmless to
    /// the pool itself but wastes GPU memory.
    pub fn get_or_create(&mut self, id: TileId) -> &PaintTarget {
        self.tiles
            .entry(id)
            .or_insert_with(|| PaintTarget::new(&self.device, TILE_SIZE, TILE_SIZE))
    }

    /// Returns the tile at `id` if it is currently resident, without
    /// creating it.
    pub fn get(&self, id: TileId) -> Option<&PaintTarget> {
        self.tiles.get(&id)
    }

    /// Drops the tile at `id` if it is resident, freeing its GPU memory.
    ///
    /// Returns whether a tile was actually present to drop. **The tile's
    /// painted contents are lost** — see the module docs' eviction
    /// section. Dropping the [`PaintTarget`] is safe to do at any time
    /// (wgpu keeps a resource alive on the GPU timeline until in-flight
    /// work referencing it finishes), but any bind group or view a
    /// consumer cached for this tile now points at a destroyed texture;
    /// consumers must re-fetch via [`TilePool::get`] /
    /// [`TilePool::get_or_create`] after an evict instead of reusing a
    /// held reference.
    pub fn evict(&mut self, id: TileId) -> bool {
        self.tiles.remove(&id).is_some()
    }

    /// Count of tiles currently resident in GPU memory.
    pub fn tiles_in_memory(&self) -> usize {
        self.tiles.len()
    }

    /// Approximate GPU memory held by resident tiles, in bytes.
    ///
    /// `TILE_SIZE² * 4` bytes per resident tile (`Rgba8Unorm`: one byte
    /// per channel, four channels), ignoring the small fixed overhead of
    /// each tile's dims uniform buffer.
    pub fn memory_bytes(&self) -> u64 {
        self.tiles.len() as u64 * u64::from(TILE_SIZE) * u64::from(TILE_SIZE) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_for_maps_interior_point_to_its_tile() {
        // The global UV whose column-3/row-0 tile and (384.5, 128.5) texel
        // center are the worked example this module's spec was reviewed
        // against: (3 + 0.75) / 4 = 0.9375, (0 + 0.25) / 4 = 0.0625.
        let tile = tile_for([0.9375, 0.0625], 2048, 2048);
        assert_eq!(tile, Some(TileId::new(3, 0)));
    }

    #[test]
    fn tile_for_exact_tile_boundary_is_lower_inclusive() {
        // 0.75 * 2048 = 1536 = 3 * TILE_SIZE exactly: the boundary belongs
        // to tile 3, not tile 2. 0.25 * 2048 = 512 = 1 * TILE_SIZE exactly:
        // belongs to tile 1, not tile 0.
        let tile = tile_for([0.75, 0.25], 2048, 2048);
        assert_eq!(tile, Some(TileId::new(3, 1)));
    }

    #[test]
    fn tile_for_rejects_out_of_range_uv() {
        assert_eq!(tile_for([-0.01, 0.5], 2048, 2048), None);
        assert_eq!(tile_for([0.5, 1.0], 2048, 2048), None);
        assert_eq!(tile_for([1.0, 1.0], 2048, 2048), None);
        assert_eq!(tile_for([f32::NAN, 0.5], 2048, 2048), None);
        assert_eq!(tile_for([0.5, f32::NAN], 2048, 2048), None);
    }

    #[test]
    fn tile_for_rejects_zero_virtual_dimensions() {
        assert_eq!(tile_for([0.5, 0.5], 0, 2048), None);
        assert_eq!(tile_for([0.5, 0.5], 2048, 0), None);
    }

    #[test]
    fn tile_for_uses_texel_space_not_tiles_per_axis_scaling() {
        // 1000 is not a multiple of TILE_SIZE: tiles_per_axis = ceil(1000
        // / 512) = 2. uv.x = 0.5 lands at texel 500, which is inside tile
        // 0's texel range [0, 512) — floor(0.5 * 2) = 1 would wrongly
        // place it in tile 1, so this guards the texel-space formula.
        let tile = tile_for([0.5, 0.0], 1000, 1000);
        assert_eq!(tile, Some(TileId::new(0, 0)));
    }

    #[test]
    fn texel_within_tile_matches_the_worked_example() {
        let (texel, tile) = texel_within_tile([0.9375, 0.0625], 2048, 2048).unwrap();
        assert_eq!(tile, TileId::new(3, 0));
        assert_eq!(texel, [384.5, 128.5]);
    }

    #[test]
    fn texel_within_tile_at_exact_boundary_centers_on_the_first_texel() {
        let (texel, tile) = texel_within_tile([0.75, 0.25], 2048, 2048).unwrap();
        assert_eq!(tile, TileId::new(3, 1));
        assert_eq!(texel, [0.5, 0.5]);
    }

    #[test]
    fn texel_within_tile_rejects_out_of_range_uv() {
        assert_eq!(texel_within_tile([1.0, 0.5], 2048, 2048), None);
    }

    #[test]
    fn tile_id_equality_and_hash_are_structural() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TileId::new(1, 2));
        assert!(set.contains(&TileId::new(1, 2)));
        assert!(!set.contains(&TileId::new(2, 1)));
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;
        use crate::paint::{Dab, PaintCompositor};

        /// Requests a device with `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
        /// enabled. Mirrors `paint_thread`'s gpu test helper: this suite
        /// runs with a required real adapter, not an optional one.
        fn request_device() -> (wgpu::Device, wgpu::Queue) {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .expect("a wgpu adapter must be available for this test");
            assert!(
                adapter
                    .features()
                    .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
                "adapter {:?} lacks TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES",
                adapter.get_info().name
            );
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }))
            .expect("device request must succeed")
        }

        #[test]
        fn get_or_create_lazily_allocates_touched_tiles_only() {
            let (device, _queue) = request_device();
            let mut pool = TilePool::new(device, 2048, 2048);
            assert_eq!(
                pool.tiles_in_memory(),
                0,
                "nothing allocated at construction"
            );

            let a = pool.get_or_create(TileId::new(0, 0));
            assert_eq!(a.dimensions(), (TILE_SIZE, TILE_SIZE));

            let b = pool.get_or_create(TileId::new(3, 2));
            assert_eq!(b.dimensions(), (TILE_SIZE, TILE_SIZE));

            assert_eq!(pool.tiles_in_memory(), 2);
            assert_eq!(
                pool.memory_bytes(),
                2 * u64::from(TILE_SIZE) * u64::from(TILE_SIZE) * 4
            );
        }

        #[test]
        fn get_or_create_returns_the_same_tile_on_repeat_touch() {
            let (device, queue) = request_device();
            let mut pool = TilePool::new(device.clone(), 2048, 2048);
            let compositor =
                PaintCompositor::new(device.clone()).expect("device has the required feature");

            let id = TileId::new(1, 1);
            let dab = Dab::new([256.0, 256.0], 32.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1.0);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_tile_pool_test_encoder"),
            });
            compositor
                .splat_dabs(&mut encoder, pool.get_or_create(id), &[dab])
                .expect("single dab batch should splat");
            queue.submit(Some(encoder.finish()));

            assert_eq!(pool.tiles_in_memory(), 1);
            // Touching the same id again must not allocate a second tile.
            let _ = pool.get_or_create(id);
            assert_eq!(pool.tiles_in_memory(), 1);

            let bytes = pool
                .get(id)
                .expect("tile is resident")
                .read_back_rgba8(&device, &queue)
                .expect("readback should succeed");
            let center_idx = (256usize * TILE_SIZE as usize + 256) * 4;
            assert!(
                bytes[center_idx] > 200,
                "re-touching an existing tile must not lose its painted contents: {:?}",
                &bytes[center_idx..center_idx + 4]
            );
        }

        #[test]
        fn evict_drops_the_tile_and_re_touch_creates_a_fresh_one() {
            let (device, _queue) = request_device();
            let mut pool = TilePool::new(device, 2048, 2048);

            pool.get_or_create(TileId::new(0, 0));
            pool.get_or_create(TileId::new(3, 2));
            assert_eq!(pool.tiles_in_memory(), 2);

            assert!(pool.evict(TileId::new(0, 0)));
            assert_eq!(pool.tiles_in_memory(), 1);
            assert!(pool.get(TileId::new(0, 0)).is_none());

            // Evicting an already-absent id is a harmless no-op.
            assert!(!pool.evict(TileId::new(0, 0)));

            let fresh = pool.get_or_create(TileId::new(0, 0));
            assert_eq!(fresh.dimensions(), (TILE_SIZE, TILE_SIZE));
            assert_eq!(pool.tiles_in_memory(), 2);
        }

        #[test]
        fn evict_then_recreate_loses_painted_contents() {
            let (device, queue) = request_device();
            let mut pool = TilePool::new(device.clone(), 2048, 2048);
            let compositor =
                PaintCompositor::new(device.clone()).expect("device has the required feature");

            let id = TileId::new(0, 0);
            let dab = Dab::new([256.0, 256.0], 32.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1.0);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_tile_pool_test_evict_encoder"),
            });
            compositor
                .splat_dabs(&mut encoder, pool.get_or_create(id), &[dab])
                .expect("single dab batch should splat");
            queue.submit(Some(encoder.finish()));

            let painted = pool
                .get(id)
                .expect("tile is resident")
                .read_back_rgba8(&device, &queue)
                .expect("readback should succeed");
            let center_idx = (256usize * TILE_SIZE as usize + 256) * 4;
            assert!(
                painted[center_idx] > 200,
                "sanity check: the dab actually painted before eviction"
            );

            assert!(pool.evict(id));
            let reborn = pool.get_or_create(id);
            let bytes = reborn
                .read_back_rgba8(&device, &queue)
                .expect("readback should succeed");
            assert!(
                bytes.iter().all(|&b| b == 0),
                "a re-created tile after evict must start fully transparent, \
                 losing the prior paint — the documented Wave-2 limitation"
            );
        }
    }
}

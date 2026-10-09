# umber-gpu tile-pool slab — landing notes

Wave-2 GPU claw: `tile_pool.rs`, the foundation slice of the tile-pool
allocator docs/specs/architecture.md's "Virtual texturing" section
describes (physical page atlas + page-table indirection + background
upload thread). This lands the addressing (UV → tile) and the lazy
create/evict lifecycle only — no atlas packing, no page table, no
readback-before-evict. `paint.rs`'s module docs already flagged this as
"a later slice"; this is that slice's first landing, consumed through
`PaintTarget`'s existing public API with zero changes to `paint.rs` or
`paint_thread.rs`.

## The example-numbers discrepancy in the task brief — read first

The task brief's worked example for the UV→tile mapping doesn't check
out arithmetically: "uv (0.75, 0.25) on 2048² → tile (3,0) with texel
center (384.5, 128.5)". Independently re-deriving it: on a 2048×2048
virtual texture with `TILE_SIZE = 512`, `uv.x = 0.75` maps to global texel
`1536` (`= 3 * 512` exactly) → tile column 3, correct. But `uv.y = 0.25`
maps to global texel `512` (`= 1 * 512` exactly) → tile **row 1**, not
row 0, and the local-in-tile position at that exact boundary is `(0.5,
0.5)`, not `(384.5, 128.5)`.

Reverse-engineering what input *does* produce the stated outputs: the
pair `(384.5, 128.5)` is exactly `(0.75, 0.25)` scaled by `TILE_SIZE` and
offset by `0.5` — i.e. `(0.75, 0.25)` is the *tile-local* fraction inside
tile `(3, 0)`, not the global UV. The global UV that actually produces
tile `(3,0)` / texel-center `(384.5, 128.5)` is `((3 + 0.75) / 4, (0 +
0.25) / 4) = (0.9375, 0.0625)`.

Both numbers are exact in `f32` (`0.9375 = 15/16`, `0.0625 = 1/16`,
`2048` is a power of two), so this isn't a float-precision artifact —
the brief's example mixed a tile-local fraction with a global-UV label.
Resolution: implemented the mapping from first principles (see below),
and `tile_pool.rs`'s tests cover **both** readings —
`tile_for_maps_interior_point_to_its_tile` uses the corrected global UV
`(0.9375, 0.0625)` to reproduce the brief's intended `(3,0)` /
`(384.5, 128.5)` outputs, and `tile_for_exact_tile_boundary_is_lower_inclusive`
uses the brief's literal `(0.75, 0.25)` to document what it actually
produces (`(3,1)` / `(0.5, 0.5)`), as a lower-inclusive-boundary
regression test.

## What was built

**`crates/umber-gpu/src/tile_pool.rs`** (new module)

- `TILE_SIZE: u32 = 512` — fixed edge length of every tile.
- `TileId { column: u32, row: u32 }` — `Debug`/`Clone`/`Copy`/`PartialEq`/
  `Eq`/`Hash`. No wraparound; no bounds validation against any particular
  virtual texture (see "Bounds checking" below).
- `tile_for(uv, virt_w, virt_h) -> Option<TileId>` (free function) —
  maps a normalized UV to its tile. `None` for `u`/`v` outside the
  half-open `[0.0, 1.0)` (negatives, `>= 1.0`, `NaN` via
  `Range::contains`'s `PartialOrd` semantics) or for a zero virtual
  width/height.
- `texel_within_tile(uv, virt_w, virt_h) -> Option<([f32; 2], TileId)>`
  (free function) — same domain as `tile_for`; also returns the
  quantized (floor + 0.5) texel-center position local to the tile.
- `TilePool` — owns `device: wgpu::Device`, the virtual
  `(width, height)`, and `tiles: HashMap<TileId, PaintTarget>`.
  - `new(device, width, height) -> Self` — allocates nothing.
  - `virtual_dimensions(&self) -> (u32, u32)`.
  - `get_or_create(&mut self, id) -> &PaintTarget` — `HashMap::entry` +
    `or_insert_with`; creates a fresh `512×512` `PaintTarget` on first
    touch.
  - `get(&self, id) -> Option<&PaintTarget>` — resident-only, non-creating
    accessor (added beyond the literal brief; `PaintThread`-style
    read-only access needs this and `get_or_create` alone can't serve it
    without mutating).
  - `evict(&mut self, id) -> bool` — drops the tile if resident, returns
    whether one was actually there.
  - `tiles_in_memory(&self) -> usize`, `memory_bytes(&self) -> u64`
    (`tiles_in_memory * TILE_SIZE² * 4`, `Rgba8Unorm`).

**`crates/umber-gpu/src/lib.rs`** — added `pub mod tile_pool;` and
re-exports (`TileId`, `TilePool`, `TILE_SIZE`, `tile_for`,
`texel_within_tile`). No other file touched; no `Cargo.toml` changes
(the module needed no new dependency — see "No thiserror" below).

## UV→tile mapping: texel-space division, not `floor(uv * tiles_per_axis)`

The brief's own formula sketch (`floor(uv * tiles_per_axis)`) only agrees
with the texel-space version when the virtual texture's dimensions are
exact multiples of `TILE_SIZE`. `tile_for_uses_texel_space_not_tiles_per_axis_scaling`
is a regression test for the case where they aren't: on a 1000-wide
virtual texture, `tiles_per_axis = ceil(1000/512) = 2`, and `uv.x = 0.5`
lands at texel 500 — inside tile 0's range `[0, 512)` — but
`floor(0.5 * 2) = 1` would wrongly place it in tile 1. `tile_for`
instead computes `floor(u * virt_w) / TILE_SIZE`, which is correct in
both cases. This matters because §2 of docs/specs/requirements.md
specs per-texture-set resolutions that aren't guaranteed multiples of
512 (4K default, 8K stretch — `4096` and `8192` happen to divide evenly,
but nothing in the requirements guarantees every future caller's virtual
size will).

## Bounds checking: `get_or_create` trusts its caller

`TilePool` stores the virtual `(width, height)` it was constructed with,
but `get_or_create` does **not** check `id` against it — an
out-of-range `TileId` silently allocates a tile with no corresponding
region of the virtual texture. This was a deliberate choice, not an
oversight: `tile_for` is the sanctioned path from UV to `TileId` and
always produces in-range ids by construction; adding a fallible
`Result`-returning `get_or_create` (considered, and flagged during
review) would have deviated from the task's specified `-> &PaintTarget`
signature for a case that only arises if a caller builds a `TileId`
directly and gets it wrong — a caller bug, not a pool invariant. Flagging
here so a reviewer can override this call if `TilePool` ends up exposed
to less-trusted callers later.

## No `thiserror` error type

The constraint list calls for `thiserror`, but nothing in this module
is fallible in a way worth modeling as an error: `get_or_create` always
succeeds (`PaintTarget::new` is infallible), `evict`/`get` return
plain `bool`/`Option`, and `tile_for`/`texel_within_tile` use `Option`
because "not every UV lands on a tile" is a normal, expected outcome,
not an error condition. No `PaintError` variant was added for the same
reason `paint_thread.rs`'s landing notes checked first: `grep -rn
PaintError --include="*.rs" .` shows it's only matched inside
`umber-gpu`, but there was nothing here that needed a new variant in the
first place.

## Eviction semantics (also documented at the top of `tile_pool.rs`)

`evict` drops the `PaintTarget` outright — **painted contents are
lost**. There is no readback-before-evict, no dirty tracking, no
disk swap. Re-touching an evicted id via `get_or_create` gets a brand
new zero-initialized `PaintTarget` (same zero-init guarantee
`PaintTarget::new` already documents), indistinguishable from a tile
that was never painted. This is the documented Wave-2 limitation the
task brief asked for, not a bug: Wave 3's bake integration is where
either a readback-before-evict or a disk-backed swap (mirroring the
undo journal's "~100 steps then disk swap" budget in
docs/specs/architecture.md) is expected to land.

`evict_then_recreate_loses_painted_contents` (GPU test) makes this
falsifiable rather than just asserted in prose: it splats a dab into a
tile, evicts it, re-creates it, and reads back the fresh tile to confirm
every byte is `0`.

Dropping the `wgpu::Texture` is safe at any point — wgpu keeps a
resource alive on the GPU timeline until in-flight work referencing it
finishes, it doesn't require draining the queue first. What `evict`
*can't* protect against: a consumer that cached a bind group or
`TextureView` for a tile before it was evicted and recreated now holds a
handle to a destroyed texture. There's no "handle" type here to make that
unrepresentable (that's an atlas-allocator-level fix, Wave 3+); documented
in `TilePool::evict`'s rustdoc instead.

## Tests

Non-GPU (pure math, `tile_pool::tests`): 9 new tests — `tile_for`'s UV
domain (interior point, exact-boundary lower-inclusivity, out-of-range
rejection including `NaN`/negative/`>=1.0`, zero virtual dims, the
texel-space-vs-tiles-per-axis regression), `texel_within_tile`'s two
worked examples plus its own out-of-range rejection, and `TileId`
structural equality/hash via a `HashSet`.

GPU (`tile_pool::tests::gpu`, feature `gpu`, real adapter required, no
skip — matches `paint_thread.rs`'s stricter convention over `paint.rs`'s
skip-if-missing one, since this suite is specified to run with a real
adapter): 4 new tests.

- `get_or_create_lazily_allocates_touched_tiles_only` — 2048×2048 pool,
  zero tiles at construction, touches `(0,0)` and `(3,2)`, asserts both
  are `512×512` and `tiles_in_memory() == 2` / `memory_bytes()` matches.
- `get_or_create_returns_the_same_tile_on_repeat_touch` — splats a dab
  into a tile, touches the same id again, confirms `tiles_in_memory()`
  stays `1` and the paint survives the second touch (proves
  `get_or_create` doesn't recreate an already-resident tile).
- `evict_drops_the_tile_and_re_touch_creates_a_fresh_one` — the task's
  literal scenario: evict `(0,0)` → count drops to `1`, `get` returns
  `None`, a second evict of the same (now-absent) id is a harmless
  `false`, re-touch creates a fresh `512×512` tile.
- `evict_then_recreate_loses_painted_contents` — the falsifiable version
  of the documented data-loss contract (see "Eviction semantics" above).

All green: `cargo fmt -p umber-gpu`, `cargo clippy -p umber-gpu
--all-targets -- -D warnings`, `cargo test -p umber-gpu` (37 — the
existing 28 plus 9 new), `cargo test -p umber-gpu --features gpu` (50 —
the existing 37 plus 13 new: the 9 pure-math tests run under this feature
too, plus 4 new GPU tests), `cargo build --workspace`.

`cargo clippy -p umber-gpu --all-targets --features gpu -- -D warnings`
(a stricter invocation than the task's literal command, which omits
`--features gpu`) currently fails, but on a **pre-existing** issue in
`renderer.rs` (`assert!(px[0] <= 255 ...)` on a `u8`, landed in
`0071bb9`, unrelated to this slab) — confirmed by checking
`git show HEAD:crates/umber-gpu/src/renderer.rs`, which already contains
the flagged lines. `tile_pool.rs` itself produces zero clippy warnings
under either feature combination.

## Reviewer checklist

- [ ] **Confirm the example-numbers resolution above independently**
  before trusting the mapping tests — this is the highest-leverage
  thing to re-derive by hand, since it means the task brief's own worked
  example was internally inconsistent and I substituted a corrected
  interpretation.
- [ ] `get_or_create`'s no-bounds-check-on-`TileId` design (see "Bounds
  checking" above) is a deliberate scope call, not a gap filled in
  later code. If `TilePool` grows a caller that builds `TileId`s by hand
  instead of going through `tile_for`, revisit whether a fallible
  `Result`-returning variant (or a separate `is_in_bounds` check) is
  worth the signature change.
- [ ] `evict`'s "stale handle" caveat (cached bind group/view surviving
  past an evict+recreate) has no type-level guard here — it's
  rustdoc-only. Whoever wires `TilePool` into `PaintThread`/the renderer
  needs to re-fetch per-frame rather than cache a `&PaintTarget` or its
  view across an evict.
- [ ] `memory_bytes()` counts only the `Rgba8Unorm` texture payload
  (`TILE_SIZE² * 4` per tile), not each tile's small `dims_buffer`
  uniform (8 bytes) — negligible today, but if `PaintTarget` grows
  heavier per-tile state later, re-check whether the diagnostic should
  include it.
- [ ] `tile_for`/`texel_within_tile` cast `u32` virtual dimensions to
  `f32` for the multiply; this is exact for the 4K/8K sizes
  docs/specs/requirements.md §2 specs (`f32` represents integers exactly
  up to 2^24), but would silently lose precision on a hypothetical
  multi-gigapixel virtual texture well beyond this project's scope.
- [ ] No atlas packing, no page table, no background upload thread —
  this is explicitly the Wave-2 foundation slice per
  docs/specs/architecture.md's "v0.1 scope honesty" note (tile-pool
  architecture ships in v0.1, atlas/bake integration is Wave 3). Don't
  mistake this `HashMap`-per-tile design for the final allocator.
- [ ] The pre-existing `renderer.rs` clippy failure under
  `--features gpu` (see "Tests" above) is out of this slab's ownership
  and was left unfixed deliberately — flag if CI runs that stricter
  invocation and expects it green.

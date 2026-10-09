# umber-export TIFF + JPEG — landing notes

Wave-3 claw: the last two formats of the requirements.md §6 list ("formats:
PNG, EXR, TIFF, JPEG") — `crates/umber-export/src/formats.rs` (new module),
the `image` crate (0.25, `default-features = false`, `features = ["tiff",
"jpeg"]`) added as a dependency at workspace + crate level, and
`OutputFormat::Tiff` added to the existing enum in `presets.rs` (one line;
`Jpeg` already existed). `png.rs`, `exr.rs`, and `umber-cli` are a parallel
lane's scope and were not touched.

## What was built

**`crates/umber-export/src/formats.rs`** (new module)
- `write_tiff_rgba8(path, width, height, rgba, transfer: png::Transfer)
  -> Result<(), TiffError>` — lossless RGBA8 TIFF via
  `image::codecs::tiff::TiffEncoder`. Same validate → create-parent-dirs →
  encode shape as `png::write_png`; `transfer` is applied with the same
  sRGB curve PNG uses (duplicated locally rather than imported — see
  below — so the two writers stay byte-for-byte consistent on the same
  input).
- `write_jpeg_rgba8(path, width, height, rgba, quality: u8)
  -> Result<(), JpegError>` — lossy RGB JPEG via
  `image::codecs::jpeg::JpegEncoder::new_with_quality`. Alpha is dropped
  before encoding (see tradeoffs below). No transfer parameter, by brief
  — but the paint buffer is linear `rgba8unorm` per `png.rs`'s module
  doc, so a basecolor-class map passed through raw will come out dark;
  callers exporting color maps as JPEG previews must apply the sRGB
  curve themselves before calling (the same conversion `png::write_png`
  does internally for `Transfer::Srgb`), the same way a caller would for
  any other linear-only writer.
- `TiffError` / `JpegError` (`thiserror`) — `SizeMismatch { actual,
  expected, width, height }`, `Encode(String)` (via `From<image::ImageError>`,
  mirroring `exr.rs`'s `From<exr::error::Error>`), `Io(#[from]
  std::io::Error)`.

**`crates/umber-export/src/presets.rs`** — one enum variant added:
`OutputFormat::Tiff`. Nothing else in the file touched (`Jpeg` was already
present from an earlier wave).

**`crates/umber-export/src/lib.rs`** — `pub mod formats;` added, the top
module doc comment's format list extended to mention TIFF/JPEG, and
`write_tiff_rgba8`/`write_jpeg_rgba8`/`TiffError`/`JpegError` re-exported
at the crate root alongside the existing `pub use presets::{…}` (this
crate's established pattern: `OutputFormat` etc. are flattened to the
root even though `png`/`exr`'s own writers are reached through their
module path — the brief named "the lib.rs re-exports" as this task's to
own, so the new public items join the flattened set).

**Cargo.toml (workspace + crate)** — `image = { version = "0.25",
default-features = false, features = ["tiff", "jpeg"] }` at the workspace
level, `image = { workspace = true }` added to `umber-export`'s deps.
`default-features = false` means *umber-export's own request* only pulls
the tiff+jpeg codecs — it does not mean the final binary excludes the
others: `image` was already in the dependency graph transitively (via
`eframe` and `arboard`, both already in the workspace for clipboard/icon
support), and Cargo's feature unification resolves one shared feature set
across every crate that depends on `image`, so PNG support (at least) is
compiled in regardless of what `umber-export` asks for. Confirmed via
`Cargo.lock`'s `image` entry, which lists `png 0.18.1` among its resolved
deps even with this crate's features pared to `["tiff", "jpeg"]`.

### Why the sRGB curve is duplicated instead of imported

`write_tiff_rgba8` takes a `png::Transfer` (reusing that enum — it is
`pub`) and needs the same linear→sRGB byte mapping `png.rs` applies, but
the mapping function itself (`linear_to_srgb_u8`) is private to that
module, and `png.rs` is a parallel lane's file this task was scoped to
leave untouched (not even a `pub(crate)` visibility bump). The eleven-line
function is copied verbatim into `formats.rs` rather than requesting a
visibility change. If a future pass merges the lanes, this is the one
obvious spot to collapse back into a shared helper.

## Format tradeoffs

**JPEG — lossy, RGB only, no alpha.** JPEG's color model is YCbCr with
chroma subsampling; there is no alpha plane in the format at all, not a
missing feature of the `image` crate. `write_jpeg_rgba8` drops the alpha
byte of every input texel before encoding — this is not a bug to fix
later, it is the ceiling of the format. Treat JPEG as the quick-preview /
thumbnail output only:
- Never route normal maps, masks, or anything read back for its alpha
  (opacity, AO coverage, UDIM padding masks) through JPEG — the channel
  that matters is silently gone, and the lossy RGB quantization alone
  already makes JPEG unfit for normal maps (a few bits of error become a
  visible shading seam).
- Fine for basecolor-style previews where a human is going to eyeball a
  thumbnail and file size matters more than exactness.

**TIFF — lossless, full RGBA8, same transfer-curve semantics as PNG.**
TIFF here is PNG's lossless sibling for pipelines/tools that specifically
want `.tiff`/`.tif` (some DCC and print tooling defaults to it over PNG).
No channel is dropped, no quantization beyond the 8-bit-per-channel input
depth already carries. Pick it over `Png8` only when the consumer
specifically requires TIFF — otherwise PNG is the equivalent, more
universally supported choice.

## Quality guidance (JPEG)

`quality` is the `image` crate's 1–100 JPEG quality scale, passed straight
through to `JpegEncoder::new_with_quality` with no clamping or validation
added on top. Rough guidance for callers wiring this into UI:
- **90–95**: visually near-lossless for basecolor previews; file size
  still meaningfully smaller than PNG for photographic-ish content.
- **70–85**: typical thumbnail/preview-grid quality — the sweet spot for
  "fast to load, good enough to recognize the asset."
- **below 50**: visible blocking/banding on anything but near-flat color;
  only reasonable for tiny thumbnails where artifacts aren't resolvable.
- `jpeg_lower_quality_yields_smaller_file` pins the one invariant that
  actually matters mechanically (lower quality ⇒ smaller file on the same
  input); it does not assert any specific size or PSNR number, since
  those are encoder-version-dependent.

## Testing

- `tiff_roundtrips_a_tiny_map_exactly` — 2×2, four distinct RGBA texels
  (including partial/zero alpha), written then decoded via `image::open`;
  every byte must survive exactly (lossless claim, pinned).
- `tiff_applies_srgb_transfer_like_png` — flat mid-gray (128) with
  `Transfer::Srgb` decodes back to 188, the same anchor point
  `png::tests::srgb_transfer_matches_spec_corners` pins for PNG — the two
  writers' curves provably agree on shared input.
- `tiff_rejects_mismatched_buffer_size` / `jpeg_rejects_mismatched_buffer_size`
  — short buffer rejected before any file is created (asserts
  `!path.exists()`, matching the `exr.rs` convention).
- `tiff_creates_parent_directories` / `jpeg_creates_parent_directories` —
  nested missing directories are created.
- `jpeg_roundtrips_dimensions_and_drops_alpha` — decodes back as
  `image::ColorType::Rgb8` (not `Rgba8`) at the right dimensions; checks
  each quadrant's dominant channel survives recognizably rather than
  exact bytes (JPEG is lossy — exact-byte assertions on a 2×2 image would
  be testing libjpeg's block-DCT rounding, not this module's contract).
- `jpeg_lower_quality_yields_smaller_file` — same 64×64 gradient input
  encoded at quality 5 and 95; asserts the low-quality file is smaller.

34/34 `umber-export` tests green (26 pre-existing + 8 new here — the 26
is the 23-test baseline this task was briefed against plus 3 PNG16 tests
the parallel lane landed in `png.rs` during this task). Workspace-wide:
`cargo check --workspace --all-targets` and `cargo test --workspace` both
stay green with the new `OutputFormat::Tiff` variant and the new public
API — no other crate currently matches on `OutputFormat` exhaustively (a
repo-wide grep turns up only `presets.rs`'s own definition/usages and the
`lib.rs` re-export), so the added variant isn't a breaking change for
`umber-cli`/`umber-app`.

## Reviewer checklist

- [ ] `png.rs`, `exr.rs`, `umber-cli` diffs are empty (parallel lane's
      scope).
- [ ] `presets.rs` diff is exactly the one `OutputFormat::Tiff` variant —
      no other line touched.
- [ ] `image` crate pulled in with `default-features = false` and only
      `["tiff", "jpeg"]` requested by `umber-export` itself — don't expect
      this to exclude other codecs from the final binary; `eframe`/
      `arboard` already pull `image` in with their own features, and
      Cargo unifies the feature set across the whole dependency graph.
- [ ] `lib.rs` re-exports `write_tiff_rgba8`/`write_jpeg_rgba8`/
      `TiffError`/`JpegError` at the crate root (this task owned the
      lib.rs re-exports; confirm they landed, not just `pub mod formats;`).
- [ ] Alpha-drop in `write_jpeg_rgba8` is treated as a documented format
      ceiling, not flagged as a bug to "fix" by keeping alpha.
- [ ] `linear_to_srgb_u8` duplication in `formats.rs` vs. `png.rs` is
      accepted as intentional (see "why duplicated" above), not merged
      without also resolving the parallel-lane file-ownership split.
- [ ] No `unwrap`/`expect` outside `#[cfg(test)]` modules.
- [ ] `cargo fmt --check`, `cargo clippy -p umber-export --all-targets --
      -D warnings` clean.
- [ ] Full `umber-export` suite green (34/34 at landing time).

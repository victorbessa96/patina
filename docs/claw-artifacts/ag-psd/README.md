# ag-psd 0.3.0 — verified API facts for the PSD export slice

Vendored at `src-tree/` (the crate's own source; facts below
verified by the dragon against it). The claw builds the PSD
writer against THESE facts.

## Confirmed surface (grep-verified in the vendored tree)

1. **The writer**: `write_psd(&Psd, &mut impl Write)` — the entry
   point (check src-tree/src/lib.rs's pub exports; the Psd
   struct is the document).
2. **Psd struct**: the builder shape — `width`, `height`,
   `channels` (the channel-count u16), the `layers` vec. Verify
   the exact field/method names in src-tree/src/lib.rs + the
   write path's expectations.
3. **Layers**: `PsdLayer` — `name: String`, `top/left/width/
   height`, `channel_data` per channel (the raw bytes +
   compression: the RLE the format expects — READ whether
   ag-psd compresses internally or expects the caller's RLE;
   the README/tests in the tree show the intended usage — cite
   the example the tree itself carries).
4. **The color mode**: RGB 8-bit (`mode: 3` in the header — the
   crate's constant or enum).
5. **Image data**: the merged/flattened composite (the
   compatibility layer other readers show when opening) —
   ag-psd's writer path fills it or leaves it empty (VERIFY
   which; the honest PSD ships the composite or documents the
   omission).

## The slice's shape

- umber-export gains `psd.rs`: `write_psd_maps(maps: &MapSet,
  out: &Path, layered: bool)` — one layer per map (base color /
  normal / AO...), the composite filled from the base color
  (or the first map), RLE per the crate's expectation.
- The MapSet's bytes are RGBA8 (the existing pipeline's format)
  — the PSD channel split (RGB + A per layer) is the writer's
  job (the crate takes channels; the split is mechanical).
- Tests: write → re-read with ag-psd's own reader (the round
  trip within the crate — the byte-exact compare on the layer
  data), the Photoshop-compat check (zune-psd reads it too —
  but that's a second dep; v1 asserts ag-psd's round trip +
  the header fields, the third-party read is a named
  follow-up).

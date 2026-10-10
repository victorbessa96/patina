# openusd 0.7.0 — verified facts for the .usdc import slice

Vendored at `src-tree/` (the crate's src; facts verified by the
dragon against it). The claw builds the .usdc IMPORT against
THESE facts.

## Confirmed surface (grep-verified in the vendored tree)

1. **The entry point**: `usdc::CrateData<R>::open(reader,
   safe: bool)` where `R: io::Read + io::Seek` (mod.rs:174) —
   opens + indexes a .usdc, path-keyed specs (`HashMap<sdf::Path,
   Spec>`), fields resolved on demand.
2. **The low-level file**: `usdc::CrateFile<R>::open(reader)` +
   `.validate()` (reader.rs:78/106) — sections, versions, the raw
   ValueRep decode.
3. **The sibling formats**: the crate reads/writes .usda, .usdc,
   AND .usdz (lib.rs:8-21); `usda` module is the text format the
   app's existing a17e1c8 parser handles — the crate is the
   eventual UNIFIER, v1 uses it for .usdc only.
4. **The examples**: README:112 shows `dump_usdc` — a read-path
   example exists in the crate's own examples (cite when wiring).

## The slice's shape

- umber-import (CHECK: does the workspace have an umber-import
  crate, or does umber-app carry import?) gains the .usdc arm:
  the existing import dispatch (usda today) + `CrateData::open`
  for .usdc — the SAME mesh-extraction logic the usda parser
  feeds, reading specs via the path index.
- Tests: the round-trip fixtures — a small .usdc written with
  openusd's own CrateWriter, read back through umber's import
  (vertex/prim counts, the material binding), the ERROR path
  (a truncated file fails cleanly, no panic).

## Dependency decision (the dragon's, pre-made)

`openusd = "0.7.0"` in umber-import (or the app's import module)
— pure Rust, no C build, the deferral recorded in the audit is
VOID now. The .usdz read is a named follow-up, not v1.

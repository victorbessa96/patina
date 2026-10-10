# Autosave + Crash Recovery — the Journal Design (§8's P1 row)

Written 2026-10-10 16:20 against `7a636a6`. The project format's
last unbuilt row: the journal that survives a crash between
saves.

## The shape: an append-only journal beside the project

1. **The journal file**: `<project>.umber/journal.jsonl` — one
   JSON line per mutation event, append-only, fsync'd per entry.
   The event carries: the operation id, the layer/texture-set
   ids it touched, and the content hash of any tile bytes it
   wrote (tiles are already hash-named per §8's format — the
   journal only records WHICH hashes became live, never blob
   data; the tiles themselves are written through the normal
   assets path immediately).
2. **The write path**: the app's mutation entry points (the
   same chokepoints the undo stack records) call
   `journal.record(op)` after the in-memory apply succeeds and
   the tile bytes hit disk. Autosave = the journal is ALWAYS
   current (every op journaled), NOT a periodic dump — the
   crash window shrinks to the op's own write ordering.
3. **Recovery**: at `load_from_dir`, if journal.jsonl exists:
   replay the events not already reflected in project.json (the
   project carries a `journal_seq` high-water mark; events past
   it replay idempotently — the ops are the same
   apply-functions the app uses, so replay = re-apply). On
   success, project.json rewrites with the new high-water mark
   and the journal truncates (the recovery's own atomicity: the
   rewrite-then-truncate order, crash-safe because replay is
   idempotent).
4. **The honest v1 scope**: journaling the document mutations
   (layer add/remove/reorder, paint-commit tile hash swaps) —
   the ops the undo stack already sees. Settings/panel state
   NOT journaled (they live in project.json, saved on Save — a
   crash loses at most the unsaved settings, an honest loss
   documented in the row).
5. **The corruption rule**: a malformed journal line = the
   journal stops replay there (the high-water discipline —
   everything up to the last valid line applies; a torn final
   write is exactly the crash case, and the last valid line is
   the recovery point). Never a hard error — recovery is
   best-effort by design, the journal is an optimization over
   losing everything since the last save.

## Tests

1. The journal write: an op through the app's path lands as a
   parseable JSONL line with the right shape.
2. The replay: a project + a journal with N events past the
   high-water mark loads with the events applied (assert the
   end state == the state a live session would reach).
3. The torn-write case: a truncated final line replays to the
   last VALID line, loads clean, no error.
4. The idempotence: replaying the same journal twice (simulating
   a crash during recovery itself) yields the same state.
5. The high-water advance: after recovery, project.json's
   journal_seq = the last replayed event, and the journal
   truncates to empty.

## Build

One claw slice, code-only: the journal module (umber-core, beside
project.rs — it owns the op vocabulary), the app-side record
chokepoints, the five tests. No new deps (serde_json, fs, the
existing assets-hash path).

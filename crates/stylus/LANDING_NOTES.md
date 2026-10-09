# Windows Ink (`WM_POINTER`) backend skeleton — landing notes

## What was built

- `Cargo.toml`: optional `windows = "0.62"` dependency (features
  `Win32_UI_Input_Pointer`, `Win32_UI_WindowsAndMessaging`,
  `Win32_Foundation`), placed under `[target.'cfg(windows)'.dependencies]`
  (see "Cargo.toml dependency placement" below for why), plus a
  `winink = ["dep:windows"]` feature. Not a default feature.
- `src/winink.rs`, gated `#[cfg(all(windows, feature = "winink"))]`:
  - `PenState` — a per-pointer accumulator holding a monotonic time epoch
    (first observed `dwTime`, ms) so every decoded event carries
    nanoseconds relative to that epoch, not a raw wall-clock tick.
  - `PenState::decode(&mut self, pointer_id: u32, msg: u32) -> Option<TabletEvent>`
    — the passive decode entry point. Calls `GetPointerInfo` to filter to
    `PT_PEN` pointers and get `dwTime`; for `WM_POINTERUPDATE` only, also
    calls `GetPointerPenInfo` for position/pressure/tilt.
  - Pure, FFI-free mapping functions factored out so they're unit-testable
    without a live HWND/pointer: `map_simple_message` (enter/leave/down/up),
    `map_pen_message` (update → `Motion`), `normalize_pressure`.
  - Unit tests, gated `#[cfg(all(windows, feature = "winink", test))]`:
    pressure normalization (incl. defensive clamp above the documented
    range), tilt presence/absence via `penMask`, message→event mapping,
    and `dwTime` epoch tracking including `u32` wraparound.
- `src/lib.rs`: added `#[cfg(all(windows, feature = "winink"))] pub mod winink;`.
  Existing `TabletEvent`/`InputSource`/`SourcedEvent` vocabulary untouched.

## API decisions + vendored-source line refs

- **Two-call sequence** (`GetPointerInfo` then conditionally `GetPointerPenInfo`)
  matches `docs/claw-artifacts/stylus/README.md:28-29`. `GetPointerInfo` is
  called for every message (needed for `pointerType` filtering and
  `dwTime`); `GetPointerPenInfo` is only called for `WM_POINTERUPDATE`,
  since `Contact`/`ProximityIn`/`ProximityOut` in the frozen vocabulary
  carry no position/pressure payload — calling it for every message would
  be pure waste.
- **Field names** (`pointerType`, `dwTime`, `ptPixelLocation`, `pressure`,
  `tiltX`, `tiltY`, `penMask`) taken verbatim, camelCase, from the vendored
  struct defs: `docs/claw-artifacts/stylus/win32_pointer/mod.rs:282-311`
  (`POINTER_INFO`/`POINTER_PEN_INFO`). Note the README's own prose
  (`README.md:23-26`) paraphrases these as snake_case (`dw_time`, `tilt_x`,
  `pointer_id`) — that's just the README's shorthand, not the real API;
  the code follows the struct source, not the README prose.
- **Pressure normalization**: `pressure as f32 / 32767.0`, clamped to
  `0.0..=1.0`. The `32767` divisor is a verified fact
  (`docs/claw-artifacts/stylus/README.md:23-24`); the clamp is defensive
  only (hardware is not expected to exceed the documented range) — covered
  by `pressure_clamps_beyond_the_documented_range`.
- **Tilt gating by `penMask`**: `PEN_MASK_TILT_X`/`PEN_MASK_TILT_Y`
  (`docs/claw-artifacts/stylus/win32_pointer/windowsandmessaging_mod.rs:5481-5485`,
  values `4u32`/`8u32`) are checked before reporting `Some([tiltX, tiltY])`.
  Devices without a tilt sensor leave `tiltX`/`tiltY` at a meaningless `0`;
  reporting `tilt: None` in that case is more honest than a fake `[0.0, 0.0]`
  and uses the existing `Option<[f32; 2]>` shape in `TabletEvent::Motion`
  rather than inventing a new field.
- **Button events are out of scope for this skeleton.** `TabletEvent::Button`
  exists in the frozen vocabulary but nothing in this decode path produces
  it yet — the task's spec for `decode` only calls for
  ProximityIn/Out/Motion/Contact. Barrel-button mapping (`PEN_FLAG_BARREL`
  / `POINTER_MESSAGE_FLAG_*`) is a natural Wave 2 addition once there's a
  live HWND to drive it.
- **No custom error type.** `decode` returns `Option<TabletEvent>`, not
  `Result`. A `GetPointerInfo` failure and "pointer isn't a pen" collapse
  to the same `None`; the caller can't act differently on either, so a
  `thiserror` enum would add a variant nobody inspects. No `unwrap()`
  outside `#[cfg(test)]`. (But see "Open risk: dropped up/leave" below —
  this isn't a cost-free simplification.)
- **`dwTime` → `time_ns` wraparound**: `dw_time.wrapping_sub(epoch)` as
  `u64`, then `* 1_000_000`. `dwTime` is `GetTickCount`-based and wraps
  every ~49.7 days; `wrapping_sub` on the `u32`s before widening handles one
  wrap correctly. Covered by `time_ns_handles_dw_time_wraparound`.

## Load-bearing assumptions not in the vendored material

The vendored artifacts cover `GetPointerInfo`/`GetPointerPenInfo` and the
`POINTER_INFO`/`POINTER_PEN_INFO`/message-id surface, but not everything
this code relies on:

- **`windows::Win32::Foundation::POINT`** (`{ x: i32, y: i32 }`) wasn't
  vendored (only the `Pointer` and `WindowsAndMessaging` modules were).
  This is well-established, stable Win32 API shape, but it rests on
  background knowledge rather than a vendored source — if wrong, it's a
  hard compile error on Windows CI, not a silent bug.
- **Coordinate space of `pos`.** `ptPixelLocation` is screen-space,
  integer-pixel, and *prediction-smoothed* by Windows Ink (a different
  value than what the pen physically reported, adjusted to hide input
  latency). `TabletEvent::Motion.pos`'s doc comment in `lib.rs` doesn't
  specify a coordinate space, and a future Wayland/X11 backend will
  naturally emit surface-local coordinates — these will disagree with this
  backend's screen-space output unless reconciled at the app layer.
  Alternatives available on `POINTER_PEN_INFO.pointerInfo` if a reviewer
  wants a different tradeoff: `ptPixelLocationRaw` (unpredicted — probably
  the better choice for authoritative stroke recording) and
  `ptHimetricLocation`/`ptHimetricLocationRaw` (sub-pixel, HIMETRIC units).
  Converting to client-area coordinates would need `ScreenToClient`, which
  needs `Win32_Graphics_Gdi` — not in the feature set the task specified,
  so that conversion is app-layer or Wave 2 work regardless.
- **Tilt units.** `tiltX`/`tiltY` are degrees (-90..=90) per Microsoft's
  Win32 docs, not in the vendored struct source (which only gives `i32`,
  no units). `TabletEvent::Motion.tilt`'s doc comment doesn't specify
  units either — worth pinning down before a second backend (which may
  report tilt differently) lands.

## What Linux could verify vs. what only Windows CI can

**Verified on this Linux box** (all green):
- `cargo check -p stylus` (default features).
- `cargo check -p stylus --no-default-features`.
- `cargo check -p stylus --features winink` (forced on, despite `winink`
  not being a default feature — see "Cargo.toml dependency placement").
- `cargo clippy -p stylus --all-targets -- -D warnings` (default features).
- `cargo clippy -p stylus --features winink --all-targets -- -D warnings`.
- `cargo test -p stylus --features winink` (still only the 1 pre-existing
  test runs — the 8 new `winink` tests stay `cfg(windows)`-gated out on a
  Linux host even with the feature forced on, exactly as expected).
- `cargo fmt -p stylus -- --check`.

**Cannot be verified here — Windows CI must check:**
- That `src/winink.rs` actually compiles against real `windows` 0.62 types
  on a Windows target, and that the `POINT` field assumption above holds.
- The 8 unit tests under `#[cfg(all(windows, feature = "winink", test))]`
  actually running and passing on `windows-latest`.
- `cargo clippy -p stylus --features winink --all-targets -- -D warnings`
  on an actual Windows target — the `unsafe` blocks and casts were checked
  clean by clippy on this Linux box, but clippy without the target's real
  `windows` types compiled in is a weaker signal than the genuine
  Windows-target run.

## Cargo.toml dependency placement

Initially added `windows` as a plain optional `[dependencies]` entry (as
literally written in the task). On testing `cargo check -p stylus
--features winink` on this Linux box to make sure a forced-on feature at
least fails inertly, it failed hard: `windows-future` (pulled in
transitively through `Win32_Foundation`) references `windows_core::imp`
and `windows_threading::submit` items that only exist on Windows targets —
unlike the `windows` crate's own generated bindings, which compile to
nothing off-Windows via their own `#![cfg(windows)]`, `windows-future` is
not gated the same way and fails to compile outright on a Linux host,
regardless of version.

Fixed by moving the dependency into `[target.'cfg(windows)'.dependencies]`
(`optional = true` and the `winink = ["dep:windows"]` feature wiring
unchanged — Cargo supports target-specific optional deps behind `dep:`
features). Cargo now never attempts to resolve or compile `windows` at all
on a non-Windows host, regardless of which features are enabled. Re-ran
the full check/clippy/test suite above after the fix — all green,
including the previously-failing forced-feature check. This also protects
anyone running `--all-features` (workspace-wide or via rust-analyzer) on a
Linux dev machine.

## Open risk: a failed `GetPointerInfo` call silently drops the event

`decode` returns `None` whenever `GetPointerInfo` fails for a given
`pointer_id`, with no distinction from "not a pen." If that failure
happens on a `WM_POINTERUP` or `WM_POINTERLEAVE`, the dropped event is a
`Contact { down: false }` or `ProximityOut` that never reaches the app —
i.e. a stroke that never un-sticks. This skeleton doesn't have enough
context (no live HWND, no persistent per-pointer-id state across calls
beyond the time epoch) to do better than return `None`; Wave 2's live-HWND
integration should decide whether `PenState` needs to track currently-down
pointer ids so it can synthesize the missing up/leave rather than silently
losing it.

## Found and deliberately did not fix

- **README documentation bug**: `docs/claw-artifacts/stylus/README.md:12`
  states `POINTER_INPUT_TYPE (PT_PEN = 2)`, but the vendored source itself
  (`win32_pointer/windowsandmessaging_mod.rs:5523`) defines
  `PT_PEN: POINTER_INPUT_TYPE = POINTER_INPUT_TYPE(3i32)`. Doesn't affect
  correctness here — the code imports the `PT_PEN` symbol rather than
  hardcoding its value — but the README's "verified facts" section should
  be corrected before anyone else copies the `2` out of it.
- **THIRD_PARTY.md gap**: `windows` is a new direct dependency of `stylus`
  and is not recorded in `/THIRD_PARTY.md`'s dependency table. License is
  MIT OR Apache-2.0 (vendored `README.md:5`), so it's GPL-3.0-compatible
  like everything else there — this is a records gap, not a licensing
  blocker. Matches the exact pattern of `6756685`'s `rfd` catch (added at
  skeleton time, recorded after the fact). Left unedited: `THIRD_PARTY.md`
  is outside `crates/stylus`, and this task's scope is the `stylus` crate
  only. `Cargo.lock` also changed as a side effect of adding the
  dependency — a workspace-root file, same scope note applies.
- **CI gap**: `.github/workflows/ci.yml` never passes `--features winink`
  (or `--all-features`) on either matrix leg — `clippy --workspace
  --all-targets`, `test --workspace`, and `build --release --workspace`
  all run with default features only. That means this module, including
  its Windows-only unit tests, **will not be compiled or run by CI as it
  stands today** — on `windows-latest` or anywhere else — until a
  follow-up adds a Windows-only step. Example of what that step could look
  like (not applied — CI is outside this task's scope):
  ```yaml
  - name: Windows Ink backend (Windows only)
    if: runner.os == 'Windows'
    run: |
      cargo clippy -p stylus --features winink --all-targets -- -D warnings
      cargo test -p stylus --features winink
  ```

## Reviewer checklist

- [ ] Record `windows` (0.62, MIT OR Apache-2.0) in `/THIRD_PARTY.md`'s
      Wave 1 dependency table.
- [ ] On a Windows machine/CI: `cargo test -p stylus --features winink`
      passes (8 new tests) and `cargo clippy -p stylus --features winink
      --all-targets -- -D warnings` is clean against the real Windows
      target (this box could only check the latter against an inert,
      not-actually-compiled dependency).
- [ ] Confirm `windows::Win32::Foundation::POINT` has `x`/`y: i32` fields
      in the resolved 0.62.x version (not vendored here; see above).
- [ ] Decide `pos`'s coordinate space: predicted screen-space
      (`ptPixelLocation`, what's implemented) vs. raw screen-space
      (`ptPixelLocationRaw`) vs. sub-pixel HIMETRIC — and whether/how it
      should reconcile with future Wayland/X11 surface-local coordinates.
      Consider documenting the chosen space on `TabletEvent::Motion.pos`
      itself in `lib.rs` so every backend is held to the same contract.
  - [ ] Confirm tilt units (degrees, per Win32 docs) belong on
      `TabletEvent::Motion.tilt`'s doc comment for the same reason.
- [ ] Decide whether/when to add a CI step that builds+tests `stylus` with
      `--features winink` on `windows-latest` — currently nothing does
      (see suggested step above).
- [ ] Fix `docs/claw-artifacts/stylus/README.md:12` (`PT_PEN = 2` → `3`) or
      drop the artifact per its own "delete when the backend lands" note.
- [ ] Decide how to handle a failed `GetPointerInfo` call on
      `WM_POINTERUP`/`WM_POINTERLEAVE` (see "Open risk" above) — likely a
      Wave 2 concern once `PenState` has a live HWND to track active
      pointer ids against.
- [ ] Confirm the decision to key `ProximityIn/Out`/`Contact` purely off
      the window message id (not `POINTER_FLAG_INRANGE`/`INCONTACT`) is
      acceptable for Wave 1 — it's simpler and matches the task's literal
      mapping spec, but `pointerFlags` is available on `POINTER_INFO` if a
      future reviewer wants cross-validation against the message id.
- [ ] Confirm barrel-button handling being deferred to Wave 2 (live HWND
      pump) rather than stubbed now is the right call.

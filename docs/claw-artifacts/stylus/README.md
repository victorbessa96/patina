# Claw artifacts: stylus (Windows Ink WM_POINTER backend)

Hand-off artifacts for the stylus-claw. The opencode/claude sandboxes cannot
read the cargo registry; these are verbatim copies of the exact `windows`
crate 0.62.2 (MIT OR Apache-2.0) source modules the backend needs. Delete
when the backend lands and is cross-reviewed.

## Files

- `win32_pointer/mod.rs` — `Windows::Win32::UI::Input::Pointer` verbatim:
  `GetPointerInfo`, `GetPointerPenInfo`, `POINTER_INFO`, `POINTER_PEN_INFO`,
  `POINTER_INPUT_TYPE` (PT_PEN = 2), pointer-id/event fns, the full surface.
- `windowsandmessaging_mod.rs` — `Windows::Win32::UI::WindowsAndMessaging`
  mod re-exports (HWND, window messages, message structs the pointer events
  arrive on).

## Verified API facts (checked against the vendored sources 2026-10-09)

- Dependency to add to `crates/stylus/Cargo.toml`:
  `windows = { version = "0.62", features = ["Win32_UI_Input_Pointer", "Win32_UI_WindowsAndMessaging", "Win32_Foundation"] }`
  (feature names map to module paths: Win32::UI::Input::Pointer ->
  "Win32_UI_Input_Pointer").
- `POINTER_PEN_INFO` carries `pressure: u32` (0..32767 range per Windows Ink,
  normalize / 32767.0), `tilt_x`/`tilt_y: i32`, `pen_flags`, plus the shared
  `POINTER_INFO` fields: `pointer_id`, `frame_id`, `pointer_type`,
  `pt_pixel_location` (POINT), `dw_time`, `history_count`.
- Events arrive as window messages: `WM_POINTERDOWN` (0x0246), `WM_POINTERUP`
  (0x0247), `WM_POINTERUPDATE` (0x0245), `WM_POINTERENTER` (0x0249),
  `WM_POINTERLEAVE` (0x024A). On each: `GET_POINTERID_WPARAM(wparam)` or
  `GetPointerInfo(pointer_id, &mut info)`, then `GetPointerPenInfo(id, &mut pen)`.
- Architecture decision (docs/specs/architecture.md): the app owns the winit
  event loop; the stylus crate must NOT create its own window. The Windows Ink
  backend is a passive listener: expose `stylus::winink::translate_wparam(wparam, lparam) -> Option<TabletEvent>`
  + a `PenState` accumulator the app drives from its WndProc hook — the app
  layer hooks its own subclass or a poll-based `peek()` API; this crate's
  Windows code compiles under `#[cfg(windows)]` + a `winink` cargo feature.
- Linux builds must compile with the feature off (CI runs ubuntu without it).

## Scope discipline

Wave 1 exit needs the backend SKELETON compiling + unit-tested on Windows
CI: message decode + pressure normalization + event mapping. Live HWND
pump integration is Wave 2 (with the stroke pipeline).

//! Windows Ink (`WM_POINTER`) backend — passive decode, no window ownership.
//!
//! Per `docs/specs/architecture.md`, `umber-app` owns the single winit event
//! loop; this crate must not create a window, register a window class, or
//! run a message pump. The app layer hooks its own `WndProc` subclass,
//! extracts a pointer id from `wparam` (`GET_POINTERID_WPARAM`), and
//! forwards `(pointer_id, msg)` to [`PenState::decode`]. This module does
//! the rest: it calls `GetPointerInfo`/`GetPointerPenInfo` itself and maps
//! the result onto the crate's backend-agnostic [`crate::TabletEvent`].
//!
//! Wave 1 scope (`docs/claw-artifacts/stylus/README.md`): skeleton decode +
//! pressure normalization + event mapping, unit-tested without a live
//! HWND. Live pointer-pump integration is Wave 2.

use crate::TabletEvent;
use windows::Win32::UI::Input::Pointer::{
    GetPointerInfo, GetPointerPenInfo, POINTER_INFO, POINTER_PEN_INFO,
};
use windows::Win32::UI::WindowsAndMessaging::{
    PEN_MASK_TILT_X, PEN_MASK_TILT_Y, PT_PEN, WM_POINTERDOWN, WM_POINTERENTER, WM_POINTERLEAVE,
    WM_POINTERUP, WM_POINTERUPDATE,
};

/// Windows Ink reports pen pressure on a fixed `0..=32767` scale (verified
/// against the vendored `POINTER_PEN_INFO` source —
/// `docs/claw-artifacts/stylus/win32_pointer/mod.rs`).
const MAX_PRESSURE: f32 = 32767.0;

/// Per-pointer decode state for the Windows Ink passive listener.
///
/// Tracks a monotonic time epoch derived from the first `dwTime` (ms,
/// `GetTickCount`-based and wrapping) it observes, so every [`TabletEvent`]
/// carries nanoseconds relative to that epoch rather than a raw wall-clock
/// tick count.
#[derive(Debug, Default)]
pub struct PenState {
    epoch_dw_time: Option<u32>,
}

impl PenState {
    /// Creates a fresh accumulator with no time epoch established yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes one `WM_POINTER*` message into a [`TabletEvent`].
    ///
    /// `pointer_id` is the id extracted from `wparam` by the caller's
    /// `WndProc` hook; `msg` is the raw window message (one of
    /// `WM_POINTERDOWN`, `WM_POINTERUP`, `WM_POINTERUPDATE`,
    /// `WM_POINTERENTER`, `WM_POINTERLEAVE`). Returns `None` for any other
    /// message, for a non-pen pointer (touch/mouse), or if `GetPointerInfo`
    /// fails for this `pointer_id`. The last case silently drops whatever
    /// event `msg` represented — including a `WM_POINTERUP`/`WM_POINTERLEAVE`
    /// — rather than treating it as an error; see `LANDING_NOTES.md` for why
    /// that's an open question for the Wave 2 live-HWND integration rather
    /// than something this skeleton can resolve.
    pub fn decode(&mut self, pointer_id: u32, msg: u32) -> Option<TabletEvent> {
        let mut info = POINTER_INFO::default();
        // SAFETY: `info` is a valid, zero-initialized out-param buffer of
        // the exact type `GetPointerInfo` expects; `pointer_id` is an
        // opaque id supplied by Windows via `wparam`, not dereferenced.
        unsafe { GetPointerInfo(pointer_id, &mut info) }.ok()?;

        if info.pointerType != PT_PEN {
            return None;
        }

        let time_ns = self.time_ns(info.dwTime);

        if msg == WM_POINTERUPDATE {
            let mut pen = POINTER_PEN_INFO::default();
            // SAFETY: same contract as the `GetPointerInfo` call above.
            unsafe { GetPointerPenInfo(pointer_id, &mut pen) }.ok()?;
            map_pen_message(msg, &pen, time_ns)
        } else {
            map_simple_message(msg, time_ns)
        }
    }

    /// Converts a raw `dwTime` (ms) sample into nanoseconds since the
    /// epoch established by the first call, handling `dwTime`'s
    /// `u32` wraparound via wrapping subtraction.
    fn time_ns(&mut self, dw_time: u32) -> u64 {
        let epoch = *self.epoch_dw_time.get_or_insert(dw_time);
        u64::from(dw_time.wrapping_sub(epoch)) * 1_000_000
    }
}

/// Maps the position-less `WM_POINTER*` messages (enter/leave/down/up) to
/// their [`TabletEvent`] equivalents.
fn map_simple_message(msg: u32, time_ns: u64) -> Option<TabletEvent> {
    match msg {
        WM_POINTERENTER => Some(TabletEvent::ProximityIn { time_ns }),
        WM_POINTERLEAVE => Some(TabletEvent::ProximityOut { time_ns }),
        WM_POINTERDOWN => Some(TabletEvent::Contact {
            down: true,
            time_ns,
        }),
        WM_POINTERUP => Some(TabletEvent::Contact {
            down: false,
            time_ns,
        }),
        _ => None,
    }
}

/// Maps a `WM_POINTERUPDATE` message plus its `POINTER_PEN_INFO` payload to
/// a [`TabletEvent::Motion`]. Tilt is only reported when the device marks
/// both tilt axes valid in `penMask` — untilt-capable digitizers leave
/// `tiltX`/`tiltY` at a meaningless zero.
fn map_pen_message(msg: u32, pen: &POINTER_PEN_INFO, time_ns: u64) -> Option<TabletEvent> {
    if msg != WM_POINTERUPDATE {
        return None;
    }

    let tilt_valid = PEN_MASK_TILT_X | PEN_MASK_TILT_Y;
    let tilt =
        (pen.penMask & tilt_valid == tilt_valid).then(|| [pen.tiltX as f32, pen.tiltY as f32]);

    Some(TabletEvent::Motion {
        pos: [
            pen.pointerInfo.ptPixelLocation.x as f32,
            pen.pointerInfo.ptPixelLocation.y as f32,
        ],
        pressure: normalize_pressure(pen.pressure),
        tilt,
        time_ns,
    })
}

/// Normalizes Windows Ink's `0..=32767` pen pressure range to `0.0..=1.0`.
/// Clamped defensively — hardware is not expected to report outside the
/// documented range, but a malformed sample must not produce an
/// out-of-range [`TabletEvent::Motion::pressure`].
fn normalize_pressure(pressure: u32) -> f32 {
    (pressure as f32 / MAX_PRESSURE).clamp(0.0, 1.0)
}

#[cfg(all(windows, feature = "winink", test))]
mod tests {
    use super::*;
    use windows::Win32::Foundation::POINT;

    fn pen_info(pressure: u32, tilt: Option<(i32, i32)>, pos: (i32, i32)) -> POINTER_PEN_INFO {
        let mut pen = POINTER_PEN_INFO::default();
        pen.pointerInfo.ptPixelLocation = POINT { x: pos.0, y: pos.1 };
        pen.pressure = pressure;
        if let Some((tilt_x, tilt_y)) = tilt {
            pen.penMask = PEN_MASK_TILT_X | PEN_MASK_TILT_Y;
            pen.tiltX = tilt_x;
            pen.tiltY = tilt_y;
        }
        pen
    }

    #[test]
    fn pressure_normalizes_into_unit_range() {
        assert_eq!(normalize_pressure(0), 0.0);
        assert_eq!(normalize_pressure(32767), 1.0);
        assert!((normalize_pressure(16384) - 0.500_015_26).abs() < 1e-6);
    }

    #[test]
    fn pressure_clamps_beyond_the_documented_range() {
        assert_eq!(normalize_pressure(u32::MAX), 1.0);
    }

    #[test]
    fn update_maps_to_motion_with_tilt() {
        let pen = pen_info(16384, Some((10, -5)), (100, 200));
        let event = map_pen_message(WM_POINTERUPDATE, &pen, 42).unwrap();
        match event {
            TabletEvent::Motion {
                pos,
                pressure,
                tilt,
                time_ns,
            } => {
                assert_eq!(pos, [100.0, 200.0]);
                assert!((pressure - 0.500_015_26).abs() < 1e-6);
                assert_eq!(tilt, Some([10.0, -5.0]));
                assert_eq!(time_ns, 42);
            }
            other => panic!("expected Motion, got {other:?}"),
        }
    }

    #[test]
    fn update_without_tilt_mask_reports_no_tilt() {
        let pen = pen_info(0, None, (0, 0));
        let event = map_pen_message(WM_POINTERUPDATE, &pen, 0).unwrap();
        assert!(matches!(event, TabletEvent::Motion { tilt: None, .. }));
    }

    #[test]
    fn non_update_message_has_no_pen_mapping() {
        let pen = pen_info(0, None, (0, 0));
        assert_eq!(map_pen_message(WM_POINTERDOWN, &pen, 0), None);
    }

    #[test]
    fn simple_messages_map_to_proximity_and_contact() {
        assert_eq!(
            map_simple_message(WM_POINTERENTER, 1),
            Some(TabletEvent::ProximityIn { time_ns: 1 })
        );
        assert_eq!(
            map_simple_message(WM_POINTERLEAVE, 2),
            Some(TabletEvent::ProximityOut { time_ns: 2 })
        );
        assert_eq!(
            map_simple_message(WM_POINTERDOWN, 3),
            Some(TabletEvent::Contact {
                down: true,
                time_ns: 3
            })
        );
        assert_eq!(
            map_simple_message(WM_POINTERUP, 4),
            Some(TabletEvent::Contact {
                down: false,
                time_ns: 4
            })
        );
        assert_eq!(map_simple_message(WM_POINTERUPDATE, 5), None);
    }

    #[test]
    fn time_ns_tracks_a_monotonic_epoch_from_the_first_sample() {
        let mut state = PenState::new();
        assert_eq!(state.time_ns(1_000), 0);
        assert_eq!(state.time_ns(1_010), 10_000_000);
    }

    #[test]
    fn time_ns_handles_dw_time_wraparound() {
        let mut state = PenState::new();
        assert_eq!(state.time_ns(u32::MAX - 4), 0);
        assert_eq!(state.time_ns(5), 10_000_000);
    }
}

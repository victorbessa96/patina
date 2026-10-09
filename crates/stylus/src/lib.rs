//! stylus — the owned stylus/tablet input crate.
//!
//! The must-own layer (docs/research/03 §2): winit has no X11 pen events,
//! no Wintab, and its 0.31 pen events are still beta. Umber therefore owns
//! a dual-path input layer:
//!   - Windows: Windows Ink (WM_POINTER) AND Wintab32 polling (runtime
//!     switch; external tablets commonly run Wintab).
//!   - Linux: Wayland zwp_tablet_unstable_v2 + X11 XI2 valuator axes.
//!
//! Wave 1 scope: the backend-agnostic event vocabulary + platform
//! dispatch skeleton. Concrete backend code is feature-gated and lands
//! incrementally (Windows Ink first per SPEC Wave 1).

use std::fmt;

/// A raw tablet event, normalized across backends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TabletEvent {
    /// Pen entered proximity.
    ProximityIn { time_ns: u64 },
    /// Pen left proximity.
    ProximityOut { time_ns: u64 },
    /// Position/pressure/tilt update.
    Motion {
        pos: [f32; 2],
        pressure: f32,
        tilt: Option<[f32; 2]>,
        time_ns: u64,
    },
    /// Tip contact began or ended.
    Contact { down: bool, time_ns: u64 },
    /// Barrel button state changed.
    Button {
        button: u8,
        down: bool,
        time_ns: u64,
    },
}

/// Which platform input path produced an event (diagnostics + tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSource {
    WindowsInk,
    Wintab,
    WaylandTablet,
    X11Xinput2,
    MouseFallback,
}

impl fmt::Display for InputSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            InputSource::WindowsInk => "windows-ink",
            InputSource::Wintab => "wintab",
            InputSource::WaylandTablet => "wayland-tablet",
            InputSource::X11Xinput2 => "x11-xinput2",
            InputSource::MouseFallback => "mouse-fallback",
        };
        f.write_str(s)
    }
}

/// Events the app loop receives, tagged with their source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourcedEvent {
    pub source: InputSource,
    pub event: TabletEvent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_vocabulary_round_trips() {
        let e = SourcedEvent {
            source: InputSource::WindowsInk,
            event: TabletEvent::Motion {
                pos: [10.0, 20.0],
                pressure: 0.8,
                tilt: None,
                time_ns: 1,
            },
        };
        assert_eq!(e.source.to_string(), "windows-ink");
        assert!(
            matches!(e.event, TabletEvent::Motion { pressure, .. } if (pressure - 0.8).abs() < 1e-6)
        );
    }
}

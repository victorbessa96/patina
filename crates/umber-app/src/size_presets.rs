//! The square texture-size presets the Bakes panel and the Export dialog
//! both offer (requirements §6, the P2 "8K" row): one list so the two
//! pickers can never disagree, plus the combo-row labels.
//!
//! The 8K row carries its VRAM cost on the label — an 8192² RGBA8 texel
//! buffer is 256 MiB, and a bake holds several at once (see
//! [`rgba8_bytes`]) — so picking it is an informed choice, not a surprise.
//!
//! 8K audit (what each size path does at 8192):
//! - Export writers (PNG 8/16, EXR, TIFF, JPEG) and the driver take
//!   explicit `u32` sizes and compute byte counts in `usize`: no cap, no
//!   truncation. `u32` texel math holds at 8K (worst case `w*h*16` for the
//!   f32 maps is 2^30); it first overflows at 16K.
//! - GPU textures: `wgpu::Limits::default()` (what every umber device
//!   requests) allows `max_texture_dimension_2d` = 8192 exactly; all bake
//!   dispatches are 2D (`width, height` or `div_ceil(8)`), well under the
//!   65535-per-dimension limit.
//! - RGBA8 readbacks: an 8K target's padded readback is exactly 256 MiB,
//!   equal to (not over) the default `max_buffer_size` — pinned by test.
//! - f32 readbacks (position / world-normal maps) were the one GPU cap:
//!   1 GiB at 8K. `umber_bake::position` now reads them back in bands.
//! - The Bakes panel's resolution list stopped at 2048, and the Export
//!   dialog had no size picker (hard-wired 512): both now offer
//!   [`SIZE_PRESETS`].

/// The offered square sizes, ascending: 512 (the default) through 8K.
pub const SIZE_PRESETS: [u32; 5] = [512, 1024, 2048, 4096, 8192];

/// The default size both pickers start on (fast bakes, fast startup).
pub const DEFAULT_SIZE: u32 = 512;

/// Sizes at or above this carry the VRAM estimate on their combo row.
const VRAM_NOTE_FROM: u32 = 8192;

/// Bytes in one `size`×`size` RGBA8 texel buffer (`w*h*4`) — the unit
/// every GPU texture, readback, and CPU map at that size is measured in.
///
/// At 8K that is 256 MiB per buffer. The AO bake holds two of them on the
/// GPU (AO + bent targets) plus the 256 MiB readback staging buffer, and
/// its position pass adds two `Rgba32Float` textures at 4× that (1 GiB
/// each): ~2.75 GiB peak, inside an 8 GB card.
pub fn rgba8_bytes(size: u32) -> u64 {
    u64::from(size) * u64::from(size) * 4
}

/// Whether `size` is one of [`SIZE_PRESETS`].
pub fn is_preset(size: u32) -> bool {
    SIZE_PRESETS.contains(&size)
}

/// The combo row / selected-text label for `size`: `"2048 × 2048"`, with
/// the per-buffer VRAM estimate appended on the 8K row
/// (`"8192 × 8192 (~256 MB/texel-buffer)"`).
pub fn preset_label(size: u32) -> String {
    if size >= VRAM_NOTE_FROM {
        let mib = rgba8_bytes(size) >> 20;
        format!("{size} × {size} (~{mib} MB/texel-buffer)")
    } else {
        format!("{size} × {size}")
    }
}

/// Draws the size combo for `current`, offering exactly [`SIZE_PRESETS`].
pub fn size_combo(ui: &mut egui::Ui, label: &str, current: &mut u32) {
    egui::ComboBox::from_label(label)
        .selected_text(preset_label(*current))
        .show_ui(ui, |ui| {
            for candidate in SIZE_PRESETS {
                ui.selectable_value(current, candidate, preset_label(candidate));
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_exactly_512_through_8k() {
        assert_eq!(SIZE_PRESETS, [512, 1024, 2048, 4096, 8192]);
        assert!(SIZE_PRESETS.windows(2).all(|w| w[1] == w[0] * 2));
        assert!(is_preset(DEFAULT_SIZE));
        for rejected in [0, 128, 256, 513, 8193, 16384, u32::MAX] {
            assert!(!is_preset(rejected), "{rejected}");
        }
    }

    #[test]
    fn only_the_8k_row_carries_the_vram_estimate() {
        assert_eq!(preset_label(8192), "8192 × 8192 (~256 MB/texel-buffer)");
        assert_eq!(preset_label(4096), "4096 × 4096");
        assert_eq!(preset_label(512), "512 × 512");
    }

    #[test]
    fn rgba8_byte_counts_are_exact() {
        assert_eq!(rgba8_bytes(512), 1 << 20);
        assert_eq!(rgba8_bytes(4096), 64 << 20);
        assert_eq!(rgba8_bytes(8192), 268_435_456);
        assert_eq!(rgba8_bytes(8192), 256 << 20);
        // No u32 truncation on the way: the 8K count fits u32 too, but
        // the helper must not wrap one past the 16K edge either.
        assert_eq!(rgba8_bytes(65_536), 1 << 34);
    }

    #[test]
    fn eight_k_rgba8_readback_fits_the_default_buffer_limit_exactly() {
        // PaintTarget::read_back_rgba8 / BakeTarget stage the whole target
        // in one buffer: 8192 * 4 = 32768-byte rows (already 256-aligned),
        // so the staging buffer is w*h*4 with no padding — and that equals
        // wgpu's default max_buffer_size (wgpu rejects only sizes OVER it).
        let padded_row = 8192_u64 * 4;
        assert_eq!(padded_row % 256, 0, "no row padding at 8K");
        assert_eq!(padded_row * 8192, rgba8_bytes(8192));
        assert_eq!(
            rgba8_bytes(8192),
            wgpu::Limits::default().max_buffer_size,
            "8K RGBA8 staging is exactly the default limit"
        );
        // ...and 8192 is exactly the default texture-dimension ceiling.
        assert_eq!(wgpu::Limits::default().max_texture_dimension_2d, 8192);
    }
}

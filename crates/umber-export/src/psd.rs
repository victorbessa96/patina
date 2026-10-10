//! PSD (Photoshop) export — one layered document per map set.
//!
//! The honest v1 per the design: a set-level writer (one `.psd`
//! file holding every map as a named layer) using [`ag_psd`]'s
//! `write_psd`/`read_psd` round-trip — the §6 P2 row. Layer
//! order = the set's map order; the composite is the first map
//! (the base color when present) so single-image viewers that
//! only show the flattened composite still see something sane.
//!
//! The PSD is a document format, unlike the per-file outputs
//! (PNG/EXR/TIFF/JPEG) the preset engine dispatches — so this
//! lives outside [`crate::driver::write_outputs`] as a set-level
//! entry point called when a preset's `psd` toggle is on.

use crate::driver::MapSet;
use crate::presets::MapKind;
use ag_psd::psd::{ColorMode, PixelData, Psd, WriteOptions};
use ag_psd::{read_psd, write_psd};

/// Writes `set` as one layered PSD at `path`. Every map becomes
/// a layer named by its [`MapKind::token`]; the composite image
/// data is the first map in the set (base color when present).
///
/// Returns the bytes written for the caller's reporting.
pub fn write_psd_maps(set: &MapSet, path: &std::path::Path) -> Result<usize, String> {
    let size = set.size();
    let layers: Vec<(MapKind, Vec<u8>)> = set
        .maps_with_bytes()
        .map(|(k, b)| (k, b.to_vec()))
        .collect();
    if layers.is_empty() {
        return Err("no maps in the set".into());
    }
    let mut psd = Psd {
        width: f64::from(size),
        height: f64::from(size),
        color_mode: Some(ColorMode::Rgb),
        ..Default::default()
    };
    let children = layers
        .iter()
        .map(|(kind, rgba8)| {
            let (top, left, bottom, right) = (0.0, 0.0, f64::from(size), f64::from(size));
            ag_psd::psd::Layer {
                additional_info: ag_psd::psd::LayerAdditionalInfo {
                    name: Some(kind.token().to_string()),
                    ..Default::default()
                },
                top: Some(top),
                left: Some(left),
                bottom: Some(bottom),
                right: Some(right),
                canvas: Some(PixelData {
                    width: size,
                    height: size,
                    data: rgba8.clone(),
                }),
                ..Default::default()
            }
        })
        .collect();
    psd.children = Some(children);
    // The composite: the first map (base color when the set carries
    // one). Both fields set — the writer reads image_data first
    // (writer.rs:822), the reader lands the composite in canvas
    // (reader.rs:1126); setting both keeps the round trip honest.
    let (_, first) = &layers[0];
    let composite = PixelData {
        width: size,
        height: size,
        data: first.clone(),
    };
    psd.image_data = Some(composite.clone());
    psd.canvas = Some(composite);
    let bytes = write_psd(&psd, &WriteOptions::default());
    std::fs::write(path, &bytes).map_err(|e| e.to_string())?;
    Ok(bytes.len())
}

/// Reads a PSD back — the test path + any future import round
/// trip. Public for the tests; not yet an import feature.
pub fn read_back(path: &std::path::Path) -> Result<Psd, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    read_psd(&bytes, &ag_psd::psd::ReadOptions::default()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(size: u32, value: u8) -> Vec<u8> {
        // Opaque RGBA8: solid RGB + alpha 255 — the real map sets'
        // shape. NOT alpha=value: ag-psd's writer premultiplies
        // partial alpha toward white ("the weird white matte",
        // writer.rs:992-1005), so an alpha=200 fixture would shift
        // RGB on write and the byte-compare would fail on the
        // composite — correctly, because the matte is lossy for
        // non-opaque composites. Map sets are opaque; the fixture
        // matches the real data.
        let mut buf = vec![0u8; (size * size * 4) as usize];
        for px in buf.chunks_exact_mut(4) {
            px[0] = value;
            px[1] = value;
            px[2] = value;
            px[3] = 255;
        }
        buf
    }

    #[test]
    fn round_trip_layers_and_composite() {
        let size = 8;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 200));
        set.set(MapKind::Roughness, solid(size, 64));
        let dir = std::env::temp_dir().join("umber-psd-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.psd");
        let written = write_psd_maps(&set, &path).unwrap();
        assert!(written > 0);

        let psd = read_back(&path).unwrap();
        assert_eq!(psd.width, size as f64);
        assert_eq!(psd.height, size as f64);
        let children = psd.children.as_ref().unwrap();
        assert_eq!(children.len(), 2);
        // Layer names == the map tokens, in insertion order.
        let names: Vec<&str> = children
            .iter()
            .map(|l| l.additional_info.name.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(names, ["baseColor", "roughness"]);
        // Layer pixel data round-trips byte-identically: layer 0
        // solid RGB 200 (baseColor), layer 1 solid RGB 64
        // (roughness); both opaque (alpha 255 — the writer skips
        // the Transparency channel for opaque background layers,
        // the reader restores alpha 255).
        for (layer, value) in children.iter().zip([200u8, 64]) {
            let data = layer.canvas.as_ref().expect("canvas round-trips");
            assert_eq!(data.width, size);
            assert_eq!(data.height, size);
            for px in data.data.chunks_exact(4) {
                assert_eq!(px[0], value);
                assert_eq!(px[1], value);
                assert_eq!(px[2], value);
                assert_eq!(px[3], 255);
            }
        }
        // The composite = the first map (baseColor, solid RGB 200,
        // opaque). The reader lands it in psd.canvas (reader.rs:1126).
        let composite = psd.canvas.as_ref().expect("composite lands in canvas");
        for px in composite.data.chunks_exact(4) {
            assert_eq!(px[0], 200);
            assert_eq!(px[1], 200);
            assert_eq!(px[2], 200);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn empty_set_is_an_error() {
        let set = MapSet::new(8);
        let path = std::env::temp_dir().join("umber-psd-test/empty.psd");
        assert!(write_psd_maps(&set, &path).is_err());
    }

    #[test]
    fn registry_psd_export_produces_file() {
        // The format-registry integration shape: a set exported
        // via the writer lands as a .psd that exists + nonzero.
        let size = 4;
        let mut set = MapSet::new(size);
        set.set(MapKind::BaseColor, solid(size, 128));
        let dir = std::env::temp_dir().join("umber-psd-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("registry.psd");
        let n = write_psd_maps(&set, &path).unwrap();
        assert!(n > 0 && path.exists());
        let _ = std::fs::remove_file(&path);
    }
}

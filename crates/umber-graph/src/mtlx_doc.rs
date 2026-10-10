//! The texture-set MaterialX document (`docs/specs/mtlx-export-design.md`,
//! the §6 P0 row): one `<surfacematerial>` + one OpenPBR Surface shader
//! whose inputs read the exported texture files through `<image>` nodes.
//!
//! [`to_mtlx_document`] is pure string building — the export driver
//! (`umber-export`) resolves the filenames and calls it. Unlike
//! [`crate::mtlx::to_mtlx`] (painter graphs, `<node type=…>` + `N{id}`
//! names) this emits the STANDARD MaterialX element shape: elements named
//! by node category (`<image>`, `<extract>`, `<normalmap>`,
//! `<open_pbr_surface>`, `<surfacematerial>`), wired with `nodename=`.
//! Mirrored from `to_mtlx`: the header, the `<materialx version="1.38">`
//! root, 2-space indentation, attribute escaping and float formatting.
//! [`crate::mtlx::from_mtlx`] skips every element here (none is a
//! `<node>`), so it parses a document to an EMPTY graph — the structural
//! checks in this module's tests walk the same reader's element tree.
//!
//! # The OpenPBR inputs (cited against the vendored nodedef)
//!
//! `docs/claw-artifacts/openpbr/open_pbr_surface.mtlx`,
//! `ND_open_pbr_surface_surfaceshader` (L6–91):
//!
//! | map | input | type | line |
//! |---|---|---|---|
//! | base color | `base_color` | `color3` | L10 |
//! | metallic | `base_metalness` | `float` | L14 |
//! | roughness | `specular_roughness` | `float` | L20 |
//! | emissive | `emission_color` | `color3` | L76 |
//! | opacity | `geometry_opacity` | `float` | L78 |
//! | normal | `geometry_normal` | `vector3` | L82 |
//!
//! Ambient occlusion and height have NO input on the nodedef (no
//! occlusion/displacement/height parameter anywhere in it): they are not
//! wired — never an invented input name. The tangent-space normal map goes
//! through a stdlib `<normalmap>` node into `geometry_normal`; `normalmap`
//! is NOT in the vendored file (it is MaterialX stdlib, `in: vector3`).
//!
//! The untextured value inputs ([`SurfaceParams`]) are the app's
//! real-time OpenPBR subset (`umber-gpu/src/material.rs`): base_weight L8,
//! base_color L10, base_metalness L14, specular_weight L16,
//! specular_color L18, specular_roughness L20, specular_ior L22,
//! coat_weight L56, coat_color L58, coat_roughness L60, coat_ior L64,
//! emission_luminance L74, emission_color L76, geometry_opacity L78.
//!
//! # Honest scope
//!
//! Full OpenPBR fidelity is NOT claimed — the document references the
//! exported maps; consuming DCCs evaluate what they support. The root
//! says `version="1.38"` (the existing writer's), while the vendored
//! nodedef is a 1.39 document (its L2) — `open_pbr_surface` ships in the
//! 1.39 stdlib, and 1.39 loaders upgrade 1.38 documents on read.

use crate::mtlx::{esc, fmt_float, fmt_vec};

/// The MaterialX filename token a UDIM-tiled texture set's `file`
/// strings carry (MaterialX spec, "Filename Substitutions"): the
/// consuming renderer resolves it per UV tile. The export driver expands
/// its own `$udim` token to this for tiled runs.
pub const UDIM_TOKEN: &str = "<UDIM>";

/// The texture maps a set can carry — mirrors `umber_export::MapKind`
/// variant-for-variant (this crate is export-free; the driver converts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureMap {
    /// Base color (color-managed).
    BaseColor,
    /// Roughness (data).
    Roughness,
    /// Metallic (data).
    Metallic,
    /// Ambient occlusion (data) — no OpenPBR input; never wired.
    AmbientOcclusion,
    /// Tangent-space normal (data).
    Normal,
    /// Height (data) — no OpenPBR input; never wired.
    Height,
    /// Opacity (data).
    Opacity,
    /// Emissive (color-managed).
    Emissive,
}

impl TextureMap {
    /// The OpenPBR input `(name, type)` this map feeds — the cited
    /// nodedef lines are in the module docs. `None` for maps the nodedef
    /// has no input for (AO, height).
    pub fn openpbr_input(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::BaseColor => Some(("base_color", "color3")),
            Self::Metallic => Some(("base_metalness", "float")),
            Self::Roughness => Some(("specular_roughness", "float")),
            Self::Emissive => Some(("emission_color", "color3")),
            Self::Opacity => Some(("geometry_opacity", "float")),
            Self::Normal => Some(("geometry_normal", "vector3")),
            Self::AmbientOcclusion | Self::Height => None,
        }
    }

    /// The §9 rule ("baseColor yes; roughness/metallic/normal = data"):
    /// base color + emissive are sRGB, everything else linear. The export
    /// driver passes the colorspace of the bytes it ACTUALLY wrote (a
    /// format can override the rule); this is the rule itself.
    pub fn default_colorspace(self) -> MtlxColorSpace {
        match self {
            Self::BaseColor | Self::Emissive => MtlxColorSpace::SrgbTexture,
            _ => MtlxColorSpace::LinRec709,
        }
    }
}

/// The `colorspace` an image file is tagged with (MaterialX's standard
/// colorspace names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtlxColorSpace {
    /// `srgb_texture` — sRGB-encoded color.
    SrgbTexture,
    /// `lin_rec709` — linear (data maps; MaterialX applies no transform
    /// to non-color image types either way).
    LinRec709,
}

impl MtlxColorSpace {
    /// The MaterialX colorspace name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SrgbTexture => "srgb_texture",
            Self::LinRec709 => "lin_rec709",
        }
    }
}

/// One exported texture file the document references.
#[derive(Debug, Clone, PartialEq)]
pub struct MtlxTexture {
    /// Which map the file carries (picks the OpenPBR input).
    pub map: TextureMap,
    /// The file string, relative to the `.mtlx` (may carry [`UDIM_TOKEN`]).
    pub file: String,
    /// The colorspace of the file's bytes.
    pub colorspace: MtlxColorSpace,
    /// `None`: the file's own RGB(A) is the map. `Some(i)`: the map is
    /// channel `i` (0..=3) of a packed file (ORM, metallicRoughness) —
    /// read through an `<extract>`. Only scalar inputs can be extracted.
    pub channel: Option<u8>,
}

/// The non-textured OpenPBR values the document carries — exactly the
/// app's real-time subset (`umber_gpu::material::OpenPbrParams`), field
/// names = the nodedef input names. [`Default`] = the nodedef's own
/// defaults (L8–78), which `OpenPbrParams::default()` also matches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceParams {
    pub base_weight: f32,
    pub base_color: [f32; 3],
    pub base_metalness: f32,
    pub specular_weight: f32,
    pub specular_color: [f32; 3],
    pub specular_roughness: f32,
    pub specular_ior: f32,
    pub coat_weight: f32,
    pub coat_color: [f32; 3],
    pub coat_roughness: f32,
    pub coat_ior: f32,
    pub emission_luminance: f32,
    pub emission_color: [f32; 3],
    pub geometry_opacity: f32,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            base_color: [0.8, 0.8, 0.8],
            base_metalness: 0.0,
            specular_weight: 1.0,
            specular_color: [1.0, 1.0, 1.0],
            specular_roughness: 0.3,
            specular_ior: 1.5,
            coat_weight: 0.0,
            coat_color: [1.0, 1.0, 1.0],
            coat_roughness: 0.0,
            coat_ior: 1.6,
            emission_luminance: 0.0,
            emission_color: [1.0, 1.0, 1.0],
            geometry_opacity: 1.0,
        }
    }
}

impl SurfaceParams {
    /// Unpacks the six `vec4` slots of `umber_gpu::material::OpenPbrParams`
    /// (`base`, `surface`, `specular`, `coat`, `emission`, `iors` — that
    /// struct's documented slot layout; this crate is GPU-free, so the
    /// caller passes the fields).
    ///
    /// The GPU struct premultiplies emission (`luminance × color`); the
    /// spec's HDR emission is `emission_color * emission_luminance`
    /// (openpbr-spec-v1.1.1.md L903), so the product is split as
    /// luminance = the largest component, color = rgb / luminance (zero
    /// emission → luminance 0, color the default white). The viewport's
    /// emission units are not calibrated nits — the product survives,
    /// the absolute scale is the app's.
    pub fn from_gpu_slots(
        base: [f32; 4],
        surface: [f32; 4],
        specular: [f32; 4],
        coat: [f32; 4],
        emission: [f32; 4],
        iors: [f32; 4],
    ) -> Self {
        let e = [emission[0], emission[1], emission[2]];
        let lum = e[0].max(e[1]).max(e[2]);
        let (emission_luminance, emission_color) = if lum > 0.0 {
            (lum, [e[0] / lum, e[1] / lum, e[2] / lum])
        } else {
            (0.0, [1.0, 1.0, 1.0])
        };
        Self {
            base_weight: base[3],
            base_color: [base[0], base[1], base[2]],
            base_metalness: surface[0],
            specular_weight: specular[3],
            specular_color: [specular[0], specular[1], specular[2]],
            specular_roughness: surface[1],
            specular_ior: iors[0],
            coat_weight: coat[3],
            coat_color: [coat[0], coat[1], coat[2]],
            coat_roughness: surface[2],
            coat_ior: iors[1],
            emission_luminance,
            emission_color,
            geometry_opacity: surface[3],
        }
    }
}

/// A valid MaterialX element name from a texture-set name: anything
/// outside `[A-Za-z0-9_]` becomes `_`; an empty or digit-led name gets an
/// `M_` prefix.
fn element_name(set: &str) -> String {
    let mut s: String = set
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert_str(0, "M_");
    }
    s
}

/// Serialize the texture set `name`'s MaterialX document: an `<image>`
/// per wired texture (+ `<extract>` for packed channels, `<normalmap>`
/// for the normal), one `<open_pbr_surface>` and one `<surfacematerial>`
/// named after the set.
///
/// Wiring rules: textures are taken in order and the FIRST one per
/// OpenPBR input wins; maps without an input (AO, height) and
/// `channel: Some(_)` on a non-scalar input are skipped. Every untextured
/// input of [`SurfaceParams`] is emitted as a value, in nodedef order.
/// A textured `emission_color` with `emission_luminance == 0` emits
/// luminance 1 (a painted emissive map would otherwise contribute
/// nothing). Deterministic: no timestamps, no absolute paths.
pub fn to_mtlx_document(name: &str, params: &SurfaceParams, textures: &[MtlxTexture]) -> String {
    let set = element_name(name);
    let mut wired: Vec<(&'static str, &'static str, &MtlxTexture)> = Vec::new();
    for tex in textures {
        let Some((input, ty)) = tex.map.openpbr_input() else {
            continue;
        };
        if wired.iter().any(|(i, _, _)| *i == input) {
            continue;
        }
        if tex.channel.is_some_and(|c| ty != "float" || c > 3) {
            continue;
        }
        wired.push((input, ty, tex));
    }

    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<materialx version=\"1.38\">\n");

    // Upstream nodes first, one chain per wired input (in wiring order);
    // `feeds` records the node each shader input connects to.
    let mut feeds: Vec<(&'static str, &'static str, String)> = Vec::new();
    for &(input, ty, tex) in &wired {
        let image = format!("{set}_{input}_image");
        let image_ty = match tex.channel {
            Some(3) => "color4",
            Some(_) => "color3",
            None => ty,
        };
        out.push_str(&format!(
            "  <image name=\"{}\" type=\"{}\">\n",
            esc(&image),
            image_ty
        ));
        out.push_str(&format!(
            "    <input name=\"file\" type=\"filename\" value=\"{}\" colorspace=\"{}\" />\n",
            esc(&tex.file),
            tex.colorspace.as_str()
        ));
        out.push_str("  </image>\n");
        let feed = if let Some(channel) = tex.channel {
            let extract = format!("{set}_{input}_extract");
            out.push_str(&format!(
                "  <extract name=\"{}\" type=\"float\">\n",
                esc(&extract)
            ));
            out.push_str(&format!(
                "    <input name=\"in\" type=\"{}\" nodename=\"{}\" />\n",
                image_ty,
                esc(&image)
            ));
            out.push_str(&format!(
                "    <input name=\"index\" type=\"integer\" value=\"{channel}\" />\n"
            ));
            out.push_str("  </extract>\n");
            extract
        } else if tex.map == TextureMap::Normal {
            let normalmap = format!("{set}_{input}_normalmap");
            out.push_str(&format!(
                "  <normalmap name=\"{}\" type=\"vector3\">\n",
                esc(&normalmap)
            ));
            out.push_str(&format!(
                "    <input name=\"in\" type=\"vector3\" nodename=\"{}\" />\n",
                esc(&image)
            ));
            out.push_str("  </normalmap>\n");
            normalmap
        } else {
            image
        };
        feeds.push((input, ty, feed));
    }

    let shader = format!("{set}_shader");
    out.push_str(&format!(
        "  <open_pbr_surface name=\"{}\" type=\"surfaceshader\">\n",
        esc(&shader)
    ));
    for (input, ty, feed) in &feeds {
        out.push_str(&format!(
            "    <input name=\"{input}\" type=\"{ty}\" nodename=\"{}\" />\n",
            esc(feed)
        ));
    }
    let p = params;
    let emission_textured = feeds.iter().any(|(i, _, _)| *i == "emission_color");
    let emission_luminance = if emission_textured && p.emission_luminance == 0.0 {
        1.0
    } else {
        p.emission_luminance
    };
    let values: [(&str, Value); 14] = [
        ("base_weight", Value::F(p.base_weight)),
        ("base_color", Value::C(p.base_color)),
        ("base_metalness", Value::F(p.base_metalness)),
        ("specular_weight", Value::F(p.specular_weight)),
        ("specular_color", Value::C(p.specular_color)),
        ("specular_roughness", Value::F(p.specular_roughness)),
        ("specular_ior", Value::F(p.specular_ior)),
        ("coat_weight", Value::F(p.coat_weight)),
        ("coat_color", Value::C(p.coat_color)),
        ("coat_roughness", Value::F(p.coat_roughness)),
        ("coat_ior", Value::F(p.coat_ior)),
        ("emission_luminance", Value::F(emission_luminance)),
        ("emission_color", Value::C(p.emission_color)),
        ("geometry_opacity", Value::F(p.geometry_opacity)),
    ];
    for (input, value) in values {
        if feeds.iter().any(|(i, _, _)| *i == input) {
            continue; // the texture wins over the constant.
        }
        let (ty, v) = match value {
            Value::F(f) => ("float", fmt_float(f)),
            Value::C(c) => ("color3", fmt_vec(&c)),
        };
        out.push_str(&format!(
            "    <input name=\"{input}\" type=\"{ty}\" value=\"{}\" />\n",
            esc(&v)
        ));
    }
    out.push_str("  </open_pbr_surface>\n");

    out.push_str(&format!(
        "  <surfacematerial name=\"{}\" type=\"material\">\n",
        esc(&set)
    ));
    out.push_str(&format!(
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"{}\" />\n",
        esc(&shader)
    ));
    out.push_str("  </surfacematerial>\n");
    out.push_str("</materialx>\n");
    out
}

enum Value {
    F(f32),
    C([f32; 3]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtlx::{from_mtlx, parse_doc, Elem};

    fn tex(map: TextureMap, file: &str, channel: Option<u8>) -> MtlxTexture {
        MtlxTexture {
            map,
            file: file.into(),
            colorspace: map.default_colorspace(),
            channel,
        }
    }

    /// Every map kind, one packed (glTF-style metallicRoughness) file.
    fn full_set() -> Vec<MtlxTexture> {
        vec![
            tex(TextureMap::BaseColor, "Sword_baseColor.png", None),
            tex(TextureMap::Metallic, "Sword_metallicRoughness.png", Some(1)),
            tex(
                TextureMap::Roughness,
                "Sword_metallicRoughness.png",
                Some(2),
            ),
            tex(TextureMap::Normal, "Sword_normal.png", None),
            tex(TextureMap::Opacity, "Sword_opacity.png", None),
            tex(TextureMap::Emissive, "Sword_emissive.png", None),
            tex(TextureMap::AmbientOcclusion, "Sword_ao.png", None),
            tex(TextureMap::Height, "Sword_height.png", None),
        ]
    }

    fn children<'a>(e: &'a Elem, name: &str) -> Vec<&'a Elem> {
        e.children.iter().filter(|c| c.name == name).collect()
    }

    fn by_name<'a>(root: &'a Elem, name: &str) -> &'a Elem {
        root.children
            .iter()
            .find(|c| c.attr("name") == Some(name))
            .unwrap_or_else(|| panic!("no element named {name}"))
    }

    fn input<'a>(e: &'a Elem, name: &str) -> &'a Elem {
        children(e, "input")
            .into_iter()
            .find(|i| i.attr("name") == Some(name))
            .unwrap_or_else(|| panic!("no input {name}"))
    }

    /// The image node an input of `shader` ultimately reads (through an
    /// extract / normalmap when present).
    fn image_feeding<'a>(root: &'a Elem, shader: &Elem, input_name: &str) -> &'a Elem {
        let mut node = by_name(root, input(shader, input_name).attr("nodename").unwrap());
        while node.name != "image" {
            node = by_name(root, input(node, "in").attr("nodename").unwrap());
        }
        node
    }

    fn vendored_nodedef() -> String {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/claw-artifacts/openpbr/open_pbr_surface.mtlx"
        );
        let text = std::fs::read_to_string(path).expect("vendored OpenPBR nodedef");
        let end = text
            .find("</nodedef>")
            .expect("the first nodedef closes (L91)");
        text[..end].to_string()
    }

    // (a) The round-trip through the existing reader.
    #[test]
    fn from_mtlx_parses_the_document_and_the_tree_is_wired() {
        let doc = to_mtlx_document("Sword", &SurfaceParams::default(), &full_set());
        // The graph reader accepts the document; it carries no painter
        // `<node>` elements, so the graph is empty (unknown elements are
        // skipped — mtlx.rs's parse rules), with no warnings.
        let (graph, nodedefs, warnings) = from_mtlx(&doc).expect("from_mtlx parses it");
        assert!(graph.nodes.is_empty() && graph.edges.is_empty());
        assert!(nodedefs.is_empty() && warnings.is_empty());

        // The same reader's element tree: the material → shader → image
        // chain is closed and typed.
        let root = parse_doc(&doc).expect("parses");
        assert_eq!(root.attr("version"), Some("1.38"));
        let materials = children(&root, "surfacematerial");
        assert_eq!(materials.len(), 1);
        assert_eq!(materials[0].attr("name"), Some("Sword"));
        assert_eq!(materials[0].attr("type"), Some("material"));
        let ss = input(materials[0], "surfaceshader");
        assert_eq!(ss.attr("type"), Some("surfaceshader"));
        let shader = by_name(&root, ss.attr("nodename").unwrap());
        assert_eq!(shader.name, "open_pbr_surface");
        assert_eq!(shader.attr("type"), Some("surfaceshader"));
        // 6 wired maps (AO + height have no input) → 6 images, 2
        // extracts (the packed file), 1 normalmap.
        assert_eq!(children(&root, "image").len(), 6);
        assert_eq!(children(&root, "extract").len(), 2);
        assert_eq!(children(&root, "normalmap").len(), 1);
        // Every connection resolves to an element of the declared type.
        for e in &root.children {
            for inp in children(e, "input") {
                if let Some(target) = inp.attr("nodename") {
                    let t = by_name(&root, target);
                    assert_eq!(t.attr("type"), inp.attr("type"), "{target} type");
                }
            }
        }
        // 15 inputs on the shader: 6 connections + the 14 values minus
        // the 5 textured ones (geometry_normal has no value form).
        assert_eq!(children(shader, "input").len(), 15);
    }

    // (b) Every input name/type cited exists in the vendored nodedef.
    #[test]
    fn every_cited_input_matches_the_vendored_nodedef() {
        let nodedef = vendored_nodedef();
        assert!(nodedef.contains(
            "<nodedef name=\"ND_open_pbr_surface_surfaceshader\" node=\"open_pbr_surface\""
        ));
        assert!(nodedef.contains("<output name=\"out\" type=\"surfaceshader\" />"));
        let doc = to_mtlx_document("Sword", &SurfaceParams::default(), &full_set());
        let root = parse_doc(&doc).unwrap();
        let shader = children(&root, "open_pbr_surface")[0];
        for inp in children(shader, "input") {
            let (n, t) = (inp.attr("name").unwrap(), inp.attr("type").unwrap());
            let needle = format!("<input name=\"{n}\" type=\"{t}\"");
            assert!(nodedef.contains(&needle), "{needle} not in the nodedef");
        }
        // The map → input table itself, including the unwired kinds.
        for map in [
            TextureMap::BaseColor,
            TextureMap::Roughness,
            TextureMap::Metallic,
            TextureMap::Normal,
            TextureMap::Opacity,
            TextureMap::Emissive,
        ] {
            let (n, t) = map.openpbr_input().unwrap();
            assert!(nodedef.contains(&format!("<input name=\"{n}\" type=\"{t}\"")));
        }
        assert_eq!(TextureMap::AmbientOcclusion.openpbr_input(), None);
        assert_eq!(TextureMap::Height.openpbr_input(), None);
        for absent in ["occlusion", "displacement", "height"] {
            assert!(
                !nodedef.contains(absent),
                "{absent} appeared in the nodedef"
            );
            assert!(!doc.contains(&format!("name=\"{absent}")));
        }
    }

    // (c) Colorspace per image node: sRGB color, linear data.
    #[test]
    fn colorspace_is_srgb_for_color_maps_and_linear_for_data() {
        let doc = to_mtlx_document("Sword", &SurfaceParams::default(), &full_set());
        let root = parse_doc(&doc).unwrap();
        let shader = children(&root, "open_pbr_surface")[0];
        for (input_name, expected) in [
            ("base_color", "srgb_texture"),
            ("emission_color", "srgb_texture"),
            ("base_metalness", "lin_rec709"),
            ("specular_roughness", "lin_rec709"),
            ("geometry_normal", "lin_rec709"),
            ("geometry_opacity", "lin_rec709"),
        ] {
            let image = image_feeding(&root, shader, input_name);
            assert_eq!(
                input(image, "file").attr("colorspace"),
                Some(expected),
                "{input_name}"
            );
        }
        // A caller-supplied colorspace (the bytes actually written) wins
        // over the §9 default.
        let mut t = tex(TextureMap::Roughness, "r.tif", None);
        t.colorspace = MtlxColorSpace::SrgbTexture;
        let doc = to_mtlx_document("S", &SurfaceParams::default(), &[t]);
        assert!(doc.contains("value=\"r.tif\" colorspace=\"srgb_texture\""));
    }

    // (d) The UDIM token survives escaping and reads back verbatim;
    // concrete names stay concrete.
    #[test]
    fn udim_token_round_trips_and_concrete_names_stay_concrete() {
        let tiled = [tex(
            TextureMap::BaseColor,
            "Sword_baseColor_<UDIM>.png",
            None,
        )];
        let doc = to_mtlx_document("Sword", &SurfaceParams::default(), &tiled);
        assert!(doc.contains("Sword_baseColor_&lt;UDIM&gt;.png"));
        let root = parse_doc(&doc).unwrap();
        let file = input(children(&root, "image")[0], "file");
        assert_eq!(file.attr("value"), Some("Sword_baseColor_<UDIM>.png"));

        let single = [tex(TextureMap::BaseColor, "Sword_baseColor.png", None)];
        let doc = to_mtlx_document("Sword", &SurfaceParams::default(), &single);
        assert!(doc.contains("value=\"Sword_baseColor.png\""));
        assert!(!doc.contains("UDIM"));
    }

    #[test]
    fn untextured_inputs_carry_the_params() {
        let params = SurfaceParams {
            specular_roughness: 0.55,
            base_metalness: 1.0,
            ..SurfaceParams::default()
        };
        let doc = to_mtlx_document("Sword", &params, &[]);
        assert!(doc.contains(&value_input("specular_roughness", "float", "0.55")));
        assert!(doc.contains(&value_input("base_metalness", "float", "1")));
        assert!(doc.contains(&value_input("base_color", "color3", "0.8, 0.8, 0.8")));
        // Textured: the constant is replaced by the connection.
        let rough = [tex(TextureMap::Roughness, "r.png", None)];
        let doc = to_mtlx_document("Sword", &params, &rough);
        assert!(!doc.contains("value=\"0.55\""));
        let connection = "<input name=\"specular_roughness\" type=\"float\" \
                          nodename=\"Sword_specular_roughness_image\" />";
        assert!(doc.contains(connection));
    }

    fn value_input(name: &str, ty: &str, value: &str) -> String {
        format!("<input name=\"{name}\" type=\"{ty}\" value=\"{value}\" />")
    }

    #[test]
    fn defaults_match_the_vendored_nodedef_values() {
        let nodedef = vendored_nodedef();
        let d = SurfaceParams::default();
        for (n, v) in [
            ("base_weight", "1.0"),
            ("base_metalness", "0.0"),
            ("specular_weight", "1.0"),
            ("specular_roughness", "0.3"),
            ("specular_ior", "1.5"),
            ("coat_weight", "0.0"),
            ("coat_roughness", "0.0"),
            ("coat_ior", "1.6"),
            ("emission_luminance", "0.0"),
            ("geometry_opacity", "1"),
        ] {
            assert!(
                nodedef.contains(&format!("<input name=\"{n}\" type=\"float\" value=\"{v}\"")),
                "{n} default"
            );
        }
        assert_eq!(d.specular_roughness, 0.3);
        assert_eq!(d.base_color, [0.8, 0.8, 0.8]);
        assert_eq!(d.coat_ior, 1.6);
        let base_color = "<input name=\"base_color\" type=\"color3\" value=\"0.8, 0.8, 0.8\"";
        assert!(nodedef.contains(base_color));
    }

    #[test]
    fn gpu_slots_unpack_and_split_emission() {
        // OpenPbrParams::default()'s slots → the nodedef defaults.
        let p = SurfaceParams::from_gpu_slots(
            [0.8, 0.8, 0.8, 1.0],
            [0.0, 0.3, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.5, 1.6, 0.0, 0.0],
        );
        assert_eq!(p, SurfaceParams::default());
        // Premultiplied emission splits so color × luminance = rgb.
        let p = SurfaceParams::from_gpu_slots(
            [0.8, 0.8, 0.8, 1.0],
            [0.0, 0.3, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [4.0, 2.0, 0.0, 0.0],
            [1.5, 1.6, 0.0, 0.0],
        );
        assert_eq!(p.emission_luminance, 4.0);
        assert_eq!(p.emission_color, [1.0, 0.5, 0.0]);
    }

    #[test]
    fn textured_emission_with_zero_luminance_emits_one() {
        let emissive = [tex(TextureMap::Emissive, "e.png", None)];
        let doc = to_mtlx_document("S", &SurfaceParams::default(), &emissive);
        assert!(doc.contains(&value_input("emission_luminance", "float", "1")));
    }

    #[test]
    fn first_texture_per_input_wins_and_bad_extracts_are_skipped() {
        let doc = to_mtlx_document(
            "S",
            &SurfaceParams::default(),
            &[
                tex(TextureMap::Roughness, "a.png", None),
                tex(TextureMap::Roughness, "b.png", None),
                // A color input cannot come from one packed channel.
                tex(TextureMap::BaseColor, "c.png", Some(0)),
            ],
        );
        assert!(doc.contains("a.png") && !doc.contains("b.png") && !doc.contains("c.png"));
        assert!(doc.contains("<input name=\"base_color\" type=\"color3\" value="));
    }

    #[test]
    fn alpha_channel_extract_reads_a_color4_image() {
        let alpha = [tex(TextureMap::Roughness, "ms.png", Some(3))];
        let doc = to_mtlx_document("S", &SurfaceParams::default(), &alpha);
        assert!(doc.contains("<image name=\"S_specular_roughness_image\" type=\"color4\">"));
        let wire = "<input name=\"in\" type=\"color4\" nodename=\"S_specular_roughness_image\" />";
        assert!(doc.contains(wire));
        assert!(doc.contains(&value_input("index", "integer", "3")));
    }

    #[test]
    fn set_names_become_valid_element_names() {
        assert_eq!(element_name("Sword"), "Sword");
        assert_eq!(element_name("my set.v2"), "my_set_v2");
        assert_eq!(element_name("1001"), "M_1001");
        assert_eq!(element_name(""), "M_");
    }

    #[test]
    fn output_is_deterministic() {
        let a = to_mtlx_document("Sword", &SurfaceParams::default(), &full_set());
        let b = to_mtlx_document("Sword", &SurfaceParams::default(), &full_set());
        assert_eq!(a, b);
    }
}

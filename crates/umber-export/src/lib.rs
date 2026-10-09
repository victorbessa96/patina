//! umber-export — template/preset export engine.
//!
//! Wave 3 scope (docs/specs/requirements.md §6): naming tokens
//! ($mesh/$textureSet/$udim/$colorSpace/$srcMap/$layerName), channel
//! packing, engine presets (glTF/Unreal/Unity/Blender), normal-convention
//! conversion, PNG 8/16 + EXR 16F/32F.
//!
//! Wave 1 scope: the naming-token substitution engine — pure string
//! logic, fully testable headless, and the natural first real code here.

/// Substitute `$token` placeholders in an export path template.
///
/// Unknown tokens pass through verbatim so user templates never break
/// silently when a token's data is missing (documented, tested behavior).
pub fn expand_template(template: &str, tokens: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (name, value) in tokens {
        let key = format!("${}", name);
        if out.contains(&key) && !value.is_empty() {
            out = out.replace(&key, value);
        }
    }
    out
}

/// Convert `/` separators in a template into subfolders (Painter
/// convention: slash creates directories).
pub fn template_to_path(template: &str) -> String {
    template.replace('/', std::path::MAIN_SEPARATOR_STR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_known_tokens() {
        let out = expand_template(
            "$mesh_$textureSet_$colorSpace",
            &[
                ("mesh", "sword"),
                ("textureSet", "Blade"),
                ("colorSpace", "basecolor"),
            ],
        );
        assert_eq!(out, "sword_Blade_basecolor");
    }

    #[test]
    fn unknown_tokens_pass_through() {
        let out = expand_template("$mesh_$unknown", &[("mesh", "sword")]);
        assert_eq!(out, "sword_$unknown");
    }

    #[test]
    fn empty_values_leave_placeholder() {
        let out = expand_template("$mesh_$udim", &[("mesh", "sword"), ("udim", "")]);
        assert_eq!(out, "sword_$udim");
    }

    #[test]
    fn slashes_pass_through_as_separators_on_unix() {
        // template_to_path only maps separators; token expansion is a
        // separate step (expand_template). Test what each function does.
        assert_eq!(template_to_path("out/$mesh/color"), "out/$mesh/color");
        let expanded = expand_template(&template_to_path("out/$mesh/color"), &[("mesh", "sword")]);
        assert_eq!(expanded, "out/sword/color");
    }
}

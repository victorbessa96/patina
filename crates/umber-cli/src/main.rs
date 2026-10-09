//! umber-cli — headless automation surface.
//!
//! Wave 6 scope: bake/export/batch over the .umber project format
//! (SPEC.md deliverable 6). Wave 1 scope: mesh inspection — a real,
//! testable command that exercises umber-mesh end to end headless.

use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => {
            let path = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: umber-cli inspect <mesh-file>"))?;
            let mesh = umber_mesh::load(std::path::Path::new(path))?;
            println!("file:            {}", path);
            println!("vertices:         {}", mesh.vertex_count());
            println!("triangles:        {}", mesh.triangle_count());
            println!("uv set entries:   {}", mesh.uvs.len());
            println!("materials:        {}", mesh.material_names.len());
            if let Some((min, max)) = mesh.bounds() {
                println!("bounds min:       {:?}", min);
                println!("bounds max:       {:?}", max);
            }
            Ok(())
        }
        Some(other) => Err(anyhow::anyhow!(
            "unknown command: {other}\nusage: umber-cli inspect <mesh-file>"
        )),
        None => {
            println!("umber-cli v{} (wave-1 skeleton)", env!("CARGO_PKG_VERSION"));
            println!("commands: inspect <mesh-file>");
            Ok(())
        }
    }
}

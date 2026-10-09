//! umber-cli — headless automation surface.
//!
//! Wave 6 scope: bake/export/batch over the .umber project format
//! (SPEC.md deliverable 6). Wave 1 scope: mesh inspection — a real,
//! testable command that exercises umber-mesh end to end headless.
//! Wave 3: `bake-ao` — headless AO bake of a mesh to PNG, the first
//! real automation path over the bake engine.

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
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
        Some("bake-ao") => bake_ao_cmd(&args[1..]),
        Some(other) => Err(anyhow::anyhow!(
            "unknown command: {other}\nusage: umber-cli <inspect|bake-ao> ..."
        )),
        None => {
            println!("umber-cli v{} (wave-3)", env!("CARGO_PKG_VERSION"));
            println!("commands: inspect <mesh-file>");
            println!("          bake-ao <mesh-file> <out.png> [--size N] [--rays N]");
            Ok(())
        }
    }
}

/// `bake-ao <mesh> <out.png> [--size N] [--rays N]` — headless AO bake
/// over the real mesh (position-map path), written as an sRGB PNG with
/// coverage alpha.
fn bake_ao_cmd(args: &[String]) -> Result<()> {
    let usage = "usage: umber-cli bake-ao <mesh-file> <out.png> [--size N] [--rays N]";
    let mesh_path = args.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let out_path = args.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?;

    let mut size: u32 = 512;
    let mut rays: u32 = 16;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                size = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--size needs a value"))?
                    .parse()?;
                i += 2;
            }
            "--rays" => {
                rays = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--rays needs a value"))?
                    .parse()?;
                i += 2;
            }
            other => return Err(anyhow::anyhow!("unknown flag: {other}")),
        }
    }

    let mesh = umber_mesh::load(std::path::Path::new(mesh_path))?;
    log::info!("mesh: {} tris", mesh.triangle_count());

    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|e| anyhow::anyhow!("no suitable GPU adapter: {e}"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    log::info!("adapter: {}", adapter.get_info().name);

    let mut params = umber_bake::AoBakeParams::new(
        // max_distance/bias per the bake tests' convention; the plane is
        // unused by bake_ao_mesh (the mesh's position map replaces it).
        10.0,
        0.01,
        umber_bake::PlaneDesc::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0],
        ),
    );
    params.rays = rays;
    let map = umber_bake::ao::bake_ao_mesh(&device, &queue, &mesh, size, size, &params)?;
    log::info!("baked {} texels", map.len() / 4);

    // RGBA8 with coverage alpha: sRGB-encode, alpha carries coverage.
    umber_export::png::write_png(
        std::path::Path::new(out_path),
        size,
        size,
        &map,
        umber_export::png::Transfer::Srgb,
    )?;
    println!("wrote {out_path} ({size}x{size}, {rays} rays)");
    Ok(())
}

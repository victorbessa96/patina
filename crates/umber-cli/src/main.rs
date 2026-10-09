//! umber-cli — headless automation surface.
//!
//! Wave 6 scope: bake/export/batch over the .umber project format
//! (SPEC.md deliverable 6). Wave 1 scope: mesh inspection — a real,
//! testable command that exercises umber-mesh end to end headless.
//! Wave 3: `bake-ao` (single-map headless bake) and `bake-all` (every
//! implemented baker, written with the `TextureSetName_map`
//! convention — the automation path over the bake engine).

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => inspect_cmd(&args[1..]),
        Some("bake-ao") => bake_ao_cmd(&args[1..]),
        Some("bake-all") => bake_all_cmd(&args[1..]),
        Some(other) => Err(anyhow::anyhow!(
            "unknown command: {other}\nusage: umber-cli <inspect|bake-ao|bake-all> ..."
        )),
        None => {
            println!("umber-cli v{} (wave-3)", env!("CARGO_PKG_VERSION"));
            println!("commands: inspect <mesh-file>");
            println!("          bake-ao <mesh-file> <out.png> [--size N] [--rays N]");
            println!("          bake-all <mesh-file> <out-dir> [--size N] [--rays N]");
            Ok(())
        }
    }
}

/// `inspect <mesh>` — mesh summary.
fn inspect_cmd(args: &[String]) -> Result<()> {
    let path = args
        .first()
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

/// Size/ray flag parsing shared by the bake commands.
struct BakeFlags {
    size: u32,
    rays: u32,
}

fn parse_bake_flags(args: &[String], start: usize) -> Result<BakeFlags> {
    let mut flags = BakeFlags {
        size: 512,
        rays: 16,
    };
    let mut i = start;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                flags.size = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--size needs a value"))?
                    .parse()?;
                i += 2;
            }
            "--rays" => {
                flags.rays = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--rays needs a value"))?
                    .parse()?;
                i += 2;
            }
            other => return Err(anyhow::anyhow!("unknown flag: {other}")),
        }
    }
    Ok(flags)
}

/// A headless bake context: adapter + device + queue, created once and
/// shared by every baker in the run.
struct BakeContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn bake_context() -> Result<BakeContext> {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|e| anyhow::anyhow!("no suitable GPU adapter: {e}"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    log::info!("adapter: {}", adapter.get_info().name);
    Ok(BakeContext { device, queue })
}

/// `bake-ao <mesh> <out.png> [--size N] [--rays N]` — headless AO bake
/// over the real mesh (position-map path), written as an sRGB PNG with
/// coverage alpha.
fn bake_ao_cmd(args: &[String]) -> Result<()> {
    let usage = "usage: umber-cli bake-ao <mesh-file> <out.png> [--size N] [--rays N]";
    let mesh_path = args.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let out_path = args.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let flags = parse_bake_flags(args, 2)?;

    let mesh = umber_mesh::load(std::path::Path::new(mesh_path))?;
    let ctx = bake_context()?;
    let map = bake_ao(&ctx, &mesh, &flags)?;
    umber_export::png::write_png(
        std::path::Path::new(out_path),
        flags.size,
        flags.size,
        &map,
        umber_export::png::Transfer::Srgb,
    )?;
    println!(
        "wrote {out_path} ({}x{}, {} rays)",
        flags.size, flags.size, flags.rays
    );
    Ok(())
}

/// `bake-all <mesh> <out-dir> [flags]` — every implemented baker,
/// written with the `TextureSetName_map` convention.
fn bake_all_cmd(args: &[String]) -> Result<()> {
    let usage = "usage: umber-cli bake-all <mesh-file> <out-dir> [--size N] [--rays N]";
    let mesh_path = args.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let out_dir = args.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let flags = parse_bake_flags(args, 2)?;

    let mesh_path = std::path::Path::new(mesh_path);
    let mesh = umber_mesh::load(mesh_path)?;
    let ctx = bake_context()?;
    let set = umber_mesh::texture_set_name(mesh_path, &mesh);
    let out_dir = std::path::Path::new(out_dir);
    std::fs::create_dir_all(out_dir)?;

    // AO map: <set>_ambient_occlusion.png.
    let ao = bake_ao(&ctx, &mesh, &flags)?;
    let ao_path = umber_mesh::format_mesh_map(
        out_dir,
        &set,
        umber_mesh::MeshMapKind::AmbientOcclusion,
        "png",
    );
    umber_export::png::write_png(
        &ao_path,
        flags.size,
        flags.size,
        &ao,
        umber_export::png::Transfer::Srgb,
    )?;
    println!("wrote {}", ao_path.display());

    println!("bake-all complete for texture set '{set}'");
    Ok(())
}

/// Shared AO bake driver (bake-ao + bake-all).
fn bake_ao(ctx: &BakeContext, mesh: &umber_mesh::MeshData, flags: &BakeFlags) -> Result<Vec<u8>> {
    log::info!("mesh: {} tris", mesh.triangle_count());
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
    params.rays = flags.rays;
    let map = umber_bake::ao::bake_ao_mesh(
        &ctx.device,
        &ctx.queue,
        mesh,
        flags.size,
        flags.size,
        &params,
    )?;
    log::info!("baked {} texels", map.len() / 4);
    Ok(map)
}

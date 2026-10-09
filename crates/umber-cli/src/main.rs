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
        Some("export") => export_cmd(&args[1..]),
        Some(other) => Err(anyhow::anyhow!(
            "unknown command: {other}\nusage: umber-cli <inspect|bake-ao|bake-all> ..."
        )),
        None => {
            println!("umber-cli v{} (wave-3)", env!("CARGO_PKG_VERSION"));
            println!("commands: inspect <mesh-file>");
            println!("          bake-ao <mesh-file> <out.png> [--size N] [--rays N]");
            println!("          bake-all <mesh-file> <out-dir> [--size N] [--rays N]");
            println!("          export <mesh-file> <out-dir> --preset <gltf|unreal|unity|blender> [--size N] [--rays N]");
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
            // export_cmd parses --preset itself; skip it (value too).
            "--preset" => i += 2,
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

    // Curvature map: <set>_curvature.png (already grayscale RGB + alpha).
    let curvature = umber_bake::curvature::bake_curvature_mesh(
        &ctx.device,
        &ctx.queue,
        &mesh,
        flags.size,
        flags.size,
        &umber_bake::CurvatureParams::default(),
    )?;
    let curv_path =
        umber_mesh::format_mesh_map(out_dir, &set, umber_mesh::MeshMapKind::Curvature, "png");
    umber_export::png::write_png(
        &curv_path,
        flags.size,
        flags.size,
        &curvature,
        umber_export::png::Transfer::Srgb,
    )?;
    println!("wrote {}", curv_path.display());

    // Position map: <set>_position.png — the f32 position data
    // re-encoded to 8-bit (each axis mapped [min,max] -> [0,255] over
    // the mesh bounds; w = coverage as alpha).
    let position_params = umber_bake::position::PositionMapParams {
        width: flags.size,
        height: flags.size,
    };
    let position_f32 =
        umber_bake::position::bake_position_map(&ctx.device, &ctx.queue, &mesh, &position_params)?;
    let position = encode_position_rgba8(&position_f32);
    let pos_path =
        umber_mesh::format_mesh_map(out_dir, &set, umber_mesh::MeshMapKind::Position, "png");
    umber_export::png::write_png(
        &pos_path,
        flags.size,
        flags.size,
        &position,
        umber_export::png::Transfer::Srgb,
    )?;
    println!("wrote {}", pos_path.display());

    // Thickness map: <set>_thickness.png.
    let thickness_params = umber_bake::thickness::ThicknessParams {
        rays: flags.rays,
        ..umber_bake::thickness::ThicknessParams::default()
    };
    let thickness = umber_bake::thickness::bake_thickness_mesh(
        &ctx.device,
        &ctx.queue,
        &mesh,
        flags.size,
        flags.size,
        &thickness_params,
    )?;
    let thick_path =
        umber_mesh::format_mesh_map(out_dir, &set, umber_mesh::MeshMapKind::Thickness, "png");
    umber_export::png::write_png(
        &thick_path,
        flags.size,
        flags.size,
        &thickness,
        umber_export::png::Transfer::Srgb,
    )?;
    println!("wrote {}", thick_path.display());

    // Dilation post-pass: spread island-edge color into UV seams so
    // exported maps show no transparent halos (requirements §6).
    let dilate = umber_bake::dilation::dilate_map(
        &ctx.device,
        &ctx.queue,
        &ao,
        flags.size,
        flags.size,
        &umber_bake::dilation::DilateParams::default(),
    )?;
    let dilated_path = out_dir.join(format!("{set}_ambient_occlusion_dilated.png"));
    umber_export::png::write_png(
        &dilated_path,
        flags.size,
        flags.size,
        &dilate,
        umber_export::png::Transfer::Srgb,
    )?;
    println!("wrote {}", dilated_path.display());

    println!("bake-all complete for texture set '{set}'");
    Ok(())
}

/// Re-encodes the f32 position map (rgba32f: xyz = world pos, w =
/// coverage) to RGBA8: each axis normalized over the mesh bounds to
/// [0,255]; alpha carries coverage. Bounds come from the map itself
/// (covered texels only) so the output is self-contained.
fn encode_position_rgba8(position_f32: &[f32]) -> Vec<u8> {
    // First pass: gather bounds over covered texels.
    let texels = position_f32.len() / 4;
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for t in 0..texels {
        if position_f32[t * 4 + 3] > 0.5 {
            for axis in 0..3 {
                let v = position_f32[t * 4 + axis];
                min[axis] = min[axis].min(v);
                max[axis] = max[axis].max(v);
            }
        }
    }
    // Degenerate (empty/zero-height map): all-black uncovered.
    let span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let mut out = vec![0u8; texels * 4];
    for t in 0..texels {
        if position_f32[t * 4 + 3] > 0.5 {
            for axis in 0..3 {
                let v = if span[axis] > f32::EPSILON {
                    (position_f32[t * 4 + axis] - min[axis]) / span[axis]
                } else {
                    0.5
                };
                out[t * 4 + axis] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            out[t * 4 + 3] = 255;
        }
    }
    out
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

/// `export <mesh> <out-dir> --preset <name> [--size N] [--rays N]` —
/// runs the full pipeline headless: bakes the P0 maps, packs them
/// through the chosen export preset, writes the named outputs.
fn export_cmd(args: &[String]) -> Result<()> {
    let usage =
        "usage: umber-cli export <mesh-file> <out-dir> --preset <gltf|unreal|unity|blender> [--size N] [--rays N]";
    let mesh_path = args.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let out_dir = args.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?;

    let mut preset_name = String::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" => {
                preset_name = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--preset needs a value"))?
                    .clone();
                i += 2;
            }
            "--size" | "--rays" => i += 2, // parsed by BakeFlags below
            other => return Err(anyhow::anyhow!("unknown flag: {other}")),
        }
    }

    let flags = parse_bake_flags(args, 2)?;
    let preset = match preset_name.as_str() {
        "gltf" => umber_export::ExportPreset::gltf_metal_rough(),
        "unreal" => umber_export::ExportPreset::unreal_orm(),
        "unity" => umber_export::ExportPreset::unity_hdrp_urp(),
        "blender" => umber_export::ExportPreset::blender_principled(),
        other => {
            return Err(anyhow::anyhow!(
                "unknown preset: {other} (gltf|unreal|unity|blender)"
            ))
        }
    };

    let mesh_path = std::path::Path::new(mesh_path);
    let mesh = umber_mesh::load(mesh_path)?;
    let ctx = bake_context()?;
    let set_name = umber_mesh::texture_set_name(mesh_path, &mesh);

    // Bake the union of the preset's map kinds (data maps as linear
    // RGBA8 sources for the driver).
    let ao = bake_ao(&ctx, &mesh, &flags)?;
    let position_params = umber_bake::position::PositionMapParams {
        width: flags.size,
        height: flags.size,
    };
    let position_f32 =
        umber_bake::position::bake_position_map(&ctx.device, &ctx.queue, &mesh, &position_params)?;
    let position = encode_position_rgba8(&position_f32);

    // The driver validates ALL outputs up front. For headless export
    // with baked-only sources (AO + flat normal today), a full engine
    // preset errors on BaseColor/Metallic — the honest behavior is
    // exporting the outputs we CAN fill and warning about the rest.
    // Strategy: filter the preset to outputs whose maps we hold.
    let mut map_set = umber_export::MapSet::new(flags.size);
    map_set.set(umber_export::MapKind::AmbientOcclusion, ao);
    let texels = (flags.size * flags.size) as usize;
    let mut flat_normal = Vec::with_capacity(texels * 4);
    for _ in 0..texels {
        flat_normal.extend_from_slice(&[128, 128, 255, 255]);
    }
    map_set.set(umber_export::MapKind::Normal, flat_normal);
    let _ = position; // position joins when a preset references it

    let available: Vec<umber_export::MapKind> = map_set.maps_iter().collect();
    let full_preset = preset;
    let filtered_outputs: Vec<_> = full_preset
        .outputs
        .iter()
        .filter(|o| o.maps.iter().all(|(k, _)| available.contains(k)))
        .cloned()
        .collect();
    for skipped in &full_preset.outputs {
        if !filtered_outputs.contains(skipped) {
            println!(
                "skipped {} (needs maps not baked headless)",
                skipped.filename
            );
        }
    }
    let preset = umber_export::ExportPreset {
        name: full_preset.name.clone(),
        outputs: filtered_outputs,
    };
    if preset.outputs.is_empty() {
        return Err(anyhow::anyhow!(
            "preset '{preset_name}' has no outputs satisfiable from baked maps"
        ));
    }

    let written =
        umber_export::run_preset(&preset, &map_set, &set_name, std::path::Path::new(out_dir))?;
    for path in &written {
        println!("wrote {}", path.display());
    }
    println!(
        "export complete: preset '{preset_name}', {} outputs",
        written.len()
    );
    Ok(())
}

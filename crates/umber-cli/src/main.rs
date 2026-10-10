//! umber-cli — headless automation surface.
//!
//! Wave 6 scope: bake/export/batch over the .umber project format
//! (SPEC.md deliverable 6). Wave 1 scope: mesh inspection — a real,
//! testable command that exercises umber-mesh end to end headless.
//! Wave 3: `bake-ao` (single-map headless bake) and `bake-all` (every
//! implemented baker, written with the `TextureSetName_map`
//! convention — the automation path over the bake engine). Wave 6:
//! `batch` — a JSON recipe scripting the commands above (see
//! [`batch`]).

mod batch;

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Deserialize;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => inspect_cmd(&args[1..]),
        Some("bake-ao") => bake_ao_cmd(&args[1..]),
        Some("bake-all") => bake_all_cmd(&args[1..]),
        Some("export") => export_cmd(&args[1..]),
        Some("batch") => batch::batch_cmd(&args[1..]),
        Some(other) => Err(anyhow::anyhow!(
            "unknown command: {other}\nusage: umber-cli <inspect|bake-ao|bake-all|export|batch> ..."
        )),
        None => {
            println!("umber-cli v{} (wave-6)", env!("CARGO_PKG_VERSION"));
            println!("commands: inspect <mesh-file>");
            println!("          bake-ao <mesh-file> <out.png> [--size N] [--rays N] [--dilate N]");
            println!("          bake-all <mesh-file> <out-dir> [--size N] [--rays N] [--dilate N]");
            println!("          export <mesh-file> <out-dir> --preset <gltf|unreal|unity|blender> [--size N] [--rays N] [--tile UDIM]...");
            println!("          batch <recipe.json>");
            Ok(())
        }
    }
}

/// `inspect <mesh>` — mesh summary.
fn inspect_cmd(args: &[String]) -> Result<()> {
    let path = args
        .first()
        .ok_or_else(|| anyhow::anyhow!("usage: umber-cli inspect <mesh-file>"))?;
    inspect_mesh(Path::new(path))?;
    Ok(())
}

/// Loads `path` and prints its summary (inspect + the batch `inspect`
/// step); returns the mesh for callers that report on it.
fn inspect_mesh(path: &Path) -> Result<umber_mesh::MeshData> {
    let mesh = umber_mesh::load(path)?;
    println!("file:            {}", path.display());
    println!("vertices:         {}", mesh.vertex_count());
    println!("triangles:        {}", mesh.triangle_count());
    println!("uv set entries:   {}", mesh.uvs.len());
    println!("materials:        {}", mesh.material_names.len());
    if let Some((min, max)) = mesh.bounds() {
        println!("bounds min:       {:?}", min);
        println!("bounds max:       {:?}", max);
    }
    Ok(mesh)
}

/// Size/ray/dilate flag parsing shared by the bake commands.
struct BakeFlags {
    size: u32,
    rays: u32,
    /// Dilation iterations for the UV-padding post-pass (0 = off).
    dilate: u32,
}

impl Default for BakeFlags {
    /// The CLI defaults (`--size 512 --rays 16`, no dilation) — also the
    /// batch recipe's defaults for omitted fields.
    fn default() -> Self {
        Self {
            size: 512,
            rays: 16,
            dilate: 0,
        }
    }
}

fn parse_bake_flags(args: &[String], start: usize) -> Result<BakeFlags> {
    let mut flags = BakeFlags::default();
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
            "--dilate" => {
                flags.dilate = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--dilate needs a value"))?
                    .parse()?;
                i += 2;
            }
            // export_cmd parses --preset/--tile itself; skip them (value too).
            "--preset" | "--tile" => i += 2,
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
    // Optional UV-padding post-pass: spread island-edge color into
    // seams when --dilate N (N > 0) is given.
    let map = if flags.dilate > 0 {
        umber_bake::dilation::dilate_map(
            &ctx.device,
            &ctx.queue,
            &map,
            flags.size,
            flags.size,
            &umber_bake::dilation::DilateParams {
                iterations: flags.dilate,
            },
        )?
    } else {
        map
    };
    umber_export::png::write_png(
        std::path::Path::new(out_path),
        flags.size,
        flags.size,
        &map,
        umber_export::png::Transfer::Srgb,
    )?;
    println!(
        "wrote {out_path} ({}x{}, {} rays{})",
        flags.size,
        flags.size,
        flags.rays,
        if flags.dilate > 0 {
            format!(", {} dilate steps", flags.dilate)
        } else {
            String::new()
        }
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

    let (set, written) = run_bake(
        Path::new(mesh_path),
        Path::new(out_dir),
        &flags,
        &BakeMap::ALL,
    )?;
    for path in &written {
        if flags.dilate > 0 {
            println!("wrote {} (dilated {} steps)", path.display(), flags.dilate);
        } else {
            println!("wrote {}", path.display());
        }
    }
    println!("bake-all complete for texture set '{set}'");
    Ok(())
}

/// The mesh-map bakers `bake-all` runs, in its write order — the ONE
/// list of bakeable map names (the batch `bake` step's `maps` filter
/// selects from it; an unknown name is a recipe validation error).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BakeMap {
    Ao,
    Curvature,
    Position,
    WorldNormal,
    TangentNormal,
    Thickness,
}

impl BakeMap {
    /// Every baker, in `bake-all` order.
    const ALL: [BakeMap; 6] = [
        BakeMap::Ao,
        BakeMap::Curvature,
        BakeMap::Position,
        BakeMap::WorldNormal,
        BakeMap::TangentNormal,
        BakeMap::Thickness,
    ];

    /// The `TextureSetName_map` naming slot. World-space normal is the
    /// WorldSpaceNormal slot, NOT the tangent-space Normal map; the baked
    /// tangent normal is `normal_base` (Substance-style naming: distinct
    /// from a user-painted normal map).
    fn kind(self) -> umber_mesh::MeshMapKind {
        match self {
            BakeMap::Ao => umber_mesh::MeshMapKind::AmbientOcclusion,
            BakeMap::Curvature => umber_mesh::MeshMapKind::Curvature,
            BakeMap::Position => umber_mesh::MeshMapKind::Position,
            BakeMap::WorldNormal => umber_mesh::MeshMapKind::WorldSpaceNormal,
            BakeMap::TangentNormal => umber_mesh::MeshMapKind::NormalBase,
            BakeMap::Thickness => umber_mesh::MeshMapKind::Thickness,
        }
    }

    /// The PNG transfer the map is written with. Unit-vector maps
    /// (world/tangent normals) stay Linear — an Srgb write would decode a
    /// flat tangent normal (128,128,255) as (188,188,255) (≈ +0.474/+0.474
    /// in tangent xy). Dilated maps use the same transfer, so a dilated
    /// file equals the undilated one except at seams.
    fn transfer(self) -> umber_export::png::Transfer {
        match self {
            BakeMap::WorldNormal | BakeMap::TangentNormal => umber_export::png::Transfer::Linear,
            _ => umber_export::png::Transfer::Srgb,
        }
    }
}

/// Bakes `maps` for the mesh at `mesh_path` into `out_dir` (created if
/// absent) with the `TextureSetName_map` convention — the shared body of
/// `bake-all` and the batch `bake` step. With `flags.dilate > 0` every
/// map is seam-filled before its write. Returns the texture-set name and
/// the written paths, in `maps` order.
fn run_bake(
    mesh_path: &Path,
    out_dir: &Path,
    flags: &BakeFlags,
    maps: &[BakeMap],
) -> Result<(String, Vec<PathBuf>)> {
    let mesh = umber_mesh::load(mesh_path)?;
    let ctx = bake_context()?;
    let set = umber_mesh::texture_set_name(mesh_path, &mesh);
    std::fs::create_dir_all(out_dir)?;

    let mut written = Vec::with_capacity(maps.len());
    for &map in maps {
        let data = bake_map(&ctx, &mesh, map, flags)?;
        let data = if flags.dilate > 0 {
            umber_bake::dilation::dilate_map(
                &ctx.device,
                &ctx.queue,
                &data,
                flags.size,
                flags.size,
                &umber_bake::dilation::DilateParams {
                    iterations: flags.dilate,
                },
            )?
        } else {
            data
        };
        let path = umber_mesh::format_mesh_map(out_dir, &set, map.kind(), "png");
        umber_export::png::write_png(&path, flags.size, flags.size, &data, map.transfer())?;
        written.push(path);
    }
    Ok((set, written))
}

/// Bakes one map as RGBA8 (`flags.size` square; alpha = coverage).
fn bake_map(
    ctx: &BakeContext,
    mesh: &umber_mesh::MeshData,
    map: BakeMap,
    flags: &BakeFlags,
) -> Result<Vec<u8>> {
    let position_params = umber_bake::position::PositionMapParams {
        width: flags.size,
        height: flags.size,
    };
    Ok(match map {
        BakeMap::Ao => bake_ao(ctx, mesh, flags)?,
        // Already grayscale RGB + alpha.
        BakeMap::Curvature => umber_bake::curvature::bake_curvature_mesh(
            &ctx.device,
            &ctx.queue,
            mesh,
            flags.size,
            flags.size,
            &umber_bake::CurvatureParams::default(),
        )?,
        // The f32 position data re-encoded to 8-bit (each axis mapped
        // [min,max] -> [0,255] over the mesh bounds; w = coverage).
        BakeMap::Position => encode_position_rgba8(&umber_bake::position::bake_position_map(
            &ctx.device,
            &ctx.queue,
            mesh,
            &position_params,
        )?),
        // The f32 unit normals re-encoded (each axis [-1,1] -> [0,255];
        // w = coverage as alpha).
        BakeMap::WorldNormal => {
            let wnormal_f32 = umber_bake::position::bake_world_normal_map(
                &ctx.device,
                &ctx.queue,
                mesh,
                &position_params,
            )?;
            let texels = (flags.size * flags.size) as usize;
            let mut wnormal = vec![0u8; texels * 4];
            for t in 0..texels {
                if wnormal_f32[t * 4 + 3] > 0.5 {
                    for axis in 0..3 {
                        let v = (wnormal_f32[t * 4 + axis] + 1.0) * 0.5;
                        wnormal[t * 4 + axis] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                    wnormal[t * 4 + 3] = 255;
                }
            }
            wnormal
        }
        // The TBN pass over the position pair (OpenGL convention).
        BakeMap::TangentNormal => umber_bake::normal_map::bake_tangent_normal_mesh(
            &ctx.device,
            &ctx.queue,
            mesh,
            flags.size,
            flags.size,
            &umber_bake::normal_map::TangentNormalParams::default(),
        )?,
        BakeMap::Thickness => umber_bake::thickness::bake_thickness_mesh(
            &ctx.device,
            &ctx.queue,
            mesh,
            flags.size,
            flags.size,
            &umber_bake::thickness::ThicknessParams {
                rays: flags.rays,
                ..umber_bake::thickness::ThicknessParams::default()
            },
        )?,
    })
}

/// The high-to-low transfer maps (`bake_transfer_mesh`'s outputs) — the
/// batch `bake_transfer` step's `maps` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum TransferMapName {
    Height,
    WorldNormal,
    TangentNormal,
}

impl TransferMapName {
    /// Every transfer map.
    const ALL: [TransferMapName; 3] = [
        TransferMapName::Height,
        TransferMapName::WorldNormal,
        TransferMapName::TangentNormal,
    ];

    fn map(self) -> umber_bake::TransferMap {
        match self {
            TransferMapName::Height => umber_bake::TransferMap::Height,
            TransferMapName::WorldNormal => umber_bake::TransferMap::WorldNormal,
            TransferMapName::TangentNormal => umber_bake::TransferMap::TangentNormal,
        }
    }

    /// The naming slot — the same slots `bake-all` writes its world and
    /// tangent normals to.
    fn kind(self) -> umber_mesh::MeshMapKind {
        match self {
            TransferMapName::Height => umber_mesh::MeshMapKind::Height,
            TransferMapName::WorldNormal => umber_mesh::MeshMapKind::WorldSpaceNormal,
            TransferMapName::TangentNormal => umber_mesh::MeshMapKind::NormalBase,
        }
    }
}

/// Ray/target settings for [`run_bake_transfer`].
struct TransferSettings {
    size: u32,
    /// Max hit distance in front of the LOW surface (world units).
    front: f32,
    /// Max hit distance behind the LOW surface (world units).
    back: f32,
    /// Ray-origin push off the LOW surface along its normal.
    offset: f32,
}

impl Default for TransferSettings {
    /// v1 defaults (no CLI precedent): a symmetric 0.1-unit cage — a
    /// tight fit for meter-scale props; recipes override per asset.
    fn default() -> Self {
        Self {
            size: BakeFlags::default().size,
            front: 0.1,
            back: 0.1,
            offset: 0.0,
        }
    }
}

/// Bakes HIGH detail onto the LOW mesh's UVs (`bake_transfer_mesh`) for
/// each of `maps`, written into `out_dir` under the LOW mesh's texture
/// set. All transfer maps are data, written Linear. Returns the
/// texture-set name and the written paths, in `maps` order.
fn run_bake_transfer(
    low_path: &Path,
    high_path: &Path,
    out_dir: &Path,
    maps: &[TransferMapName],
    settings: &TransferSettings,
) -> Result<(String, Vec<PathBuf>)> {
    let low = umber_mesh::load(low_path)?;
    let high = umber_mesh::load(high_path)?;
    let ctx = bake_context()?;
    let set = umber_mesh::texture_set_name(low_path, &low);
    std::fs::create_dir_all(out_dir)?;

    let mut written = Vec::with_capacity(maps.len());
    for &map in maps {
        let params = umber_bake::TransferParams::new(
            settings.front,
            settings.back,
            settings.offset,
            settings.size,
            settings.size,
        )
        .with_map(map.map());
        let data = umber_bake::bake_transfer_mesh(&ctx.device, &ctx.queue, &low, &high, &params)?;
        let path = umber_mesh::format_mesh_map(out_dir, &set, map.kind(), "png");
        umber_export::png::write_png(
            &path,
            settings.size,
            settings.size,
            &data,
            umber_export::png::Transfer::Linear,
        )?;
        written.push(path);
    }
    Ok((set, written))
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

/// The UDIM tiles a mesh's geometry occupies, ascending and unique —
/// `tile_of_triangle`'s distinct values.
///
/// A mesh whose UVs all lie in the CLOSED `[0, 1]` square is `[1001]`
/// outright (the design's single-tile rule): `tile_of_triangle` tags by
/// the first vertex and `floor(1.0) = 1`, so a unit quad whose triangles
/// start on its `u = 1` / `v = 1` edge would otherwise read as tile
/// 1002/1011/1012 and lose the byte-identical whole-mesh export.
fn present_tiles(mesh: &umber_mesh::MeshData) -> Vec<u16> {
    let unit = |c: f32| (0.0..=1.0).contains(&c);
    if mesh.uvs.iter().all(|uv| unit(uv[0]) && unit(uv[1])) {
        return vec![umber_mesh::FIRST_TILE];
    }
    let mut tiles = umber_mesh::tile_of_triangle(mesh);
    tiles.sort_unstable();
    tiles.dedup();
    tiles
}

/// The tiles an export writes: every present tile when `requested` is
/// empty (the default), else the requested ones (ascending, unique),
/// each of which must hold geometry — a tile with no triangles has
/// nothing to bake.
fn select_tiles(requested: &[u16], present: &[u16]) -> Result<Vec<u16>> {
    if requested.is_empty() {
        return Ok(present.to_vec());
    }
    let mut tiles = requested.to_vec();
    tiles.sort_unstable();
    tiles.dedup();
    if let Some(missing) = tiles.iter().find(|t| !present.contains(t)) {
        return Err(anyhow::anyhow!(
            "--tile {missing}: the mesh has no geometry there (present tiles: {present:?})"
        ));
    }
    Ok(tiles)
}

/// The UDIM 10x10 grid.
const UDIM_GRID: std::ops::RangeInclusive<u16> = 1001..=1100;

/// Parses one `--tile` value: a UDIM number in the 10x10 grid
/// (1001..=1100).
fn parse_tile(value: &str) -> Result<u16> {
    let tile: u16 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("--tile {value}: not a UDIM number"))?;
    if !UDIM_GRID.contains(&tile) {
        return Err(anyhow::anyhow!(
            "--tile {tile}: outside the UDIM grid 1001..=1100"
        ));
    }
    Ok(tile)
}

/// `export <mesh> <out-dir> --preset <name> [--size N] [--rays N]
/// [--tile UDIM]...` — runs the full pipeline headless: bakes the P0
/// maps, packs them through the chosen export preset, writes the named
/// outputs.
///
/// UDIM (wave-5 slice 5): the export runs once per tile. `--tile` is
/// repeatable; without it, every tile the mesh's geometry occupies
/// ([`present_tiles`], from `tile_of_triangle`) is exported. A mesh
/// living only in tile 1001 bakes exactly as before (whole mesh) and
/// writes the same file names; otherwise each tile bakes its own
/// triangles over its own UV window (`filter_mesh_for_tile` + the
/// unchanged bakers) and `$udim` expands per tile — templates without
/// `$udim` get `_<tile>` before the extension so tiles never collide.
fn export_cmd(args: &[String]) -> Result<()> {
    let usage =
        "usage: umber-cli export <mesh-file> <out-dir> --preset <gltf|unreal|unity|blender> [--size N] [--rays N] [--tile UDIM]...";
    let mesh_path = args.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    let out_dir = args.get(1).ok_or_else(|| anyhow::anyhow!("{usage}"))?;

    let mut preset_name = String::new();
    let mut requested_tiles = Vec::new();
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
            "--tile" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--tile needs a value"))?;
                requested_tiles.push(parse_tile(value)?);
                i += 2;
            }
            "--size" | "--rays" => i += 2, // parsed by BakeFlags below
            other => return Err(anyhow::anyhow!("unknown flag: {other}")),
        }
    }

    let flags = parse_bake_flags(args, 2)?;
    let (written, tiles) = run_export(
        Path::new(mesh_path),
        Path::new(out_dir),
        &preset_name,
        &flags,
        &requested_tiles,
    )?;
    for path in &written {
        println!("wrote {}", path.display());
    }
    println!(
        "export complete: preset '{preset_name}', {} outputs across tiles {tiles:?}",
        written.len()
    );
    Ok(())
}

/// The export preset named `name` (the `--preset` / recipe values).
fn preset_by_name(name: &str) -> Result<umber_export::ExportPreset> {
    Ok(match name {
        "gltf" => umber_export::ExportPreset::gltf_metal_rough(),
        "unreal" => umber_export::ExportPreset::unreal_orm(),
        "unity" => umber_export::ExportPreset::unity_hdrp_urp(),
        "blender" => umber_export::ExportPreset::blender_principled(),
        other => {
            return Err(anyhow::anyhow!(
                "unknown preset: {other} (gltf|unreal|unity|blender)"
            ))
        }
    })
}

/// The export pipeline body shared by `export` and the batch `export`
/// step: bakes the P0 maps per tile, packs them through the preset
/// named `preset_name`, writes into `out_dir`. `requested_tiles` empty =
/// every present tile. Returns the written paths and the exported tiles.
fn run_export(
    mesh_path: &Path,
    out_dir: &Path,
    preset_name: &str,
    flags: &BakeFlags,
    requested_tiles: &[u16],
) -> Result<(Vec<PathBuf>, Vec<u16>)> {
    let preset = preset_by_name(preset_name)?;
    let mesh = umber_mesh::load(mesh_path)?;
    let ctx = bake_context()?;
    let set_name = umber_mesh::texture_set_name(mesh_path, &mesh);
    let present = present_tiles(&mesh);
    let tiles = select_tiles(requested_tiles, &present)?;
    // The pre-UDIM path: a mesh living only in tile 1001 bakes whole.
    let whole_mesh = present == [umber_mesh::FIRST_TILE];

    let position_params = umber_bake::position::PositionMapParams {
        width: flags.size,
        height: flags.size,
    };
    let position_f32 =
        umber_bake::position::bake_position_map(&ctx.device, &ctx.queue, &mesh, &position_params)?;
    let position = encode_position_rgba8(&position_f32);
    let _ = position; // position joins when a preset references it

    // The driver validates ALL outputs up front. For headless export
    // with baked-only sources (AO + baked tangent normal today), a
    // full engine preset still errors on BaseColor/Metallic — the
    // honest behavior is exporting the outputs we CAN fill and
    // warning about the rest. Strategy: filter the preset to outputs
    // whose maps we hold.
    let mut tile_sets = Vec::with_capacity(tiles.len());
    for &tile in &tiles {
        // Per-tile source: the tile's triangles with UVs rebased onto
        // [0, 1] (the documented filter + unchanged-core bake pattern).
        let filtered;
        let tile_mesh = if whole_mesh {
            &mesh
        } else {
            filtered = umber_bake::ao::filter_mesh_for_tile(&mesh, tile);
            &filtered
        };
        // Bake the union of the preset's map kinds (data maps as linear
        // RGBA8 sources for the driver).
        let ao = bake_ao(&ctx, tile_mesh, flags)?;
        let mut map_set = umber_export::MapSet::new(flags.size);
        map_set.set(umber_export::MapKind::AmbientOcclusion, ao);
        // The REAL baked tangent-space normal (OpenGL working convention;
        // the driver flips green for DirectX presets).
        let tnormal = umber_bake::normal_map::bake_tangent_normal_mesh(
            &ctx.device,
            &ctx.queue,
            tile_mesh,
            flags.size,
            flags.size,
            &umber_bake::normal_map::TangentNormalParams::default(),
        )?;
        map_set.set(umber_export::MapKind::Normal, tnormal);
        tile_sets.push((tile, map_set));
    }

    // Every tile's set holds the same kinds, so one filter serves all.
    let available: Vec<umber_export::MapKind> = tile_sets
        .first()
        .map(|(_, set)| set.maps_iter().collect())
        .unwrap_or_default();
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

    let pairs: Vec<(u16, &umber_export::MapSet)> =
        tile_sets.iter().map(|(tile, set)| (*tile, set)).collect();
    let written = umber_export::run_preset_tiled(
        &preset,
        &pairs,
        &umber_export::TokenSources {
            texture_set: &set_name,
            mesh: umber_export::mesh_stem(mesh_path),
            layer_name: "",
            // Overridden per tile by the tiled driver.
            udim: umber_export::SINGLE_TILE_UDIM,
        },
        out_dir,
    )?;
    Ok((written, tiles))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seam-free 2x1 strip spanning UV [0,2]x[0,1]: two triangles in
    /// tile 1001 (first vertex at u=0), two in 1002 (first vertex u=1).
    fn two_tile_strip() -> umber_mesh::MeshData {
        let uvs = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ];
        umber_mesh::MeshData {
            positions: uvs
                .iter()
                .map(|uv: &[f32; 2]| [uv[0], uv[1], 0.0])
                .collect(),
            normals: vec![[0.0, 0.0, 1.0]; 6],
            uvs,
            indices: vec![0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4],
            material_names: vec![],
        }
    }

    #[test]
    fn present_tiles_of_a_two_tile_mesh() {
        // tile_of_triangle = [1001, 1001, 1002, 1002] → distinct, sorted.
        assert_eq!(present_tiles(&two_tile_strip()), vec![1001, 1002]);
        // Reversed triangle order: still ascending.
        let mut mesh = two_tile_strip();
        mesh.indices = vec![1, 2, 5, 1, 5, 4, 0, 1, 4, 0, 4, 3];
        assert_eq!(present_tiles(&mesh), vec![1001, 1002]);
    }

    #[test]
    fn present_tiles_of_a_unit_square_mesh_is_1001() {
        let mut mesh = two_tile_strip();
        mesh.indices = vec![0, 1, 4, 0, 4, 3];
        assert_eq!(present_tiles(&mesh), vec![1001]);
    }

    #[test]
    fn unit_quad_starting_on_its_far_corner_is_still_1001() {
        // Both triangles start at UV (1,1): tile_of_triangle tags them
        // 1012 (floor(1.0) = 1 on both axes), but every UV is in the
        // closed [0,1] square, so the mesh is single-tile 1001.
        let mesh = umber_mesh::MeshData {
            positions: vec![[0.0; 3]; 4],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![2, 3, 0, 2, 0, 1],
            material_names: vec![],
        };
        assert_eq!(umber_mesh::tile_of_triangle(&mesh), vec![1012, 1012]);
        assert_eq!(present_tiles(&mesh), vec![1001]);
    }

    #[test]
    fn select_tiles_defaults_to_present_and_validates_requests() {
        let present = [1001, 1002];
        assert_eq!(select_tiles(&[], &present).unwrap(), vec![1001, 1002]);
        // Repeated/unsorted flags: ascending, unique.
        assert_eq!(
            select_tiles(&[1002, 1001, 1002], &present).unwrap(),
            vec![1001, 1002]
        );
        assert_eq!(select_tiles(&[1002], &present).unwrap(), vec![1002]);
        // A tile with no geometry is an error naming the tile.
        let err = select_tiles(&[1003], &present).unwrap_err();
        assert!(err.to_string().contains("1003"), "{err}");
    }

    #[test]
    fn parse_tile_accepts_the_grid_only() {
        assert_eq!(parse_tile("1001").unwrap(), 1001);
        assert_eq!(parse_tile("1100").unwrap(), 1100);
        assert!(parse_tile("1000").is_err());
        assert!(parse_tile("1101").is_err());
        assert!(parse_tile("abc").is_err());
    }

    #[test]
    fn bake_flag_parser_skips_the_tile_flag() {
        // export_cmd hands its whole arg list to parse_bake_flags, which
        // must not reject --tile as unknown.
        let args: Vec<String> = ["mesh.obj", "out", "--tile", "1002", "--size", "64"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let flags = parse_bake_flags(&args, 2).expect("--tile is skipped");
        assert_eq!(flags.size, 64);
    }
}

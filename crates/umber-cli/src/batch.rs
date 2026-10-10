//! `batch <recipe.json>` — the wave-6 scripting surface
//! (docs/specs/cli-batch-design.md): a JSON recipe of steps, each mapping
//! 1:1 onto an existing command's runner (`inspect` -> [`inspect_mesh`],
//! `bake` -> [`run_bake`], `bake_transfer` -> [`run_bake_transfer`],
//! `export` -> [`run_export`]).
//!
//! The whole recipe validates before any step runs; errors name the
//! JSON path (`steps[1].bake: unknown field ...`). serde_json's own
//! messages carry only line/column, so the steps parse in two stages —
//! the top level with `steps` as raw values, then each step on its own
//! with its index prefixed. Each executed step prints `[i] <name> ok
//! (<detail>)` or `[i] <name> FAILED: <error>` (`i` = the `steps[i]`
//! index), then a final `batch: N steps, M ok, K failed` line. Mesh and
//! output paths are taken as given (relative = the working directory),
//! like the other commands.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

use crate::{
    inspect_mesh, preset_by_name, run_bake, run_bake_transfer, run_export, BakeFlags, BakeMap,
    TransferMapName, TransferSettings, UDIM_GRID,
};

/// The recipe format version this build reads.
const RECIPE_VERSION: u32 = 1;

/// What a failing step does to the rest of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OnError {
    /// Halt at the first failure (the default).
    #[default]
    Stop,
    /// Run every step; the batch still fails if any step did.
    Continue,
}

/// A validated recipe ([`parse_recipe`]'s output).
#[derive(Debug)]
struct BatchRecipe {
    #[allow(dead_code)] // validated == RECIPE_VERSION; kept for the record
    version: u32,
    steps: Vec<BatchStep>,
    on_error: OnError,
}

/// The recipe's top level, `steps` still raw — stage one of the parse
/// (each step deserializes separately so its error can name its index).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecipe {
    /// Raw so a mistyped value (`"1"`) still gets the `version:` error.
    version: serde_json::Value,
    steps: Vec<serde_json::Value>,
    #[serde(default)]
    on_error: OnError,
}

/// One recipe step: `{ "<step-name>": { fields } }`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BatchStep {
    Inspect(InspectStep),
    Bake(BakeStep),
    BakeTransfer(BakeTransferStep),
    Export(ExportStep),
}

impl BatchStep {
    /// The step's recipe key (the `[i] <name>` in the output).
    fn name(&self) -> &'static str {
        match self {
            BatchStep::Inspect(_) => "inspect",
            BatchStep::Bake(_) => "bake",
            BatchStep::BakeTransfer(_) => "bake_transfer",
            BatchStep::Export(_) => "export",
        }
    }
}

fn default_size() -> u32 {
    BakeFlags::default().size
}

fn default_rays() -> u32 {
    BakeFlags::default().rays
}

fn default_front() -> f32 {
    TransferSettings::default().front
}

fn default_back() -> f32 {
    TransferSettings::default().back
}

fn default_offset() -> f32 {
    TransferSettings::default().offset
}

/// `inspect` — the `inspect` command.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct InspectStep {
    mesh: PathBuf,
}

/// `bake` — `bake-all`, filtered to `maps` (empty/absent = every map).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct BakeStep {
    mesh: PathBuf,
    #[serde(default)]
    maps: Vec<BakeMap>,
    out_dir: PathBuf,
    #[serde(default = "default_size")]
    size: u32,
    #[serde(default = "default_rays")]
    rays: u32,
    #[serde(default)]
    dilate: u32,
}

/// `bake_transfer` — HIGH-to-LOW transfer (`maps` empty/absent = every
/// transfer map).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct BakeTransferStep {
    low: PathBuf,
    high: PathBuf,
    #[serde(default)]
    maps: Vec<TransferMapName>,
    out_dir: PathBuf,
    #[serde(default = "default_size")]
    size: u32,
    #[serde(default = "default_front")]
    front: f32,
    #[serde(default = "default_back")]
    back: f32,
    #[serde(default = "default_offset")]
    offset: f32,
}

/// `export` — the `export` command (`tiles` empty/absent = every
/// present tile).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportStep {
    mesh: PathBuf,
    preset: String,
    out_dir: PathBuf,
    #[serde(default = "default_size")]
    size: u32,
    #[serde(default = "default_rays")]
    rays: u32,
    #[serde(default)]
    tiles: Vec<u16>,
}

/// `steps[i]`, plus `.<key>` when the step is a single-key object — the
/// JSON path prefixed onto a step's errors.
fn step_path(index: usize, value: &serde_json::Value) -> String {
    match value.as_object() {
        Some(obj) if obj.len() == 1 => {
            format!("steps[{index}].{}", obj.keys().next().expect("one key"))
        }
        _ => format!("steps[{index}]"),
    }
}

/// Parses and validates a recipe — everything checked before any step
/// runs: the version, every step's shape (unknown step/field/map name =
/// an error naming the JSON path), the export presets and tiles.
fn parse_recipe(text: &str) -> Result<BatchRecipe> {
    let raw: RawRecipe = serde_json::from_str(text).map_err(|e| anyhow::anyhow!("recipe: {e}"))?;
    if raw.version.as_u64() != Some(u64::from(RECIPE_VERSION)) {
        return Err(anyhow::anyhow!(
            "version: unsupported recipe version {} (this build reads {RECIPE_VERSION})",
            raw.version
        ));
    }
    if raw.steps.is_empty() {
        return Err(anyhow::anyhow!("steps: the recipe has no steps"));
    }
    let mut steps = Vec::with_capacity(raw.steps.len());
    for (i, value) in raw.steps.into_iter().enumerate() {
        let path = step_path(i, &value);
        let step: BatchStep =
            serde_json::from_value(value).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        validate_step(&step).map_err(|e| anyhow::anyhow!("{path}.{e}"))?;
        steps.push(step);
    }
    Ok(BatchRecipe {
        version: RECIPE_VERSION,
        steps,
        on_error: raw.on_error,
    })
}

/// The checks serde can't express; errors start with the field name.
fn validate_step(step: &BatchStep) -> Result<()> {
    if let BatchStep::Export(s) = step {
        preset_by_name(&s.preset).map_err(|e| anyhow::anyhow!("preset: {e}"))?;
        for (j, tile) in s.tiles.iter().enumerate() {
            if !UDIM_GRID.contains(tile) {
                return Err(anyhow::anyhow!(
                    "tiles[{j}]: {tile} is outside the UDIM grid 1001..=1100"
                ));
            }
        }
    }
    Ok(())
}

/// Runs one step through its command's runner; `Ok` = the `ok (...)`
/// detail.
fn run_step(step: &BatchStep) -> Result<String> {
    match step {
        BatchStep::Inspect(s) => {
            let mesh = inspect_mesh(&s.mesh)?;
            Ok(format!(
                "{} vertices, {} triangles",
                mesh.vertex_count(),
                mesh.triangle_count()
            ))
        }
        BatchStep::Bake(s) => {
            let maps = if s.maps.is_empty() {
                BakeMap::ALL.to_vec()
            } else {
                s.maps.clone()
            };
            let flags = BakeFlags {
                size: s.size,
                rays: s.rays,
                dilate: s.dilate,
            };
            let (set, written) = run_bake(&s.mesh, &s.out_dir, &flags, &maps)?;
            Ok(format!(
                "{} maps for '{set}' -> {}",
                written.len(),
                s.out_dir.display()
            ))
        }
        BatchStep::BakeTransfer(s) => {
            let maps = if s.maps.is_empty() {
                TransferMapName::ALL.to_vec()
            } else {
                s.maps.clone()
            };
            let settings = TransferSettings {
                size: s.size,
                front: s.front,
                back: s.back,
                offset: s.offset,
            };
            let (set, written) = run_bake_transfer(&s.low, &s.high, &s.out_dir, &maps, &settings)?;
            Ok(format!(
                "{} maps for '{set}' -> {}",
                written.len(),
                s.out_dir.display()
            ))
        }
        BatchStep::Export(s) => {
            let flags = BakeFlags {
                size: s.size,
                rays: s.rays,
                dilate: 0,
            };
            let (written, tiles) = run_export(&s.mesh, &s.out_dir, &s.preset, &flags, &s.tiles)?;
            Ok(format!(
                "{} outputs, preset '{}', tiles {tiles:?} -> {}",
                written.len(),
                s.preset,
                s.out_dir.display()
            ))
        }
    }
}

/// One executed step's result: `Ok(detail)` or `Err(message)`.
struct StepResult {
    name: &'static str,
    outcome: std::result::Result<String, String>,
}

/// The run's per-step results (only the steps that ran: `stop` halts
/// early).
struct BatchReport {
    /// The recipe's step count (`N` in the summary).
    total: usize,
    results: Vec<StepResult>,
    on_error: OnError,
}

impl BatchReport {
    fn failed(&self) -> usize {
        self.results.iter().filter(|r| r.outcome.is_err()).count()
    }

    fn ok(&self) -> usize {
        self.results.len() - self.failed()
    }

    /// `batch: N steps, M ok, K failed` — under `stop`, `M + K < N` when
    /// steps were left unrun.
    fn summary_line(&self) -> String {
        format!(
            "batch: {} steps, {} ok, {} failed",
            self.total,
            self.ok(),
            self.failed()
        )
    }

    /// `Ok` iff no step failed (exit 0 only when K = 0); under `stop` the
    /// error names the failing step's index.
    fn into_result(self) -> Result<()> {
        let Some(i) = self.results.iter().position(|r| r.outcome.is_err()) else {
            return Ok(());
        };
        match self.on_error {
            OnError::Stop => {
                let r = &self.results[i];
                let err = r.outcome.as_ref().err().map_or("", String::as_str);
                Err(anyhow::anyhow!(
                    "batch stopped at step [{i}] {}: {err}",
                    r.name
                ))
            }
            OnError::Continue => Err(anyhow::anyhow!(
                "batch: {} of {} steps failed",
                self.failed(),
                self.total
            )),
        }
    }
}

/// Executes the recipe's steps in order, writing each step's `[i]` line
/// to `out` (the summary line is the caller's: [`BatchReport::summary_line`]).
fn run_batch(recipe: &BatchRecipe, out: &mut impl Write) -> BatchReport {
    let mut results = Vec::with_capacity(recipe.steps.len());
    for (i, step) in recipe.steps.iter().enumerate() {
        let name = step.name();
        // One line per step: flatten any multi-line error.
        let outcome = run_step(step).map_err(|e| format!("{e:#}").replace('\n', " "));
        let _ = match &outcome {
            Ok(detail) => writeln!(out, "[{i}] {name} ok ({detail})"),
            Err(err) => writeln!(out, "[{i}] {name} FAILED: {err}"),
        };
        let failed = outcome.is_err();
        results.push(StepResult { name, outcome });
        if failed && recipe.on_error == OnError::Stop {
            break;
        }
    }
    BatchReport {
        total: recipe.steps.len(),
        results,
        on_error: recipe.on_error,
    }
}

/// `batch <recipe.json>` — validate the whole recipe, run its steps,
/// print the summary; errors (exit 1) when any step failed.
pub(crate) fn batch_cmd(args: &[String]) -> Result<()> {
    let usage = "usage: umber-cli batch <recipe.json>";
    let path = args.first().ok_or_else(|| anyhow::anyhow!("{usage}"))?;
    if let Some(extra) = args.get(1) {
        return Err(anyhow::anyhow!("unexpected argument: {extra}\n{usage}"));
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading recipe {path}: {e}"))?;
    let recipe = parse_recipe(&text)?;
    let report = run_batch(&recipe, &mut std::io::stdout());
    println!("{}", report.summary_line());
    report.into_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The design doc's full example (`on_error` picked from its
    /// `"stop" | "continue"` placeholder).
    const DESIGN_EXAMPLE: &str = r#"{
      "version": 1,
      "steps": [
        { "bake": { "mesh": "sword.fbx", "maps": ["ao", "curvature", "tangent-normal"],
                     "out_dir": "bakes/", "size": 2048, "dilate": 8 } },
        { "bake_transfer": { "low": "sword_low.obj", "high": "sword_high.fbx",
                     "maps": ["height", "world-normal"], "out_dir": "transfer/" } },
        { "export": { "mesh": "sword.fbx", "preset": "unreal", "out_dir": "out/",
                     "tiles": [1001, 1002] } },
        { "inspect": { "mesh": "sword.fbx" } }
      ],
      "on_error": "stop"
    }"#;

    /// A fresh scratch dir per test (tests share one process: the tag
    /// keeps them apart).
    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("umber-cli-batch-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// A unit quad OBJ (UVs over [0,1], +Z normal) — the CLI has no
    /// checked-in mesh fixture, so the tests write one.
    fn write_quad_obj(dir: &Path) -> PathBuf {
        let path = dir.join("quad.obj");
        std::fs::write(
            &path,
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
             vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
             vn 0 0 1\n\
             f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n",
        )
        .expect("write quad.obj");
        path
    }

    /// The bake output path for `kind` (the runner's naming).
    fn map_path(mesh: &Path, out_dir: &Path, kind: umber_mesh::MeshMapKind) -> PathBuf {
        let set = umber_mesh::texture_set_name(mesh, &umber_mesh::load(mesh).expect("load"));
        umber_mesh::format_mesh_map(out_dir, &set, kind, "png")
    }

    /// Skips gracefully (umber-bake's GPU-test convention) when no
    /// adapter is available in this environment.
    fn gpu_available() -> bool {
        let instance = wgpu::Instance::default();
        if pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .is_err()
        {
            eprintln!("skipping: no wgpu adapter available");
            return false;
        }
        true
    }

    fn file_count(dir: &Path) -> usize {
        std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
    }

    fn run(recipe: &serde_json::Value) -> (BatchReport, String) {
        let recipe = parse_recipe(&recipe.to_string()).expect("recipe validates");
        let mut out = Vec::new();
        let report = run_batch(&recipe, &mut out);
        (report, String::from_utf8(out).expect("utf-8 output"))
    }

    #[test]
    fn the_design_example_validates() {
        let recipe = parse_recipe(DESIGN_EXAMPLE).expect("the design example validates");
        assert_eq!(recipe.version, 1);
        assert_eq!(recipe.on_error, OnError::Stop);
        let names: Vec<_> = recipe.steps.iter().map(BatchStep::name).collect();
        assert_eq!(names, ["bake", "bake_transfer", "export", "inspect"]);
        let BatchStep::Bake(bake) = &recipe.steps[0] else {
            panic!("step 0 is bake")
        };
        assert_eq!(
            bake.maps,
            [BakeMap::Ao, BakeMap::Curvature, BakeMap::TangentNormal]
        );
        assert_eq!((bake.size, bake.rays, bake.dilate), (2048, 16, 8));
        let BatchStep::BakeTransfer(transfer) = &recipe.steps[1] else {
            panic!("step 1 is bake_transfer")
        };
        assert_eq!(
            transfer.maps,
            [TransferMapName::Height, TransferMapName::WorldNormal]
        );
        assert_eq!(transfer.size, 512);
        let BatchStep::Export(export) = &recipe.steps[2] else {
            panic!("step 2 is export")
        };
        assert_eq!(export.preset, "unreal");
        assert_eq!(export.tiles, [1001, 1002]);
    }

    #[test]
    fn on_error_defaults_to_stop() {
        let recipe =
            parse_recipe(r#"{ "version": 1, "steps": [{ "inspect": { "mesh": "a.obj" } }] }"#)
                .expect("validates");
        assert_eq!(recipe.on_error, OnError::Stop);
    }

    #[test]
    fn validation_errors_name_the_json_path() {
        let err = |text: &str| parse_recipe(text).unwrap_err().to_string();

        // Unknown step name: serde's message carries no path, so the
        // step index is prefixed.
        let e = err(r#"{ "version": 1, "steps": [
            { "inspect": { "mesh": "a.obj" } },
            { "bakee": { "mesh": "a.obj", "out_dir": "o" } } ] }"#);
        assert!(e.starts_with("steps[1].bakee:"), "{e}");
        assert!(e.contains("unknown variant `bakee`"), "{e}");

        // Unknown field inside a step.
        let e = err(r#"{ "version": 1, "steps": [
            { "bake": { "mesh": "a.obj", "out_dir": "o", "sise": 64 } } ] }"#);
        assert!(e.starts_with("steps[0].bake:"), "{e}");
        assert!(e.contains("unknown field `sise`"), "{e}");

        // Unknown map name (bake-all's baker list only).
        let e = err(r#"{ "version": 1, "steps": [
            { "bake": { "mesh": "a.obj", "out_dir": "o", "maps": ["ao", "id"] } } ] }"#);
        assert!(e.starts_with("steps[0].bake:"), "{e}");
        assert!(e.contains("`id`"), "{e}");

        // Unknown transfer map.
        let e = err(r#"{ "version": 1, "steps": [ { "bake_transfer": {
            "low": "l.obj", "high": "h.obj", "out_dir": "o", "maps": ["ao"] } } ] }"#);
        assert!(e.starts_with("steps[0].bake_transfer:"), "{e}");

        // Unknown preset / off-grid tile.
        let e = err(
            r#"{ "version": 1, "steps": [ { "inspect": { "mesh": "a.obj" } },
            { "export": { "mesh": "a.obj", "out_dir": "o", "preset": "maya" } } ] }"#,
        );
        assert!(e.starts_with("steps[1].export.preset:"), "{e}");
        let e = err(r#"{ "version": 1, "steps": [ { "export": {
            "mesh": "a.obj", "out_dir": "o", "preset": "gltf", "tiles": [1001, 999] } } ] }"#);
        assert!(e.starts_with("steps[0].export.tiles[1]:"), "{e}");

        // Version, unknown top-level field, bad on_error.
        let e = err(r#"{ "version": 2, "steps": [{ "inspect": { "mesh": "a.obj" } }] }"#);
        assert!(e.starts_with("version:"), "{e}");
        let e = err(r#"{ "version": "1", "steps": [{ "inspect": { "mesh": "a.obj" } }] }"#);
        assert!(e.starts_with("version:"), "{e}");
        let e = err(r#"{ "version": 1, "steps": [], "on_eror": "stop" }"#);
        assert!(e.contains("on_eror"), "{e}");
        let e = err(
            r#"{ "version": 1, "steps": [{ "inspect": { "mesh": "a.obj" } }],
            "on_error": "retry" }"#,
        );
        assert!(e.contains("retry"), "{e}");
    }

    #[test]
    fn bake_then_export_writes_both_and_summarizes() {
        if !gpu_available() {
            return;
        }
        let dir = scratch("bake-export");
        let mesh = write_quad_obj(&dir);
        let bakes = dir.join("bakes");
        let out = dir.join("out");
        let (report, lines) = run(&serde_json::json!({
            "version": 1,
            "steps": [
                { "bake": { "mesh": mesh, "maps": ["ao", "tangent-normal"],
                            "out_dir": bakes, "size": 32, "rays": 4 } },
                { "export": { "mesh": mesh, "preset": "gltf", "out_dir": out,
                              "size": 32, "rays": 4 } }
            ]
        }));

        let lines: Vec<&str> = lines.lines().collect();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].starts_with("[0] bake ok ("), "{lines:?}");
        assert!(lines[1].starts_with("[1] export ok ("), "{lines:?}");
        assert!(map_path(&mesh, &bakes, umber_mesh::MeshMapKind::AmbientOcclusion).is_file());
        assert!(map_path(&mesh, &bakes, umber_mesh::MeshMapKind::NormalBase).is_file());
        assert!(file_count(&out) >= 1, "export wrote nothing");

        // The summary line parses: `batch: N steps, M ok, K failed`.
        let summary = report.summary_line();
        let nums: Vec<usize> = summary
            .strip_prefix("batch: ")
            .expect("summary prefix")
            .split(", ")
            .map(|part| {
                part.split(' ')
                    .next()
                    .and_then(|n| n.parse().ok())
                    .expect("count")
            })
            .collect();
        assert_eq!(nums, [2, 2, 0], "{summary}");
        assert_eq!(summary, "batch: 2 steps, 2 ok, 0 failed");
        assert!(report.into_result().is_ok());
    }

    /// Step 0 bakes a missing mesh (fails at load, before the GPU);
    /// step 1 is a good AO bake.
    fn bad_then_good(dir: &Path, on_error: &str) -> (serde_json::Value, PathBuf, PathBuf) {
        let mesh = write_quad_obj(dir);
        let out = dir.join("good");
        let recipe = serde_json::json!({
            "version": 1,
            "steps": [
                { "bake": { "mesh": dir.join("missing.obj"), "maps": ["ao"],
                            "out_dir": dir.join("bad"), "size": 32, "rays": 4 } },
                { "bake": { "mesh": mesh, "maps": ["ao"], "out_dir": out,
                            "size": 32, "rays": 4 } }
            ],
            "on_error": on_error
        });
        (recipe, mesh, out)
    }

    #[test]
    fn on_error_continue_runs_every_step_and_fails() {
        if !gpu_available() {
            return;
        }
        let dir = scratch("continue");
        let (recipe, mesh, out) = bad_then_good(&dir, "continue");
        let (report, lines) = run(&recipe);

        assert_eq!(report.results.len(), 2);
        assert!(report.results[0].outcome.is_err());
        assert!(report.results[1].outcome.is_ok());
        assert!(lines.contains("[0] bake FAILED: "), "{lines}");
        assert!(lines.contains("[1] bake ok ("), "{lines}");
        // Step 1 still ran.
        assert!(map_path(&mesh, &out, umber_mesh::MeshMapKind::AmbientOcclusion).is_file());
        assert_eq!(report.summary_line(), "batch: 2 steps, 1 ok, 1 failed");
        // Exit 1.
        let err = report.into_result().unwrap_err().to_string();
        assert!(err.contains("1 of 2"), "{err}");
    }

    #[test]
    fn on_error_stop_halts_at_the_first_failure() {
        // No GPU needed: step 0 fails at mesh load and step 1 never runs.
        let dir = scratch("stop");
        let (recipe, mesh, out) = bad_then_good(&dir, "stop");
        let (report, lines) = run(&recipe);

        assert_eq!(report.results.len(), 1);
        assert!(lines.starts_with("[0] bake FAILED: "), "{lines}");
        assert!(!lines.contains("[1]"), "{lines}");
        // Step 1 did NOT run.
        assert!(!map_path(&mesh, &out, umber_mesh::MeshMapKind::AmbientOcclusion).exists());
        assert!(!out.exists());
        assert_eq!(report.summary_line(), "batch: 2 steps, 0 ok, 1 failed");
        // Exit 1, naming the step index.
        let err = report.into_result().unwrap_err().to_string();
        assert!(err.contains("step [0] bake"), "{err}");
    }

    #[test]
    fn bake_maps_filter_bakes_only_the_listed_maps() {
        if !gpu_available() {
            return;
        }
        let dir = scratch("filter");
        let mesh = write_quad_obj(&dir);
        let out = dir.join("ao-only");
        let (report, _) = run(&serde_json::json!({
            "version": 1,
            "steps": [ { "bake": { "mesh": mesh, "maps": ["ao"], "out_dir": out,
                                   "size": 32, "rays": 4 } } ]
        }));

        assert!(report.into_result().is_ok());
        assert_eq!(file_count(&out), 1, "exactly one map written");
        assert!(map_path(&mesh, &out, umber_mesh::MeshMapKind::AmbientOcclusion).is_file());
    }
}

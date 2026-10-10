//! `agent` — the AI interface hook (docs/specs/ai-hook-design.md): an
//! interactive JSONL session over the batch step vocabulary, for an
//! external agent (an AI assistant, a pipeline robot, a script) driving
//! umber without the GUI.
//!
//! Each stdin line is one step, written in the recipe's shape
//! (`{ "<step>": { fields } }`). The no-field steps also accept a bare
//! `"<step>"`, `{ "<step>": null }` or `{ "<step>": {} }`. Blank lines
//! are skipped. Each step gets exactly one stdout line in reply:
//! `{"ok": true, "step": "<name>", "index": i, ...result fields}` or
//! `{"ok": false, "step": "<name>", "index": i, "error": "..."}`, where
//! `i` counts the non-blank lines and `step` is `null` when the line
//! names no step. A failing step never ends the session. `quit` replies
//! and then exits 0; stdin EOF also exits 0, so a crashed harness never
//! leaves an orphaned process. Stdout carries only these JSON lines.
//! Logs and diagnostics go to stderr.
//!
//! The session holds one mesh (`load`). `state`, `bake` and `export`
//! work on that mesh. `inspect` and `bake_transfer` are the batch steps
//! unchanged and stay stateless.
//!
//! `bake` runs asynchronously on a session-owned thread, and the client
//! polls it with `bake_status`. The app's async bake worker (7bf6140)
//! lives in umber-app, which umber-cli does not depend on and should not
//! (that would pull egui into the headless binary). So this module has
//! its own minimal job: a `JoinHandle` plus a shared status. The thread
//! runs the same bake functions the batch `bake` step uses
//! ([`bake_context`], [`bake_mesh_maps`]). The panel's worker stays on
//! the app side.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::batch::{default_rays, default_size, BakeTransferStep, InspectStep};
use crate::{
    bake_context, bake_mesh_maps, export_mesh, mesh_summary, select_tiles, BakeFlags, BakeMap,
    MeshSummary, UDIM_GRID,
};

/// One session step. The batch's `inspect` / `bake_transfer` reuse the
/// recipe structs. `bake` / `export` drop the recipe's `mesh` field
/// because they act on the held mesh.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AgentStep {
    Load(LoadStep),
    State(NoFields),
    Inspect(InspectStep),
    Bake(AgentBakeStep),
    BakeStatus(NoFields),
    BakeTransfer(BakeTransferStep),
    Export(AgentExportStep),
    Quit(NoFields),
}

/// The body of a step that takes no fields (`{}`; `null` and the bare
/// name are normalized to it by [`normalize_step`]).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoFields {}

/// `load` — load a mesh and hold it for the session.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadStep {
    mesh: PathBuf,
}

/// `bake` — the batch `bake` run on the held mesh. `maps` empty or
/// absent bakes every map. `tiles` empty or absent bakes the whole mesh
/// with the batch's untiled names. Otherwise each requested tile is
/// baked from its own triangles and named `<set>_<map>_<tile>.png`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentBakeStep {
    #[serde(default)]
    maps: Vec<BakeMap>,
    out_dir: PathBuf,
    #[serde(default = "default_size")]
    size: u32,
    #[serde(default = "default_rays")]
    rays: u32,
    #[serde(default)]
    dilate: u32,
    #[serde(default)]
    tiles: Vec<u16>,
}

/// `export` — the batch `export` run on the held mesh. `tiles` empty or
/// absent exports every present tile.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentExportStep {
    preset: String,
    out_dir: PathBuf,
    #[serde(default = "default_size")]
    size: u32,
    #[serde(default = "default_rays")]
    rays: u32,
    #[serde(default)]
    tiles: Vec<u16>,
}

/// The async bake's state, as `bake_status` reports it
/// (`{"status": "pending" | "running" | "done" | "failed", ...}`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum BakeStatus {
    /// The thread is spawned but has not started baking yet.
    Pending,
    Running,
    Done {
        texture_set: String,
        written: Vec<PathBuf>,
    },
    Failed {
        error: String,
    },
}

impl BakeStatus {
    fn is_finished(&self) -> bool {
        matches!(self, BakeStatus::Done { .. } | BakeStatus::Failed { .. })
    }
}

/// The session's bake job: the worker thread and the status it reports.
struct BakeJob {
    status: Arc<Mutex<BakeStatus>>,
    /// Taken (joined) once the thread has finished.
    handle: Option<JoinHandle<()>>,
}

/// Locks the status. A panicking bake thread never holds the lock while
/// baking, but a poisoned lock still carries the last status written.
fn lock(status: &Mutex<BakeStatus>) -> MutexGuard<'_, BakeStatus> {
    status
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl BakeJob {
    /// The current status. Once the thread has finished it is joined
    /// here. A thread that ended without writing a final status
    /// (a panic) is reported as `failed`, never left `running`.
    fn poll(&mut self) -> BakeStatus {
        if self.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            let joined = self.handle.take().expect("checked above").join();
            let mut status = lock(&self.status);
            if !status.is_finished() {
                *status = BakeStatus::Failed {
                    error: match joined {
                        Err(_) => "the bake thread panicked".to_string(),
                        Ok(()) => "the bake thread ended without a result".to_string(),
                    },
                };
            }
        }
        lock(&self.status).clone()
    }
}

/// The held mesh: its path (which names the texture set and the
/// `$mesh` token) and its summary.
struct Loaded {
    path: PathBuf,
    mesh: Arc<umber_mesh::MeshData>,
    summary: MeshSummary,
}

/// A session's state between steps.
#[derive(Default)]
struct Session {
    loaded: Option<Loaded>,
    bake: Option<BakeJob>,
}

/// A step's outcome: the result fields to merge into the reply.
type Fields = Map<String, Value>;

fn fields(value: impl Serialize) -> Result<Fields> {
    match serde_json::to_value(value)? {
        Value::Object(map) => Ok(map),
        other => Err(anyhow::anyhow!("internal: non-object result {other}")),
    }
}

/// `{"summary": ...}`, the reply to `load`, `state` and `inspect`.
fn summary_fields(summary: &MeshSummary) -> Result<Fields> {
    let mut map = Fields::new();
    map.insert("summary".into(), serde_json::to_value(summary)?);
    Ok(map)
}

/// The step name a line's JSON carries: the bare string, or the single
/// key of a one-key object.
fn step_name(value: &Value) -> Option<String> {
    match value {
        Value::String(name) => Some(name.clone()),
        Value::Object(obj) if obj.len() == 1 => obj.keys().next().cloned(),
        _ => None,
    }
}

/// Converts the shorthand forms of a no-field step to serde's struct
/// form: `"quit"` and `{"quit": null}` become `{"quit": {}}`. Any other
/// value is returned unchanged.
fn normalize_step(value: Value) -> Value {
    match value {
        Value::String(name) => {
            let mut obj = Map::new();
            obj.insert(name, Value::Object(Map::new()));
            Value::Object(obj)
        }
        Value::Object(mut obj) if obj.len() == 1 => {
            if let Some(body) = obj.values_mut().next() {
                if body.is_null() {
                    *body = Value::Object(Map::new());
                }
            }
            Value::Object(obj)
        }
        other => other,
    }
}

impl Session {
    /// The held mesh, or the honest error.
    fn loaded(&self) -> Result<&Loaded> {
        self.loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no mesh loaded (send a `load` step first)"))
    }

    /// Runs one step and returns its result fields. `Quit` is handled by
    /// the caller ([`Session::reply`]).
    fn run(&mut self, step: AgentStep) -> Result<Fields> {
        match step {
            AgentStep::Load(s) => {
                let mesh = umber_mesh::load(&s.mesh)?;
                let summary = mesh_summary(&s.mesh, &mesh);
                let reply = summary_fields(&summary)?;
                self.loaded = Some(Loaded {
                    path: s.mesh,
                    mesh: Arc::new(mesh),
                    summary,
                });
                Ok(reply)
            }
            AgentStep::State(NoFields {}) => summary_fields(&self.loaded()?.summary),
            AgentStep::Inspect(s) => {
                let mesh = umber_mesh::load(&s.mesh)?;
                summary_fields(&mesh_summary(&s.mesh, &mesh))
            }
            AgentStep::Bake(s) => self.start_bake(s),
            AgentStep::BakeStatus(NoFields {}) => {
                let job = self
                    .bake
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("no bake started (send `bake` first)"))?;
                fields(job.poll())
            }
            AgentStep::BakeTransfer(s) => {
                let (texture_set, written) = s.run()?;
                fields(serde_json::json!({ "texture_set": texture_set, "written": written }))
            }
            AgentStep::Export(s) => {
                for (j, tile) in s.tiles.iter().enumerate() {
                    if !UDIM_GRID.contains(tile) {
                        return Err(anyhow::anyhow!(
                            "tiles[{j}]: {tile} is outside the UDIM grid 1001..=1100"
                        ));
                    }
                }
                let loaded = self.loaded()?;
                let flags = BakeFlags {
                    size: s.size,
                    rays: s.rays,
                    dilate: 0,
                };
                let mut skipped = Vec::new();
                let (written, tiles) = export_mesh(
                    &loaded.path,
                    &loaded.mesh,
                    &s.out_dir,
                    &s.preset,
                    &flags,
                    &s.tiles,
                    &mut |name: &str| skipped.push(name.to_string()),
                )?;
                fields(serde_json::json!({
                    "written": written,
                    "tiles": tiles,
                    "skipped": skipped,
                }))
            }
            AgentStep::Quit(NoFields {}) => unreachable!("quit is answered by Session::reply"),
        }
    }

    /// `bake`: validates the request synchronously (held mesh, no bake in
    /// flight, tiles), then starts the bake thread and replies
    /// `{"status": "started"}`.
    fn start_bake(&mut self, s: AgentBakeStep) -> Result<Fields> {
        if let Some(job) = &mut self.bake {
            if !job.poll().is_finished() {
                return Err(anyhow::anyhow!(
                    "a bake is already in flight (poll `bake_status` until it is done or failed)"
                ));
            }
        }
        let loaded = self.loaded()?;
        let present = &loaded.summary.tiles;
        let tiles = if s.tiles.is_empty() {
            Vec::new()
        } else {
            let tiles = select_tiles(&s.tiles, present)?;
            // Export's whole-mesh rule: a mesh living only in 1001 bakes
            // whole, with the untiled names (its triangles may be tagged
            // 1012 by a UV on the far edge, so filtering would drop them).
            if *present == [umber_mesh::FIRST_TILE] {
                Vec::new()
            } else {
                tiles
            }
        };
        let maps = if s.maps.is_empty() {
            BakeMap::ALL.to_vec()
        } else {
            s.maps
        };
        let flags = BakeFlags {
            size: s.size,
            rays: s.rays,
            dilate: s.dilate,
        };
        let set = loaded.summary.texture_set.clone();
        let mesh = Arc::clone(&loaded.mesh);
        let out_dir = s.out_dir;

        let status = Arc::new(Mutex::new(BakeStatus::Pending));
        let worker_status = Arc::clone(&status);
        let handle = std::thread::Builder::new()
            .name("umber-agent-bake".into())
            .spawn(move || {
                *lock(&worker_status) = BakeStatus::Running;
                let result = bake_held(&mesh, &set, &out_dir, &flags, &maps, &tiles);
                *lock(&worker_status) = match result {
                    Ok(written) => BakeStatus::Done {
                        texture_set: set,
                        written,
                    },
                    Err(e) => BakeStatus::Failed {
                        error: format!("{e:#}"),
                    },
                };
            })?;
        self.bake = Some(BakeJob {
            status,
            handle: Some(handle),
        });
        fields(serde_json::json!({ "status": "started" }))
    }

    /// The reply line for input line `line` (step `index`), and whether
    /// the session should end.
    fn reply(&mut self, index: usize, line: &str) -> (Value, bool) {
        let (name, step) = parse_line(line);
        let (outcome, quit) = match step {
            Ok(AgentStep::Quit(NoFields {})) => {
                let in_flight = self
                    .bake
                    .as_mut()
                    .is_some_and(|job| !job.poll().is_finished());
                let mut map = Fields::new();
                // An unfinished bake is abandoned at exit; say so.
                map.insert("bake_in_flight".into(), Value::Bool(in_flight));
                (Ok(map), true)
            }
            Ok(step) => (self.run(step), false),
            Err(e) => (Err(e), false),
        };
        (envelope(index, name, outcome), quit)
    }
}

/// Parses one input line: the step name it carries (for the reply, even
/// when the step itself is invalid) and the typed step.
fn parse_line(line: &str) -> (Option<String>, Result<AgentStep>) {
    let value: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(e) => return (None, Err(anyhow::anyhow!("not a JSON step: {e}"))),
    };
    let name = step_name(&value);
    let step = serde_json::from_value(normalize_step(value)).map_err(anyhow::Error::from);
    (name, step)
}

/// The reply line: `ok` / `step` / `index`, then the result fields or
/// the `error`.
fn envelope(index: usize, step: Option<String>, outcome: Result<Fields>) -> Value {
    let mut map = Fields::new();
    map.insert("ok".into(), Value::Bool(outcome.is_ok()));
    map.insert("step".into(), step.map_or(Value::Null, Value::String));
    map.insert("index".into(), Value::from(index));
    match outcome {
        Ok(fields) => {
            for (key, value) in fields {
                map.insert(key, value);
            }
        }
        Err(e) => {
            map.insert("error".into(), Value::String(format!("{e:#}")));
        }
    }
    Value::Object(map)
}

/// Bakes the held mesh: the whole mesh when `tiles` is empty (the batch
/// `bake` behavior), otherwise each tile's own triangles.
fn bake_held(
    mesh: &umber_mesh::MeshData,
    set: &str,
    out_dir: &Path,
    flags: &BakeFlags,
    maps: &[BakeMap],
    tiles: &[u16],
) -> Result<Vec<PathBuf>> {
    let ctx = bake_context()?;
    if tiles.is_empty() {
        return bake_mesh_maps(&ctx, mesh, set, out_dir, flags, maps, None);
    }
    let mut written = Vec::new();
    for &tile in tiles {
        let tile_mesh = umber_bake::ao::filter_mesh_for_tile(mesh, tile);
        written.extend(bake_mesh_maps(
            &ctx,
            &tile_mesh,
            set,
            out_dir,
            flags,
            maps,
            Some(tile),
        )?);
    }
    Ok(written)
}

/// Runs the session over `input` until `quit` or EOF, writing one reply
/// line per step to `out` (flushed per line, because the harness is
/// waiting on it).
fn run_session(input: impl BufRead, out: &mut impl Write) -> Result<()> {
    let mut session = Session::default();
    let mut index = 0;
    for line in input.lines() {
        let line = match line {
            Ok(line) => line,
            // Invalid UTF-8: the line is consumed; answer and go on.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                let reply = envelope(index, None, Err(anyhow::anyhow!("not a JSON step: {e}")));
                index += 1;
                if !write_line(out, &reply)? {
                    return Ok(());
                }
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        if line.trim().is_empty() {
            continue;
        }
        let (reply, quit) = session.reply(index, &line);
        index += 1;
        if !write_line(out, &reply)? || quit {
            return Ok(());
        }
    }
    // EOF: the harness is gone (or done). A clean exit either way.
    Ok(())
}

/// Writes and flushes one reply line. Returns `Ok(false)` when the
/// reader has gone away (a broken pipe), which ends the session cleanly
/// just as EOF does.
fn write_line(out: &mut impl Write, reply: &Value) -> Result<bool> {
    match writeln!(out, "{reply}").and_then(|()| out.flush()) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// `agent` — the JSONL session on stdin/stdout.
pub(crate) fn agent_cmd(args: &[String]) -> Result<()> {
    if let Some(extra) = args.first() {
        return Err(anyhow::anyhow!(
            "unexpected argument: {extra}\nusage: umber-cli agent (steps as JSON lines on stdin)"
        ));
    }
    // Stdout is locked per write, not for the session. A stray println
    // elsewhere then shows up as a non-JSON line (which the tests catch)
    // instead of deadlocking on the lock.
    run_session(std::io::stdin().lock(), &mut std::io::stdout())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `input` through a session and parses every reply line (each
    /// must be JSON: the stdout discipline).
    fn session(input: &str) -> Vec<Value> {
        let mut out = Vec::new();
        run_session(input.as_bytes(), &mut out).expect("session runs");
        String::from_utf8(out)
            .expect("utf-8")
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON ({e}): {l}")))
            .collect()
    }

    #[test]
    fn no_field_steps_accept_every_shorthand() {
        let replies = session("\"state\"\n{\"state\": null}\n{\"state\": {}}\n");
        assert_eq!(replies.len(), 3);
        for (i, r) in replies.iter().enumerate() {
            // Each parsed as `state` (no mesh yet: the honest error).
            assert_eq!(r["step"], "state", "{r}");
            assert_eq!(r["index"], i, "{r}");
            assert_eq!(r["ok"], false, "{r}");
            assert!(
                r["error"].as_str().unwrap().contains("no mesh loaded"),
                "{r}"
            );
        }
    }

    #[test]
    fn bad_lines_get_error_envelopes_and_the_loop_goes_on() {
        let replies = session(
            "{\"bakee\": {}}\n\nnot json\n{\"state\": {\"x\": 1}}\n\"bake_status\"\n\"quit\"\n\"state\"\n",
        );
        // The blank line is skipped; nothing after quit runs.
        assert_eq!(replies.len(), 5, "{replies:?}");
        assert_eq!(replies[0]["step"], "bakee");
        assert_eq!(replies[0]["ok"], false);
        assert!(replies[0]["error"]
            .as_str()
            .unwrap()
            .contains("unknown variant"));
        assert_eq!(replies[1]["step"], Value::Null);
        assert_eq!(replies[1]["index"], 1);
        assert!(replies[1]["error"]
            .as_str()
            .unwrap()
            .contains("not a JSON step"));
        assert_eq!(replies[2]["step"], "state");
        assert!(replies[2]["error"]
            .as_str()
            .unwrap()
            .contains("unknown field"));
        assert_eq!(replies[3]["step"], "bake_status");
        assert!(replies[3]["error"]
            .as_str()
            .unwrap()
            .contains("no bake started"));
        assert_eq!(replies[4]["step"], "quit");
        assert_eq!(replies[4]["ok"], true);
        assert_eq!(replies[4]["bake_in_flight"], false);
    }

    #[test]
    fn bake_and_export_need_a_loaded_mesh() {
        let replies = session(
            "{\"bake\": {\"out_dir\": \"o\"}}\n{\"export\": {\"preset\": \"gltf\", \"out_dir\": \"o\"}}\n",
        );
        for r in &replies {
            assert_eq!(r["ok"], false, "{r}");
            assert!(
                r["error"].as_str().unwrap().contains("no mesh loaded"),
                "{r}"
            );
        }
    }

    #[test]
    fn bake_status_serializes_tagged() {
        let done = BakeStatus::Done {
            texture_set: "quad".into(),
            written: vec![PathBuf::from("o/quad_ambient_occlusion.png")],
        };
        let v = serde_json::to_value(&done).unwrap();
        assert_eq!(v["status"], "done");
        assert_eq!(v["texture_set"], "quad");
        assert_eq!(
            serde_json::to_value(BakeStatus::Pending).unwrap()["status"],
            "pending"
        );
    }
}

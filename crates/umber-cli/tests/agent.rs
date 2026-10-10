//! The `agent` session over the process boundary
//! (docs/specs/ai-hook-design.md, tests 1-5). Each test spawns the real
//! binary with piped stdio, the way a harness does.
//!
//! These live in `tests/` rather than the bin's test module because
//! `CARGO_BIN_EXE_umber-cli` (the built binary's path) exists only for
//! integration tests. Every stdout line must parse as JSON, and nothing
//! may follow the last reply: that is the stdout-discipline check. The
//! waits are generous because adapter init on lavapipe or a cold driver
//! can be slow, and a hang fails the test instead of wedging the run.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// A non-GPU step's reply deadline.
const STEP_TIMEOUT: Duration = Duration::from_secs(60);
/// The async bake's deadline (lavapipe is slow).
const BAKE_TIMEOUT: Duration = Duration::from_secs(600);
/// The exit deadline after quit/EOF.
const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// A spawned `umber-cli agent`. Stdout is read on a thread into a
/// channel so every wait has a deadline.
struct Agent {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
}

impl Agent {
    fn spawn() -> Self {
        // The environment passes through (the ICD pin reaches the child).
        let mut child = Command::new(env!("CARGO_BIN_EXE_umber-cli"))
            .arg("agent")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn umber-cli agent");
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let stdin = child.stdin.take();
        Self {
            child,
            stdin,
            lines,
        }
    }

    fn send_raw(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{line}").expect("write step");
        stdin.flush().expect("flush step");
    }

    /// The next reply line, parsed. It must be JSON.
    fn recv(&self, timeout: Duration) -> Value {
        let line = self
            .lines
            .recv_timeout(timeout)
            .unwrap_or_else(|e| panic!("no reply within {timeout:?}: {e}"));
        serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("stdout line is not JSON ({e}): {line}"))
    }

    /// Sends one step and returns its reply.
    fn call(&mut self, step: &Value) -> Value {
        self.send_raw(&step.to_string());
        self.recv(STEP_TIMEOUT)
    }

    /// Closes stdin (EOF).
    fn close_stdin(&mut self) {
        drop(self.stdin.take());
    }

    /// Waits for the exit (killing the child past the deadline), then
    /// asserts stdout carried nothing after the last reply.
    fn wait_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + EXIT_TIMEOUT;
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                break status;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                panic!("agent did not exit within {EXIT_TIMEOUT:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        match self.lines.recv_timeout(STEP_TIMEOUT) {
            Err(RecvTimeoutError::Disconnected) => {}
            Ok(line) => panic!("stray stdout after the last reply: {line}"),
            Err(RecvTimeoutError::Timeout) => panic!("stdout never closed"),
        }
        status
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A fresh scratch dir per test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("umber-cli-agent-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A unit quad OBJ (UVs over [0,1], +Z normal), the same fixture the
/// batch tests write.
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

fn assert_ok(reply: &Value, step: &str) {
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["step"], step, "{reply}");
}

/// Test 1: load + inspect + quit. Three replies that parse and carry
/// the mesh summary.
#[test]
fn session_loop_answers_load_inspect_quit() {
    let dir = scratch("loop");
    let mesh = write_quad_obj(&dir);
    let expected = umber_mesh::load(&mesh).expect("fixture loads");
    let mut agent = Agent::spawn();

    let load = agent.call(&json!({ "load": { "mesh": mesh } }));
    assert_ok(&load, "load");
    assert_eq!(load["index"], 0);
    let summary = &load["summary"];
    assert_eq!(summary["vertices"], expected.vertex_count(), "{load}");
    assert_eq!(summary["triangles"], 2, "{load}");
    assert_eq!(
        summary["texture_set"],
        umber_mesh::texture_set_name(&mesh, &expected),
        "{load}"
    );
    assert_eq!(summary["tiles"], json!([1001]), "{load}");
    let (min, max) = expected.bounds().expect("non-empty fixture");
    assert_eq!(summary["bounds"], json!([min.to_array(), max.to_array()]));

    let inspect = agent.call(&json!({ "inspect": { "mesh": mesh } }));
    assert_ok(&inspect, "inspect");
    assert_eq!(inspect["index"], 1);
    // The same summary function: the stateless inspect matches the load.
    assert_eq!(inspect["summary"], load["summary"]);

    let quit = agent.call(&json!("quit"));
    assert_ok(&quit, "quit");
    assert_eq!(quit["index"], 2);
    assert!(agent.wait_exit().success());
}

/// Test 2: `state` after `load` returns exactly the load's summary, and
/// asking again changes nothing.
#[test]
fn state_after_load_is_the_load_summary() {
    let dir = scratch("state");
    let mesh = write_quad_obj(&dir);
    let mut agent = Agent::spawn();

    // Before any load: the explicit error, not an empty summary.
    let empty = agent.call(&json!("state"));
    assert_eq!(empty["ok"], false, "{empty}");
    assert!(empty["error"].as_str().unwrap().contains("no mesh loaded"));

    let load = agent.call(&json!({ "load": { "mesh": mesh } }));
    assert_ok(&load, "load");
    let state = agent.call(&json!({ "state": {} }));
    assert_ok(&state, "state");
    assert_eq!(state["summary"], load["summary"]);
    let again = agent.call(&json!({ "state": null }));
    assert_eq!(again["summary"], state["summary"]);

    agent.close_stdin();
    assert!(agent.wait_exit().success());
}

/// Test 3: `bake` starts the job, and `bake_status` follows it to a
/// final state. If the machine has no GPU adapter the bake still starts
/// and the status must report that exact failure. That is the only
/// accepted skip; any other failure fails the test.
#[test]
fn bake_runs_async_and_bake_status_follows_it() {
    let dir = scratch("bake");
    let mesh = write_quad_obj(&dir);
    let out = dir.join("bakes");
    let mut agent = Agent::spawn();

    // Polling with no bake started is an error.
    let none = agent.call(&json!("bake_status"));
    assert_eq!(none["ok"], false, "{none}");

    let load = agent.call(&json!({ "load": { "mesh": mesh } }));
    assert_ok(&load, "load");
    let started = agent.call(&json!({ "bake": {
        "maps": ["ao", "tangent-normal"], "out_dir": out, "size": 32, "rays": 4 } }));
    assert_ok(&started, "bake");
    assert_eq!(started["status"], "started");

    let deadline = Instant::now() + BAKE_TIMEOUT;
    let mut seen = Vec::new();
    let last = loop {
        let status = agent.call(&json!("bake_status"));
        assert_ok(&status, "bake_status");
        let name = status["status"]
            .as_str()
            .expect("status string")
            .to_string();
        assert!(
            ["pending", "running", "done", "failed"].contains(&name.as_str()),
            "{status}"
        );
        if seen.last() != Some(&name) {
            seen.push(name.clone());
        }
        if name == "done" || name == "failed" {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "bake still {name} after {BAKE_TIMEOUT:?}"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    eprintln!("bake_status sequence: {seen:?}");
    // pending/running can only ever precede the final state.
    assert!(seen[..seen.len() - 1]
        .iter()
        .all(|s| s == "pending" || s == "running"));

    if last["status"] == "failed" {
        let error = last["error"].as_str().unwrap_or_default();
        assert!(
            error.contains("no suitable GPU adapter"),
            "the bake failed for a reason other than a missing adapter: {last}"
        );
        eprintln!("skipping the bake outputs: no wgpu adapter ({error})");
    } else {
        assert_eq!(
            last["texture_set"], load["summary"]["texture_set"],
            "{last}"
        );
        let written = last["written"].as_array().expect("written paths");
        assert_eq!(written.len(), 2, "{last}");
        for path in written {
            assert!(
                Path::new(path.as_str().unwrap()).is_file(),
                "{path} missing"
            );
        }
    }

    assert_ok(&agent.call(&json!("quit")), "quit");
    assert!(agent.wait_exit().success());
}

/// Test 4: a failing step gets the error envelope with its index, and
/// the session survives (the next step still answers).
#[test]
fn an_error_step_does_not_kill_the_session() {
    let dir = scratch("error");
    let mesh = write_quad_obj(&dir);
    let mut agent = Agent::spawn();

    let unknown = agent.call(&json!({ "bakee": { "out_dir": "o" } }));
    assert_eq!(unknown["ok"], false, "{unknown}");
    assert_eq!(unknown["step"], "bakee");
    assert_eq!(unknown["index"], 0);
    assert!(unknown["error"]
        .as_str()
        .unwrap()
        .contains("unknown variant"));

    agent.send_raw("this is not json");
    let garbage = agent.recv(STEP_TIMEOUT);
    assert_eq!(garbage["ok"], false, "{garbage}");
    assert_eq!(garbage["step"], Value::Null);
    assert_eq!(garbage["index"], 1);

    let missing = agent.call(&json!({ "load": { "mesh": dir.join("missing.obj") } }));
    assert_eq!(missing["ok"], false, "{missing}");
    assert_eq!(missing["step"], "load");
    assert_eq!(missing["index"], 2);
    assert!(!missing["error"].as_str().unwrap().is_empty());

    // Still alive, and the failed load held nothing.
    let state = agent.call(&json!("state"));
    assert_eq!(state["ok"], false, "{state}");
    let load = agent.call(&json!({ "load": { "mesh": mesh } }));
    assert_ok(&load, "load");
    assert_eq!(load["index"], 4);

    assert_ok(&agent.call(&json!("quit")), "quit");
    assert!(agent.wait_exit().success());
}

/// Test 5a: `quit` exits 0 (with its reply).
#[test]
fn quit_exits_cleanly() {
    let mut agent = Agent::spawn();
    let quit = agent.call(&json!({ "quit": {} }));
    assert_ok(&quit, "quit");
    assert_eq!(quit["bake_in_flight"], false);
    assert_eq!(agent.wait_exit().code(), Some(0));
}

/// Test 5b: stdin EOF exits 0 too (the harness-crash rule), whether
/// mid-session or before any step.
#[test]
fn stdin_eof_exits_cleanly() {
    let dir = scratch("eof");
    let mesh = write_quad_obj(&dir);
    let mut agent = Agent::spawn();
    assert_ok(&agent.call(&json!({ "load": { "mesh": mesh } })), "load");
    agent.close_stdin();
    assert_eq!(agent.wait_exit().code(), Some(0));

    let mut silent = Agent::spawn();
    silent.close_stdin();
    assert_eq!(silent.wait_exit().code(), Some(0));
}

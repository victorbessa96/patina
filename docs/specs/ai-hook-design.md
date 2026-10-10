# AI Interface Hook — Wave-6 Design (the final named item)

Wave-6's last slice. Written 2026-10-10 09:34 against the tree at
`e493c60`. The audit's wave-6 list names "AI interface hook" — for
a DCC, the honest v1 is NOT an embedded LLM; it's the **headless
automation surface**: a structured command channel an external
agent (an AI assistant driving the machine, a pipeline robot, a
scripted workflow) uses to operate umber without the GUI.

## The shape: the CLI's structured session

The batch driver (a6d297f) runs recipes one-shot. The AI hook
extends the same step vocabulary to an INTERACTIVE session:

`umber agent` — a stdin/stdout JSONL loop (the MCP-adjacent
shape every agent harness already speaks):
- Each stdin line: a step object (the SAME BatchStep schema as
  the recipe file — inspect/bake/bake_transfer/export — plus
  three session-only steps: `state` (the loaded mesh + tiles +
  texture-set summary), `bake_status` (the async job's state),
  and `quit`).
- Each stdout line: the step's result as JSON (mesh summary, bake
  records, export paths, or the error with the step index) —
  machine-readable, no prose interleaved (errors carry a
  `message` field; the harness parses, the human reads).
- The session holds state between steps (the loaded mesh — a
  `load` step once, then bake/export reuse it — the batch
  recipe's implicit assumption made explicit and interactive).
- GPU access: the same wgpu device request as the batch path
  (works on the 3070 with the ICD pin, on CI's lavapipe, and on
  any headless adapter — no window, no egui).

**Why JSONL and not sockets/RPC:** the process-boundary contract
is the simplest that composes — an agent spawns `umber agent`,
writes lines, reads lines. Every harness (Hermes' terminal tool,
a Python driver, a future MCP server wrapper) wraps that in one
function. Sockets add auth/lifecycle surface with zero v1 gain.

## The steps table (v1 surface)

| Step | Fields | Notes |
|---|---|---|
| load | mesh | once per session; the mesh summary returns |
| state | — | the loaded mesh, present tiles, texture sets |
| inspect | mesh | the batch step, stateless variant |
| bake | maps, out_dir, size, rays, dilate, tiles | the async bake STARTS; `bake_status` polls |
| bake_status | — | pending/running/done(records)/failed(err) |
| bake_transfer | low, high, maps, out_dir | sync (the transfer is one-shot) |
| export | preset, out_dir, tiles | sync |
| quit | — | clean exit |

The async-bake integration reuses the merged worker (7bf6140) —
the agent polls instead of the panel polling; the SAME job
machinery, a different poller.

## Tests (headless; the GPU-gated paths follow the adapter rule)

1. The session loop: spawn the binary (std::process in the test),
   write load+inspect+quit lines, assert the three JSON responses
   parse + carry the expected fields (the mesh summary's vertex
   count, etc.).
2. The state step after load == the load's summary (idempotence).
3. bake → bake_status pending → (join/timeout) done(records
   exist) — the async path through the loop.
4. An error step → the error JSON with the index; the session
   SURVIVES (the next step still answers — never a dead loop).
5. quit → clean exit code 0; stdin EOF → clean exit too (the
   harness crash rule).

## Build

One lean claw slice: the session loop in umber-cli (reusing the
batch step internals — the 1:1 rule), the three session-only
steps, the five tests. No new deps (serde_json is in).

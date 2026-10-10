# CLI Scripting (Batch Mode) — Wave-6 Design

Wave-6's second slice. Written 2026-10-10 07:25 against the tree at
`ec0d9cc`. The audit's wave-6 list names "scripting API on the CLI"
— the honest v1 of that is NOT an embedded language; it's the batch
driver: a JSON recipe file that scripts the existing commands
(inspect/bake/export pipelines), because that's what automation
actually consumes (CI, farm pipelines, artist scripts).

## The contract

`umber batch <recipe.json>` — one command, one file, exit code
reports success:

```json
{
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
  "on_error": "stop" | "continue"
}
```

Each step maps 1:1 to an existing command's internals (the cmd fns
refactor lightly into reusable pieces — the same code paths, no
duplicated logic). Unknown step/field = a validation error naming
the JSON path (fail fast, no silent skips). `on_error: stop` (the
default) halts at the first failure and exits 1 with the step index
+ error; `continue` runs all steps, collects per-step results,
exits 1 if ANY failed — the summary printed as a results table
(step, ok/err, message).

**The output convention**: each step prints `[i] <step-name> ok
(<detail>)` or `[i] <step-name> FAILED: <error>` — line-delimited,
parseable by shell scripts (the automation contract). A final
`batch: N steps, M ok, K failed` line. Exit 0 only when K=0.

## What it deliberately is NOT

- No conditionals/variables/templating — the recipe is data, not a
  language (the honest v1; if recipes need composition, the
  shell/python AROUND the batch command composes — that's what
  pipelines already do).
- No new deps: the recipe is JSON — serde_json is already in the
  workspace tree.

## Steps table (the full v1 surface)

| Step | Fields | Maps to |
|---|---|---|
| inspect | mesh | inspect_cmd |
| bake | mesh, maps[], out_dir, size, rays, dilate | bake_all_cmd (per-map filter — the maps list selects from the baker set: ao/position/world-normal/curvature/thickness/tangent-normal/id/bent/height/world-space... the merged baker names; unknown map = validation error) |
| bake_transfer | low, high, maps[], out_dir, front/back/offset | bake_transfer_mesh (the maps: height/world-normal/tangent-normal) |
| export | mesh, preset, out_dir, size, tiles[] | export_cmd (tiles optional = all present) |

## Tests (headless; the meshes are the checked-in fixtures)

1. Recipe parse: the full example above validates; a bad step
   name/field errors with the JSON path in the message (assert
   both).
2. A two-step recipe (bake + export) on the quad fixture -> exit
   0, both steps' outputs exist, the summary line parses.
3. on_error continue: a recipe with a deliberately bad mesh path
   in step 1 + a good step 2 -> exit 1, step 2 still ran (its
   output exists), the table shows both results.
4. on_error stop: the same recipe -> exit 1, step 2 did NOT run
   (its output absent).
5. The bake maps filter: maps:["ao"] bakes only AO (one output
   file), not the full set.

## Build

One lean claw slice: the recipe types + validation (with the
path-naming errors), the step dispatch reusing the cmd internals,
the five tests. umber-cli only.

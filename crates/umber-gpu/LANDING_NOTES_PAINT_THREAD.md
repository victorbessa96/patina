# umber-gpu paint-thread slab — landing notes

Wave-2 GPU claw: `paint_thread.rs`, the orchestration layer between input
(or whatever else decides *what* to paint) and the already-landed paint
core in `paint.rs` (`PaintTarget` + `PaintCompositor` + `Dab`/`DabBuffer`,
see `LANDING_NOTES_PAINT.md`). `paint.rs` was not rewritten — this consumes
it through its existing public API, plus one additive `PaintError` variant
(see below).

## The contract conflict this task's own brief contained — read first

The task brief's point 3 talks itself into two different, mutually
exclusive conclusions about the overlap contract in the same paragraph:
first stating overlapping dabs must **not** be split across dispatches,
then reversing to "within one dispatch... overlap across workgroups... is
a benign race? NO", and finally landing on "PaintThread serializes all
dabs of a stroke segment through ONE dispatch in exact order."

That final conclusion is wrong, and contradicts the actual, landed,
cross-reviewed contract documented in `paint.rs`'s module docs (quoted
here verbatim since it's the authority):

> a single `PaintCompositor::splat_dabs` call is one compute dispatch, one
> workgroup per dab, with no atomics on the shared storage texture. Dabs
> within one call must not have overlapping bounding circles, or the two
> workgroups race on the shared texels. Overlapping dabs... must be split
> across separate `splat_dabs` calls — wgpu's resource hazard tracking
> orders successive compute passes on the same texture, so a second call
> always sees the first call's writes.

A compute dispatch has no defined order between workgroups and no
atomics on the shared storage texture (confirmed again by reading
`shaders::PAINT_COMPUTE_SHADER::cs_main`: plain `textureLoad`/
`textureStore`, no `atomic<_>` types anywhere). Premultiplied "over" is
not commutative (`splat(red) then splat(blue)` ≠ `splat(blue) then
splat(red)` on their overlap), so "serialize a whole segment through one
dispatch" cannot produce a defined composite the moment that segment's
dabs overlap — which is the *normal* case for a stroke, since consecutive
dabs are spaced sub-radius by design. The task's own GPU-test requirement
(two overlapping dabs must land at ~0.75 alpha, "same math as paint.rs's
test") only has a defined answer at all if those two dabs go through
separate dispatches — paint.rs's own `splat_two_dabs_composites_overlap_alpha_over`
test proves this by construction (two dabs, two `splat_dabs` calls, not
one call with two dabs).

Given the conflict, this implementation follows `paint.rs` (the verified,
landed source) over the task brief's own self-contradicting reasoning.
Flagging this explicitly rather than silently picking a side, since a
reviewer skimming only the task brief would reasonably expect the opposite
design.

## What was built

**`crates/umber-gpu/src/paint_thread.rs`** (new module)

- `PaintThreadCommand` — `Stage { dabs: Vec<Dab> }` / `Clear` / `Resize {
  width, height }`. `Clone`, `Debug`.
- `FrameStats` — `dabs_composited`, `dispatches`, `clears`, `segments`,
  all `u64`. `Copy`, `Default`, `Debug`, `PartialEq`/`Eq`. Cumulative over
  the `PaintThread`'s lifetime; `process_pending` returns the running
  total, already updated, every call — no separate accessor needed.
- `PaintThread` — owns `device`, `queue`, a `PaintCompositor`, a
  `PaintTarget`, a `DabBuffer` (capacity `min(4096,
  compositor.max_dabs_per_batch())` — see "Capacity" below), the `mpsc`
  channel's both ends, and cumulative `FrameStats`.
  - `new(device, queue, width, height) -> Result<Self, PaintError>` —
    fails fast with `PaintError::MissingDeviceFeature` via
    `PaintCompositor::new`, exactly like the paint core itself.
  - `publish(&mut self, cmds: Vec<PaintThreadCommand>) -> Result<(),
    PaintError>` — sends each command on the channel in order.
  - `process_pending(&mut self) -> Result<FrameStats, PaintError>` —
    drains everything queued, applies it to the target on **one**
    `wgpu::CommandEncoder`, submits once. No-ops (no encoder created or
    submitted) when nothing is queued.
  - `paint_target(&self) -> &PaintTarget` — read-only accessor.
- `dabs_may_overlap(a, b) -> bool` (private) — the overlap predicate
  `stage_dabs` uses to decide dispatch boundaries. See "Overlap test"
  below.

**`crates/umber-gpu/src/paint.rs`** — one additive change: a
`PaintError::ChannelClosed` variant, for `mpsc::Sender::send`'s `Result`
(unreachable in practice, since `PaintThread` owns both channel ends for
its whole lifetime, but the API requires handling it and "no unwrap
outside tests" is a hard constraint). Confirmed via `grep -rn PaintError
--include="*.rs" .` across the whole repo before adding it: `PaintError`
is only referenced inside `umber-gpu` itself (no exhaustive `match` in
`umber-app` or elsewhere that a new variant could break).

**`crates/umber-gpu/src/lib.rs`** — added `pub mod paint_thread;` and
re-exports (`PaintThread`, `PaintThreadCommand`, `FrameStats`).

No `Cargo.toml` changes.

## The real contract: segments vs. dispatches

`process_pending`'s `Stage` handling splits a batch into two independent,
differently-motivated layers:

- **Segments** — `dabs.chunks(capacity)`. Exists purely to respect
  `DabBuffer`'s capacity; knows nothing about spatial overlap. The
  capacity-overflow test (4097 dabs, capacity 4096) produces exactly 2
  segments: 4096 + 1.
- **Dispatches** — *within* one segment, dabs are walked in stroke order
  and accumulated into a group; the moment the next dab might overlap
  *any* dab already in the current group, that group flushes as its own
  `splat_dabs` call (one GPU compute dispatch) before the new dab starts
  a fresh group. Non-overlapping dabs share a dispatch (cheaper); any
  pair that might overlap never does.

So `dispatches >= segments` always, and they measure different things —
this is why `FrameStats` has both fields rather than one. Within a
`process_pending` call, every dispatch for a given `Stage` command lands
on the same encoder in order, so wgpu's inter-pass hazard tracking (the
same mechanism `paint.rs`'s own overlap test relies on) serializes them
correctly: dispatch *N+1* is guaranteed to see dispatch *N*'s writes,
preserving the stroke's original compositing order end to end — even
though multiple segments and many dispatches are involved.

## Overlap test: conservative AABB, not exact circle-circle

`dabs_may_overlap` doesn't test circle-circle intersection against
`cs_main`'s actual per-texel disc check (`dist <= safe_radius` before any
`textureLoad`/`textureStore`). It tests axis-aligned square overlap on
`pos ± radius`, padded by a 2-texel margin (`BOX_ROUNDING_MARGIN`) to cover
the shader's `floor(pos - radius)` / `ceil(pos + radius)` bounding-box
rounding. This is deliberately a superset of the real hazard region:

- False positive (flagged as overlapping, but the actual discs/texels
  never collide) → one extra dispatch split. Costs a little batching
  efficiency, nothing else.
- False negative (missed an actual overlap) → reintroduces the exact race
  this module exists to prevent.

Given that asymmetry, biasing toward "may overlap" was the only
defensible choice. The margin is a texel-space heuristic, not derived from
`cs_main` line-by-line — if a future shader change widens the box
computation further (e.g. multi-texel supersampling), `BOX_ROUNDING_MARGIN`
needs re-deriving.

## Capacity: `min(4096, compositor.max_dabs_per_batch())`

The task's literal "capacity: 4096" is clamped against
`PaintCompositor::max_dabs_per_batch()` (backed by
`wgpu::Limits::max_compute_workgroups_per_dimension`, typically 65535, but
not guaranteed on every adapter). Without the clamp, a segment built to
the raw 4096 cap could still trip `PaintCompositor::splat_dabs`'s own
`CapacityOverflow` on a device whose limit is below 4096. In this
sandbox's adapter the clamp is a no-op (65535 > 4096), but it's one
`.min()` call to make the two capacities consistent on any device.

## Resize: contents dropped, not preserved

`PaintThreadCommand::Resize { width, height }` replaces `self.target` with
a fresh `PaintTarget::new(&device, width, height)` — the old texture is
dropped, the new one starts zero-initialized (same spec guarantee
`PaintTarget::new` already relies on). No read-back-and-recomposite
attempt is made. Any command before a `Resize` in the same
`process_pending` drain still executes against the old target (already
recorded on the encoder); anything after runs against the new one. A
caller that needs to preserve paint data across a resize must read it
back (or re-bake it) before publishing `Resize`.

## GPU tests (feature `gpu`, real adapter required, no skips)

Unlike `paint.rs`'s existing GPU tests (which print a message and return
early if no adapter or feature is available — appropriate for a test that
might run on a CI box without a GPU), `paint_thread`'s GPU tests `expect`/
panic instead: this suite is specified to run with a real adapter, so a
silent skip would hide a real failure.

- `end_to_end_overlapping_stage_splits_into_two_dispatches` — 64×64
  target, one `Stage` with two overlapping half-alpha dabs: red centered
  at the target's center `(32,32)`, blue offset `(+8,+8)` at `(40,40)`,
  both radius 16 (the task's literal placement). Asserts `stats.segments
  == 1`, `stats.dispatches == 2` (the deterministic proof that overlap
  forced a split — pixel checks alone wouldn't catch a broken
  single-dispatch version on an adapter that happens to run workgroups in
  a convenient order), `dabs_composited == 2`. Readback confirms a
  red-only crescent reads red, a blue-only crescent reads blue, the
  overlap region lands at ~0.75 alpha coverage with blue (drawn second)
  dominant and red still showing through, and all four corners stay
  untouched.

  Note on "center red": the task brief's literal wording asks for a
  "center red" assertion, but with these literal coordinates it's
  geometrically impossible — the two dabs' centers are only `sqrt(8²+8²)
  ≈ 11.3` texels apart, well inside each other's radius-16 circle, so
  *both* centers sit inside the overlap region, not in a color-pure zone.
  There is no point that is "red center, blue nowhere nearby." The test
  instead probes each dab's far crescent (`(20,32)` for red-only,
  `(52,40)` for blue-only — distances given in the test's own comments)
  and a genuinely-overlapping point `(36,36)` for the 0.75-coverage
  check, which is what the task's compositing-math assertion actually
  needs.
- `capacity_overflow_batch_splits_into_two_segments` — 256×256 target, a
  64×64 grid of tiny (radius 0.5) dabs spaced 4 texels apart (so no two
  grid dabs' padded bounding boxes touch) plus one extra dab, 4097 total.
  Asserts `stats.segments == 2` (4096 + 1), `stats.dispatches == 2` (the
  grid is spaced wide enough that neither segment needs an internal
  overlap split — a second deterministic check, not just "no error"),
  `dabs_composited == 4097`, no `PaintError`.

Plus 4 new non-GPU unit tests for `dabs_may_overlap` and `FrameStats`'s
`Default`.

All green: `cargo fmt -p umber-gpu -- --check`, `cargo clippy -p umber-gpu
--all-targets -- -D warnings`, `cargo clippy -p umber-gpu --all-targets
--features gpu -- -D warnings`, `cargo test -p umber-gpu` (21 — the
existing 17 plus 4 new), `cargo test -p umber-gpu --features gpu` (28 —
the existing 22 plus 6 new), `cargo build --workspace`.

## Reviewer checklist

- [ ] **Confirm the contract-conflict call above is actually correct**
  before trusting anything downstream of it — this is the single
  highest-leverage thing to re-derive independently rather than take on
  faith, since the task brief itself argued (briefly) for the opposite
  design.
- [ ] `dabs_may_overlap`'s AABB-with-margin test is a heuristic, not a
  mechanical derivation from `cs_main`. If `PAINT_COMPUTE_SHADER`'s
  bounding-box math changes, re-check `BOX_ROUNDING_MARGIN` (currently
  2.0 texels: ~1 for `floor` plus ~1 for `ceil`, summed conservatively
  rather than precisely).
- [ ] `segments` vs. `dispatches` is a two-field design specifically so
  the capacity-driven count and the overlap-driven count stay separately
  observable. If a future caller only wants one number, check which one
  — they are not interchangeable (`dispatches >= segments`, not `==`, in
  general; they only coincide when a `Stage` batch has no internal
  overlaps at all, as in the capacity-overflow test's grid).
- [ ] `PaintThread::new` takes `device`/`queue` by value (matches the
  task signature), not borrowed — same rationale as `PaintCompositor::new`
  in the existing paint core (wgpu handles are `Arc`-backed, cheap to
  clone/move). Whoever wires this into `umber-app` needs the same
  `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` device-creation fix flagged
  as reviewer checklist item #1 in `LANDING_NOTES_PAINT.md` — not
  re-solved here, still outstanding.
- [ ] The `PaintError::ChannelClosed` variant is additive and currently
  unreachable from normal use (self-owned channel). If `PaintThread` is
  ever restructured so the two channel ends live in different places
  (e.g. a real OS thread), re-examine whether it's still actually
  unreachable.
- [ ] No tile-pool / multi-target wiring here — one `PaintThread` owns
  exactly one `PaintTarget`, matching `paint.rs`'s own one-tile Wave-2
  scope note.
- [ ] On the (currently unreachable) error path inside `stage_dabs` —
  `flush_group` returning `Err` partway through a `Stage` command —
  `self.stats` has already counted the dispatches/dabs that were
  recorded on the encoder *before* the failing group, even though that
  encoder is dropped without being submitted (so none of those
  dispatches actually ran), and any commands still in `process_pending`'s
  local `commands` Vec after the failing one are silently lost rather
  than re-queued. No code change made for this, since the error path is
  unreachable by construction (groups are built to respect `DabBuffer`'s
  capacity) — but if that invariant is ever loosened, stats and dropped
  commands both need revisiting.
- [ ] `stage_dabs`'s overlap grouping is O(n²) within a segment (each new
  dab is checked against every dab already in the current group). The
  capacity-overflow test's 4096-dab non-overlapping segment is close to
  the worst case for this (every dab checked against a growing group
  before any flush) — still fast in practice (sub-second), but worth
  knowing if segment sizes grow well past 4096 in the future.

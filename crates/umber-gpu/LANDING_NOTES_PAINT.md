# umber-gpu paint pass — landing notes

Wave-2 GPU claw: the dab-compositing compute pass — `PaintTarget` (the GPU
paint surface), `Dab`/`DabBuffer` (the CPU-side stamp batch + its GPU
upload), `PaintCompositor` (the compute pipeline + dispatch), and the WGSL
splat shader. Reference artifacts are at `docs/claw-artifacts/umber-gpu/`
(delete per that directory's own README once this is cross-reviewed).

## What was built

**`crates/umber-gpu/src/paint.rs`** (new module)
- `Dab` — `#[repr(C)]`, `bytemuck::Pod`/`Zeroable`, 48 bytes, byte-for-byte
  matching the WGSL `Dab` struct in `shaders::PAINT_COMPUTE_SHADER`.
  Construct via `Dab::new(pos, radius, color, alpha, hardness)` — the
  struct's one private padding field means a bare struct literal can't be
  built any other way, so the padding can never be forgotten at a call
  site.
- `PaintError` — `thiserror`, three variants: `EmptyBatch`,
  `CapacityOverflow { requested, capacity }`, `MissingDeviceFeature`.
- `DabBuffer` — CPU-side `Vec<Dab>` staging with a configured `capacity`.
  `stage(&mut self, dabs)` validates (empty/overflow) and replaces the
  staged batch; `upload(&self, device) -> Result<wgpu::Buffer, PaintError>`
  creates a fresh GPU storage buffer via `create_buffer_init`, erroring on
  `EmptyBatch` instead of handing wgpu a zero-size buffer.
- `PaintTarget` — owns the `Rgba8Unorm` storage texture + view + an 8-byte
  `vec2<u32>` dims uniform buffer. `new(device, width, height)`,
  `clear(&self, encoder)` (render-pass `LoadOp::Clear`), `dimensions()`,
  `texture()` (for readback).
- `PaintCompositor` — owns the compute pipeline + the 3-entry bind-group
  layout (dab storage buffer / storage texture / dims uniform) + a cloned
  `device` handle. `new(device) -> Result<Self, PaintError>` (see the
  landmine below — fails with `MissingDeviceFeature` up front rather than
  panicking partway through pipeline construction), `max_dabs_per_batch()`,
  and `splat_dabs(&self, encoder, target, dabs) -> Result<(), PaintError>`:
  stages + uploads a fresh dab buffer, builds a fresh bind group, records
  one compute pass (`dispatch_workgroups(dabs.len(), 1, 1)`).

**`crates/umber-gpu/src/shaders.rs`**
- `PAINT_COMPUTE_SHADER` — the WGSL splat shader. `@workgroup_size(64)`,
  one workgroup per dab (`workgroup_id.x` indexes `dabs[]`); each
  invocation strides the dab's clamped bounding box by
  `local_invocation_index` so invocations *within* a workgroup never touch
  the same texel. `radial_falloff` implements the hardness curve without
  WGSL's built-in `smoothstep` (see below — avoids a NaN at `hardness ==
  1.0`). Compositing is `src = color * (alpha * falloff); out = src + dst *
  (1 - src.a)` — standard premultiplied "A over B". The storage-texture
  binding variable is named `paint_tex`, not the task sketch's literal
  `target` — `target` reads as a plausible WGSL reserved word and renaming
  costs nothing, so it was changed defensively rather than risking a parse
  error that only the `gpu`-featured test run would ever catch.

**`crates/umber-gpu/src/lib.rs`** — added `pub mod paint;` and re-exports
(`Dab`, `DabBuffer`, `PaintCompositor`, `PaintError`, `PaintTarget`).

No `Cargo.toml` changes — everything needed (`wgpu`, `bytemuck`, `thiserror`,
`pollster`) was already a dependency from the Wave-1 render pass.

## Live-verified in this sandbox

Like the Wave-1 claw, this sandbox has a real wgpu adapter — and it's a
**real GPU, not a software rasterizer**: `adapter.get_info()` reports
`Intel(R) HD Graphics 530 (SKL GT2)`, `device_type: IntegratedGpu`,
`driver: "Intel open-source Mesa driver"` (Mesa ANV), `backend: Vulkan`.
So the `gpu` feature's tests exercised a real Vulkan driver end to end, not
llvmpipe/lavapipe (confirmed: no "skipping" line in `cargo test --features
gpu -- --nocapture` output for any of the three paint GPU tests). All
three pass:

- `splat_one_dab_paints_center_leaves_edges_untouched` — one red,
  radius-16, hardness-1.0 dab on a 64×64 target: center pixel strongly red
  and opaque, a point 2px outside the radius untouched (`[0, 0, 0, 0]`),
  the far corner untouched.
- `splat_two_dabs_composites_overlap_alpha_over` — two half-alpha dabs
  (red then blue) with overlapping footprints, splatted as **two separate
  `splat_dabs` calls on one encoder** (per the overlap contract below):
  the red-only region reads red, the blue-only region reads blue, and the
  overlap region's alpha lands at ~191/255 — exactly `1 - (1-0.5)² = 0.75`
  coverage, confirming both the compositing math and that the second
  dispatch correctly saw the first dispatch's writes (no explicit barrier
  was coded — this is wgpu's automatic inter-pass hazard tracking).
- `splat_three_dabs_one_batch_clamps_edges_and_falls_off_then_clears` —
  one batch, three non-overlapping dabs, added specifically to exercise
  what the first two tests didn't: `workgroup_id.x` indexing `dabs[1]`/
  `dabs[2]` (proving the 48-byte `Dab` stride is right, not just that a
  single `dabs[0]` read works), bounding-box clamping against *both* the
  min `(0,0)` and max `(63,63)` corners of a 64×64 target, a non-hard
  (`hardness = 0.0`) falloff (asserts alpha strictly decreases from the
  dab's center outward), and finally `PaintTarget::clear` (asserts every
  byte reads back as `0` afterward).

Also green: `cargo fmt -p umber-gpu -- --check`, `cargo clippy -p
umber-gpu --all-targets -- -D warnings`, `cargo clippy -p umber-gpu
--all-targets --features gpu -- -D warnings`, `cargo test -p umber-gpu`
(17 tests) and `cargo test -p umber-gpu --features gpu` (22 tests, all 15
pre-existing tests plus the 7 new ones), `cargo build --workspace`.

## The landmine: `read_write` storage textures need a device feature

The vendored README documented `texture_storage_2d<rgba8unorm,
read_write>` and `StorageTextureAccess::ReadWrite` as verified facts, and
they're syntactically/type-correct — but **the first real GPU test run hit
a wgpu validation panic**, not a compile error:

```
Binding index 1: ReadWrite access to storage textures with format Rgba8Unorm is not supported
```

(Separately, the same README's bind-group-layout text — "`Texture {
access: StorageTextureAccess::WriteOnly|ReadWrite, format, view_dimension}`
for write" — doesn't name a real variant: the compiler confirms the actual
type is `wgpu::BindingType::StorageTexture { access, format,
view_dimension }`, not `Texture { access, .. }`. `BindingType::Texture`
exists but is the *sampled*-texture variant and has no `access` field.
Not fixed here — it's the vendored README's own text, not owned by this
task — but worth flagging for the next claw that trusts it as "verified".)

Diagnosis (via `adapter.features()` / `adapter.get_texture_format_features`,
printed temporarily in a test — not in any vendored doc): `Rgba8Unorm`
*does* report `STORAGE_READ_WRITE` in its format-feature flags on this
adapter, but wgpu only grants non-portable per-format capabilities —
read-write storage access on anything other than the core `r32float`/
`r32uint`/`r32sint` formats — when the device is created with
`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` requested. The
default `DeviceDescriptor` requests no features, so the bind-group-layout
creation that worked on paper failed at the validation layer.

Fixed two ways:
1. `PaintCompositor::new` now checks `device.features().contains(..)` up
   front and returns `PaintError::MissingDeviceFeature` instead of letting
   wgpu's bind-group-layout validation panic partway through construction.
2. This crate's own GPU tests (`paint::tests::gpu::try_request_device`)
   check the *adapter* supports the feature (skip gracefully, printing
   which feature is missing, if not) and then request it in
   `required_features`.

**This is not fixed for production**, and it's out of this task's scope
(umber-gpu owns no device-creation call in the real app — `umber-app`
captures `eframe`'s `wgpu_render_state.device`, created by egui-wgpu, not
by this crate). **Reviewer checklist item #1**: whoever wires
`PaintCompositor` into `umber-app` must get
`TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` into that device. The hook is
real and vendored: `egui_wgpu_lib.rs:225-257` shows `WgpuSetup::CreateNew`
carries a `device_descriptor: Arc<dyn Fn(&wgpu::Adapter) ->
wgpu::DeviceDescriptor>`-shaped closure (field at `:230`), called as
`adapter.request_device(&(*device_descriptor)(&adapter))` at `:255`. So
`umber-app`'s `NativeOptions.wgpu_options.wgpu_setup` needs to be a
`WgpuSetup::CreateNew` whose `device_descriptor` closure sets
`required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
(merged with whatever else that closure already requests) instead of
relying on `WgpuConfiguration::default()`. Without this,
`PaintCompositor::new` returns `Err(MissingDeviceFeature)` the first time
`umber-app` calls it — a typed error now, not a panic, but still a hard
stop until this is wired.

## WGSL decisions

- **`Dab` field order (Rust) differs from the task's literal listing.**
  The task describes `Dab { pos: [f32;2], radius: f32, color: [f32;4],
  alpha: f32, hardness: f32 }`. Declared in that literal order under
  `#[repr(C)]`, `color` lands at byte offset 12 — but WGSL's `vec4<f32>`
  forces 16-byte alignment, so the WGSL-side `color` must sit at offset
  16. Rust's `#[repr(C)]` won't insert that padding on its own, because
  `[f32; 4]`'s *Rust* alignment is 4 (array alignment = element
  alignment, not element-count-scaled) — unlike WGSL's `vec4<f32>`. Fixed
  by declaring the struct as `pos, radius, alpha, color, hardness, _pad:
  [f32; 3]` (alpha moved before color, which costs nothing at call sites
  since `Dab::new`'s *parameter* order still matches the task's
  conceptual order) and adding a `const _: () = assert!(size_of::<Dab>()
  == 48)` to pin it. Named-field structs don't care about declaration
  order for a constructor's parameter list, only for `Pod`'s byte layout.
- **Hardness falloff avoids WGSL's built-in `smoothstep`.** The natural
  formula is `smoothstep(hardness, 1.0, t)`, but `smoothstep`'s reference
  implementation divides by `(edge1 - edge0)`, which is exactly zero when
  `hardness == 1.0` (a fully hard brush) — a real, reachable input, not a
  corner case. `radial_falloff` in the shader reimplements the same cubic
  curve with the denominator clamped to `1e-4`, so hardness `1.0` degrades
  to "full strength everywhere inside the radius, hard cutoff at the
  boundary" instead of `NaN`. The new `hardness = 0.0` GPU test confirms
  the soft end of the curve actually falls off monotonically, not just
  that it avoids NaN at the hard end.
- **Workgroup-local striding, not atomics.** Per the task's note that
  "storage-texture atomics are NOT needed," each workgroup owns exactly
  one dab and strides its own clamped bounding box by
  `local_invocation_index` so invocations *within* a workgroup never
  collide. There is deliberately no synchronization *between* workgroups
  in the same dispatch — see the overlap contract below.
- **`@binding(2) var<uniform> dims: vec2<u32>`** is exactly the task's
  literal snippet (not wrapped in a named struct) — it matches a plain
  Rust `[u32; 2]` byte-for-byte, so `PaintTarget::new` writes
  `bytemuck::cast_slice(&[width, height])` directly with no intermediate
  type.
- **Compute entry point is named `cs_main`**, following this crate's
  existing `vs_main`/`fs_main` convention (`shaders::MESH_SHADER`) rather
  than the task's descriptive-name suggestion.
- **The storage-texture binding variable is `paint_tex`, not `target`**
  (see "What was built" above) — a defensive rename against WGSL's
  reserved-word list, since a parse error here would only surface in the
  `gpu`-featured test run, not plain `cargo check`.

## The overlap contract (read this before calling `splat_dabs` from brush code)

One `splat_dabs` call is one dispatch, one workgroup per dab, no atomics.
**Dabs within one call must not have overlapping bounding circles** — two
workgroups racing a `textureLoad`/`textureStore` pair on the same texel is
a real data race with no defined winner. Overlapping dabs (the normal
case for a stroke: consecutive dabs along a path almost always overlap at
typical spacing) must go in **separate `splat_dabs` calls**. This is safe
and sequenced correctly — wgpu's automatic resource hazard tracking orders
successive compute passes that touch the same storage texture — and the
two-dab GPU test exercises exactly this (two calls on one encoder, not one
call with two dabs). The caller-side batching policy (how dabs get
grouped into non-overlapping calls — e.g. by spatial partitioning, or just
one dab per call as the simplest-correct starting point) is **not**
umber-gpu's problem; it belongs to whichever crate turns a stroke's
`DabPlan` stream into GPU `Dab` batches.

## Signature deviations from the task sketch (and why)

- **One combined 3-entry `BindGroupLayout`, owned by `PaintCompositor`**,
  not a 2-entry layout owned by `PaintTarget` as the task's point 1
  literally states. A single `@group(0)` in WGSL needs one
  `wgpu::BindGroup`/`BindGroupLayout` pair covering all three bindings
  (dabs, texture, dims) — there's no API for splicing two
  independently-owned layouts into one bind group. `PaintTarget` owns the
  texture/view/dims-buffer (the two resources the task's point 1
  mentions); `PaintCompositor` owns the layout that describes all three
  bindings, since it also owns the pipeline that the layout must match
  (mirrors the existing `GpuContext`-owns-pipeline-and-layout pattern in
  `renderer.rs`).
- **A fresh dab storage buffer + bind group are built on every
  `splat_dabs` call**, not "the per-target bind group" the task's point 4
  describes (singular, implying one cached object). The dab buffer's
  *contents* change on every call, and a `wgpu::BindGroup` binds to a
  buffer's identity, not a snapshot of its bytes — so a cached bind group
  would need rebuilding anyway, or the dab buffer would need to be a
  persistent buffer updated via `queue.write_buffer`. The latter was
  considered and rejected (not implemented, just reasoned through):
  `queue.write_buffer` writes land at *submit* time, not *encoder-record*
  time, so two `splat_dabs` calls recorded onto one encoder (exactly what
  the overlap contract requires for overlapping dabs) would both see
  whichever write happened to land last, not each call's own batch.
  `create_buffer_init` writes synchronously at buffer creation, which is
  why `splat_dabs` takes `&self` rather than `&mut self` — there's no
  cached, mutated state to protect.
- **`PaintCompositor::new` takes an owned `wgpu::Device`** (cloned at the
  call site, cheap — wgpu handles are `Arc`-backed), not a borrowed
  `&wgpu::Device`, so `splat_dabs` can create buffers/bind groups without
  threading a device parameter through every call. Mirrors
  `GpuContext::device` in `renderer.rs`.
- **`PaintCompositor::new` and `DabBuffer::upload` return `Result`**
  rather than the task's implied infallible construction, so the
  device-feature gap above and an empty-batch upload surface as typed
  `PaintError`s instead of wgpu validation panics.

## Needs golden-image hardening later

The three GPU tests here are hand-picked-pixel assertions, not
golden-image comparisons against `golden::compare_rgba8`. That harness
(landed in the previous wave) is the right long-term home for this pass's
correctness tests — a reference PNG of a known stroke's dab batch,
compared with tolerance — but wiring a `PaintTarget`'s readback into
`golden::RenderTarget` (or giving `golden.rs` a second `readback` path
that isn't tied to `MeshBuffers`) wasn't attempted here, to stay inside
this task's scope. Also not covered:
- Batches larger than a handful of dabs, or dabs whose bounding boxes are
  large enough to need many workgroup-strided passes per invocation —
  only small (6–16px radius) dabs on a 64×64 target were exercised.
- The tile-pool / multi-tile wiring (explicitly out of Wave-2 scope per
  the task and `docs/specs/architecture.md`'s virtual-texturing section).

## Reviewer checklist

- [ ] **#1 (blocking for real use):** get
      `wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` into the
      device `umber-app` captures from eframe before `PaintCompositor::new`
      is ever called there — via `NativeOptions.wgpu_options.wgpu_setup`'s
      `device_descriptor` closure (see "The landmine" above for the exact
      vendored citation). Confirm on real hardware (Vulkan/Metal/Dx12),
      not just this sandbox's adapter.
  - [ ] If some target adapter genuinely doesn't support
        `STORAGE_READ_WRITE` on `Rgba8Unorm` even with the feature
        requested, the fallback is splitting the shader's single
        `read_write` binding into a `read`-only sampled/storage binding
        for `dst` plus a separate `write`-only storage binding for the
        output — more bindings, more plumbing, not attempted here since
        this sandbox's adapter didn't need it.
- [ ] Verify the overlap contract against the real brush→dab-batch
      pipeline once it lands: does whatever turns `DabPlan` streams into
      `Dab` batches actually split overlapping dabs across calls, or
      does it need to?
- [ ] Decide whether `PaintTarget::clear` should run automatically in
      `PaintTarget::new` — currently relies on wgpu's spec-guaranteed
      zero-initialization of new textures instead, which is correct but
      easy to forget when *reusing* a target (e.g. a tile pulled back out
      of a future pool). `clear()` itself is GPU-tested (see above).
- [ ] The vendored README's bind-group-layout text names a non-existent
      `BindingType::Texture { access, .. }` variant — the real one is
      `BindingType::StorageTexture { .. }`. Not fixed here (not this
      task's file to edit), but flag it if `docs/claw-artifacts/` outlives
      this review.
- [ ] Delete `docs/claw-artifacts/umber-gpu/` once this and the Wave-1
      render pass are both cross-reviewed (per that directory's own
      README).

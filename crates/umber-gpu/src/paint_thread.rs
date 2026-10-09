//! Orchestration between input and the paint core ([`crate::paint`]): a
//! command channel that lets a producer (input handling, undo/redo, resize
//! events) describe *what* should happen to the paint surface without
//! owning the device, and a consumer ([`PaintThread::process_pending`])
//! that turns queued commands into actual [`PaintCompositor`]/[`PaintTarget`]
//! calls on demand (e.g. once per displayed frame).
//!
//! # The dispatch-batching contract
//!
//! [`paint`][`crate::paint`]'s module docs are the authority here: one
//! [`PaintCompositor::splat_dabs`] call is one compute dispatch, one
//! workgroup per dab, with **no atomics** on the shared storage texture.
//! Dabs within a single dispatch that touch overlapping texels race —
//! there is no defined order between workgroups — so compositing order
//! (which matters, since premultiplied-alpha "over" is not commutative)
//! can only be guaranteed *across* separate dispatches, where wgpu's
//! resource hazard tracking serializes successive compute passes on the
//! same texture.
//!
//! This means a whole stroke segment can **not** be flushed through one
//! dispatch just because it arrived as one [`PaintThreadCommand::Stage`]
//! batch: consecutive dabs along a stroke routinely overlap by design
//! (spacing is sub-radius), and packing them into one dispatch would
//! silently race instead of compositing them in order.
//!
//! `process_pending` instead splits each `Stage` batch into two independent
//! layers of grouping:
//!
//! - **Segments** — `dabs.chunks(capacity)`, where `capacity` is
//!   [`PaintThread`]'s configured staging limit. This exists purely to
//!   respect [`DabBuffer`]'s capacity; it knows nothing about overlap.
//! - **Dispatches** — within one segment, dabs are walked in order and
//!   accumulated into a dispatch group; a dab that could overlap *any* dab
//!   already in the current group forces that group to flush as its own
//!   [`PaintCompositor::splat_dabs`] call before the new dab starts a fresh
//!   group. Non-overlapping dabs (e.g. separate brush strokes painted far
//!   apart) share a dispatch; overlapping ones never do.
//!
//! So `dispatches >= segments` always, and the two [`FrameStats`] counters
//! track different things: `segments` is the capacity-driven chunk count,
//! `dispatches` is the actual number of `splat_dabs` calls (and GPU compute
//! dispatches) issued. Within one `process_pending` call, dispatches for a
//! given `Stage` command are recorded on one [`wgpu::CommandEncoder`] in
//! order, so the hazard tracking that serializes them preserves the
//! stroke's original compositing order end to end.
//!
//! Overlap is tested conservatively via axis-aligned bounding squares
//! (`pos ± radius`, plus a rounding margin — see [`dabs_may_overlap`]) that
//! over-approximate `shaders::PAINT_COMPUTE_SHADER`'s per-dab footprint:
//! false positives (an extra dispatch split) just cost a little batching
//! efficiency, but false negatives would reintroduce the race, so the test
//! is deliberately biased to never under-split.

use std::sync::mpsc;

use crate::paint::{Dab, DabBuffer, PaintCompositor, PaintError, PaintTarget};

/// The default cap on how many dabs [`PaintThread`] will group into one
/// segment before splitting, independent of the device's own
/// `max_compute_workgroups_per_dimension` limit (which is typically much
/// larger, e.g. 65535, and is enforced separately inside
/// [`PaintCompositor::splat_dabs`]).
const STAGING_CAPACITY: usize = 4096;

/// A queued mutation to a [`PaintThread`]'s paint surface.
///
/// Sent via [`PaintThread::publish`] and drained by
/// [`PaintThread::process_pending`] in FIFO order.
#[derive(Debug, Clone)]
pub enum PaintThreadCommand {
    /// Composite `dabs` onto the paint target, in order (see the
    /// module-level dispatch-batching contract).
    Stage {
        /// The dabs to composite, in stroke order.
        dabs: Vec<Dab>,
    },
    /// Reset the paint target to fully transparent.
    Clear,
    /// Recreate the paint target at a new size.
    ///
    /// The target's existing contents are **not** preserved — the old
    /// texture is dropped and a fresh zero-initialized one takes its
    /// place. Callers that need to preserve paint data across a resize
    /// must read it back (or re-bake it) before publishing this command.
    Resize {
        /// New target width, in texels.
        width: u32,
        /// New target height, in texels.
        height: u32,
    },
}

/// Cumulative counters describing the work [`PaintThread::process_pending`]
/// has done over this [`PaintThread`]'s lifetime.
///
/// Returned fresh (and already updated) from every `process_pending` call,
/// so callers that just want a running total never need a separate
/// accessor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Total dabs composited via [`PaintCompositor::splat_dabs`].
    pub dabs_composited: u64,
    /// Total `splat_dabs` calls (one GPU compute dispatch each).
    pub dispatches: u64,
    /// Total [`PaintThreadCommand::Clear`] commands processed.
    pub clears: u64,
    /// Total capacity-bounded segments a `Stage` command's dabs were split
    /// into (see the module-level contract). `dispatches >= segments`.
    pub segments: u64,
}

/// Orchestrates [`crate::paint`] on behalf of a decoupled producer: owns the
/// paint surface, the compositor, and the command channel that lets a
/// caller describe paint-surface mutations without touching wgpu directly.
pub struct PaintThread {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compositor: PaintCompositor,
    target: PaintTarget,
    dab_buffer: DabBuffer,
    sender: mpsc::Sender<PaintThreadCommand>,
    receiver: mpsc::Receiver<PaintThreadCommand>,
    stats: FrameStats,
}

impl PaintThread {
    /// Builds a new `width`x`height` paint thread on `device`/`queue`.
    ///
    /// # Errors
    ///
    /// Returns [`PaintError::MissingDeviceFeature`] if `device` lacks
    /// `wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` — checked
    /// by [`PaintCompositor::new`], which this fails fast through.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Result<Self, PaintError> {
        let compositor = PaintCompositor::new(device.clone())?;
        let target = PaintTarget::new(&device, width, height);
        let capacity = STAGING_CAPACITY.min(compositor.max_dabs_per_batch());
        let dab_buffer = DabBuffer::new(capacity);
        let (sender, receiver) = mpsc::channel();
        Ok(Self {
            device,
            queue,
            compositor,
            target,
            dab_buffer,
            sender,
            receiver,
            stats: FrameStats::default(),
        })
    }

    /// Queues `cmds` for the next [`PaintThread::process_pending`] call, in
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`PaintError::ChannelClosed`] if the internal channel's
    /// receiver has been dropped — unreachable in normal use, since
    /// `PaintThread` owns both ends.
    pub fn publish(&mut self, cmds: Vec<PaintThreadCommand>) -> Result<(), PaintError> {
        for cmd in cmds {
            self.sender
                .send(cmd)
                .map_err(|_| PaintError::ChannelClosed)?;
        }
        Ok(())
    }

    /// Drains every command queued since the last call, applies them to the
    /// paint target in order on a single [`wgpu::CommandEncoder`], and
    /// submits it. Returns the updated cumulative [`FrameStats`].
    ///
    /// If nothing is queued, this is a no-op (no encoder is created or
    /// submitted) and the previous stats are returned unchanged.
    ///
    /// See the module-level docs for how `Stage` batches are split into
    /// segments and dispatches.
    ///
    /// # Errors
    ///
    /// Propagates [`PaintError`] from staging or splatting a dab group —
    /// in practice unreachable, since groups are built to respect
    /// [`DabBuffer`]'s capacity by construction.
    pub fn process_pending(&mut self) -> Result<FrameStats, PaintError> {
        let mut commands = Vec::new();
        while let Ok(cmd) = self.receiver.try_recv() {
            commands.push(cmd);
        }
        if commands.is_empty() {
            return Ok(self.stats);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_paint_thread_encoder"),
            });

        for cmd in commands {
            match cmd {
                PaintThreadCommand::Clear => {
                    self.target.clear(&mut encoder);
                    self.stats.clears += 1;
                }
                PaintThreadCommand::Resize { width, height } => {
                    self.target = PaintTarget::new(&self.device, width, height);
                }
                PaintThreadCommand::Stage { dabs } => {
                    self.stage_dabs(&mut encoder, &dabs)?;
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(self.stats)
    }

    /// Splits `dabs` into capacity-bounded segments, then each segment into
    /// overlap-safe dispatch groups, recording one `splat_dabs` call per
    /// group onto `encoder`. Updates `self.stats` as it goes.
    fn stage_dabs(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dabs: &[Dab],
    ) -> Result<(), PaintError> {
        let capacity = self.dab_buffer.capacity();
        for segment in dabs.chunks(capacity) {
            let mut group: Vec<Dab> = Vec::with_capacity(segment.len());
            for dab in segment.iter().copied() {
                if group.iter().any(|g| dabs_may_overlap(g, &dab)) {
                    self.flush_group(encoder, &group)?;
                    group.clear();
                }
                group.push(dab);
            }
            if !group.is_empty() {
                self.flush_group(encoder, &group)?;
            }
            // Counted after the segment fully succeeded (review #2): a
            // mid-segment failure must not report work that never ran.
            self.stats.segments += 1;
        }
        Ok(())
    }

    /// Stages and splats one overlap-safe dispatch group, bumping
    /// `dispatches`/`dabs_composited`.
    fn flush_group(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        group: &[Dab],
    ) -> Result<(), PaintError> {
        self.dab_buffer.stage(group)?;
        self.compositor
            .splat_dabs(encoder, &self.target, self.dab_buffer.staged())?;
        self.stats.dispatches += 1;
        self.stats.dabs_composited += group.len() as u64;
        Ok(())
    }

    /// Read-only access to the paint surface, for display or further
    /// compositing.
    pub fn paint_target(&self) -> &PaintTarget {
        &self.target
    }
}

/// Conservative overlap test between two dabs' shader-side footprints.
///
/// `shaders::PAINT_COMPUTE_SHADER`'s `cs_main` computes each dab's touched
/// region as the texel box `[floor(pos - radius), ceil(pos + radius)]`
/// (intersected with the target bounds, which only shrinks it further). Two
/// dabs can only race if their *unclamped* boxes intersect, so testing
/// against the clamped, true-circle footprint is unnecessary — this checks
/// axis-aligned square overlap on `pos ± radius`, padded by
/// [`BOX_ROUNDING_MARGIN`] texels to cover the `floor`/`ceil` rounding.
/// Over-approximating only costs a little batching efficiency (an
/// unnecessary dispatch split); under-approximating would reintroduce the
/// race, so this is deliberately biased toward "may overlap".
fn dabs_may_overlap(a: &Dab, b: &Dab) -> bool {
    let ra = a.radius.max(1e-5);
    let rb = b.radius.max(1e-5);
    let threshold = ra + rb + BOX_ROUNDING_MARGIN;
    (a.pos[0] - b.pos[0]).abs() <= threshold && (a.pos[1] - b.pos[1]).abs() <= threshold
}

/// Safety margin added to `dabs_may_overlap`'s overlap threshold to cover
/// the shader's `floor`/`ceil` box rounding (each dab's box can extend up
/// to ~1 texel beyond `pos ± radius` on every side).
const BOX_ROUNDING_MARGIN: f32 = 2.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn dab_at(x: f32, y: f32, radius: f32) -> Dab {
        Dab::new([x, y], radius, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0)
    }

    #[test]
    fn overlap_detects_touching_bounding_boxes() {
        let a = dab_at(10.0, 10.0, 4.0);
        let b = dab_at(16.0, 10.0, 4.0);
        assert!(dabs_may_overlap(&a, &b));
    }

    #[test]
    fn overlap_rejects_well_separated_dabs() {
        let a = dab_at(10.0, 10.0, 4.0);
        let b = dab_at(100.0, 100.0, 4.0);
        assert!(!dabs_may_overlap(&a, &b));
    }

    #[test]
    fn overlap_is_symmetric() {
        let a = dab_at(0.0, 0.0, 2.0);
        let b = dab_at(5.0, 0.0, 2.0);
        assert_eq!(dabs_may_overlap(&a, &b), dabs_may_overlap(&b, &a));
    }

    #[test]
    fn frame_stats_default_is_all_zero() {
        assert_eq!(
            FrameStats::default(),
            FrameStats {
                dabs_composited: 0,
                dispatches: 0,
                clears: 0,
                segments: 0,
            }
        );
    }

    #[cfg(feature = "gpu")]
    mod gpu {
        use super::super::*;

        /// Requests a device with
        /// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` enabled. Unlike
        /// `paint`'s gpu tests, this panics rather than skipping when no
        /// adapter is available: this suite runs with a required real
        /// adapter, not an optional one.
        fn request_device() -> (wgpu::Device, wgpu::Queue) {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .expect("a wgpu adapter must be available for this test");
            assert!(
                adapter
                    .features()
                    .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
                "adapter {:?} lacks TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES",
                adapter.get_info().name
            );
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }))
            .expect("device request must succeed")
        }

        fn read_back_rgba8(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            target: &PaintTarget,
        ) -> Vec<u8> {
            let (width, height) = target.dimensions();
            let unpadded_row = width * 4;
            let padding = (256 - (unpadded_row % 256)) % 256;
            let padded_row = unpadded_row + padding;
            let size = padded_row as u64 * height as u64;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("umber_paint_thread_test_readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("umber_paint_thread_test_readback_encoder"),
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: target.texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(Some(encoder.finish()));

            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("poll should succeed in tests");
            receiver
                .recv()
                .expect("map_async callback should fire")
                .expect("buffer map should succeed");

            let data = slice.get_mapped_range().expect("mapped range");
            let bytes: &[u8] = &data;
            let mut out = Vec::with_capacity(unpadded_row as usize * height as usize);
            for row in 0..height as usize {
                let start = row * padded_row as usize;
                out.extend_from_slice(&bytes[start..start + unpadded_row as usize]);
            }
            drop(data);
            buffer.unmap();
            out
        }

        fn pixel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
            let i = (y * width + x) as usize * 4;
            [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
        }

        #[test]
        fn end_to_end_overlapping_stage_splits_into_two_dispatches() {
            let (device, queue) = request_device();
            let mut thread = PaintThread::new(device.clone(), queue.clone(), 64, 64)
                .expect("device has the required feature");

            // red centered at the 64x64 target's center (32,32), blue
            // offset by (+8,+8) at (40,40), both radius 16 — their
            // bounding boxes overlap heavily, so this Stage must become
            // two dispatches (red first, blue over it) rather than one
            // racy dispatch of two dabs. Note: the two dabs' *centers* are
            // only 11.3 texels apart, both well inside the other's
            // radius-16 circle, so there is no "pure center" probe point
            // for either color alone — see the red-only/blue-only probes
            // below, picked on each dab's far crescent instead.
            let red = Dab::new([32.0, 32.0], 16.0, [1.0, 0.0, 0.0, 1.0], 0.5, 1.0);
            let blue = Dab::new([40.0, 40.0], 16.0, [0.0, 0.0, 1.0, 1.0], 0.5, 1.0);

            thread
                .publish(vec![PaintThreadCommand::Stage {
                    dabs: vec![red, blue],
                }])
                .expect("publish should succeed");

            let stats = thread
                .process_pending()
                .expect("process_pending should succeed");

            assert_eq!(stats.segments, 1, "one Stage command fits in one segment");
            assert_eq!(
                stats.dispatches, 2,
                "overlapping dabs must split into separate dispatches: {stats:?}"
            );
            assert_eq!(stats.dabs_composited, 2);
            assert_eq!(stats.clears, 0);

            let bytes = read_back_rgba8(&device, &queue, thread.paint_target());

            // Red-only crescent: inside red's radius (center 32,32 r=16,
            // distance ~11.5), outside blue's (center 40,40 r=16,
            // distance ~20.9).
            let red_only = pixel(&bytes, 64, 20, 32);
            assert!(
                red_only[0] > 100,
                "red-only region should be red-ish: {red_only:?}"
            );
            assert!(
                red_only[2] < 20,
                "red-only region should have no blue: {red_only:?}"
            );

            // Blue-only crescent: inside blue's radius (distance ~12.5),
            // outside red's (distance ~22.2).
            let blue_only = pixel(&bytes, 64, 52, 40);
            assert!(
                blue_only[2] > 100,
                "blue-only region should be blue-ish: {blue_only:?}"
            );
            assert!(
                blue_only[0] < 20,
                "blue-only region should have no red: {blue_only:?}"
            );

            // Overlap region (36,36): inside both radii (distance ~6.4
            // from red, ~5.0 from blue). Blue drawn second (over red) at
            // half alpha each — two 0.5-alpha overs compose to
            // 1 - (1-0.5)^2 = 0.75 coverage (~191/255), with blue
            // dominant and red still showing through underneath. This,
            // plus `stats.dispatches == 2` above, is what actually proves
            // ordering — the alpha value alone is symmetric in red/blue
            // and wouldn't catch a reversed-order bug.
            let overlap = pixel(&bytes, 64, 36, 36);
            assert!(
                (170..210).contains(&overlap[3]),
                "two half-alpha overs should land near 0.75 coverage: {overlap:?}"
            );
            assert!(
                overlap[2] > overlap[0],
                "blue drawn over red should dominate: {overlap:?}"
            );
            assert!(
                overlap[0] > 20,
                "red should still show through underneath: {overlap:?}"
            );

            // All four corners, far from both dabs, stay untouched.
            assert_eq!(pixel(&bytes, 64, 0, 0), [0, 0, 0, 0]);
            assert_eq!(pixel(&bytes, 64, 63, 0), [0, 0, 0, 0]);
            assert_eq!(pixel(&bytes, 64, 0, 63), [0, 0, 0, 0]);
            assert_eq!(pixel(&bytes, 64, 63, 63), [0, 0, 0, 0]);
        }

        #[test]
        fn capacity_overflow_batch_splits_into_two_segments() {
            let (device, queue) = request_device();
            let mut thread = PaintThread::new(device.clone(), queue.clone(), 256, 256)
                .expect("device has the required feature");

            // A 64x64 grid of tiny, widely-spaced (non-overlapping) dabs,
            // plus one extra — 4097 total, one more than STAGING_CAPACITY.
            // Positions sit on texel centers (x.5): a radius-0.5 dab at an
            // integer position is >= 0.707 from every texel center and
            // paints nothing (shader probes centers at +0.5).
            let mut dabs = Vec::with_capacity(4097);
            for j in 0..64u32 {
                for i in 0..64u32 {
                    let x = 2.5 + 4.0 * i as f32;
                    let y = 2.5 + 4.0 * j as f32;
                    dabs.push(Dab::new([x, y], 0.5, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0));
                }
            }
            // Segment-2 sentinel at (0.5, 254.5): deliberately far from
            // every grid dab (grid x starts at 2.5), so its texel (0, 254)
            // can ONLY be white if the second segment actually rendered.
            // (The original placement at the grid corner coincided with
            // dab #4096 — a dropped segment 2 would have passed it.)
            dabs.push(Dab::new([0.5, 254.5], 0.5, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0));
            assert_eq!(dabs.len(), 4097);

            thread
                .publish(vec![PaintThreadCommand::Stage { dabs }])
                .expect("publish should succeed");

            let stats = thread
                .process_pending()
                .expect("4097 dabs must stage across two segments without error");

            assert_eq!(
                stats.segments, 2,
                "4096 + 1 must split into two segments: {stats:?}"
            );
            assert_eq!(
                stats.dispatches, 2,
                "the grid is spaced wide enough that neither segment needs an overlap split: {stats:?}"
            );
            assert_eq!(stats.dabs_composited, 4097);

            // Pixel readback (review blocker): counters alone can pass while
            // the second segment never rendered. The sentinel sits where no
            // segment-1 dab can reach, so texel (0, 254) is white ONLY if
            // segment 2 dispatched. Also probe a segment-1 texel and an
            // untouched corner.
            let bytes = read_back_rgba8(&device, &queue, thread.paint_target());
            let idx = |x: usize, y: usize| (y * 256 + x) * 4;
            // Second-segment sentinel: white at texel (0, 254).
            let i = idx(0, 254);
            assert_eq!(
                &bytes[i..i + 4],
                &[255, 255, 255, 255][..],
                "the 4097th dab (second segment) must actually render"
            );
            // First-segment spot: first grid dab at texel (2, 2).
            let i = idx(2, 2);
            assert_eq!(&bytes[i..i + 4], &[255, 255, 255, 255][..]);
            // Untouched: far corner of the target.
            let i = idx(0, 0);
            assert_eq!(&bytes[i..i + 4], &[0, 0, 0, 0][..]);
        }
    }
}

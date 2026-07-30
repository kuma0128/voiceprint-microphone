//! Live session: mic → `StreamingPipeline` → speaker.
//!
//! Owns the three threads the data flow needs (cpal input cb, worker,
//! cpal output cb) and the two bounded queues between them. Drop the
//! [`LiveSession`] to stop; the cpal streams are torn down and the
//! worker exits on the next iteration.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use mellonella_core::enrollment::EmbeddingPool;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::PipelineComponents;
use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};

use crate::devices::DeviceKind;
use crate::resampler::StreamingResampler;
use crate::separation::{SeparatorTuning, TargetSpeakerSeparator};
use crate::{AudioIoError, ChannelStrategy, INTERNAL_SAMPLE_RATE};

/// Input capture must continue while the one-second SepFormer block is
/// inferred (~125-250 ms on a typical CPU). 64 ordinary ~10 ms cpal
/// chunks provide enough headroom without ever blocking the real-time
/// callback; the worker drains them immediately after each inference.
const INPUT_RING_CHUNKS: usize = 64;
/// Enough output headroom for short scheduler / model-inference stalls.
/// The callback drains this channel into a sample-based elastic buffer,
/// so a larger slot count does not itself add latency.
const OUTPUT_RING_CHUNKS: usize = 64;
/// Prime the output path before playback begins. The old callback started
/// draining immediately, so every normal 32 ms pipeline burst was followed
/// by a race against the next device callback and periodic silence sounded
/// like a broken / crackling signal on the Discord side.
const OUTPUT_PREBUFFER_MS: u32 = 64;
/// Fill level the elastic clock bridge gently steers toward.
const OUTPUT_TARGET_BUFFER_MS: u32 = 96;
/// Maximum input/output hardware-clock correction. Two independent devices
/// are never exactly 48 kHz; ±0.2 % is ample for real hardware drift and is
/// small enough to be inaudible as pitch movement.
const MAX_CLOCK_CORRECTION: f64 = 0.002;
const PLAYBACK_FADE_MS: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaybackStatus {
    Rendered,
    Prebuffering,
    Underrun,
}

/// Sample-based jitter buffer and asynchronous clock bridge between the
/// microphone and output devices.
///
/// Even when both devices advertise 48 kHz, their physical oscillators drift.
/// A plain FIFO must therefore eventually either empty or overflow. This
/// buffer uses linear interpolation with a tiny fill-dependent read-rate
/// correction, keeping the queue centred without dropping whole 32 ms chunks.
struct PlaybackBuffer {
    samples: VecDeque<f32>,
    read_position: f64,
    target_fill: usize,
    start_fill: usize,
    fade_total: usize,
    fade_remaining: usize,
    started: bool,
    last_output: f32,
}

impl PlaybackBuffer {
    fn new(sample_rate: u32) -> Self {
        let millis_to_samples = |ms: u32| {
            (u64::from(sample_rate) * u64::from(ms) / 1_000)
                .try_into()
                .unwrap_or(usize::MAX)
        };
        Self {
            samples: VecDeque::new(),
            read_position: 0.0,
            target_fill: millis_to_samples(OUTPUT_TARGET_BUFFER_MS),
            start_fill: millis_to_samples(OUTPUT_PREBUFFER_MS),
            fade_total: millis_to_samples(PLAYBACK_FADE_MS).max(1),
            fade_remaining: 0,
            started: false,
            last_output: 0.0,
        }
    }

    fn push(&mut self, samples: &[f32]) {
        self.samples.extend(samples.iter().copied());
    }

    fn buffered_samples(&self) -> usize {
        self.samples.len()
    }

    fn playback_step(&self) -> f64 {
        if self.target_fill == 0 {
            return 1.0;
        }
        let error = self.samples.len() as f64 / self.target_fill as f64 - 1.0;
        1.0 + (error * MAX_CLOCK_CORRECTION).clamp(-MAX_CLOCK_CORRECTION, MAX_CLOCK_CORRECTION)
    }

    fn render(&mut self, output: &mut [f32]) -> PlaybackStatus {
        output.fill(0.0);
        if output.is_empty() {
            return PlaybackStatus::Rendered;
        }
        if !self.started {
            // Two callbacks of margin cover unusually large device periods;
            // the ordinary path starts at the fixed 64 ms prebuffer.
            let required = self.start_fill.max(output.len().saturating_mul(2));
            if self.samples.len() < required {
                return PlaybackStatus::Prebuffering;
            }
            self.started = true;
            self.fade_remaining = self.fade_total;
        }

        let step = self.playback_step();
        let final_position = self.read_position + step * output.len().saturating_sub(1) as f64;
        let final_index = final_position.floor() as usize;
        if final_index.saturating_add(1) >= self.samples.len() {
            // This should be exceptional after priming. Fade the previous
            // sample to zero instead of making a full-scale discontinuity,
            // then wait for the prebuffer to refill.
            let fade = self.fade_total.min(output.len());
            for (index, slot) in output.iter_mut().take(fade).enumerate() {
                *slot = self.last_output * (1.0 - (index + 1) as f32 / fade as f32);
            }
            self.started = false;
            self.read_position = 0.0;
            self.last_output = 0.0;
            return PlaybackStatus::Underrun;
        }

        for slot in output.iter_mut() {
            let index = self.read_position.floor() as usize;
            let fraction = (self.read_position - index as f64) as f32;
            let left = self.samples[index];
            let right = self.samples[index + 1];
            let mut sample = left + (right - left) * fraction;
            if self.fade_remaining > 0 {
                let gain = 1.0 - self.fade_remaining as f32 / self.fade_total as f32;
                sample *= gain;
                self.fade_remaining -= 1;
            }
            *slot = sample;
            self.last_output = sample;
            self.read_position += step;
        }

        let consumed = self.read_position.floor() as usize;
        self.samples.drain(..consumed);
        self.read_position -= consumed as f64;
        PlaybackStatus::Rendered
    }
}

/// Lock-free gate controls and diagnostics shared by the GUI and audio
/// worker. The worker snapshots these immediately before each pipeline
/// push, so changes take effect during an active call without rebuilding
/// model sessions or interrupting the stream.
#[derive(Debug)]
pub struct GateTuning {
    threshold: AtomicU32,
    hangover_ms: AtomicU32,
    release_ms: AtomicU32,
    last_score: AtomicU32,
    effective_threshold: AtomicU32,
}

impl GateTuning {
    #[must_use]
    pub fn new(config: GateConfig) -> Self {
        Self {
            threshold: AtomicU32::new(config.theta_pass.to_bits()),
            hangover_ms: AtomicU32::new(config.hangover_ms.to_bits()),
            release_ms: AtomicU32::new(config.release_ms.to_bits()),
            last_score: AtomicU32::new(0.0_f32.to_bits()),
            effective_threshold: AtomicU32::new(config.theta_pass.to_bits()),
        }
    }

    #[must_use]
    pub fn threshold(&self) -> f32 {
        f32::from_bits(self.threshold.load(Ordering::Relaxed))
    }

    pub fn set_threshold(&self, value: f32) {
        if value.is_finite() {
            self.threshold
                .store(value.clamp(0.05, 0.95).to_bits(), Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn hangover_ms(&self) -> f32 {
        f32::from_bits(self.hangover_ms.load(Ordering::Relaxed))
    }

    pub fn set_hangover_ms(&self, value: f32) {
        if value.is_finite() {
            self.hangover_ms
                .store(value.clamp(0.0, 2_000.0).to_bits(), Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn release_ms(&self) -> f32 {
        f32::from_bits(self.release_ms.load(Ordering::Relaxed))
    }

    pub fn set_release_ms(&self, value: f32) {
        if value.is_finite() {
            self.release_ms
                .store(value.clamp(0.0, 1_000.0).to_bits(), Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn last_score(&self) -> f32 {
        f32::from_bits(self.last_score.load(Ordering::Relaxed))
    }

    #[must_use]
    pub fn effective_threshold(&self) -> f32 {
        f32::from_bits(self.effective_threshold.load(Ordering::Relaxed))
    }

    fn apply_to(&self, mut config: GateConfig) -> GateConfig {
        config.theta_pass = self.threshold();
        config.hangover_ms = self.hangover_ms();
        config.release_ms = self.release_ms();
        // A manual threshold must stay exact and predictable.
        config.adaptive_theta = false;
        config
    }

    fn update_diagnostics(&self, score: f32, effective_threshold: f32) {
        self.last_score.store(score.to_bits(), Ordering::Relaxed);
        self.effective_threshold
            .store(effective_threshold.to_bits(), Ordering::Relaxed);
    }
}

/// Caller-tunable knobs for [`LiveSession::new`].
///
/// Most users want `SessionConfig::default()`; named device fields
/// let the CLI / GUI honour user choice from the device picker.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    /// Input device name (as returned by [`crate::list_input_devices`]).
    /// `None` → host default input.
    pub input_device: Option<String>,
    /// Output device name. `None` → host default output.
    pub output_device: Option<String>,
    /// Streaming pipeline configuration. `audio_sample_rate` is
    /// overridden internally to [`INTERNAL_SAMPLE_RATE`] (48 kHz) —
    /// the cpal-side resamplers convert from the device rate.
    pub streaming: StreamingConfig,
    /// When `Some(path)`, the streaming engine runs DFN3 noise
    /// suppression at 48 kHz **after** the optional TSE stage and
    /// before the gate envelope. `None` → no NS.
    ///
    /// Latency cost: ~20 ms (2-frame DFN3 conv lookahead at the
    /// 48 kHz hop). Forwarded into [`StreamingConfig::dfn3_onnx_path`]
    /// at session construction.
    pub dfn3_onnx_path: Option<PathBuf>,
    /// When `Some(path)`, the streaming engine wires in the pyannote
    /// 3.0 segmentation ONNX as an overlap detector and routes the
    /// chain adaptively: solo speaker → DFN3 only, overlap → TSE only.
    /// `None` → legacy TSE → DFN3 cascade (which attenuates by ~28 dB
    /// — definitely not what users want).
    ///
    /// Forwarded into [`StreamingConfig::overlap_onnx_path`] at
    /// session construction.
    pub overlap_onnx_path: Option<PathBuf>,
    /// Optional two-speaker SepFormer model. When both this and
    /// `speaker_embedding_onnx_path` are present, input is split into
    /// two anonymous speakers and only the stream closest to the
    /// enrollment is forwarded to the normal gate/noise-removal chain.
    pub sepformer_onnx_path: Option<PathBuf>,
    /// ECAPA embedding-only ONNX used to identify the two streams
    /// emitted by SepFormer. This is a separate session from the
    /// streaming gate's ECAPA instance.
    pub speaker_embedding_onnx_path: Option<PathBuf>,
    /// Companion enrollment containing separator-domain anchors. Falls
    /// back to the ordinary raw-microphone pool for compatibility with
    /// profiles created before guided separator calibration existed.
    pub speaker_selection_pool: Option<EmbeddingPool>,
    /// Live-tunable separator knobs (fail-closed threshold + latest
    /// best-score readout) shared with the UI. `None` → the session
    /// uses private defaults and the threshold is fixed for its
    /// lifetime.
    pub separator_tuning: Option<Arc<SeparatorTuning>>,
    /// Live speaker-gate controls and score readout shared with the UI.
    /// `None` keeps the values in [`Self::streaming`] fixed.
    pub gate_tuning: Option<Arc<GateTuning>>,
    /// How to fold a multi-channel input device's interleaved
    /// frames down to mono. Default `Average` — same as step 12's
    /// hard-coded behaviour. Use `Channel(n)` to pick a specific
    /// channel for setups where one channel is the target signal
    /// and the others are room mics / reference channels (podcast
    /// interface with host on ch 0, guest on ch 1, etc.).
    pub input_channel: ChannelStrategy,
}

/// Stats surfaced when [`LiveSession::stop`] returns. Useful as a
/// post-run "did anything weird happen?" summary for the CLI.
#[derive(Debug, Default, Clone, Copy)]
pub struct LiveSessionStats {
    /// Total samples that flowed through the worker (at 48 kHz mono).
    pub samples_processed: u64,
    /// Input chunks dropped because the input ring was full (worker
    /// can't keep up).
    pub input_overruns: u64,
    /// Output chunks where the output ring was empty and silence
    /// was emitted after playback had already started (worker fell
    /// behind). Intentional startup prebuffering is not counted.
    pub output_underruns: u64,
}

/// Async event the worker can emit. Step 12 uses only `Error` for
/// pipeline failures; later steps will add diagnostics (gate state,
/// auto-learn events) for the GUI.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Worker-side pipeline error. The worker exits after emitting
    /// this; subsequent `stop()` returns the partial stats.
    Error(String),
}

/// Live audio session. Construct, then optionally drain
/// [`SessionEvent`]s from `events()`, then call `stop()`.
pub struct LiveSession {
    // Streams must outlive the session for cpal to keep callbacks
    // firing. Held as Option so `stop()` can drop them
    // deterministically before joining the worker.
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    worker: Option<JoinHandle<()>>,
    events_rx: Receiver<SessionEvent>,
    samples_processed: Arc<AtomicU64>,
    input_overruns: Arc<AtomicU64>,
    output_underruns: Arc<AtomicU64>,
    /// f32 bits of the most recent input chunk's RMS — updated by
    /// the worker each iteration. Polled by the GUI's level meter.
    input_rms_bits: Arc<AtomicU32>,
    /// f32 bits of the most recent output chunk's RMS — gate ×
    /// envelope smoothing. Together with `input_rms_bits` gives the
    /// "how much is being suppressed" reading.
    output_rms_bits: Arc<AtomicU32>,
    /// Latest gate state (true = the pipeline is currently passing
    /// audio through). Updated whenever the streaming engine emits
    /// a new gate transition.
    gate_on: Arc<AtomicBool>,
}

impl LiveSession {
    /// Open input + output devices, spawn the worker thread, start
    /// streaming. Returns once everything is running; the caller
    /// holds the session and calls `stop()` when done.
    ///
    /// # Errors
    ///
    /// Surfaces device-query, format-not-supported, stream-construction
    /// and worker-spawn failures as `AudioIoError`.
    pub fn new(
        pool: EmbeddingPool,
        components: PipelineComponents,
        config: SessionConfig,
    ) -> Result<Self, AudioIoError> {
        let host = cpal::default_host();
        let input_dev = pick_device(&host, DeviceKind::Input, config.input_device.as_deref())?;
        let output_dev = pick_device(&host, DeviceKind::Output, config.output_device.as_deref())?;

        let input_cfg = input_dev
            .default_input_config()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;
        let output_cfg = output_dev
            .default_output_config()
            .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?;

        let input_format = input_cfg.sample_format();
        let output_format = output_cfg.sample_format();
        if input_format != SampleFormat::F32 {
            return Err(AudioIoError::UnsupportedSampleFormat {
                format: format!("input {input_format:?}"),
            });
        }
        if output_format != SampleFormat::F32 {
            return Err(AudioIoError::UnsupportedSampleFormat {
                format: format!("output {output_format:?}"),
            });
        }

        let input_sr = input_cfg.sample_rate().0;
        let output_sr = output_cfg.sample_rate().0;
        let input_channels = input_cfg.channels();
        let output_channels = output_cfg.channels();

        eprintln!(
            "[audio-io] input  : {} ({} Hz, {} ch, f32)",
            input_dev.name().unwrap_or_else(|_| "?".into()),
            input_sr,
            input_channels
        );
        eprintln!(
            "[audio-io] output : {} ({} Hz, {} ch, f32)",
            output_dev.name().unwrap_or_else(|_| "?".into()),
            output_sr,
            output_channels
        );

        let (input_tx, input_rx) = bounded::<Vec<f32>>(INPUT_RING_CHUNKS);
        let (output_tx, output_rx) = bounded::<Vec<f32>>(OUTPUT_RING_CHUNKS);
        let (events_tx, events_rx) = bounded::<SessionEvent>(64);

        let samples_processed = Arc::new(AtomicU64::new(0));
        let input_overruns = Arc::new(AtomicU64::new(0));
        let output_underruns = Arc::new(AtomicU64::new(0));
        let input_rms_bits = Arc::new(AtomicU32::new(0));
        let output_rms_bits = Arc::new(AtomicU32::new(0));
        let gate_on = Arc::new(AtomicBool::new(false));

        let input_stream = build_input_stream(
            &input_dev,
            input_cfg.clone().into(),
            input_channels,
            input_sr,
            input_tx,
            input_overruns.clone(),
            config.input_channel,
        )?;
        let output_stream = build_output_stream(
            &output_dev,
            output_cfg.clone().into(),
            output_channels,
            output_sr,
            output_rx,
            output_underruns.clone(),
        )?;

        // Override audio_sample_rate so the pipeline sees the
        // post-resample 48 kHz the worker actually feeds it, and
        // forward the DFN3 path so the engine wires NS as the
        // post-TSE stage in the audio chain.
        let mut streaming_cfg = config.streaming.clone();
        streaming_cfg.audio_sample_rate = INTERNAL_SAMPLE_RATE;
        streaming_cfg
            .dfn3_onnx_path
            .clone_from(&config.dfn3_onnx_path);
        streaming_cfg
            .overlap_onnx_path
            .clone_from(&config.overlap_onnx_path);
        let dfn3_enabled = streaming_cfg.dfn3_onnx_path.is_some();
        let adaptive_enabled = streaming_cfg.overlap_onnx_path.is_some()
            && streaming_cfg.pipeline.tse.is_some()
            && dfn3_enabled;
        eprintln!(
            "[audio-io] chain routing: {}",
            if adaptive_enabled {
                "ADAPTIVE (Solo→DFN3, Overlap→TSE)"
            } else {
                "legacy cascade (TSE → DFN3) — set overlap_onnx_path for adaptive mode"
            }
        );
        let selection_pool = config
            .speaker_selection_pool
            .clone()
            .unwrap_or_else(|| pool.clone());
        let separator = match (
            config.sepformer_onnx_path.as_deref(),
            config.speaker_embedding_onnx_path.as_deref(),
        ) {
            (Some(sepformer), Some(ecapa)) => {
                let tuning = config
                    .separator_tuning
                    .clone()
                    .unwrap_or_else(|| Arc::new(SeparatorTuning::default()));
                eprintln!(
                    "[audio-io] target speaker separation: ENABLED \
                     (1 s SepFormer → enrolled ECAPA stream selection, \
                     threshold {:.2})",
                    tuning.threshold(),
                );
                Some(
                    TargetSpeakerSeparator::new(sepformer, ecapa, selection_pool, tuning)
                        .map_err(AudioIoError::Pipeline)?,
                )
            }
            (None, None) => None,
            _ => {
                return Err(AudioIoError::Pipeline(
                    "SepFormer and speaker-embedding ONNX paths must be configured together"
                        .to_string(),
                ));
            }
        };
        if let Some(tuning) = config.gate_tuning.as_deref() {
            tuning.update_diagnostics(0.0, tuning.threshold());
        }
        let pipeline = StreamingPipeline::new(pool, streaming_cfg, components)
            .map_err(|e| AudioIoError::Pipeline(e.to_string()))?;
        if dfn3_enabled {
            eprintln!("[audio-io] noise suppression: ENABLED (+ ~20 ms latency, post-TSE)");
        }

        let worker = spawn_worker(
            pipeline,
            separator,
            config.gate_tuning,
            input_rx,
            output_tx,
            events_tx,
            samples_processed.clone(),
            input_rms_bits.clone(),
            output_rms_bits.clone(),
            gate_on.clone(),
            input_overruns.clone(),
            output_underruns.clone(),
        )?;

        input_stream
            .play()
            .map_err(|e| AudioIoError::Stream(e.to_string()))?;
        output_stream
            .play()
            .map_err(|e| AudioIoError::Stream(e.to_string()))?;

        Ok(Self {
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
            worker: Some(worker),
            events_rx,
            samples_processed,
            input_overruns,
            output_underruns,
            input_rms_bits,
            output_rms_bits,
            gate_on,
        })
    }

    /// Snapshot of the latest input chunk's RMS — useful for the
    /// GUI's live level meter. Updated by the worker on each
    /// `push_samples` iteration; reads are lock-free atomic loads
    /// of an f32 stored as `u32` bits.
    #[must_use]
    pub fn input_rms(&self) -> f32 {
        f32::from_bits(self.input_rms_bits.load(Ordering::Relaxed))
    }

    /// Snapshot of the latest output chunk's RMS. Compared with
    /// `input_rms` this shows how much the gate × envelope is
    /// suppressing right now.
    #[must_use]
    pub fn output_rms(&self) -> f32 {
        f32::from_bits(self.output_rms_bits.load(Ordering::Relaxed))
    }

    /// Latest gate state (true = audio is currently being passed
    /// through). Useful for a "live filter ON / OFF" indicator in
    /// the GUI.
    #[must_use]
    pub fn gate_on(&self) -> bool {
        self.gate_on.load(Ordering::Relaxed)
    }

    /// Try to receive a pending session event without blocking.
    /// Returns `None` when no event is queued.
    pub fn try_recv_event(&self) -> Option<SessionEvent> {
        self.events_rx.try_recv().ok()
    }

    /// Live snapshot of the running counters. Useful for the CLI to
    /// print a periodic "still going" line.
    #[must_use]
    pub fn stats_snapshot(&self) -> LiveSessionStats {
        LiveSessionStats {
            samples_processed: self.samples_processed.load(Ordering::Relaxed),
            input_overruns: self.input_overruns.load(Ordering::Relaxed),
            output_underruns: self.output_underruns.load(Ordering::Relaxed),
        }
    }

    /// Stop the session and return the final stats. Drops the cpal
    /// streams first (so the input queue stops growing), then joins
    /// the worker (which exits when the input channel closes).
    ///
    /// # Errors
    ///
    /// Returns `WorkerDied` if the worker thread panicked.
    pub fn stop(mut self) -> Result<LiveSessionStats, AudioIoError> {
        self.input_stream.take();
        // Output stream stays alive until the worker has flushed —
        // we just drop it after joining so trailing audio still
        // flows out.
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| AudioIoError::WorkerDied("panicked".into()))?;
        }
        self.output_stream.take();
        Ok(self.stats_snapshot())
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        // If the caller forgot `stop()`, do the same shutdown
        // sequence on Drop. Errors are swallowed — there's nowhere
        // useful to report them from a destructor.
        self.input_stream.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.output_stream.take();
    }
}

/// One worker iteration drains as much input as is available, runs
/// the pipeline, sends output downstream. Falls out of the loop when
/// the input channel disconnects (i.e. all senders dropped because
/// the input cpal stream was torn down).
fn spawn_worker(
    mut pipeline: StreamingPipeline,
    mut separator: Option<TargetSpeakerSeparator>,
    gate_tuning: Option<Arc<GateTuning>>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
    events_tx: Sender<SessionEvent>,
    samples_processed: Arc<AtomicU64>,
    input_rms_bits: Arc<AtomicU32>,
    output_rms_bits: Arc<AtomicU32>,
    gate_on: Arc<AtomicBool>,
    input_overruns: Arc<AtomicU64>,
    output_underruns: Arc<AtomicU64>,
) -> Result<JoinHandle<()>, AudioIoError> {
    std::thread::Builder::new()
        .name("mellonella-audio-io-worker".into())
        .spawn(move || {
            let mut first_input_logged = false;
            let mut first_nonempty_output_logged = false;
            let mut first_producer_drop_logged = false;
            // Producer-side back-pressure: incremented whenever the
            // ring is full and the worker drops a freshly produced
            // chunk. This means the output device is slower than the
            // pipeline (or the worker is bursting after a stall).
            // Separate from `output_underruns` (= cpal callback fired
            // with an empty ring) so the two distinct failure modes
            // are visible.
            let mut producer_drops: u64 = 0;
            // Periodic stats: every ~1 s of wall clock, log delta
            // counters so the user can see if underruns / overruns
            // continue past startup (= persistent CPU shortfall) or
            // were a one-shot stream-startup artifact.
            let mut last_stats_at = std::time::Instant::now();
            let mut last_underruns = 0_u64;
            let mut last_overruns = 0_u64;
            let mut last_processed = 0_u64;
            let mut last_producer_drops = 0_u64;
            while let Ok(chunk) = input_rx.recv() {
                if !first_input_logged && !chunk.is_empty() {
                    let r = rms(&chunk);
                    let nan_count = chunk.iter().filter(|s| s.is_nan()).count();
                    eprintln!(
                        "[audio-io] worker: first input chunk received ({} samples, RMS {:.4}, NaN count {})",
                        chunk.len(),
                        r,
                        nan_count,
                    );
                    first_input_logged = true;
                }
                if !chunk.is_empty() {
                    input_rms_bits.store(rms(&chunk).to_bits(), Ordering::Relaxed);
                }
                let separated_chunks = match separator.as_mut() {
                    Some(stage) => match stage.push(&chunk) {
                        Ok(chunks) => chunks,
                        Err(error) => {
                            let _ = events_tx.send(SessionEvent::Error(format!(
                                "target speaker separation: {error}"
                            )));
                            return;
                        }
                    },
                    None => vec![chunk],
                };
                for selected_chunk in separated_chunks {
                    match forward_pipeline_audio(
                        &mut pipeline,
                        &selected_chunk,
                        &output_tx,
                        &samples_processed,
                        &output_rms_bits,
                        &gate_on,
                        &mut first_nonempty_output_logged,
                        &mut first_producer_drop_logged,
                        &mut producer_drops,
                        gate_tuning.as_deref(),
                    ) {
                        Ok(true) => {}
                        Ok(false) => return,
                        Err(error) => {
                            let _ = events_tx.send(SessionEvent::Error(error));
                            return;
                        }
                    }
                }
                // Once every ~1 s of wall clock, dump rolling stats so
                // post-startup behaviour (persistent underruns vs.
                // clean steady state) is visible without a debugger.
                if last_stats_at.elapsed() >= std::time::Duration::from_secs(1) {
                    let u = output_underruns.load(Ordering::Relaxed);
                    let o = input_overruns.load(Ordering::Relaxed);
                    let p = samples_processed.load(Ordering::Relaxed);
                    let d = producer_drops;
                    eprintln!(
                        "[audio-io] stats: +{du} underruns, +{do_} overruns, +{dd} producer \
                         drops, +{dp} samples in last {ms} ms (totals: underruns={u}, \
                         overruns={o}, producer_drops={d}, samples={p})",
                        du = u.saturating_sub(last_underruns),
                        do_ = o.saturating_sub(last_overruns),
                        dd = d.saturating_sub(last_producer_drops),
                        dp = p.saturating_sub(last_processed),
                        ms = last_stats_at.elapsed().as_millis(),
                    );
                    last_underruns = u;
                    last_overruns = o;
                    last_processed = p;
                    last_producer_drops = d;
                    last_stats_at = std::time::Instant::now();
                }
            }
            if let Some(stage) = separator.as_mut() {
                match stage.flush() {
                    Ok(Some(selected_chunk)) => {
                        let _ = forward_pipeline_audio(
                            &mut pipeline,
                            &selected_chunk,
                            &output_tx,
                            &samples_processed,
                            &output_rms_bits,
                            &gate_on,
                            &mut first_nonempty_output_logged,
                            &mut first_producer_drop_logged,
                            &mut producer_drops,
                            gate_tuning.as_deref(),
                        );
                    }
                    Ok(None) => {}
                    Err(error) => {
                        let _ = events_tx.send(SessionEvent::Error(format!(
                            "target speaker separation flush: {error}"
                        )));
                    }
                }
            }
            if let Ok(tail) = pipeline.flush() {
                let n = tail.audio.len() as u64;
                if n > 0 {
                    let _ = output_tx.try_send(tail.audio);
                }
                samples_processed.fetch_add(n, Ordering::Relaxed);
            }
        })
        .map_err(|e| AudioIoError::Stream(format!("spawn worker: {e}")))
}

/// Push one (possibly one-second) selected-speaker block through the
/// ordinary streaming pipeline and enqueue its audio for cpal.
#[allow(clippy::too_many_arguments)]
fn forward_pipeline_audio(
    pipeline: &mut StreamingPipeline,
    audio: &[f32],
    output_tx: &Sender<Vec<f32>>,
    samples_processed: &AtomicU64,
    output_rms_bits: &AtomicU32,
    gate_on: &AtomicBool,
    first_nonempty_output_logged: &mut bool,
    first_producer_drop_logged: &mut bool,
    producer_drops: &mut u64,
    gate_tuning: Option<&GateTuning>,
) -> Result<bool, String> {
    if let Some(tuning) = gate_tuning {
        // Preserve non-UI gate fields (attack, scoring mode, learning
        // guards) and replace only the controls surfaced to the user.
        let gate = tuning.apply_to(pipeline.gate_config());
        pipeline.set_gate_config(gate);
    }
    let out = pipeline.push_samples(audio).map_err(|e| e.to_string())?;
    if let Some(tuning) = gate_tuning {
        tuning.update_diagnostics(pipeline.last_score(), pipeline.effective_gate_threshold());
    }
    if let Some(&(_, is_on)) = out.gate_decisions.last() {
        gate_on.store(is_on, Ordering::Relaxed);
    }
    if !out.audio.is_empty() {
        output_rms_bits.store(rms(&out.audio).to_bits(), Ordering::Relaxed);
        if !*first_nonempty_output_logged {
            eprintln!(
                "[audio-io] worker: first non-empty pipeline output \
                 ({} samples, RMS {:.4})",
                out.audio.len(),
                f32::from_bits(output_rms_bits.load(Ordering::Relaxed)),
            );
            *first_nonempty_output_logged = true;
        }
    }
    let n = out.audio.len() as u64;
    if n == 0 {
        return Ok(true);
    }
    match output_tx.try_send(out.audio) {
        Ok(()) => {
            samples_processed.fetch_add(n, Ordering::Relaxed);
            Ok(true)
        }
        Err(TrySendError::Full(_)) => {
            *producer_drops += 1;
            if !*first_producer_drop_logged {
                eprintln!(
                    "[audio-io] worker: first producer drop — output ring full \
                     ({OUTPUT_RING_CHUNKS} chunks). Further drops are silent; \
                     check the periodic stats line."
                );
                *first_producer_drop_logged = true;
            }
            Ok(true)
        }
        Err(TrySendError::Disconnected(_)) => Ok(false),
    }
}

/// Convert a sample-count ring capacity to a sensible chunk count.
/// `crossbeam_channel::bounded` takes a slot count, not a sample
/// count; chunks vary in size by device callback period, but a
/// 1-ms-equivalent slot count is a fine ceiling: 24 000 samples ≈
/// 500 slots of 1 ms each.
/// Root-mean-square of an audio chunk. Used by the worker thread
/// to surface a level-meter reading via the atomic snapshot
/// without forcing any new allocation.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn pick_device(
    host: &cpal::Host,
    kind: DeviceKind,
    name: Option<&str>,
) -> Result<Device, AudioIoError> {
    if let Some(name) = name {
        let iter = match kind {
            DeviceKind::Input => host
                .input_devices()
                .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?,
            DeviceKind::Output => host
                .output_devices()
                .map_err(|e| AudioIoError::DeviceQuery(e.to_string()))?,
        };
        for dev in iter {
            if dev.name().ok().as_deref() == Some(name) {
                return Ok(dev);
            }
        }
        Err(AudioIoError::DeviceNotFound {
            kind,
            name: name.to_string(),
        })
    } else {
        let dev = match kind {
            DeviceKind::Input => host.default_input_device(),
            DeviceKind::Output => host.default_output_device(),
        };
        dev.ok_or(AudioIoError::NoDefaultDevice(kind))
    }
}

fn build_input_stream(
    device: &Device,
    config: StreamConfig,
    channels: u16,
    device_sr: u32,
    input_tx: Sender<Vec<f32>>,
    overruns: Arc<AtomicU64>,
    channel_strategy: ChannelStrategy,
) -> Result<Stream, AudioIoError> {
    let mut resampler =
        StreamingResampler::new(device_sr, INTERNAL_SAMPLE_RATE).map_err(AudioIoError::Resample)?;
    let channels_usize = channels as usize;
    let mut first_call = true;
    let err_fn = |e: cpal::StreamError| eprintln!("[audio-io] input stream error: {e}");
    device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if first_call {
                    eprintln!(
                        "[audio-io] input: first cpal callback fired ({} samples × {} ch \
                         @ {} Hz)",
                        data.len() / channels_usize,
                        channels_usize,
                        device_sr,
                    );
                    first_call = false;
                }
                let mono = channel_strategy.downmix(data, channels_usize);
                let processed = match &mut resampler {
                    Some(r) => r.process(&mono),
                    None => Ok(mono),
                };
                match processed {
                    Ok(chunk) if !chunk.is_empty() => {
                        if let Err(TrySendError::Full(_)) = input_tx.try_send(chunk) {
                            overruns.fetch_add(1, Ordering::Relaxed);
                        }
                        // Disconnected means the session is shutting
                        // down; the dropped chunk is fine.
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("[audio-io] input resample: {e}"),
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| AudioIoError::Stream(e.to_string()))
}

fn build_output_stream(
    device: &Device,
    config: StreamConfig,
    channels: u16,
    device_sr: u32,
    output_rx: Receiver<Vec<f32>>,
    underruns: Arc<AtomicU64>,
) -> Result<Stream, AudioIoError> {
    let mut resampler =
        StreamingResampler::new(INTERNAL_SAMPLE_RATE, device_sr).map_err(AudioIoError::Resample)?;
    let channels_usize = channels as usize;
    let mut playback = PlaybackBuffer::new(device_sr);
    let mut mono_scratch: Vec<f32> = Vec::new();
    // Diagnostics: print the first callback so users can confirm cpal
    // is actually calling us, and the first underrun so silent output
    // bugs surface in the console. Heavy logging in the steady state
    // would flood at audio-callback frequency (~100 Hz), so these are
    // one-shot flags.
    let mut first_call = true;
    let mut first_underrun_logged = false;
    let mut first_ring_chunk_logged = false;
    let err_fn = |e: cpal::StreamError| eprintln!("[audio-io] output stream error: {e}");
    device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let frames_needed = data.len() / channels_usize;
                if first_call {
                    eprintln!(
                        "[audio-io] output: first cpal callback fired ({frames_needed} frames × {channels_usize} ch)"
                    );
                    first_call = false;
                }
                // Drain every produced chunk. `PlaybackBuffer` owns the
                // sample-level fill target and clock correction; leaving
                // chunks parked in this slot-based ring would hide its
                // true fill level and eventually force a whole-chunk drop.
                while let Ok(chunk) = output_rx.try_recv() {
                    if !first_ring_chunk_logged && !chunk.is_empty() {
                        eprintln!(
                            "[audio-io] output: first ring chunk received ({} samples @ {} Hz)",
                            chunk.len(),
                            INTERNAL_SAMPLE_RATE
                        );
                        first_ring_chunk_logged = true;
                    }
                    let resampled = match &mut resampler {
                        Some(r) => r.process(&chunk).unwrap_or_default(),
                        None => chunk,
                    };
                    playback.push(&resampled);
                }

                mono_scratch.resize(frames_needed, 0.0);
                let buffered_before = playback.buffered_samples();
                let status = playback.render(&mut mono_scratch);
                if status == PlaybackStatus::Underrun {
                    underruns.fetch_add(1, Ordering::Relaxed);
                    if !first_underrun_logged {
                        eprintln!(
                            "[audio-io] output: first underrun after playback start — wanted \
                             {frames_needed} frames with {buffered_before} buffered (worker \
                             behind). Rebuffering before audio resumes."
                        );
                        first_underrun_logged = true;
                    }
                }
                // Broadcast mono → all output channels. Silence on
                // startup prebuffer / underrun.
                for (frame_idx, &mono) in mono_scratch.iter().enumerate() {
                    let base = frame_idx * channels_usize;
                    for ch in 0..channels_usize {
                        data[base + ch] = Sample::from_sample(mono);
                    }
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| AudioIoError::Stream(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn playback_buffer_primes_before_emitting() {
        let mut playback = PlaybackBuffer::new(1_000);
        let mut output = vec![1.0; 10];
        playback.push(&vec![0.25; 63]);
        assert_eq!(playback.render(&mut output), PlaybackStatus::Prebuffering);
        assert!(output.iter().all(|&sample| sample == 0.0));

        playback.push(&[0.25; 40]);
        assert_eq!(playback.render(&mut output), PlaybackStatus::Rendered);
        assert!(output.iter().all(|sample| (0.0..=0.25).contains(sample)));
        assert!(output.iter().skip(5).any(|&sample| sample > 0.0));
    }

    #[test]
    fn playback_buffer_corrects_independent_device_clock_drift() {
        let mut playback = PlaybackBuffer::new(48_000);
        playback.push(&vec![0.0; playback.target_fill]);
        let centred = playback.playback_step();
        playback.push(&vec![0.0; playback.target_fill]);
        let overfilled = playback.playback_step();
        playback.samples.truncate(playback.target_fill / 2);
        let underfilled = playback.playback_step();
        assert!((centred - 1.0).abs() < f64::EPSILON);
        assert!(overfilled > 1.0, "overfilled queue must drain faster");
        assert!(underfilled < 1.0, "underfilled queue must drain slower");
        assert!(overfilled <= 1.0 + MAX_CLOCK_CORRECTION);
        assert!(underfilled >= 1.0 - MAX_CLOCK_CORRECTION);
    }

    #[test]
    fn playback_buffer_timings_follow_the_output_device_sample_rate() {
        for sample_rate in [44_100_u32, 48_000, 96_000] {
            let playback = PlaybackBuffer::new(sample_rate);
            let expected = |milliseconds: u32| {
                usize::try_from(u64::from(sample_rate) * u64::from(milliseconds) / 1_000)
                    .expect("test sample count fits usize")
            };
            assert_eq!(playback.start_fill, expected(OUTPUT_PREBUFFER_MS));
            assert_eq!(playback.target_fill, expected(OUTPUT_TARGET_BUFFER_MS));
            assert_eq!(playback.fade_total, expected(PLAYBACK_FADE_MS));
        }
    }

    #[test]
    fn gate_tuning_updates_only_exposed_fields() {
        let base = GateConfig {
            theta_pass: 0.30,
            hangover_ms: 200.0,
            release_ms: 80.0,
            attack_ms: 27.0,
            theta_f0: 0.61,
            adaptive_theta: true,
            ..GateConfig::default()
        };
        let tuning = GateTuning::new(base);
        tuning.set_threshold(0.57);
        tuning.set_hangover_ms(625.0);
        tuning.set_release_ms(170.0);
        let changed = tuning.apply_to(base);
        assert_eq!(changed.theta_pass, 0.57);
        assert_eq!(changed.hangover_ms, 625.0);
        assert_eq!(changed.release_ms, 170.0);
        assert_eq!(changed.attack_ms, 27.0);
        assert_eq!(changed.theta_f0, 0.61);
        assert!(!changed.adaptive_theta);
    }

    #[test]
    fn gate_tuning_rejects_non_finite_and_clamps_extremes() {
        let tuning = GateTuning::new(GateConfig::default());
        tuning.set_threshold(f32::NAN);
        assert_eq!(tuning.threshold(), GateConfig::default().theta_pass);
        tuning.set_threshold(5.0);
        tuning.set_hangover_ms(-10.0);
        tuning.set_release_ms(5_000.0);
        assert_eq!(tuning.threshold(), 0.95);
        assert_eq!(tuning.hangover_ms(), 0.0);
        assert_eq!(tuning.release_ms(), 1_000.0);
    }
}

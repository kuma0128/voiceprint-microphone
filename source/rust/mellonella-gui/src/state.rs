//! Mutable state owned by the eframe app.
//!
//! Modelled as one flat struct rather than a `SessionState` enum
//! because the UI inspects most fields independently of whether a
//! live session is currently running (enrollment, device selection,
//! last error are all sticky across start/stop cycles).
//!
//! Enrollment is held as an **in-memory `EmbeddingPool`** built from
//! the mic recording flow and updated by the auto-learn pool during
//! a live session. The pool is persisted to
//! `~/.config/mellonella/enrollment.json` and auto-loaded on next
//! launch, so file-level import / export controls are unnecessary.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mellonella_audio_io::{
    build_separator_enrollment_pool, list_input_devices, list_output_devices, AudioDevice,
    GateTuning, LiveSession, LiveSessionStats, Recorder, SeparatorTuning, SessionConfig,
    SessionEvent,
};
use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::{cos_similarity, GateConfig};
use mellonella_core::pipeline::{enroll_from_recording, PipelineComponents, PipelineConfig};
use mellonella_core::resample::resample_to;
use mellonella_core::streaming::StreamingConfig;
use mellonella_core::tse_stage::TseStageConfig;
use mellonella_core::vad::SileroVad;

/// Output sample rate used end-to-end (matches the CLI's offline
/// constant and `StreamingConfig::default().audio_sample_rate`).
pub const OUTPUT_SAMPLE_RATE: u32 = 48_000;

/// Decision sample rate for VAD / ECAPA / F0 inside the pipeline.
pub const DECISION_SAMPLE_RATE: u32 = 16_000;

/// Twenty seconds produces roughly twelve overlapping 3-second ECAPA
/// anchors (1.5-second shift), enough to represent normal variation in
/// pitch, vowels and articulation. The old five-second default produced
/// only two anchors and was not reliable against a similar friend.
pub const DEFAULT_RECORD_SECS: f32 = 20.0;
/// Reject a capture that contains too little usable speech. Keeping the
/// previous saved profile is safer than replacing it with 1-2 anchors.
pub const MIN_ENROLLMENT_ANCHORS: usize = 6;
/// Enrollment anchors from one person should remain moderately close
/// even across different vowels and pitch. Below this median the
/// profile is usually dominated by noise, clipping, or another voice.
const MIN_ENROLLMENT_CONSISTENCY: f32 = 0.40;
/// Voiceprint controls calibrated for the bundled microphone/profile.
/// The old 0.30 threshold was permissive enough for a loud same-gender
/// interferer; the longer hangover compensates for the stricter score
/// without chattering on a single low-scoring vowel.
pub const DEFAULT_GATE_THRESHOLD: f32 = 0.45;
pub const DEFAULT_GATE_HANGOVER_MS: f32 = 500.0;
pub const DEFAULT_GATE_RELEASE_MS: f32 = 120.0;

/// Where the current enrollment came from. Surfaced in the UI so
/// users see "Recorded 5.0 s" vs the auto-loaded persistent pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentOrigin {
    None,
    Mic {
        secs: u32,
    },
    /// Auto-loaded from `default_enrollment_path()` on launch. Path
    /// retained for display so the user can see which file backs the
    /// in-memory pool.
    AutoLoaded(PathBuf),
}

pub struct AppState {
    /// Current speaker pool. `None` until the user enrolls a voice.
    /// Held in-memory so `Start` doesn't have to re-read JSON.
    pub pool: Option<EmbeddingPool>,
    /// Companion pool containing embeddings measured after SepFormer.
    /// Kept separate so separator artifacts cannot alter the ordinary
    /// raw-microphone gate centroid.
    pub separator_pool: Option<EmbeddingPool>,
    pub origin: EnrollmentOrigin,
    pub pool_anchors: usize,
    pub pool_f0_mu: f32,
    pub pool_f0_sigma: f32,
    /// Human-readable, locally computed quality summary. No recording
    /// or biometric vector leaves the machine.
    pub enrollment_quality: Option<String>,
    pub available_inputs: Vec<AudioDevice>,
    pub available_outputs: Vec<AudioDevice>,
    /// Selected input device name; `None` → host default. Stored as
    /// `String` rather than indexes into `available_inputs` so the
    /// selection survives a device-list refresh.
    pub selected_input: Option<String>,
    pub selected_output: Option<String>,
    /// Path to the TSE streaming ONNX (Stage C). Picked via a file
    /// dialog in the Settings panel; persisted in-memory only for this
    /// session.
    pub tse_onnx_path: Option<PathBuf>,
    /// Mic-enrollment recording duration in seconds. Step 20 made
    /// this user-configurable from the GUI (a slider next to the
    /// Record button); default matches the previous fixed
    /// [`DEFAULT_RECORD_SECS`] value.
    pub record_duration_secs: f32,
    /// User-adjustable gate / envelope parameters. Sliders in the
    /// Settings panel mutate these in place; `start()` reads them
    /// when building the `SessionConfig`. Defaults match
    /// `GateConfig::default()`.
    pub gate_cfg: GateConfig,
    /// Lock-free live controls shared with the running audio worker.
    pub gate_tuning: Arc<GateTuning>,
    /// User-adjustable pipeline cadence (currently just
    /// `sv_update_samples` — ECAPA refresh interval). Sliders in
    /// the Settings panel mutate this; defaults match
    /// `PipelineConfig::default()`.
    pub pipeline_cfg: PipelineConfig,
    /// Shared live-tunable separator knobs: the Settings slider writes
    /// the fail-closed threshold, the running session reads it per
    /// block (no restart needed), and the latest best-score readout
    /// flows back for display. The threshold persists in
    /// `separator-threshold.txt` next to the enrollment JSON.
    pub separator_tuning: Arc<SeparatorTuning>,
    /// When `true`, the recording currently in flight extends the
    /// saved enrollment instead of replacing it: the new anchors are
    /// appended to the existing pools so one profile can hold several
    /// registrations (morning voice, evening voice, …). Scoring takes
    /// the max over all anchors, so extra registrations only ever help
    /// recall.
    pub pending_append: bool,
    pub session: Option<LiveSession>,
    pub recorder: Option<Recorder>,
    pub last_error: Option<String>,
    pub last_stats: LiveSessionStats,
}

impl Default for AppState {
    fn default() -> Self {
        let available_inputs = list_input_devices().unwrap_or_default();
        let available_outputs = list_output_devices().unwrap_or_default();
        let selected_input =
            preferred_device_from_env("VOICEPRINTMIC_INPUT_DEVICE", &available_inputs);
        let selected_output =
            preferred_device_from_env("VOICEPRINTMIC_OUTPUT_DEVICE", &available_outputs);
        let tse_onnx_path = std::env::var_os("MELLONELLA_TSE_PROD_48K_ONNX")
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .or_else(|| {
                mellonella_core::hf_fetch::cached_path(
                    mellonella_core::hf_fetch::TSE_PROD_48K_REPO,
                    mellonella_core::hf_fetch::TSE_PROD_48K_FILES[0],
                )
                .ok()
                .filter(|path| path.exists())
            });
        let gate_cfg = load_gate_config();
        let gate_tuning = Arc::new(GateTuning::new(gate_cfg));
        let mut state = Self {
            pool: None,
            separator_pool: None,
            origin: EnrollmentOrigin::None,
            pool_anchors: 0,
            pool_f0_mu: 0.0,
            pool_f0_sigma: 0.0,
            enrollment_quality: None,
            available_inputs,
            available_outputs,
            selected_input,
            selected_output,
            tse_onnx_path,
            record_duration_secs: DEFAULT_RECORD_SECS,
            gate_cfg,
            gate_tuning,
            pipeline_cfg: default_live_pipeline_cfg(),
            separator_tuning: Arc::new(SeparatorTuning::new(load_separator_threshold())),
            pending_append: false,
            session: None,
            recorder: None,
            last_error: None,
            last_stats: LiveSessionStats::default(),
        };
        // Auto-load the persistent enrollment if one exists. The user
        // shouldn't have to re-enrol on every launch; the first-run
        // wizard in `crate::app` keeps prompting only until this fires.
        if let Some(path) = default_enrollment_path() {
            if path.exists() {
                state.load_enrollment_json(&path);
            }
        }
        state
    }
}

/// Reject captures that would create a misleading biometric profile.
/// Thresholds are intentionally broad: they catch disconnected/quiet
/// inputs, long silence, DC faults and actual clipping without asking
/// the user to hit a studio-perfect level.
fn validate_enrollment_capture(audio: &[f32], sample_rate: u32) -> Result<(), String> {
    if audio.is_empty() || audio.iter().any(|sample| !sample.is_finite()) {
        return Err("マイク入力が空か壊れています".to_string());
    }

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f32;
    let mut clipped = 0_usize;
    for &sample in audio {
        sum += f64::from(sample);
        sum_sq += f64::from(sample) * f64::from(sample);
        peak = peak.max(sample.abs());
        if sample.abs() >= 0.995 {
            clipped += 1;
        }
    }
    let rms = (sum_sq / audio.len() as f64).sqrt() as f32;
    let rms_dbfs = 20.0 * rms.max(1.0e-9).log10();
    let dc = (sum / audio.len() as f64).abs() as f32;
    let clipped_ratio = clipped as f32 / audio.len() as f32;

    if rms_dbfs < -42.0 || peak < 0.025 {
        return Err(format!(
            "声が小さすぎます（平均 {rms_dbfs:.0} dBFS）。Windowsのマイク音量を上げ、15〜25cmの距離で話してください"
        ));
    }
    if clipped_ratio > 0.003 {
        return Err(format!(
            "音割れしています（クリップ {:.2}%）。Windowsのマイク音量を下げるか、マイクから少し離れてください",
            clipped_ratio * 100.0
        ));
    }
    if dc > 0.05 {
        return Err(
            "マイク信号に大きな直流ずれがあります。入力デバイスを選び直してください".to_string(),
        );
    }

    let frame_len = (sample_rate as usize / 50).max(1); // 20 ms
    let mut active = 0_usize;
    let mut frames = 0_usize;
    for frame in audio.chunks(frame_len) {
        if frame.len() < frame_len / 2 {
            continue;
        }
        frames += 1;
        let frame_rms =
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32).sqrt();
        if frame_rms >= 0.006 {
            active += 1;
        }
    }
    let active_ratio = active as f32 / frames.max(1) as f32;
    if active_ratio < 0.45 {
        return Err(format!(
            "発話区間が少なすぎます（{:.0}%）。20秒間、文章を止まらず繰り返してください",
            active_ratio * 100.0
        ));
    }
    Ok(())
}

/// Median pairwise cosine similarity is robust to one unusual vowel
/// while still exposing a recording contaminated by noise/other voices.
fn enrollment_consistency(anchors: &[Vec<f32>]) -> f32 {
    if anchors.len() < 2 {
        return 0.0;
    }
    let mut similarities = Vec::with_capacity(anchors.len() * (anchors.len() - 1) / 2);
    for (index, anchor) in anchors.iter().enumerate() {
        for other in &anchors[index + 1..] {
            similarities.push(cos_similarity(anchor, other));
        }
    }
    similarities.sort_by(f32::total_cmp);
    similarities[similarities.len() / 2]
}

/// Resolve a launcher's preferred audio device without hard-coding a
/// machine-specific full device name.  An exact match wins; otherwise the
/// value is treated as a case-insensitive substring (for example `HyperX`).
fn preferred_device_from_env(env_var: &str, devices: &[AudioDevice]) -> Option<String> {
    let requested = std::env::var(env_var).ok()?;
    devices
        .iter()
        .find(|device| device.name == requested)
        .or_else(|| {
            let requested = requested.to_lowercase();
            devices
                .iter()
                .find(|device| device.name.to_lowercase().contains(&requested))
        })
        .map(|device| device.name.clone())
}

/// Default on-disk location for the auto-saved enrollment:
/// `<dirs::config_dir>/mellonella/enrollment.json`. Returns `None` on
/// platforms where `dirs::config_dir()` is unavailable (rare).
fn mellonella_config_dir() -> Option<PathBuf> {
    #[cfg(not(test))]
    {
        Some(dirs::config_dir()?.join("mellonella"))
    }
    #[cfg(test)]
    {
        // Tests that exercise auto-loading must never rename or delete the
        // user's real biometric profile. A thread-local directory also keeps
        // concurrently-running tests from racing over one shared fixture.
        thread_local! {
            static TEST_CONFIG_DIR: PathBuf = std::env::temp_dir().join(format!(
                "mellonella-tests-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            )).join("mellonella");
        }
        TEST_CONFIG_DIR.with(Clone::clone).into()
    }
}

#[must_use]
pub fn default_enrollment_path() -> Option<PathBuf> {
    Some(mellonella_config_dir()?.join("enrollment.json"))
}

/// Companion profile used only to identify SepFormer's anonymous
/// output streams.
#[must_use]
pub fn default_separator_enrollment_path() -> Option<PathBuf> {
    Some(mellonella_config_dir()?.join("enrollment-separator.json"))
}

/// On-disk location of the persisted separator fail-closed threshold —
/// a single plain-text float next to the enrollment JSON.
#[must_use]
pub fn separator_threshold_path() -> Option<PathBuf> {
    Some(mellonella_config_dir()?.join("separator-threshold.txt"))
}

/// Persisted live gate controls. A small text format keeps this
/// backward-compatible and lets a damaged file fall back atomically to
/// safe defaults: `v2 <threshold> <hangover_ms> <release_ms>`.
///
/// `v2` identifies settings calibrated for max-over-reference identity
/// scoring. Older thresholds are accepted only after clamping them to the
/// current safe default; stricter user settings remain intact.
const GATE_SETTINGS_FORMAT: &str = "v2";

#[must_use]
pub fn gate_settings_path() -> Option<PathBuf> {
    Some(mellonella_config_dir()?.join("gate-settings.txt"))
}

/// Threshold file written by packaged builds before the live gate UI
/// started persisting all three controls together.
fn legacy_voiceprint_threshold_path() -> Option<PathBuf> {
    Some(mellonella_config_dir()?.join("voiceprint-threshold.txt"))
}

fn default_gate_config() -> GateConfig {
    GateConfig {
        theta_pass: DEFAULT_GATE_THRESHOLD,
        adaptive_theta: false,
        hangover_ms: DEFAULT_GATE_HANGOVER_MS,
        release_ms: DEFAULT_GATE_RELEASE_MS,
        ..GateConfig::default()
    }
}

fn load_gate_config() -> GateConfig {
    let mut config = default_gate_config();
    if let Some(text) = gate_settings_path().and_then(|path| std::fs::read_to_string(path).ok()) {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let versioned = tokens.first().copied() == Some(GATE_SETTINGS_FORMAT);
        // A future version must fail closed instead of being interpreted as
        // today's three-number layout. Unversioned files are the v1 layout.
        let unknown_version = tokens
            .first()
            .is_some_and(|token| token.starts_with('v') && !versioned);
        let value_tokens = if versioned { &tokens[1..] } else { &tokens[..] };
        let values: Vec<f32> = value_tokens
            .iter()
            .filter_map(|part| part.parse::<f32>().ok())
            .collect();
        if !unknown_version {
            if let [threshold, hangover_ms, release_ms, ..] = values.as_slice() {
                if (0.15..=0.85).contains(threshold)
                    && (100.0..=1_200.0).contains(hangover_ms)
                    && (30.0..=400.0).contains(release_ms)
                {
                    config.theta_pass = if versioned {
                        *threshold
                    } else {
                        threshold.max(DEFAULT_GATE_THRESHOLD)
                    };
                    config.hangover_ms = *hangover_ms;
                    config.release_ms = *release_ms;
                    return config;
                }
            }
        }
    }

    // The oldest builds stored only the threshold. Max-over-reference
    // scoring raises genuine and impostor scores relative to the old
    // centroid-only scale, so never migrate a value below today's safe
    // default. A stricter personal value remains useful and is preserved.
    if let Some(threshold) = legacy_voiceprint_threshold_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| text.trim().parse::<f32>().ok())
        .filter(|value| (0.15..=0.85).contains(value))
    {
        config.theta_pass = threshold.max(DEFAULT_GATE_THRESHOLD);
    }
    config
}

/// Load the persisted separator threshold, falling back to the library
/// default when the file is missing, unreadable, or out of range.
fn load_separator_threshold() -> f32 {
    separator_threshold_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| text.trim().parse::<f32>().ok())
        .filter(|value| (0.05..=0.95).contains(value))
        .unwrap_or(SeparatorTuning::DEFAULT_THRESHOLD)
}

/// Live-tuned `PipelineConfig` for GUI sessions: takes the library
/// defaults and overrides the fields whose library defaults are
/// backward-compat no-ops (so existing tests against
/// `PipelineConfig::default()` keep passing) but that matter for
/// real-time mic use:
///
/// * `silence_force_off_ms = 700` — ignore ordinary inter-word pauses
///   while still preventing an old score from holding the gate open
///   indefinitely in an empty room.
/// * `sv_window_samples = 16_000`, `sv_update_samples = 4_000` — use a
///   one-second identity window refreshed every 250 ms. The longer window
///   restores ECAPA score margin; the refresh cadence, not the window
///   length, controls how quickly a changed speaker is noticed.
/// * `score_ema_alpha = 0.9` — react quickly to the newer speaker
///   embedding; gate hangover handles isolated target-score dips.
/// * `async_refresh = true` — ECAPA/Fbank runs off the audio worker so
///   its periodic inference cannot starve the output callback and turn
///   otherwise clean speech into regular gaps.
/// * `sv_min_new_samples_after_silence = 1600` — fire the
///   post-silence early refresh after only 100 ms of new speech
///   (instead of the library-default 250 ms) so `last_score` catches
///   up to the current speaker faster on resume. Cheap for a
///   single-target system since there's no cross-speaker risk.
#[must_use]
pub fn default_live_pipeline_cfg() -> PipelineConfig {
    PipelineConfig {
        // Do not close inside a natural sentence pause. Score-side
        // hangover still rejects a changed speaker; this rule exists
        // only to prevent a stale gate from remaining open in silence.
        silence_force_off_ms: 700.0,
        // One second is the shortest measured window that keeps the
        // enrolled speaker comfortably separated from an impostor.
        sv_window_samples: 16_000,
        sv_update_samples: 4_000,
        score_ema_alpha: 0.90,
        async_refresh: true,
        sv_min_new_samples_after_silence: 1_600,
        // Never adapt a biometric reference from live Discord audio:
        // a sustained friend could otherwise drift the target profile.
        enable_auto_learn: false,
        ..PipelineConfig::default()
    }
}

/// `Some(path)` when the DFN3 ONNX is reachable — either via the
/// `MELLONELLA_DFN3_ONNX` env var or the on-disk cache populated by
/// [`mellonella_core::hf_fetch::ensure_dfn3_onnx`]. Used by the UI's
/// status row and by [`AppState::start`] to decide whether to wire
/// DFN3 into the live engine. Cheap (no network) — the actual fetch
/// happens elsewhere; this is just a "is the file there yet?" probe.
#[must_use]
pub fn dfn3_path_from_env() -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("MELLONELLA_DFN3_ONNX") {
        let p = PathBuf::from(raw);
        if p.exists() {
            return Some(p);
        }
    }
    let cached = mellonella_core::hf_fetch::cached_path(
        mellonella_core::hf_fetch::DFN3_REPO,
        mellonella_core::hf_fetch::DFN3_FILE,
    )
    .ok()?;
    cached.exists().then_some(cached)
}

/// Resolve the optional strong two-speaker separator bundled by the
/// VoiceprintMic launcher. This is intentionally cache-free: the
/// community model is distributed with the portable package and is
/// activated only when the launcher provides its exact path.
#[must_use]
pub fn sepformer_path_from_env() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("MELLONELLA_SEPFORMER_ONNX")?);
    path.exists().then_some(path)
}

impl AppState {
    /// Re-enumerate input/output devices from cpal. Preserves the
    /// existing selection if the named device is still present.
    pub fn refresh_devices(&mut self) {
        self.available_inputs = list_input_devices().unwrap_or_default();
        self.available_outputs = list_output_devices().unwrap_or_default();
        if let Some(name) = &self.selected_input {
            if !self.available_inputs.iter().any(|d| &d.name == name) {
                self.selected_input = None;
            }
        }
        if let Some(name) = &self.selected_output {
            if !self.available_outputs.iter().any(|d| &d.name == name) {
                self.selected_output = None;
            }
        }
    }

    /// Build a fresh `PipelineComponents`, resolving ONNX paths through
    /// the cache-first fallback chain in [`mellonella_core::hf_fetch`]
    /// (env var → cache → HuggingFace fetch). Falls back gracefully to
    /// the legacy env-var-only setup for models without an HF mirror.
    fn build_components() -> Result<PipelineComponents, String> {
        let ecapa_path = mellonella_core::hf_fetch::ensure_ecapa_onnx(|_, _| {})
            .map_err(|e| format!("ECAPA ONNX: {e}"))?;
        let vad_path = mellonella_core::hf_fetch::ensure_vad_onnx(|_, _| {})
            .map_err(|e| format!("VAD ONNX: {e}"))?;
        let fbank = Fbank::with_speechbrain_filterbank().map_err(|e| format!("Fbank init: {e}"))?;
        let ecapa =
            EcapaTdnn::from_onnx_path(&ecapa_path).map_err(|e| format!("ECAPA load: {e}"))?;
        let vad = SileroVad::from_onnx_path(&vad_path, DECISION_SAMPLE_RATE)
            .map_err(|e| format!("VAD load: {e}"))?;
        Ok(PipelineComponents {
            vad,
            fbank,
            ecapa,
            cohort: Vec::new(),
            tse: None,
        })
    }

    fn store_pool(&mut self, pool: EmbeddingPool, origin: EnrollmentOrigin) {
        let m = pool.metadata();
        self.pool_anchors = pool.anchors().len();
        self.pool_f0_mu = m.f0_mu;
        self.pool_f0_sigma = m.f0_sigma;
        let consistency = enrollment_consistency(pool.anchors());
        let grade = if consistency >= 0.60 {
            "良好"
        } else if consistency >= MIN_ENROLLMENT_CONSISTENCY {
            "使用可能"
        } else {
            "要再登録"
        };
        self.enrollment_quality = Some(format!("登録品質: {grade}（声紋一貫度 {consistency:.2}）"));
        self.origin = origin;
        self.pool = Some(pool);
        self.last_error = None;
    }

    fn clear_pool(&mut self) {
        self.pool = None;
        self.separator_pool = None;
        self.origin = EnrollmentOrigin::None;
        self.pool_anchors = 0;
        self.pool_f0_mu = 0.0;
        self.pool_f0_sigma = 0.0;
        self.enrollment_quality = None;
    }

    /// Auto-load the persistent enrollment from
    /// `default_enrollment_path()` on launch. The enrollment pool is
    /// otherwise managed entirely in-memory: mic recording builds it,
    /// auto-learn updates it during a live session, and
    /// [`Self::persist_enrollment_to_default_path`] writes it back.
    fn load_enrollment_json(&mut self, path: &Path) {
        match EmbeddingPool::load(path, EmbeddingPoolConfig::default()) {
            Ok(pool) => {
                self.store_pool(pool, EnrollmentOrigin::AutoLoaded(path.to_path_buf()));
                self.separator_pool = default_separator_enrollment_path()
                    .filter(|companion| companion.exists())
                    .and_then(|companion| {
                        EmbeddingPool::load(companion, EmbeddingPoolConfig::default()).ok()
                    });
            }
            Err(e) => {
                self.clear_pool();
                self.last_error = Some(format!("load enrollment: {e}"));
            }
        }
    }

    /// Kick off a mic recording of `secs` seconds at 48 kHz mono —
    /// matches the live audio path's rate so an optional DFN3
    /// pre-pass during enrollment runs on the same distribution as
    /// the live ECAPA refresh path. Call `poll_recorder` once per
    /// frame to detect completion.
    /// `append == true` keeps the saved enrollment and adds the new
    /// recording's anchors to it (extra voice registration); `false`
    /// replaces the profile as before.
    pub fn start_recording(&mut self, secs: f32, append: bool) {
        if self.recorder.is_some() {
            return;
        }
        self.pending_append = append && self.pool.is_some();
        match Recorder::start(self.selected_input.clone(), OUTPUT_SAMPLE_RATE, secs) {
            Ok(r) => {
                self.recorder = Some(r);
                self.last_error = None;
            }
            Err(e) => self.last_error = Some(format!("start recording: {e}")),
        }
    }

    /// Cancel an in-flight recording. The worker still returns
    /// whatever it has collected, which the next `poll_recorder`
    /// converts into an enrollment.
    pub fn cancel_recording(&mut self) {
        if let Some(r) = self.recorder.as_ref() {
            r.cancel();
        }
    }

    /// Poll the active recorder for completion. On success runs
    /// enrolment on the captured buffer and stores the resulting
    /// pool. Returns silently when no recorder is active or it's
    /// still capturing.
    pub fn poll_recorder(&mut self) {
        let Some(recorder) = self.recorder.as_mut() else {
            return;
        };
        let Some(result) = recorder.try_finish() else {
            return;
        };
        // Recorder finished; pull it out and consume the result.
        let target_secs = recorder.target_seconds().round() as u32;
        self.recorder = None;
        match result {
            Ok(audio) => {
                if audio.len() < OUTPUT_SAMPLE_RATE as usize {
                    self.last_error =
                        Some("recording too short for ECAPA enrolment (need ≥ 1 s)".to_string());
                    return;
                }
                self.run_enrollment(&audio, EnrollmentOrigin::Mic { secs: target_secs });
            }
            Err(e) => self.last_error = Some(format!("recording failed: {e}")),
        }
    }

    /// Enroll from 48 kHz mono audio. Downsamples to 16 kHz and runs
    /// ECAPA on the raw signal, matching the live engine's decision
    /// path: after the Phase 5 refactor, DFN3 lives post-TSE in the
    /// audio chain, while VAD / ECAPA / F0 see the raw mic input.
    /// Keeping enrollment on raw audio matches that distribution so
    /// anchor embeddings live in the same space as runtime refreshes.
    fn run_enrollment(&mut self, audio_48k: &[f32], origin: EnrollmentOrigin) {
        // Consume the append flag up front so an early error return
        // can't leak it into an unrelated later recording.
        let appending = std::mem::take(&mut self.pending_append);
        if let Err(message) = validate_enrollment_capture(audio_48k, OUTPUT_SAMPLE_RATE) {
            self.last_error = Some(format!(
                "声紋登録を中止しました: {message}。保存済みの声紋は変更していません。"
            ));
            return;
        }
        let audio_16k = match resample_to(audio_48k, OUTPUT_SAMPLE_RATE, DECISION_SAMPLE_RATE) {
            Ok(a) => a,
            Err(e) => {
                self.last_error = Some(format!("resample 48 kHz → 16 kHz for ECAPA: {e}"));
                return;
            }
        };
        let mut components = match Self::build_components() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        match enroll_from_recording(&audio_16k, &mut components, EmbeddingPoolConfig::default()) {
            Ok(pool) => {
                let anchors = pool.anchors().len();
                if anchors < MIN_ENROLLMENT_ANCHORS {
                    self.last_error = Some(format!(
                        "声の特徴点が{anchors}個しか取れませんでした（{MIN_ENROLLMENT_ANCHORS}個以上必要）。\
                         マイクに近づき、20秒間止まらず読み上げてください。保存済みの声紋は変更していません。"
                    ));
                    return;
                }
                let consistency = enrollment_consistency(pool.anchors());
                if consistency < MIN_ENROLLMENT_CONSISTENCY {
                    self.last_error = Some(format!(
                        "声の特徴が安定していません（一貫度 {consistency:.2}、必要値 {MIN_ENROLLMENT_CONSISTENCY:.2}）。\
                         スピーカー音を止め、マイクから15〜25cm離れて同じ声量で読み直してください。\
                         保存済みの声紋は変更していません。"
                    ));
                    return;
                }
                let separator_pool = if let Some(sepformer) = sepformer_path_from_env() {
                    let ecapa_path = match mellonella_core::hf_fetch::ensure_ecapa_onnx(|_, _| {}) {
                        Ok(path) => path,
                        Err(error) => {
                            self.last_error =
                                Some(format!("分離後の声紋登録用ECAPAを読み込めません: {error}"));
                            return;
                        }
                    };
                    match build_separator_enrollment_pool(&pool, sepformer, ecapa_path, audio_48k) {
                        Ok(separator_pool) => Some(separator_pool),
                        Err(error) => {
                            self.last_error = Some(format!(
                                "分離後の本人声紋を登録できませんでした: {error}。\
                                 保存済みの声紋は変更していません。"
                            ));
                            return;
                        }
                    }
                } else {
                    None
                };
                if appending {
                    self.append_enrollment(pool, separator_pool, origin);
                } else {
                    self.separator_pool = separator_pool;
                    self.store_pool(pool, origin);
                }
                self.persist_enrollment_to_default_path();
            }
            Err(e) => self.last_error = Some(format!("enrol: {e}")),
        }
    }

    /// Fold a freshly recorded enrollment into the existing profile
    /// instead of replacing it. Mic anchors and separator-domain
    /// anchors are appended to their respective pools, and the F0
    /// Gaussian is widened to the anchor-count-weighted combination of
    /// both recordings so morning/evening pitch shifts stay inside the
    /// distribution.
    fn append_enrollment(
        &mut self,
        new_pool: EmbeddingPool,
        new_separator_pool: Option<EmbeddingPool>,
        origin: EnrollmentOrigin,
    ) {
        let Some(mut merged) = self.pool.clone() else {
            // No saved profile after all — behave like a replace.
            self.separator_pool = new_separator_pool;
            self.store_pool(new_pool, origin);
            return;
        };
        let old_anchors: Vec<Vec<f32>> = merged.anchors().to_vec();
        let old_meta = merged.metadata();
        let new_meta = new_pool.metadata();
        #[allow(clippy::cast_precision_loss)]
        let (n_old, n_new) = (old_anchors.len() as f32, new_pool.anchors().len() as f32);
        let total = n_old + n_new;
        let mu = (n_old * old_meta.f0_mu + n_new * new_meta.f0_mu) / total;
        let ex2_old = old_meta.f0_sigma * old_meta.f0_sigma + old_meta.f0_mu * old_meta.f0_mu;
        let ex2_new = new_meta.f0_sigma * new_meta.f0_sigma + new_meta.f0_mu * new_meta.f0_mu;
        let variance = ((n_old * ex2_old + n_new * ex2_new) / total - mu * mu).max(0.0);
        merged.add_anchors(new_pool.anchors().iter().cloned());
        merged.set_f0_stats(mu, variance.sqrt());

        self.separator_pool = match (self.separator_pool.take(), new_separator_pool) {
            (Some(mut old_sep), Some(new_sep)) => {
                old_sep.add_anchors(new_sep.anchors().iter().cloned());
                Some(old_sep)
            }
            (Some(old_sep), None) => Some(old_sep),
            (None, Some(mut new_sep)) => {
                // The saved profile predates separator calibration —
                // fold its raw-mic anchors in so they keep
                // participating in stream selection.
                new_sep.add_anchors(old_anchors.iter().cloned());
                Some(new_sep)
            }
            (None, None) => None,
        };
        self.store_pool(merged, origin);
    }

    /// Save the current enrollment pool to `default_enrollment_path()`
    /// so the next launch auto-loads it. No-op when no pool is loaded
    /// or no platform config dir is configured.
    fn persist_enrollment_to_default_path(&mut self) {
        let Some(pool) = self.pool.as_ref() else {
            return;
        };
        let Some(path) = default_enrollment_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.last_error = Some(format!("create config dir: {e}"));
                return;
            }
        }
        if let Err(e) = pool.save(&path) {
            self.last_error = Some(format!("auto-save enrollment: {e}"));
            return;
        }
        if let (Some(separator_pool), Some(separator_path)) = (
            self.separator_pool.as_ref(),
            default_separator_enrollment_path(),
        ) {
            if let Err(error) = separator_pool.save(separator_path) {
                self.last_error = Some(format!("分離後の声紋を保存できません: {error}"));
            }
        }
    }

    /// Spin up a `LiveSession` using the current in-memory pool.
    ///
    /// DFN3 and TSE are always enabled when their ONNX models are
    /// reachable (DFN3 via `MELLONELLA_DFN3_ONNX`, TSE via
    /// [`Self::tse_onnx_path`]). The GUI no longer exposes toggles
    /// for them — the engine handles dual-stage absence gracefully
    /// when an ONNX is unavailable.
    pub fn start(&mut self) {
        if self.session.is_some() {
            return;
        }
        let Some(pool) = self.pool.clone() else {
            self.last_error = Some("enrol a voice first".into());
            return;
        };
        let components = match Self::build_components() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };
        let dfn3_onnx_path = dfn3_path_from_env();
        let sepformer_onnx_path = sepformer_path_from_env();
        if sepformer_onnx_path.is_some() && self.separator_pool.is_none() {
            self.last_error = Some(
                "強力2話者分離を使うには、新しい20秒方式で声紋を再登録してください。".to_string(),
            );
            return;
        }
        let speaker_embedding_onnx_path = if sepformer_onnx_path.is_some() {
            match mellonella_core::hf_fetch::ensure_ecapa_onnx(|_, _| {}) {
                Ok(path) => Some(path),
                Err(error) => {
                    self.last_error =
                        Some(format!("2話者分離用のECAPA ONNXを読み込めません: {error}"));
                    return;
                }
            }
        } else {
            None
        };
        // Auto-resolve the pyannote-3.0 segmentation ONNX so the
        // streaming engine picks the adaptive (Solo→DFN3 /
        // Overlap→TSE) routing path. Without this, the engine
        // falls back to the legacy TSE → DFN3 cascade which both
        // (a) attenuates voice by ~28 dB and (b) runs both stages
        // simultaneously per chunk, ~doubling the CPU cost and
        // producing the steady ~15 underruns/s we saw in the
        // user's first-run logs.
        let mut overlap_onnx_path =
            mellonella_core::hf_fetch::ensure_overlap_seg_onnx(|_, _| {}).ok();
        let mut pipeline_cfg = self.pipeline_cfg.clone();
        // Auto-resolve TSE ONNX if the user hasn't picked one — the
        // fetcher's cache hit path makes this cheap on every launch
        // after the first.
        if sepformer_onnx_path.is_none() && self.tse_onnx_path.is_none() {
            if let Ok(p) = mellonella_core::hf_fetch::ensure_tse_prod_48k_onnx(|_, _, _| {}) {
                self.tse_onnx_path = Some(p);
            }
        }
        if sepformer_onnx_path.is_some() {
            // SepFormer already separates both speakers before the gate.
            // Disable the weaker English-trained TSE/overlap router to
            // avoid double separation, excess attenuation and CPU cost.
            pipeline_cfg.tse = None;
            overlap_onnx_path = None;
        } else if let Some(onnx) = self.tse_onnx_path.as_ref() {
            if !onnx.exists() {
                self.last_error = Some(format!("TSE ONNX path does not exist: {}", onnx.display()));
                return;
            }
            pipeline_cfg.tse = Some(TseStageConfig::new_prod_48k(onnx.clone()));
        }
        let gate_tuning = sepformer_onnx_path
            .is_none()
            .then(|| self.gate_tuning.clone());
        let cfg = SessionConfig {
            input_device: self.selected_input.clone(),
            output_device: self.selected_output.clone(),
            streaming: StreamingConfig {
                pipeline: pipeline_cfg,
                // SepFormer already fails closed using its calibrated
                // companion voiceprint. Let the ordinary stage act as
                // VAD/envelope/noise-removal instead of rejecting the
                // separator-domain voice a second time.
                gate: if sepformer_onnx_path.is_some() {
                    GateConfig {
                        theta_pass: 0.0,
                        adaptive_theta: false,
                        ..self.gate_cfg
                    }
                } else {
                    self.gate_cfg
                },
                audio_sample_rate: OUTPUT_SAMPLE_RATE,
                diagnostics: false,
                // `LiveSession::new` overwrites this from `dfn3_onnx_path`
                // on `SessionConfig` below — keep `None` here as the
                // construction-time default.
                dfn3_onnx_path: None,
                // During overlap, TSE is already the target-speaker
                // classifier. Do not let a louder raw mixture mute the
                // correctly extracted enrolled speaker, and never mix
                // the raw friend's voice back into the result.
                overlap_bypass_speaker_gate: true,
                overlap_wet_dry_alpha: 1.0,
                // Require three consecutive detector decisions. One
                // 250 ms decision made normal conversation flap between
                // cold DFN3 and TSE states and repeatedly cut the tail.
                overlap_hold_on_ms: 750.0,
                overlap_hold_off_ms: 1_250.0,
                ..Default::default()
            },
            dfn3_onnx_path,
            overlap_onnx_path,
            sepformer_onnx_path,
            speaker_embedding_onnx_path,
            speaker_selection_pool: self.separator_pool.clone(),
            separator_tuning: Some(self.separator_tuning.clone()),
            gate_tuning,
            // GUI uses the safe default; multi-channel mic users
            // who want a specific channel use the CLI's
            // `mellonella live --input-channel N` for now. A GUI
            // dropdown is a small follow-up.
            input_channel: mellonella_audio_io::ChannelStrategy::default(),
        };
        match LiveSession::new(pool, components, cfg) {
            Ok(s) => {
                self.session = Some(s);
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("open live session: {e}"));
            }
        }
    }

    /// Tear down the live session and capture its final stats.
    pub fn stop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        match session.stop() {
            Ok(stats) => {
                self.last_stats = stats;
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("stop session: {e}"));
            }
        }
    }

    /// Persist the current separator threshold so the slider value
    /// survives restarts. Failures are silently ignored — a missing
    /// file just means the default is used next launch.
    pub fn save_separator_threshold(&self) {
        let Some(path) = separator_threshold_path() else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, format!("{:.3}\n", self.separator_tuning.threshold()));
    }

    /// Persist the live gate controls. The in-memory values have
    /// already reached the worker through atomics; this write only makes
    /// them survive the next launch.
    pub fn save_gate_settings(&mut self) {
        self.gate_cfg.theta_pass = self.gate_tuning.threshold();
        self.gate_cfg.hangover_ms = self.gate_tuning.hangover_ms();
        self.gate_cfg.release_ms = self.gate_tuning.release_ms();
        let Some(path) = gate_settings_path() else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(
            path,
            format!(
                "{GATE_SETTINGS_FORMAT} {:.3} {:.0} {:.0}\n",
                self.gate_cfg.theta_pass, self.gate_cfg.hangover_ms, self.gate_cfg.release_ms,
            ),
        );
    }

    /// Apply one coherent live preset, then persist it.
    pub fn set_gate_preset(&mut self, threshold: f32, hangover_ms: f32, release_ms: f32) {
        self.gate_tuning.set_threshold(threshold);
        self.gate_tuning.set_hangover_ms(hangover_ms);
        self.gate_tuning.set_release_ms(release_ms);
        self.save_gate_settings();
    }

    /// Poll the live session for stats + events. Call once per UI
    /// frame so the displayed counters stay fresh and worker-side
    /// errors propagate into `last_error`.
    pub fn poll_session(&mut self) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        self.last_stats = session.stats_snapshot();
        if let Some(SessionEvent::Error(msg)) = session.try_recv_event() {
            self.last_error = Some(format!("pipeline error: {msg}"));
            self.stop();
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.session.is_some()
    }

    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Whether DFN3 noise suppression is reachable in this process
    /// (env var set, file exists). Used by the status row to indicate
    /// "NS active" and by [`Self::estimated_latency_ms`] to factor in
    /// the DFN3 lookahead.
    #[must_use]
    #[allow(clippy::unused_self)] // method form is more discoverable from app.rs
    pub fn dfn3_available(&self) -> bool {
        dfn3_path_from_env().is_some()
    }

    /// Whether a TSE ONNX path has been configured and still exists
    /// on disk. Used by the status row.
    #[must_use]
    pub fn tse_available(&self) -> bool {
        self.tse_onnx_path.as_deref().is_some_and(Path::exists)
    }

    /// Whether the strong language-independent two-speaker separation
    /// model is active for the next live session.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn sepformer_available(&self) -> bool {
        sepformer_path_from_env().is_some()
    }

    /// Download the canonical Stage C TSE prod_48k model from
    /// HuggingFace into the local cache (synchronous; the UI freezes
    /// briefly during the ~10 MB download) and update
    /// [`Self::tse_onnx_path`] to point at the cached file. Reuses an
    /// already-cached copy on subsequent calls.
    ///
    /// Surfaces failures via [`Self::last_error`] rather than
    /// `Result` so the egui call site stays click-handler-shaped.
    pub fn fetch_tse_from_hf(&mut self) {
        match mellonella_core::hf_fetch::fetch_tse_prod_48k(|_, _, _| {}) {
            Ok(path) => {
                self.tse_onnx_path = Some(path);
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(format!("HuggingFace fetch failed: {e}"));
            }
        }
    }

    /// Latest input RMS from the worker (0.0 when no session is
    /// running). Used by the GUI's level meter.
    #[must_use]
    pub fn input_rms(&self) -> f32 {
        self.session.as_ref().map_or(0.0, LiveSession::input_rms)
    }

    /// Latest output (gate × envelope) RMS from the worker.
    #[must_use]
    pub fn output_rms(&self) -> f32 {
        self.session.as_ref().map_or(0.0, LiveSession::output_rms)
    }

    /// Latest gate state — `true` when audio is currently being
    /// passed through. `false` for both "gated off" and "no session".
    #[must_use]
    pub fn gate_on(&self) -> bool {
        self.session.as_ref().is_some_and(LiveSession::gate_on)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_idle() {
        let s = AppState::default();
        // A real saved enrollment is intentionally auto-loaded, so
        // "idle" means no active capture/session rather than no profile.
        assert!(!s.is_running());
        assert!(!s.is_recording());
        assert!(s.last_error.is_none());
    }

    #[test]
    fn loading_a_missing_enrollment_records_an_error() {
        let mut s = AppState::default();
        s.load_enrollment_json(std::path::Path::new(
            "/no/such/enrollment-mellonella-gui-test.json",
        ));
        assert!(s.last_error.is_some(), "expected an error to be recorded");
        assert!(s.pool.is_none());
        assert!(matches!(s.origin, EnrollmentOrigin::None));
        assert!(s.pool.is_none());
    }

    #[test]
    fn refresh_devices_clears_invalid_selection() {
        let mut s = AppState {
            selected_input: Some("__not-a-real-device__".into()),
            ..AppState::default()
        };
        s.refresh_devices();
        assert!(s.selected_input.is_none());
    }

    #[test]
    fn sample_rates_match_streaming_and_audio_io_constants() {
        assert_eq!(
            OUTPUT_SAMPLE_RATE,
            mellonella_audio_io::INTERNAL_SAMPLE_RATE,
            "OUTPUT_SAMPLE_RATE must match mellonella-audio-io's INTERNAL_SAMPLE_RATE",
        );
    }

    #[test]
    fn live_identity_window_is_at_least_one_second() {
        let config = default_live_pipeline_cfg();
        assert!(config.sv_window_samples >= DECISION_SAMPLE_RATE as usize);
        assert!(
            config.async_refresh,
            "live ECAPA must stay off the audio thread"
        );
        assert!(
            !config.enable_auto_learn,
            "never adapt from live call audio"
        );
    }

    #[test]
    fn default_enrollment_path_lives_under_config_dir() {
        let Some(p) = default_enrollment_path() else {
            eprintln!("[skip] no config dir on this platform");
            return;
        };
        assert!(
            p.ends_with("mellonella/enrollment.json") || p.ends_with("mellonella\\enrollment.json"),
            "unexpected suffix: {}",
            p.display()
        );
    }

    // ----------------------------------------------------------------
    // ONNX-backed end-to-end checks. Gated on the same env vars as the
    // mellonella-core integration tests (`MELLONELLA_ECAPA_ONNX`,
    // `MELLONELLA_VAD_ONNX`, `ORT_DYLIB_PATH`) so a contributor without
    // the model artefacts still gets a green `cargo test`. The
    // persistence helpers below use the thread-local test config root;
    // they never open or rename a real biometric profile.
    // ----------------------------------------------------------------

    fn skip_if_no_onnx() -> Option<(String, String)> {
        let Ok(ecapa) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
            eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
            return None;
        };
        let Ok(vad) = std::env::var("MELLONELLA_VAD_ONNX") else {
            eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
            return None;
        };
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("[skip] ORT_DYLIB_PATH not set");
            return None;
        }
        Some((ecapa, vad))
    }

    fn enroll_pool_from_fixture(ecapa: &str, vad: &str) -> EmbeddingPool {
        use mellonella_core::embedding::EcapaTdnn;
        use mellonella_core::features::Fbank;
        use mellonella_core::vad::SileroVad;
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("mellonella-core")
            .join("tests")
            .join("fixtures")
            .join("pipeline_input.bin");
        let bytes = std::fs::read(&fixture).expect("read pipeline_input.bin");
        assert!(bytes.len().is_multiple_of(4));
        let audio_16k: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut components = PipelineComponents {
            vad: SileroVad::from_onnx_path(vad, 16_000).expect("load VAD"),
            fbank: Fbank::with_speechbrain_filterbank().expect("Fbank from speechbrain filterbank"),
            ecapa: EcapaTdnn::from_onnx_path(ecapa).expect("load ECAPA"),
            cohort: Vec::new(),
            tse: None,
        };
        enroll_from_recording(&audio_16k, &mut components, EmbeddingPoolConfig::default())
            .expect("enroll_from_recording")
    }

    /// Swap the on-disk `default_enrollment_path()` with `pool` for
    /// the duration of `body`, restoring whatever was there.
    fn with_test_enrollment(pool: &EmbeddingPool, body: impl FnOnce()) {
        let path = default_enrollment_path().expect("config dir available");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create config dir");
        }
        let backup = path.with_extension("json.testbackup");
        let had_backup = path.exists();
        if had_backup {
            std::fs::rename(&path, &backup).expect("backup existing enrollment");
        }
        pool.save(&path).expect("save test pool");
        body();
        let _ = std::fs::remove_file(&path);
        if had_backup {
            std::fs::rename(&backup, &path).expect("restore existing enrollment");
        }
    }

    #[test]
    fn default_app_state_auto_loads_persisted_enrollment() {
        let Some((ecapa, vad)) = skip_if_no_onnx() else {
            return;
        };
        let pool = enroll_pool_from_fixture(&ecapa, &vad);
        let expected_dim = pool.anchor_centroid().expect("anchor dim").len();
        with_test_enrollment(&pool, || {
            let state = AppState::default();
            assert!(state.pool.is_some(), "should auto-load enrollment.json");
            assert!(
                matches!(state.origin, EnrollmentOrigin::AutoLoaded(_)),
                "expected AutoLoaded origin, got {:?}",
                state.origin
            );
            assert!(state.pool_anchors >= 1);
            let loaded_dim = state
                .pool
                .as_ref()
                .unwrap()
                .anchor_centroid()
                .unwrap()
                .len();
            assert_eq!(loaded_dim, expected_dim);
        });
    }

    /// Drive the GUI's exact pipeline configuration end-to-end on an
    /// offline buffer, no cpal involved. This is the headless analog
    /// of the user pressing Live monitor — it exercises the same TSE
    /// → DFN3 → gate → envelope chain that the worker runs.
    ///
    /// Gated on `MELLONELLA_TSE_PROD_48K_ONNX` and
    /// `MELLONELLA_DFN3_ONNX` *in addition* to the ECAPA / VAD pair —
    /// the GUI auto-enables both stages when the models are present,
    /// so the test follows the same rule.
    ///
    /// When `MELLONELLA_DUMP_OFFLINE_WAV` is set, the input (resampled)
    /// and chain output are also written to `/tmp/mellonella_offline_*.wav`
    /// so a developer reproducing a "weird audio output" complaint
    /// can listen to what the pipeline actually produced.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn streaming_pipeline_runs_clean_on_offline_speech() {
        use mellonella_core::resample::resample_to;
        use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
        use mellonella_core::tse_stage::TseStageConfig;

        let Some((ecapa, vad)) = skip_if_no_onnx() else {
            return;
        };
        let Ok(tse_path) = std::env::var("MELLONELLA_TSE_PROD_48K_ONNX") else {
            eprintln!("[skip] MELLONELLA_TSE_PROD_48K_ONNX not set");
            return;
        };
        let Ok(dfn3_path) = std::env::var("MELLONELLA_DFN3_ONNX") else {
            eprintln!("[skip] MELLONELLA_DFN3_ONNX not set");
            return;
        };

        // 1) Enroll a pool on the canonical 16 kHz fixture and grab
        //    the pipeline components.
        let pool = enroll_pool_from_fixture(&ecapa, &vad);
        let components = PipelineComponents {
            vad: mellonella_core::vad::SileroVad::from_onnx_path(&vad, 16_000).unwrap(),
            fbank: mellonella_core::features::Fbank::with_speechbrain_filterbank().unwrap(),
            ecapa: mellonella_core::embedding::EcapaTdnn::from_onnx_path(&ecapa).unwrap(),
            cohort: Vec::new(),
            tse: None,
        };

        // 2) Load the audio path. By default uses the 2 s 16 kHz
        //    `pipeline_input.bin` fixture and upsamples to 48 kHz.
        //    Set `MELLONELLA_OFFLINE_INPUT_WAV=<path.wav>` to point
        //    the test at a longer, real-world recording for
        //    repro-grade ear-checks; the test reads it as 16-bit
        //    signed mono and resamples to 48 kHz internally.
        let audio_48k: Vec<f32> = if let Ok(p) = std::env::var("MELLONELLA_OFFLINE_INPUT_WAV") {
            let (samples_native, native_sr) = read_pcm16_mono_wav(&p);
            if native_sr == OUTPUT_SAMPLE_RATE {
                samples_native
            } else {
                resample_to(&samples_native, native_sr, OUTPUT_SAMPLE_RATE)
                    .expect("input WAV → 48 kHz resample")
            }
        } else {
            let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("mellonella-core")
                .join("tests")
                .join("fixtures")
                .join("pipeline_input.bin");
            let bytes = std::fs::read(&fixture).unwrap();
            let audio_16k: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            resample_to(&audio_16k, 16_000, OUTPUT_SAMPLE_RATE)
                .expect("16k → 48k resample for offline streaming run")
        };

        // 3) Build a StreamingConfig that mirrors the GUI's
        //    `AppState::start()` — same gate, identity cadence, TSE
        //    Prod48k, and DFN3. The deterministic parity fixture is a
        //    synthetic ECAPA signal rather than ordinary speech, so it
        //    only produces three Silero-positive frames. Force VAD on
        //    here to exercise the live identity/async/gate path. VAD
        //    model behavior itself has a separate parity test.
        let mut pipeline_cfg = default_live_pipeline_cfg();
        pipeline_cfg.vad_threshold = -1.0;
        pipeline_cfg.tse = Some(TseStageConfig::new_prod_48k(PathBuf::from(&tse_path)));
        let cfg = StreamingConfig {
            pipeline: pipeline_cfg,
            gate: GateConfig::default(),
            audio_sample_rate: OUTPUT_SAMPLE_RATE,
            diagnostics: false,
            dfn3_onnx_path: Some(PathBuf::from(&dfn3_path)),
            ..Default::default()
        };

        let mut pipeline = StreamingPipeline::new(pool, cfg, components)
            .expect("StreamingPipeline accepts the GUI config");

        // 4) Push the audio in 480-sample (10 ms) chunks, mirroring
        //    the worker's actual cadence. Track diagnostics that
        //    the user can use to decide whether the chain is doing
        //    something pathological on real speech.
        let chunk_size: usize = 480;
        let mut all_output: Vec<f32> = Vec::with_capacity(audio_48k.len() + chunk_size);
        let mut nan_count = 0_usize;
        let mut inf_count = 0_usize;
        let mut gate_transitions = 0_usize;
        let mut last_gate_state: Option<bool> = None;
        let mut gate_on_samples = 0_u64;
        let mut chunks_pushed = 0_u64;
        let mut zero_output_chunks = 0_u64;
        let mut max_live_score = 0.0_f32;
        let t0 = std::time::Instant::now();
        for chunk in audio_48k.chunks(chunk_size) {
            let out = pipeline
                .push_samples(chunk)
                .expect("push_samples in offline streaming run");
            chunks_pushed += 1;
            if out.audio.is_empty() {
                zero_output_chunks += 1;
            }
            for &s in &out.audio {
                if s.is_nan() {
                    nan_count += 1;
                } else if !s.is_finite() {
                    inf_count += 1;
                }
            }
            for &(_, is_on) in &out.gate_decisions {
                if Some(is_on) != last_gate_state {
                    gate_transitions += 1;
                    last_gate_state = Some(is_on);
                }
            }
            if let Some(true) = last_gate_state {
                gate_on_samples += out.audio.len() as u64;
            }
            all_output.extend_from_slice(&out.audio);
            max_live_score = max_live_score.max(pipeline.last_score());
            // The live ECAPA worker runs concurrently with real 10 ms cpal
            // callbacks. Feeding two seconds of audio in a tight loop lets
            // the producer outrun inference and used to make this test pass
            // with an entirely silent output. Preserve the live cadence so
            // this exercises async result delivery and an actually open gate.
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let score_before_flush = pipeline.last_score();
        let tail = pipeline.flush().expect("flush");
        let score_after_flush = pipeline.last_score();
        for &s in &tail.audio {
            if s.is_nan() {
                nan_count += 1;
            } else if !s.is_finite() {
                inf_count += 1;
            }
        }
        all_output.extend_from_slice(&tail.audio);
        let wall_ms = t0.elapsed().as_millis();
        let audio_ms = audio_48k.len() as f64 / f64::from(OUTPUT_SAMPLE_RATE) * 1000.0;
        let realtime_factor = audio_ms / wall_ms.max(1) as f64;

        let rms_in = (audio_48k.iter().map(|s| s * s).sum::<f32>() / audio_48k.len() as f32).sqrt();
        let rms_out =
            (all_output.iter().map(|s| s * s).sum::<f32>() / all_output.len().max(1) as f32).sqrt();
        let peak_in = audio_48k.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let peak_out = all_output.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let total_out = all_output.len();
        let rms_db = 20.0 * (rms_out.max(1e-12) / rms_in.max(1e-12)).log10();
        let peak_db = 20.0 * (peak_out.max(1e-12) / peak_in.max(1e-12)).log10();

        eprintln!(
            "[offline] {chunks_pushed} chunks pushed, {zero_output_chunks} produced no output, \
             {wall_ms} ms wall ({realtime_factor:.2}× realtime)"
        );
        eprintln!(
            "[offline] gate transitions: {gate_transitions}, samples gate-on / total: \
             {gate_on_samples} / {total_out}"
        );
        eprintln!(
            "[offline] speaker score max-live={max_live_score:.4}, \
             before-flush={score_before_flush:.4}, after-flush={score_after_flush:.4}"
        );
        eprintln!("[offline] RMS  in={rms_in:.4}  out={rms_out:.4}  ({rms_db:+.1} dB)");
        eprintln!("[offline] PEAK in={peak_in:.4}  out={peak_out:.4}  ({peak_db:+.1} dB)");

        // 5) Assertions: the chain must produce finite samples and a
        //    length within shouting distance of the input (chain
        //    buffering can trim or pad by a few frames; we allow
        //    ±200 ms = ±9600 samples of slack at 48 kHz).
        assert_eq!(
            nan_count, 0,
            "pipeline emitted {nan_count} NaN samples — TSE / DFN3 state is going non-finite"
        );
        assert_eq!(
            inf_count, 0,
            "pipeline emitted {inf_count} Inf samples — clipping or division-by-zero somewhere"
        );
        let slack = OUTPUT_SAMPLE_RATE as usize / 5;
        let len_delta = all_output.len().abs_diff(audio_48k.len());
        let total_in = audio_48k.len();
        assert!(
            len_delta <= slack,
            "output length {total_out} too far from input {total_in} \
             (delta {len_delta}, slack {slack})"
        );
        assert!(
            gate_on_samples > u64::try_from(total_out).expect("audio length fits u64") / 10,
            "enrolled speech never opened the live gate: {gate_on_samples} / {total_out} samples"
        );
        assert!(
            rms_out > 1.0e-4,
            "live pipeline emitted silence for enrolled speech (RMS {rms_out:.3e})"
        );

        // 6) Optional artefact dump for ear-checking the chain.
        if std::env::var_os("MELLONELLA_DUMP_OFFLINE_WAV").is_some() {
            let in_path = "/tmp/mellonella_offline_input_48k.wav";
            let out_path = "/tmp/mellonella_offline_output_48k.wav";
            write_f32_wav(in_path, &audio_48k, hound_lite_spec());
            write_f32_wav(out_path, &all_output, hound_lite_spec());
            eprintln!(
                "[offline] wrote {} ({} samples) and {} ({} samples)",
                in_path,
                audio_48k.len(),
                out_path,
                all_output.len()
            );
        }
    }

    /// Minimal 16-bit / signed / mono WAV reader. Returns
    /// `(samples in [-1, 1], native sample rate)`. Panics on
    /// malformed input — fine for a developer harness.
    fn read_pcm16_mono_wav(path: &str) -> (Vec<f32>, u32) {
        let bytes = std::fs::read(path).expect("read input WAV");
        assert!(bytes.len() > 44, "WAV too short: {}", bytes.len());
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        // Find the `fmt ` and `data` chunks (the standard offsets
        // are only valid when there are no extension chunks before
        // `data`, which is overwhelmingly common but not guaranteed).
        let mut i = 12_usize;
        let mut fmt_off: Option<usize> = None;
        let mut data_off: Option<(usize, usize)> = None;
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            match id {
                b"fmt " => fmt_off = Some(i + 8),
                b"data" => data_off = Some((i + 8, sz)),
                _ => {}
            }
            i += 8 + sz;
        }
        let fmt = fmt_off.expect("WAV has no fmt chunk");
        let (data, data_len) = data_off.expect("WAV has no data chunk");
        let audio_format = u16::from_le_bytes([bytes[fmt], bytes[fmt + 1]]);
        let channels = u16::from_le_bytes([bytes[fmt + 2], bytes[fmt + 3]]);
        let sample_rate = u32::from_le_bytes([
            bytes[fmt + 4],
            bytes[fmt + 5],
            bytes[fmt + 6],
            bytes[fmt + 7],
        ]);
        let bits = u16::from_le_bytes([bytes[fmt + 14], bytes[fmt + 15]]);
        assert_eq!(audio_format, 1, "expected PCM (1), got {audio_format}");
        assert_eq!(channels, 1, "expected mono, got {channels} channels");
        assert_eq!(bits, 16, "expected 16-bit, got {bits}-bit");
        let scale = 1.0_f32 / f32::from(i16::MAX);
        let samples: Vec<f32> = bytes[data..data + data_len]
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) * scale)
            .collect();
        (samples, sample_rate)
    }

    /// Minimal 16-bit / 48 kHz / mono WAV writer config. Inlined so
    /// the test stays dependency-free (the GUI crate no longer pulls
    /// `hound` after the latest cleanup).
    #[derive(Clone, Copy)]
    struct WavSpec {
        channels: u16,
        sample_rate: u32,
    }
    fn hound_lite_spec() -> WavSpec {
        WavSpec {
            channels: 1,
            sample_rate: OUTPUT_SAMPLE_RATE,
        }
    }
    fn write_f32_wav(path: &str, samples: &[f32], spec: WavSpec) {
        use std::io::Write;
        let bits_per_sample = 16_u16;
        let byte_rate =
            spec.sample_rate * u32::from(spec.channels) * u32::from(bits_per_sample / 8);
        let block_align = spec.channels * (bits_per_sample / 8);
        let data_bytes: u32 = samples.len() as u32 * u32::from(bits_per_sample / 8);
        let mut f = std::fs::File::create(path).expect("create wav");
        f.write_all(b"RIFF").unwrap();
        f.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16_u32.to_le_bytes()).unwrap();
        f.write_all(&1_u16.to_le_bytes()).unwrap(); // PCM
        f.write_all(&spec.channels.to_le_bytes()).unwrap();
        f.write_all(&spec.sample_rate.to_le_bytes()).unwrap();
        f.write_all(&byte_rate.to_le_bytes()).unwrap();
        f.write_all(&block_align.to_le_bytes()).unwrap();
        f.write_all(&bits_per_sample.to_le_bytes()).unwrap();
        f.write_all(b"data").unwrap();
        f.write_all(&data_bytes.to_le_bytes()).unwrap();
        for &s in samples {
            let q = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            f.write_all(&q.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn start_records_an_error_for_an_invalid_audio_device() {
        let Some((ecapa, vad)) = skip_if_no_onnx() else {
            return;
        };
        let pool = enroll_pool_from_fixture(&ecapa, &vad);
        with_test_enrollment(&pool, || {
            let mut state = AppState::default();
            assert!(state.pool.is_some(), "precondition: pool auto-loaded");
            // Never open the developer/user's real microphone from a unit
            // test. An impossible explicit name deterministically exercises
            // the same construction-error path on every machine.
            state.selected_input = Some("__mellonella_test_missing_input__".to_string());
            state.start();
            assert!(state.last_error.is_some());
            assert!(state.session.is_none());
        });
    }

    #[test]
    fn enrollment_capture_quality_accepts_clean_speech_level_signal() {
        let audio: Vec<f32> = (0..OUTPUT_SAMPLE_RATE as usize * 3)
            .map(|index| {
                let t = index as f32 / OUTPUT_SAMPLE_RATE as f32;
                0.10 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
            })
            .collect();
        assert!(validate_enrollment_capture(&audio, OUTPUT_SAMPLE_RATE).is_ok());
    }

    #[test]
    fn legacy_voiceprint_threshold_is_clamped_to_the_new_safe_scale() {
        let legacy = legacy_voiceprint_threshold_path().expect("test config dir");
        let current = gate_settings_path().expect("test config dir");
        if let Some(parent) = legacy.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let _ = std::fs::remove_file(&current);
        std::fs::write(&legacy, "0.398\n").unwrap();
        let config = load_gate_config();
        assert!((config.theta_pass - DEFAULT_GATE_THRESHOLD).abs() < f32::EPSILON);
        assert!((config.hangover_ms - DEFAULT_GATE_HANGOVER_MS).abs() < f32::EPSILON);
        assert!((config.release_ms - DEFAULT_GATE_RELEASE_MS).abs() < f32::EPSILON);
        let _ = std::fs::remove_file(legacy);
    }

    #[test]
    fn versioned_gate_settings_preserve_a_user_tuned_threshold() {
        let legacy = legacy_voiceprint_threshold_path().expect("test config dir");
        let current = gate_settings_path().expect("test config dir");
        if let Some(parent) = current.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let _ = std::fs::remove_file(&legacy);
        std::fs::write(&current, "v2 0.398 650 140\n").unwrap();
        let config = load_gate_config();
        assert!((config.theta_pass - 0.398).abs() < f32::EPSILON);
        assert!((config.hangover_ms - 650.0).abs() < f32::EPSILON);
        assert!((config.release_ms - 140.0).abs() < f32::EPSILON);
        let _ = std::fs::remove_file(current);
    }

    #[test]
    fn unversioned_gate_settings_keep_stricter_values_but_clamp_looser_ones() {
        let legacy = legacy_voiceprint_threshold_path().expect("test config dir");
        let current = gate_settings_path().expect("test config dir");
        if let Some(parent) = current.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let _ = std::fs::remove_file(&legacy);

        std::fs::write(&current, "0.600 650 140\n").unwrap();
        assert!((load_gate_config().theta_pass - 0.600).abs() < f32::EPSILON);

        std::fs::write(&current, "0.300 650 140\n").unwrap();
        assert!((load_gate_config().theta_pass - DEFAULT_GATE_THRESHOLD).abs() < f32::EPSILON);
        let _ = std::fs::remove_file(current);
    }

    #[test]
    fn enrollment_capture_quality_rejects_silence_and_clipping() {
        assert!(validate_enrollment_capture(
            &vec![0.0; OUTPUT_SAMPLE_RATE as usize * 3],
            OUTPUT_SAMPLE_RATE
        )
        .is_err());
        assert!(validate_enrollment_capture(
            &vec![1.0; OUTPUT_SAMPLE_RATE as usize * 3],
            OUTPUT_SAMPLE_RATE
        )
        .is_err());
    }

    #[test]
    fn enrollment_consistency_separates_coherent_and_conflicting_profiles() {
        let coherent = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.10, 0.0],
            vec![0.95, -0.10, 0.0],
        ];
        let conflicting = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert!(enrollment_consistency(&coherent) > 0.9);
        assert!(enrollment_consistency(&conflicting) < MIN_ENROLLMENT_CONSISTENCY);
    }
}

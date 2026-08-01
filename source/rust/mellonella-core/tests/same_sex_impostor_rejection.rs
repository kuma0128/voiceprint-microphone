//! End-to-end check that a *similar-sounding* speaker is filtered out.
//!
//! The regression this guards against: with only `cos(window, profile)`
//! versus a fixed threshold, an unrelated speaker of the same sex and
//! language passes. Their 1 s scores reach a median of ~0.35 and a
//! maximum of ~0.53 against the enrolled profile, which overlaps the
//! enrolled speaker's own lower tail — so the threshold that blocks them
//! also cuts their owner. The fix is two extra terms, both exercised
//! here: acoustic-condition grouping of the enrollment anchors
//! (`EmbeddingPool::match_score`) and the session-scoped other-speaker
//! margin (`mellonella_core::nontarget`).
//!
//! Drives the real [`StreamingPipeline`] over a simulated call — the two
//! speakers alternating in ~4 s turns — and measures how much audio
//! survives the gate in each speaker's turns.
//!
//! Gated on `ORT_DYLIB_PATH`, `MELLONELLA_ECAPA_ONNX`,
//! `MELLONELLA_VAD_ONNX` plus two caller-supplied 16 kHz mono WAVs:
//!
//! * `MELLONELLA_SPK_TARGET_WAV` — ≥40 s of the speaker to enrol
//! * `MELLONELLA_SPK_IMPOSTOR_WAV` — ≥20 s of a *similar* other speaker
//!   (same sex and language is the case that matters; an opposite-sex
//!   impostor has margin to spare and proves nothing)
//!
//! Skips quietly when anything is missing.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::EmbeddingPoolConfig;
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{enroll_from_recording, PipelineComponents, PipelineConfig};
use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
use mellonella_core::vad::SileroVad;

const SR: u32 = 16_000;
/// Turn length in samples (~4 s), long enough for several identity
/// refreshes and for the other-speaker model to establish a cluster.
const TURN: usize = 64_000;
/// Pause between turns (~500 ms), i.e. ordinary turn-taking rather than
/// a barge-in. This is what the turn-boundary purge keys off.
const TURN_GAP: usize = 8_000;
/// Live push size — 10 ms, mirroring the audio worker's chunking.
const PUSH: usize = 160;

fn env_path(name: &str) -> Option<PathBuf> {
    let p = std::env::var_os(name).map(PathBuf::from)?;
    if !p.exists() {
        eprintln!("[skip] {name} → {} not found", p.display());
        return None;
    }
    Some(p)
}

fn skip_if_missing() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("[skip] ORT_DYLIB_PATH not set");
        return None;
    }
    Some((
        env_path("MELLONELLA_ECAPA_ONNX")?,
        env_path("MELLONELLA_VAD_ONNX")?,
        env_path("MELLONELLA_SPK_TARGET_WAV")?,
        env_path("MELLONELLA_SPK_IMPOSTOR_WAV")?,
    ))
}

fn read_pcm16_mono_wav(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read WAV");
    let mut i = 12_usize;
    let mut fmt_off: Option<usize> = None;
    let mut data_off: Option<(usize, usize)> = None;
    while i + 8 <= bytes.len() {
        let sz =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        match &bytes[i..i + 4] {
            b"fmt " => fmt_off = Some(i + 8),
            b"data" => data_off = Some((i + 8, sz.min(bytes.len() - i - 8))),
            _ => {}
        }
        i += 8 + sz + (sz & 1);
    }
    let fmt = fmt_off.expect("fmt chunk");
    let (data, dlen) = data_off.expect("data chunk");
    let sr = u32::from_le_bytes([
        bytes[fmt + 4],
        bytes[fmt + 5],
        bytes[fmt + 6],
        bytes[fmt + 7],
    ]);
    assert_eq!(sr, SR, "{}: expected 16 kHz mono", path.display());
    let scale = 1.0_f32 / f32::from(i16::MAX);
    bytes[data..data + dlen]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) * scale)
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn components(ecapa: &PathBuf, vad: &PathBuf) -> PipelineComponents {
    PipelineComponents {
        vad: SileroVad::from_onnx_path(vad, SR).expect("VAD load"),
        fbank: Fbank::with_speechbrain_filterbank().expect("filterbank"),
        ecapa: EcapaTdnn::from_onnx_path(ecapa).expect("ECAPA load"),
        cohort: Vec::new(),
        tse: None,
    }
}

/// The live GUI's identity cadence, at the decision rate so the test
/// needs no resampler.
fn live_pipeline_cfg(turn_boundary_silence_ms: f32, async_refresh: bool) -> PipelineConfig {
    PipelineConfig {
        sample_rate: SR,
        silence_force_off_ms: 700.0,
        sv_window_samples: 16_000,
        sv_update_samples: 4_000,
        score_ema_alpha: 0.90,
        async_refresh,
        sv_min_new_samples_after_silence: 1_600,
        enable_auto_learn: false,
        turn_boundary_silence_ms,
        sv_reopen_window_samples: 8_000,
        ..PipelineConfig::default()
    }
}

/// One measured run: what fraction of each speaker's turns survived.
struct Survival {
    target: f32,
    impostor: f32,
}

/// Alternate the two speakers in `TURN`-sized turns separated by
/// `TURN_GAP` of silence, push the result through the pipeline, and
/// report the output-vs-input RMS ratio over each speaker's own turns.
///
/// Passing the same recording as both speakers turns this into a solo
/// control.
fn run_call(
    gate: GateConfig,
    turn_boundary_silence_ms: f32,
    async_refresh: bool,
    target: &[f32],
    impostor: &[f32],
    comp: PipelineComponents,
) -> Survival {
    let mut comp = comp;
    let split = target.len() * 6 / 10;
    let pool = enroll_from_recording(&target[..split], &mut comp, EmbeddingPoolConfig::default())
        .expect("enrolment");

    // Build the call and remember which samples belong to whom. The gap
    // is attributed to the speaker who just finished, so it never
    // inflates the next speaker's numbers.
    let held = &target[split..];
    let mut audio: Vec<f32> = Vec::new();
    let mut is_target: Vec<bool> = Vec::new();
    let turns = (held.len() / TURN).min(impostor.len() / TURN);
    assert!(turns >= 2, "need at least 2 turns of each speaker");
    for t in 0..turns {
        for (src, tag) in [(impostor, false), (held, true)] {
            audio.extend_from_slice(&src[t * TURN..(t + 1) * TURN]);
            // `repeat_n` is Rust 1.82; keep the workspace's 1.75 MSRV.
            is_target.extend(std::iter::repeat(tag).take(TURN));
            audio.extend(std::iter::repeat(0.0).take(TURN_GAP));
            is_target.extend(std::iter::repeat(tag).take(TURN_GAP));
        }
    }

    let cfg = StreamingConfig {
        pipeline: live_pipeline_cfg(turn_boundary_silence_ms, async_refresh),
        gate,
        audio_sample_rate: SR,
        ..Default::default()
    };
    let mut pipeline = StreamingPipeline::new(pool, cfg, comp).expect("pipeline");
    let mut out: Vec<f32> = Vec::new();
    for (chunk_idx, chunk) in audio.chunks(PUSH).enumerate() {
        out.extend(pipeline.push_samples(chunk).expect("push").audio);
        // The live async worker receives one identity job per 250 ms
        // of audio. This test pushes synthetic device chunks much faster
        // than real time, so yield at that same boundary; otherwise it
        // measures an intentionally overloaded queue instead of the
        // shipped live path.
        if async_refresh && (chunk_idx + 1) % (4_000 / PUSH) == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    out.extend(pipeline.flush().expect("flush").audio);

    // The engine emits in step with its input, so index i of the output
    // corresponds to index i of the input.
    let n = out.len().min(audio.len());
    let mut tgt_in = Vec::new();
    let mut tgt_out = Vec::new();
    let mut imp_in = Vec::new();
    let mut imp_out = Vec::new();
    for i in 0..n {
        if is_target[i] {
            tgt_in.push(audio[i]);
            tgt_out.push(out[i]);
        } else {
            imp_in.push(audio[i]);
            imp_out.push(out[i]);
        }
    }
    Survival {
        target: rms(&tgt_out) / rms(&tgt_in).max(1e-9),
        impostor: rms(&imp_out) / rms(&imp_in).max(1e-9),
    }
}

/// The shipped live gate.
fn live_gate() -> GateConfig {
    GateConfig {
        theta_pass: 0.45,
        adaptive_theta: false,
        hangover_ms: 500.0,
        release_ms: 120.0,
        other_speaker_margin: 0.08,
        hangover_floor_frac: 0.6,
        ..GateConfig::default()
    }
}

#[test]
fn similar_impostor_is_filtered_while_the_enrolled_speaker_survives() {
    let Some((ecapa, vad, target_wav, impostor_wav)) = skip_if_missing() else {
        return;
    };
    let target = read_pcm16_mono_wav(&target_wav);
    let impostor = read_pcm16_mono_wav(&impostor_wav);

    let run = run_call(
        live_gate(),
        250.0,
        false,
        &target,
        &impostor,
        components(&ecapa, &vad),
    );
    eprintln!(
        "shipped gate: target survives {:.1}%, impostor leaks {:.1}%",
        100.0 * run.target,
        100.0 * run.impostor
    );

    // The residual is the head of each of the impostor's turns: ECAPA
    // needs ~0.5 s of their audio before anything can be said about it.
    // Sustained speech from them must be gone entirely.
    assert!(
        run.impostor < 0.10,
        "a same-sex impostor must be nearly silenced, leaked {:.1}% of their level",
        100.0 * run.impostor
    );
    assert!(
        run.target > 0.70,
        "the enrolled speaker must still come through, only {:.1}% survived",
        100.0 * run.target
    );
    assert!(
        run.target > run.impostor * 8.0,
        "target {:.3} must dominate impostor {:.3}",
        run.target,
        run.impostor
    );
}

#[test]
fn async_live_path_filters_the_same_sex_impostor() {
    let Some((ecapa, vad, target_wav, impostor_wav)) = skip_if_missing() else {
        return;
    };
    let target = read_pcm16_mono_wav(&target_wav);
    let impostor = read_pcm16_mono_wav(&impostor_wav);

    let run = run_call(
        live_gate(),
        250.0,
        true,
        &target,
        &impostor,
        components(&ecapa, &vad),
    );
    eprintln!(
        "async shipped gate: target survives {:.1}%, impostor leaks {:.1}%",
        100.0 * run.target,
        100.0 * run.impostor
    );
    assert!(
        run.impostor < 0.15,
        "async same-sex impostor leakage is too high: {:.1}%",
        100.0 * run.impostor
    );
    assert!(
        run.target > 0.65,
        "async path suppressed too much enrolled speech: {:.1}%",
        100.0 * run.target
    );
    assert!(run.target > run.impostor * 5.0);
}

#[test]
fn the_turn_boundary_purge_is_what_removes_the_bulk_of_the_leak() {
    let Some((ecapa, vad, target_wav, impostor_wav)) = skip_if_missing() else {
        return;
    };
    let target = read_pcm16_mono_wav(&target_wav);
    let impostor = read_pcm16_mono_wav(&impostor_wav);

    // Scoring alone leaves the ~1 s of every turn during which the
    // identity window still holds the previous speaker.
    let without = run_call(
        live_gate(),
        0.0,
        false,
        &target,
        &impostor,
        components(&ecapa, &vad),
    );
    let with = run_call(
        live_gate(),
        250.0,
        false,
        &target,
        &impostor,
        components(&ecapa, &vad),
    );
    eprintln!(
        "turn purge — impostor leak {:.1}% -> {:.1}%, target {:.1}% -> {:.1}%",
        100.0 * without.impostor,
        100.0 * with.impostor,
        100.0 * without.target,
        100.0 * with.target
    );
    assert!(
        with.impostor < without.impostor * 0.5,
        "the purge must at least halve leakage: {:.3} -> {:.3}",
        without.impostor,
        with.impostor
    );
}

#[test]
fn a_solo_session_pays_nothing_for_the_turn_boundary_purge() {
    let Some((ecapa, vad, target_wav, _)) = skip_if_missing() else {
        return;
    };
    let target = read_pcm16_mono_wav(&target_wav);

    // Both "speakers" are the enrolled one, so no other-speaker cluster
    // is ever formed and the purge must stay disarmed. Discarding the
    // identity window after every pause would otherwise cost this
    // speaker ~13% of their own audio for no benefit.
    let armed = run_call(
        live_gate(),
        250.0,
        false,
        &target,
        &target,
        components(&ecapa, &vad),
    );
    let disabled = run_call(
        live_gate(),
        0.0,
        false,
        &target,
        &target,
        components(&ecapa, &vad),
    );
    eprintln!(
        "solo: {:.1}% survives with the purge configured, {:.1}% with it off",
        100.0 * armed.target,
        100.0 * disabled.target
    );
    assert!(
        armed.target >= disabled.target - 0.02,
        "a solo session must be unaffected: {:.3} vs {:.3}",
        armed.target,
        disabled.target
    );
    assert!(
        armed.target > 0.90,
        "the enrolled speaker alone must pass essentially untouched, got {:.1}%",
        100.0 * armed.target
    );
}

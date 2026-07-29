//! Live target-speaker selection built on two-speaker SepFormer.
//!
//! Input is buffered into one-second 48 kHz blocks, resampled to the
//! model's native 8 kHz, separated into two anonymous streams, and each
//! stream is scored against the enrolled ECAPA speaker profile.  Only
//! the closer stream is returned to the ordinary VAD/gate/noise-removal
//! pipeline.

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::EmbeddingPool;
use mellonella_core::features::{Fbank, N_MELS};
use mellonella_core::gating::cos_sim_max_iter;
use mellonella_core::resample::resample_to;
use mellonella_core::sepformer::{
    SepformerSession, SEPFORMER_BLOCK_SAMPLES, SEPFORMER_SAMPLE_RATE, SEPFORMER_STREAMS,
};

use crate::INTERNAL_SAMPLE_RATE;

/// One second at the live pipeline's 48 kHz rate.
const LIVE_BLOCK_SAMPLES: usize = INTERNAL_SAMPLE_RATE as usize;
/// Rate the ECAPA Fbank front-end expects.
const ECAPA_SAMPLE_RATE: u32 = 16_000;
/// A short block-edge fade prevents clicks if source polarity or scale
/// changes at a SepFormer block boundary.
const EDGE_FADE_SAMPLES: usize = 240; // 5 ms @ 48 kHz
const SILENCE_RMS: f32 = 1.0e-5;
/// Fail closed when neither anonymous stream resembles any trusted
/// enrollment anchor. Separator-domain anchors added during the guided
/// registration normally score well above this; an unrelated speaker
/// remains below it.
const MIN_TARGET_SCORE: f32 = 0.50;

/// Live-tunable separator knobs shared between the UI thread and the
/// audio worker. Values are f32 bits inside atomics so a settings
/// slider can move mid-session without locks or a session restart.
#[derive(Debug)]
pub struct SeparatorTuning {
    /// Fail-closed pass threshold: blocks whose best stream score
    /// falls below this are muted.
    threshold_bits: AtomicU32,
    /// Best stream score of the most recently scored block — surfaced
    /// in the UI so the user can see where their own voice lands while
    /// adjusting the threshold.
    last_best_score_bits: AtomicU32,
}

impl SeparatorTuning {
    /// Library default for the fail-closed threshold.
    pub const DEFAULT_THRESHOLD: f32 = MIN_TARGET_SCORE;

    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold_bits: AtomicU32::new(threshold.to_bits()),
            last_best_score_bits: AtomicU32::new(0.0_f32.to_bits()),
        }
    }

    #[must_use]
    pub fn threshold(&self) -> f32 {
        f32::from_bits(self.threshold_bits.load(Ordering::Relaxed))
    }

    pub fn set_threshold(&self, value: f32) {
        self.threshold_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    /// Best stream score of the most recently scored block. `0.0`
    /// until the first non-silent block is processed.
    #[must_use]
    pub fn last_best_score(&self) -> f32 {
        f32::from_bits(self.last_best_score_bits.load(Ordering::Relaxed))
    }

    fn store_last_best_score(&self, value: f32) {
        self.last_best_score_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }
}

impl Default for SeparatorTuning {
    fn default() -> Self {
        Self::new(Self::DEFAULT_THRESHOLD)
    }
}
/// Calibrating every second of a 20-second recording would make the GUI
/// wait unnecessarily. Eight evenly distributed blocks cover the
/// passage while keeping post-record analysis to a few seconds.
const MAX_SEPARATOR_ENROLLMENT_BLOCKS: usize = 8;

/// How one anonymous SepFormer stream matched the enrolled speaker.
struct StreamMatch {
    /// The stream's ECAPA embedding, or empty when the stream was below
    /// the silence floor and never reached the model.
    embedding: Vec<f32>,
    /// Best cosine similarity against any enrollment anchor. `0.0` for a
    /// silent stream.
    score: f32,
}

impl StreamMatch {
    /// A stream that never reached the model.
    fn silent() -> Self {
        Self {
            embedding: Vec::new(),
            score: 0.0,
        }
    }
}

/// Buffered live separator and enrolled-speaker stream selector.
pub(crate) struct TargetSpeakerSeparator {
    separator: SepformerSession,
    ecapa: EcapaTdnn,
    fbank: Fbank,
    pool: EmbeddingPool,
    tuning: Arc<SeparatorTuning>,
    pending: Vec<f32>,
    /// Reused row-major `(frames, N_MELS)` Fbank scratch for one stream.
    features: Vec<f32>,
    /// Reused RMS-normalised 8 kHz scratch for one stream.
    normalized: Vec<f32>,
    block_index: u64,
}

impl TargetSpeakerSeparator {
    pub(crate) fn new(
        sepformer_path: impl AsRef<Path>,
        ecapa_path: impl AsRef<Path>,
        pool: EmbeddingPool,
        tuning: Arc<SeparatorTuning>,
    ) -> Result<Self, String> {
        if pool.is_empty() {
            return Err("SepFormer target selection requires an enrollment".to_string());
        }
        let separator =
            SepformerSession::from_onnx_path(sepformer_path).map_err(|e| e.to_string())?;
        let ecapa = EcapaTdnn::from_onnx_path(ecapa_path).map_err(|e| e.to_string())?;
        let fbank = Fbank::with_speechbrain_filterbank().map_err(|e| e.to_string())?;
        Ok(Self {
            separator,
            ecapa,
            fbank,
            pool,
            tuning,
            pending: Vec::with_capacity(LIVE_BLOCK_SAMPLES * 2),
            features: Vec::new(),
            normalized: Vec::with_capacity(SEPFORMER_BLOCK_SAMPLES),
            block_index: 0,
        })
    }

    /// Append live input and return every complete selected 1-second
    /// output block. Usually this is empty; once per second it contains
    /// one block.
    pub(crate) fn push(&mut self, input_48k: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        self.pending.extend_from_slice(input_48k);
        if self.pending.len() < LIVE_BLOCK_SAMPLES {
            return Ok(Vec::new());
        }
        // Move the buffer aside so `process_block` can take `&mut self`
        // while reading from it. `mem::take` leaves an empty `Vec`
        // behind and the original allocation is handed straight back, so
        // the steady state allocates nothing here — the previous
        // `split_off` + `mem::replace` pair allocated and copied a fresh
        // buffer for every block.
        let mut buffer = std::mem::take(&mut self.pending);
        let mut outputs = Vec::new();
        let mut consumed = 0;
        let mut failure = None;
        while buffer.len() - consumed >= LIVE_BLOCK_SAMPLES {
            let block = &buffer[consumed..consumed + LIVE_BLOCK_SAMPLES];
            match self.process_block(block, LIVE_BLOCK_SAMPLES) {
                Ok(output) => outputs.push(output),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
            consumed += LIVE_BLOCK_SAMPLES;
        }
        // Drop the consumed blocks and restore the buffer even on error,
        // so a failed block can't be reprocessed on the next push.
        buffer.drain(..consumed);
        self.pending = buffer;
        match failure {
            Some(error) => Err(error),
            None => Ok(outputs),
        }
    }

    /// Zero-pad and process the final partial block, returning only the
    /// number of samples that were actually captured.
    pub(crate) fn flush(&mut self) -> Result<Option<Vec<f32>>, String> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let valid = self.pending.len();
        let mut block = std::mem::take(&mut self.pending);
        block.resize(LIVE_BLOCK_SAMPLES, 0.0);
        Ok(Some(self.process_block(&block, valid)?))
    }

    fn process_block(&mut self, mixture_48k: &[f32], valid: usize) -> Result<Vec<f32>, String> {
        let started_at = Instant::now();
        let input_rms = rms(&mixture_48k[..valid]);
        if input_rms < SILENCE_RMS {
            return Ok(vec![0.0; valid]);
        }

        let mut mixture_8k = resample_to(mixture_48k, INTERNAL_SAMPLE_RATE, SEPFORMER_SAMPLE_RATE)
            .map_err(|e| e.to_string())?;
        force_length(&mut mixture_8k, SEPFORMER_BLOCK_SAMPLES);

        let streams = self
            .separator
            .separate(&mixture_8k)
            .map_err(|e| e.to_string())?;
        let [first, second] = self.match_streams(&streams)?;
        let (score_0, score_1) = (first.score, second.score);
        let selected = usize::from(score_1 > score_0);
        let min_target_score = self.tuning.threshold();
        let best_score = score_0.max(score_1);
        self.tuning.store_last_best_score(best_score);
        let accepted = best_score >= min_target_score;

        self.block_index += 1;
        if !accepted {
            eprintln!(
                "[audio-io] speaker separation block {}: rejected both streams, \
                 scores {:.3}/{:.3} (< {:.2}), {} ms",
                self.block_index,
                score_0,
                score_1,
                min_target_score,
                started_at.elapsed().as_millis(),
            );
            return Ok(vec![0.0; valid]);
        }

        // The quantized community export has an arbitrary output scale.
        // Apply one common gain to both hypothetical streams so their
        // combined energy matches the input mixture; this preserves the
        // model's relative source levels (equal speakers land near
        // -3 dB each) instead of boosting the selected voice to the full
        // two-speaker mixture level.
        let r0 = rms(&streams[0]);
        let r1 = rms(&streams[1]);
        let separated_rms = (r0 * r0 + r1 * r1).sqrt();
        let common_gain = (input_rms / separated_rms.max(1.0e-9)).min(100.0);
        // Move the selected stream out of the array rather than cloning it.
        let [stream_0, stream_1] = streams;
        let mut chosen_8k = if selected == 0 { stream_0 } else { stream_1 };

        // A global polarity inversion is inaudible, but choosing the
        // polarity most correlated with the mixture avoids a discontinuity
        // when the anonymous source order changes between blocks. Folding
        // the sign into the gain keeps this to one multiply per sample.
        let correlation: f32 = chosen_8k
            .iter()
            .zip(mixture_8k.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let scale = if correlation < 0.0 {
            -common_gain
        } else {
            common_gain
        };
        for sample in &mut chosen_8k {
            *sample *= scale;
        }

        let mut chosen_48k = resample_to(&chosen_8k, SEPFORMER_SAMPLE_RATE, INTERNAL_SAMPLE_RATE)
            .map_err(|e| e.to_string())?;
        force_length(&mut chosen_48k, LIVE_BLOCK_SAMPLES);
        chosen_48k.truncate(valid);
        fade_and_clamp(&mut chosen_48k);

        eprintln!(
            "[audio-io] speaker separation block {}: selected stream {}, scores {:.3}/{:.3}, \
             input RMS {:.4}, {} ms",
            self.block_index,
            selected + 1,
            score_0,
            score_1,
            input_rms,
            started_at.elapsed().as_millis(),
        );
        Ok(chosen_48k)
    }

    /// Score both anonymous streams against the enrolled pool.
    ///
    /// One inference per stream, not a single `(2, frames, N_MELS)` batch:
    /// ORT parallelises *within* an inference, and on this ECAPA export
    /// the batched graph barely uses the intra-op pool while two
    /// single-clip calls scale with it. Measured on an i7-14700 with
    /// `examples/perf_probe.rs`, 7 intra-op threads:
    ///
    /// | shape           | median |
    /// |-----------------|-------:|
    /// | 2 × single clip | 26.6 ms |
    /// | 1 × batch of 2  | 53.5 ms |
    ///
    /// Batching only wins when the pool is pinned to 2 threads (56.3 ms
    /// vs 61.0 ms), i.e. exactly the case the thread heuristic no longer
    /// produces on a wide CPU. Don't "optimise" this into a batch without
    /// measuring it first — `examples/perf_probe.rs` reports the
    /// single-clip baseline this shape depends on.
    ///
    /// A stream below [`SILENCE_RMS`] is skipped entirely, so a silent
    /// half costs no inference and yields no garbage embedding.
    fn match_streams(
        &mut self,
        streams: &[Vec<f32>; SEPFORMER_STREAMS],
    ) -> Result<[StreamMatch; SEPFORMER_STREAMS], String> {
        let mut matches = [StreamMatch::silent(), StreamMatch::silent()];
        for (index, source) in streams.iter().enumerate() {
            if rms(source) < SILENCE_RMS {
                continue;
            }
            let frames = self.stream_features(source)?;
            let embedding = self
                .ecapa
                .embed_features(&self.features, frames, N_MELS)
                .map_err(|e| e.to_string())?;
            // Source selection is a short one-second comparison. Use the
            // closest immutable enrollment anchor rather than their
            // centroid: a varied 20-second enrollment intentionally spans
            // different phonetic/pitch regions, and averaging them can
            // push every short utterance toward a low score even when one
            // anchor is a close match. Live auto-learn is disabled, so
            // only trusted enrollment anchors participate here.
            let score = cos_sim_max_iter(&embedding, self.pool.anchors().iter().map(Vec::as_slice));
            matches[index] = StreamMatch { embedding, score };
        }
        Ok(matches)
    }

    /// Normalise one 8 kHz stream, resample it to the ECAPA rate and
    /// extract its Fbank into the reused [`Self::features`] scratch.
    /// Returns the frame count.
    fn stream_features(&mut self, source_8k: &[f32]) -> Result<usize, String> {
        // ECAPA should compare voice timbre, not the converter's arbitrary
        // gain. Put both anonymous streams on the same safe RMS before
        // extracting their embeddings.
        let gain = (0.08 / rms(source_8k)).clamp(0.01, 100.0);
        self.normalized.clear();
        self.normalized.extend(
            source_8k
                .iter()
                .map(|&sample| (sample * gain).clamp(-1.0, 1.0)),
        );
        let mut audio_16k = resample_to(&self.normalized, SEPFORMER_SAMPLE_RATE, ECAPA_SAMPLE_RATE)
            .map_err(|e| e.to_string())?;
        force_length(&mut audio_16k, ECAPA_SAMPLE_RATE as usize);
        self.features.clear();
        self.features
            .extend_from_slice(&self.fbank.compute(&audio_16k));
        Ok(self.features.len() / N_MELS)
    }

    /// Produce separator-domain target embeddings from a clean guided
    /// enrollment recording. No fail-closed threshold is applied here:
    /// the base pool was already built from the same trusted recording,
    /// so the closer of the two anonymous streams is the calibration
    /// target by construction.
    fn calibration_anchors(&mut self, audio_48k: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let total_blocks = audio_48k.len() / LIVE_BLOCK_SAMPLES;
        if total_blocks == 0 {
            return Ok(Vec::new());
        }
        let wanted = total_blocks.min(MAX_SEPARATOR_ENROLLMENT_BLOCKS);
        let mut anchors = Vec::with_capacity(wanted);
        for sample_index in 0..wanted {
            let block_index = sample_index * total_blocks / wanted;
            let start = block_index * LIVE_BLOCK_SAMPLES;
            let block = &audio_48k[start..start + LIVE_BLOCK_SAMPLES];
            if rms(block) < SILENCE_RMS {
                continue;
            }
            let mut mixture_8k = resample_to(block, INTERNAL_SAMPLE_RATE, SEPFORMER_SAMPLE_RATE)
                .map_err(|e| e.to_string())?;
            force_length(&mut mixture_8k, SEPFORMER_BLOCK_SAMPLES);
            let streams = self
                .separator
                .separate(&mixture_8k)
                .map_err(|e| e.to_string())?;
            let [first, second] = self.match_streams(&streams)?;
            let selected = if second.score > first.score {
                second
            } else {
                first
            };
            if !selected.embedding.is_empty() {
                anchors.push(selected.embedding);
            }
        }
        Ok(anchors)
    }
}

/// Build the companion speaker-selection profile used after SepFormer.
///
/// The returned pool retains the clean enrollment anchors and adds up to
/// eight embeddings measured after the same source-separation transform
/// used live. It is stored separately from the ordinary gate profile so
/// separator artifacts cannot shift the raw-microphone centroid.
pub fn build_separator_enrollment_pool(
    base_pool: &EmbeddingPool,
    sepformer_path: impl AsRef<Path>,
    ecapa_path: impl AsRef<Path>,
    audio_48k: &[f32],
) -> Result<EmbeddingPool, String> {
    let mut stage = TargetSpeakerSeparator::new(
        sepformer_path,
        ecapa_path,
        base_pool.clone(),
        Arc::new(SeparatorTuning::default()),
    )?;
    let anchors = stage.calibration_anchors(audio_48k)?;
    if anchors.len() < 4 {
        return Err(format!(
            "分離後の本人声紋が{}個しか作れませんでした（4個以上必要）",
            anchors.len()
        ));
    }
    let mut selector_pool = base_pool.clone();
    selector_pool.add_anchors(anchors);
    Ok(selector_pool)
}

fn force_length(samples: &mut Vec<f32>, expected: usize) {
    samples.truncate(expected);
    samples.resize(expected, 0.0);
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    mean_square.sqrt()
}

/// Apply the block-edge fade and the output clamp in one pass over the
/// block, instead of walking all 48 000 samples twice.
fn fade_and_clamp(samples: &mut [f32]) {
    let fade = EDGE_FADE_SAMPLES.min(samples.len() / 2);
    let last = samples.len().saturating_sub(1);
    for (index, sample) in samples.iter_mut().enumerate() {
        // `fade <= len / 2 <= last` for every non-empty block, so neither
        // bound underflows and the two ramps never overlap.
        let gain = if index < fade {
            index as f32 / fade as f32
        } else if index > last - fade {
            (last - index) as f32 / fade as f32
        } else {
            1.0
        };
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_length_truncates_and_pads() {
        let mut long = vec![1.0; 10];
        force_length(&mut long, 4);
        assert_eq!(long, vec![1.0; 4]);
        let mut short = vec![1.0; 2];
        force_length(&mut short, 4);
        assert_eq!(short, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn edge_fade_reaches_zero_at_both_ends() {
        let mut audio = vec![1.0; EDGE_FADE_SAMPLES * 3];
        fade_and_clamp(&mut audio);
        assert_eq!(audio[0], 0.0);
        assert_eq!(audio[audio.len() - 1], 0.0);
        assert_eq!(audio[EDGE_FADE_SAMPLES], 1.0);
    }

    /// The fused pass must still clamp the untouched interior, which the
    /// separate clamp loop used to handle.
    #[test]
    fn fade_and_clamp_bounds_the_interior() {
        let mut audio = vec![4.0; EDGE_FADE_SAMPLES * 3];
        audio[EDGE_FADE_SAMPLES * 2] = -7.0;
        fade_and_clamp(&mut audio);
        assert_eq!(audio[EDGE_FADE_SAMPLES], 1.0);
        assert_eq!(audio[EDGE_FADE_SAMPLES * 2], -1.0);
    }

    /// Degenerate lengths must not underflow the tail-ramp bound.
    #[test]
    fn fade_and_clamp_handles_tiny_blocks() {
        for len in 0..4_usize {
            let mut audio = vec![2.0; len];
            fade_and_clamp(&mut audio);
            assert!(audio.iter().all(|s| (-1.0..=1.0).contains(s)));
        }
    }

    /// Optional end-to-end calibration fixture. It is ignored in normal
    /// CI because it needs three large external models/audio files.
    /// Set the four paths below and run this exact test manually.
    #[test]
    #[ignore = "requires external SepFormer, ECAPA, enrollment and raw-f32 fixtures"]
    fn external_equal_loudness_fixture_selects_target() {
        let sepformer = std::env::var("TEST_SEPFORMER_ONNX").unwrap();
        let ecapa = std::env::var("TEST_ECAPA_ONNX").unwrap();
        let enrollment = std::env::var("TEST_ENROLLMENT_JSON").unwrap();
        let calibration_path = std::env::var("TEST_CALIBRATION_F32").unwrap();
        let mixture_path = std::env::var("TEST_MIXTURE_F32").unwrap();
        let target_path = std::env::var("TEST_TARGET_F32").unwrap();
        let impostor_path = std::env::var("TEST_IMPOSTOR_F32").unwrap();
        let base_pool = EmbeddingPool::load(
            enrollment,
            mellonella_core::enrollment::EmbeddingPoolConfig::default(),
        )
        .unwrap();
        let calibration = read_f32(&calibration_path);
        let pool =
            build_separator_enrollment_pool(&base_pool, &sepformer, &ecapa, &calibration).unwrap();
        let mixture = read_f32(&mixture_path);
        let target = read_f32(&target_path);
        let impostor = read_f32(&impostor_path);
        assert_eq!(mixture.len(), LIVE_BLOCK_SAMPLES);
        assert_eq!(target.len(), LIVE_BLOCK_SAMPLES);
        assert_eq!(impostor.len(), LIVE_BLOCK_SAMPLES);

        let mut stage = TargetSpeakerSeparator::new(
            sepformer,
            ecapa,
            pool,
            Arc::new(SeparatorTuning::default()),
        )
        .unwrap();
        let output = stage.push(&mixture).unwrap().pop().unwrap();
        let (target_coef, impostor_coef) = two_source_projection(&output, &target, &impostor);
        let preference_db = 20.0 * (target_coef.abs() / impostor_coef.abs().max(1.0e-9)).log10();
        eprintln!(
            "target_coef={target_coef:.4}, impostor_coef={impostor_coef:.4}, \
             preference={preference_db:+.2} dB"
        );
        assert!(
            preference_db > 6.0,
            "selected output should prefer enrolled target by >6 dB, got {preference_db:+.2} dB"
        );
    }

    #[test]
    #[ignore = "requires external SepFormer, ECAPA, enrollment and raw-f32 fixtures"]
    fn external_unenrolled_speaker_is_rejected() {
        let sepformer = std::env::var("TEST_SEPFORMER_ONNX").unwrap();
        let ecapa = std::env::var("TEST_ECAPA_ONNX").unwrap();
        let enrollment = std::env::var("TEST_ENROLLMENT_JSON").unwrap();
        let calibration_path = std::env::var("TEST_CALIBRATION_F32").unwrap();
        let impostor_path = std::env::var("TEST_IMPOSTOR_F32").unwrap();
        let base_pool = EmbeddingPool::load(
            enrollment,
            mellonella_core::enrollment::EmbeddingPoolConfig::default(),
        )
        .unwrap();
        let calibration = read_f32(&calibration_path);
        let pool =
            build_separator_enrollment_pool(&base_pool, &sepformer, &ecapa, &calibration).unwrap();
        let impostor = read_f32(&impostor_path);
        let mut stage = TargetSpeakerSeparator::new(
            sepformer,
            ecapa,
            pool,
            Arc::new(SeparatorTuning::default()),
        )
        .unwrap();
        let output = stage.push(&impostor).unwrap().pop().unwrap();
        assert!(
            rms(&output) < 1.0e-7,
            "unenrolled speaker should be muted, output RMS={}",
            rms(&output)
        );
    }

    fn read_f32(path: &str) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect()
    }

    fn dot(left: &[f32], right: &[f32]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum()
    }

    fn two_source_projection(output: &[f32], target: &[f32], impostor: &[f32]) -> (f64, f64) {
        let tt = dot(target, target);
        let ti = dot(target, impostor);
        let ii = dot(impostor, impostor);
        let ty = dot(target, output);
        let iy = dot(impostor, output);
        let determinant = tt * ii - ti * ti;
        (
            (ty * ii - iy * ti) / determinant,
            (iy * tt - ty * ti) / determinant,
        )
    }
}

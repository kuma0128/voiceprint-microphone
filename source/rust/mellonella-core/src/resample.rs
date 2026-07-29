//! Audio resampling for the SV-rate pipeline.
//!
//! Wraps `rubato::SincFixedIn` (windowed-sinc interpolation) to convert
//! arbitrary input sample rates to the 16 kHz the rest of
//! `mellonella-core` operates at. Mirrors what
//! `mellonella_poc.pipeline.resample` does on the Python side with
//! `scipy.signal.resample_poly`, with the practical caveat that the
//! two algorithms (windowed-sinc vs polyphase + low-pass) produce
//! samples that are equivalent under any reasonable error metric but
//! not byte-equal.
//!
//! Empirical agreement on a synthesised 180 Hz harmonic stack
//! (44.1 kHz → 16 kHz):
//!
//! | metric                | value        | tolerance |
//! |-----------------------|--------------|-----------|
//! | per-sample `max\|Δ\|`  | ~5 × 10⁻³    | 1 × 10⁻²  |
//! | post-Fbank gate state | byte-equal   | n/a       |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::cell::RefCell;
use std::collections::HashMap;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// A cached resampler together with its reusable output scratch.
///
/// `Resampler::process` allocates a fresh `Vec<Vec<f32>>` on every
/// call, and its `&[V]` input signature made the caller wrap a clone of
/// the input (`vec![audio.to_vec()]` — 192 KB for a one-second 48 kHz
/// block). Holding the output buffer here and passing the caller's
/// slice straight through `process_into_buffer` leaves only the
/// returned `Vec` to allocate. `process` is itself just
/// `output_frames_next()`-sized allocation + `process_into_buffer` +
/// truncate to the reported output length, so this is a faithful
/// substitution.
struct Cached {
    resampler: SincFixedIn<f32>,
    /// `output_frames_max()`-sized, so no reallocation can occur inside
    /// `process_into_buffer`.
    output: Vec<Vec<f32>>,
}

thread_local! {
    /// Per-thread cache of constructed resamplers keyed by
    /// `(src_sr, dst_sr, chunk_size)`. `SincFixedIn::new` precomputes a
    /// large windowed-sinc table (sinc_len 256 × oversampling 256), and
    /// the live separator calls [`resample_to`] several times per
    /// second with the same fixed shapes — rebuilding that table each
    /// call dominated the non-inference cost of a block. `reset()`
    /// restores construction-time state, so a cache hit produces output
    /// identical to a freshly built instance. The key space is a
    /// handful of fixed shapes per worker thread, so the map stays
    /// tiny.
    static RESAMPLERS: RefCell<HashMap<(u32, u32, usize), Cached>> =
        RefCell::new(HashMap::new());
}

/// Errors returned by [`resample_to`].
#[derive(Debug)]
pub enum ResampleError {
    /// `rubato` rejected the configuration or the input shape.
    Rubato(String),
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rubato(s) => write!(f, "rubato resampler error: {s}"),
        }
    }
}

impl std::error::Error for ResampleError {}

/// Resample `audio` from `src_sr` to `dst_sr` using a windowed-sinc
/// interpolator. Returns the resampled buffer; samples are clipped to
/// `[-1, 1]` upstream by [`SincFixedIn`] internals if necessary.
///
/// Identity (`src_sr == dst_sr`) returns a cheap clone — useful for
/// pipelines that don't know upfront whether the input matches the
/// target rate.
///
/// # Errors
/// Returns [`ResampleError::Rubato`] when `SincFixedIn::new` or
/// `process` fails (typically: sample rates out of range, or
/// per-channel buffer mismatch — neither should happen for our
/// `mono → mono` call sites).
pub fn resample_to(audio: &[f32], src_sr: u32, dst_sr: u32) -> Result<Vec<f32>, ResampleError> {
    if src_sr == dst_sr {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = audio.len();
    RESAMPLERS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let cached = match cache.entry((src_sr, dst_sr, chunk_size)) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let cached: &mut Cached = entry.into_mut();
                cached.resampler.reset();
                cached
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let ratio = f64::from(dst_sr) / f64::from(src_sr);
                let params = SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris2,
                };
                let resampler = SincFixedIn::<f32>::new(ratio, 1.1, params, chunk_size, 1)
                    .map_err(|e| ResampleError::Rubato(e.to_string()))?;
                let output = resampler.output_buffer_allocate(true);
                entry.insert(Cached { resampler, output })
            }
        };

        // Split the borrow so the resampler and its scratch can be held
        // mutably at the same time.
        let Cached { resampler, output } = cached;
        let (_, produced) = resampler
            .process_into_buffer(&[audio], output, None)
            .map_err(|e| ResampleError::Rubato(e.to_string()))?;
        Ok(output[0][..produced].to_vec())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_returns_input() {
        let audio = vec![0.1_f32, -0.2, 0.3];
        let out = resample_to(&audio, 16_000, 16_000).unwrap();
        assert_eq!(out, audio);
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = resample_to(&[], 44_100, 16_000).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn downsample_changes_length_proportionally() {
        // 1 s @ 48 kHz → ≈ 16 kHz worth of samples (allow ±1 % slack
        // because windowed-sinc with delay correction doesn't land on
        // an exact ratio).
        let audio = vec![0.0_f32; 48_000];
        let out = resample_to(&audio, 48_000, 16_000).unwrap();
        let expected = 16_000_i32;
        let slack = expected / 100; // ±1 %
        let got = out.len() as i32;
        assert!(
            (got - expected).abs() <= slack,
            "expected ≈ {expected} samples (±{slack}), got {got}"
        );
    }
}

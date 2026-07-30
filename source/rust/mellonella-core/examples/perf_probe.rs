//! Measures the per-block costs the live target-speaker separator pays
//! once a second, so changes to the hot path can be checked instead of
//! argued about.
//!
//! ```text
//! ORT_DYLIB_PATH=... MELLONELLA_ECAPA_ONNX=... \
//!   cargo run --release --example perf_probe
//! ```
//!
//! Reports, for the two anonymous SepFormer streams a block produces:
//! * `Fbank::compute` — the log-mel front-end (2 calls per block)
//! * ECAPA embedding — 2 calls per block, the dominant non-SepFormer cost
//!
//! Set `MELLONELLA_ORT_INTRA_THREADS` to compare thread counts; that is
//! how the heuristic in [`mellonella_core::ort_threads`] was chosen.

#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::features::{Fbank, N_MELS};

/// One second at the ECAPA rate: `1 + 16000 / 160`.
const FRAMES: usize = 101;
const ITERATIONS: usize = 30;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let Ok(ecapa_path) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("set MELLONELLA_ECAPA_ONNX to the embedding-only ECAPA export");
        std::process::exit(2);
    };
    println!(
        "intra-op threads: {}",
        mellonella_core::ort_threads::intra_op_threads()
    );

    // A one-second 16 kHz stream per SepFormer output.
    let audio: Vec<f32> = (0..16_000)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.1 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
        })
        .collect();

    let mut fbank = Fbank::with_speechbrain_filterbank().expect("filterbank");
    let mut features = fbank.compute(&audio);
    assert_eq!(features.len() / N_MELS, FRAMES, "unexpected frame count");

    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        features = fbank.compute(&audio);
        timings.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let fbank_ms = median(timings);
    println!(
        "Fbank::compute        {fbank_ms:7.3} ms  (x2 per block = {:.3} ms)",
        fbank_ms * 2.0
    );

    let mut model = EcapaTdnn::from_onnx_path(&ecapa_path).expect("load ECAPA");
    // Warm up: the first inference pays lazy-allocation costs.
    let _ = model
        .embed_features(&features, FRAMES, N_MELS)
        .expect("warmup");

    let mut timings = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let _ = model
            .embed_features(&features, FRAMES, N_MELS)
            .expect("embed");
        timings.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let ecapa_ms = median(timings);
    println!(
        "ECAPA embed_features  {ecapa_ms:7.3} ms  (x2 per block = {:.3} ms)",
        ecapa_ms * 2.0
    );
    println!(
        "\nnon-SepFormer total   {:7.3} ms of the 1000 ms block budget",
        2.0 * (fbank_ms + ecapa_ms)
    );
}

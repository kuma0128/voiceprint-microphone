//! Parity check between [`mellonella_core::vad::SileroVad`] and the
//! Python silero-vad ONNX wrapper for a deterministic synthesised
//! speech-like waveform.
//!
//! Gated on `MELLONELLA_VAD_ONNX` (path to `silero_vad.onnx`) and
//! `ORT_DYLIB_PATH` (libonnxruntime). Without those, the test prints
//! a skip notice and passes — keeps CI green without vendoring the
//! 2.3 MB ONNX file.

use mellonella_core::vad::{SileroVad, CHUNK_SAMPLES_16K};

const INPUT: &[u8] = include_bytes!("fixtures/vad_input.bin");

// Generated with the official Silero `OnnxWrapper` against the model
// distributed with VoiceprintMic (SHA-256
// a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808).
// Keep this textual so the model revision used by the golden values is
// reviewable; the older binary fixture came from a different Silero model.
const EXPECTED: &[f32] = &[
    0.816_423_54,
    0.732_929_5,
    0.607_571_6,
    0.453_404_25,
    0.343_447_95,
    0.221_228_27,
    0.114_739_12,
    0.081_901_46,
    0.071_787_3,
    0.051_949_77,
    0.047_697_097,
    0.044_141_62,
    0.018_758_029,
    0.011_038_87,
    0.012_243_241,
    0.020_993_322,
    0.027_639_449,
    0.034_621_567,
    0.033_105_075,
    0.021_397_322,
    0.007_980_704,
    0.004_800_737,
    0.008_052_438,
    0.010_716_528,
    0.011_371_553,
    0.021_459_848,
    0.022_306_174,
    0.013_300_061,
    0.004_272_521,
    0.002_723_128,
    0.004_530_102,
];

const TOL: f32 = 1e-3;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn vad_matches_silero_fixture() {
    let Ok(path) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return;
    };
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skip] MELLONELLA_VAD_ONNX={path} does not exist");
        return;
    }

    let audio = read_f32_buffer(INPUT);
    let n_chunks = audio.len() / CHUNK_SAMPLES_16K;
    assert_eq!(
        n_chunks,
        EXPECTED.len(),
        "chunk count mismatch: {} vs {}",
        n_chunks,
        EXPECTED.len()
    );

    let mut vad = SileroVad::from_onnx_path(&path, 16_000).expect("load VAD ONNX");
    let mut max_delta = 0.0_f32;
    let mut argmax = 0_usize;
    for i in 0..n_chunks {
        let chunk = &audio[i * CHUNK_SAMPLES_16K..(i + 1) * CHUNK_SAMPLES_16K];
        let actual = vad.score(chunk).expect("VAD inference");
        let delta = (actual - EXPECTED[i]).abs();
        if delta > max_delta {
            max_delta = delta;
            argmax = i;
        }
    }
    assert!(
        max_delta <= TOL,
        "max|Δ|={max_delta:.3e} at chunk={argmax} exceeds tol={TOL:.3e}"
    );
}

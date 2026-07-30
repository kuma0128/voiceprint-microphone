#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::path::PathBuf;

use mellonella_core::resample::resample_to;
use mellonella_core::tse::{TseConfig, TseSession, TSE_COND_DIM};

fn read_pcm16_mono_wav(path: &PathBuf) -> (Vec<f32>, u32) {
    let bytes = std::fs::read(path).expect("read WAV");
    let mut offset = 12_usize;
    let mut sample_rate = None;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                sample_rate = Some(u32::from_le_bytes(
                    bytes[offset + 12..offset + 16].try_into().unwrap(),
                ));
            }
            b"data" => data = Some((offset + 8, size)),
            _ => {}
        }
        offset += 8 + size;
    }
    let (start, size) = data.expect("data chunk");
    let samples = bytes[start..start + size]
        .chunks_exact(2)
        .map(|chunk| f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 32768.0)
        .collect();
    (samples, sample_rate.expect("sample rate"))
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(args.next().expect("model path"));
    let enrollment = PathBuf::from(args.next().expect("enrollment json path"));
    let input_wav = args.next().map(PathBuf::from);
    let payload: serde_json::Value =
        serde_json::from_slice(&std::fs::read(enrollment).expect("read enrollment"))
            .expect("parse enrollment");
    let anchors = payload["anchors"].as_array().expect("anchors");
    let mut cond = [0.0_f32; TSE_COND_DIM];
    for (index, slot) in cond.iter_mut().enumerate() {
        *slot = anchors[0][index].as_f64().expect("anchor value") as f32;
    }

    let mut session =
        TseSession::from_onnx_path_with_config(model, TseConfig::prod_48k()).expect("load TSE");
    let input = if let Some(path) = input_wav {
        let (samples, rate) = read_pcm16_mono_wav(&path);
        resample_to(&samples, rate, 48_000).expect("resample")
    } else {
        (0..96_000)
            .map(|index| 0.1 * (2.0 * std::f32::consts::PI * 180.0 * index as f32 / 48_000.0).sin())
            .collect()
    };
    let mut output = Vec::new();
    let mut first_non_finite_chunk = None;
    for (chunk_index, chunk) in input.chunks(480).enumerate() {
        let mut padded = chunk.to_vec();
        padded.resize(480, 0.0);
        let extracted = session.process_chunk(&padded, &cond).expect("process");
        if first_non_finite_chunk.is_none() && extracted.iter().any(|sample| !sample.is_finite()) {
            first_non_finite_chunk = Some(chunk_index);
        }
        output.extend(extracted);
    }
    let rms =
        (output.iter().map(|sample| sample * sample).sum::<f32>() / output.len() as f32).sqrt();
    let peak = output
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    let nonzero = output.iter().filter(|sample| **sample != 0.0).count();
    let cond_norm = cond.iter().map(|value| value * value).sum::<f32>().sqrt();
    println!(
        "samples={} cond_norm={cond_norm:.6} rms={rms:.12} peak={peak:.12} nonzero={nonzero}",
        output.len()
    );
    println!("first_non_finite_chunk={first_non_finite_chunk:?}");
}

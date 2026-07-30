//! Two-speaker source separation with the quantized SepFormer ONNX.
//!
//! The shipped community model accepts one second of 8 kHz mono audio
//! (`mix: [1, 8000]`) and returns two separated streams
//! (`streams: [1, 8000, 2]`).  Speaker identity is intentionally not
//! assigned by the separation model; callers compare both streams with
//! an enrolled speaker embedding and keep the closer one.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::path::Path;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

use crate::embedding::EmbeddingError;

/// Native sample rate of the compact WSJ0-2mix SepFormer export.
pub const SEPFORMER_SAMPLE_RATE: u32 = 8_000;
/// Fixed block size used by the live wrapper: one second.
pub const SEPFORMER_BLOCK_SAMPLES: usize = 8_000;
/// The model separates exactly two speakers.
pub const SEPFORMER_STREAMS: usize = 2;

/// Stateful ONNX Runtime session for the source-separation model.
pub struct SepformerSession {
    session: Session,
}

impl SepformerSession {
    /// Load a SepFormer ONNX model from disk.
    ///
    /// # Errors
    /// Returns [`EmbeddingError`] for ONNX Runtime load failures.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(crate::ort_threads::intra_op_threads())?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(Self { session })
    }

    /// Separate one second of 8 kHz mono audio into two streams.
    ///
    /// # Errors
    /// Returns an incompatible-shape error unless `mixture.len() == 8000`,
    /// or when the ONNX output is not `[1, 8000, 2]` (the known export
    /// contract).
    pub fn separate(
        &mut self,
        mixture: &[f32],
    ) -> Result<[Vec<f32>; SEPFORMER_STREAMS], EmbeddingError> {
        if mixture.len() != SEPFORMER_BLOCK_SAMPLES {
            return Err(EmbeddingError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }

        // `TensorRef` borrows a `(shape, data)` pair directly, so the
        // mixture goes to ONNX Runtime without an intermediate copy.
        let outputs = self.session.run(ort::inputs![
            "mix" => TensorRef::from_array_view(([1_usize, SEPFORMER_BLOCK_SAMPLES], mixture))?,
        ])?;
        let (shape, data) = outputs["streams"].try_extract_tensor::<f32>()?;
        let dims: &[i64] = shape;
        if dims != [1, SEPFORMER_BLOCK_SAMPLES as i64, SEPFORMER_STREAMS as i64] {
            return Err(EmbeddingError::UnexpectedOutputShape {
                got: dims.to_vec(),
                expected_dim: SEPFORMER_BLOCK_SAMPLES * SEPFORMER_STREAMS,
            });
        }

        // De-interleave and validate in one pass over the output — the
        // separate `is_finite` scan walked all 16 000 samples twice.
        let mut streams = [
            Vec::with_capacity(SEPFORMER_BLOCK_SAMPLES),
            Vec::with_capacity(SEPFORMER_BLOCK_SAMPLES),
        ];
        for frame in data.chunks_exact(SEPFORMER_STREAMS) {
            let (first, second) = (frame[0], frame[1]);
            if !first.is_finite() || !second.is_finite() {
                return Err(EmbeddingError::Ort(
                    "SepFormer produced non-finite audio".to_string(),
                ));
            }
            streams[0].push(first);
            streams[1].push(second);
        }
        Ok(streams)
    }
}

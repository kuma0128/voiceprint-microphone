//! Stateful sample-rate conversion for OS audio callbacks.
//!
//! Device callbacks are not guaranteed to contain the fixed number of
//! frames requested by `rubato::SincFixedIn`.  Padding every callback
//! independently inserts artificial silence and resets the effective
//! time line every few milliseconds.  This adapter keeps the residue
//! until a complete sinc input block is available, so callback
//! boundaries are inaudible and the long-term sample count is correct.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub(crate) struct StreamingResampler {
    inner: SincFixedIn<f32>,
    pending: Vec<f32>,
    output: Vec<Vec<f32>>,
}

impl StreamingResampler {
    pub(crate) fn new(src_sr: u32, dst_sr: u32) -> Result<Option<Self>, String> {
        if src_sr == dst_sr {
            return Ok(None);
        }
        let ratio = f64::from(dst_sr) / f64::from(src_sr);
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = SincFixedIn::<f32>::new(ratio, 1.1, params, 1024, 1)
            .map_err(|error| error.to_string())?;
        let output = inner.output_buffer_allocate(true);
        Ok(Some(Self {
            inner,
            pending: Vec::with_capacity(2048),
            output,
        }))
    }

    /// Append one arbitrary callback and emit only complete converted
    /// blocks. The final incomplete input is deliberately retained; it
    /// is prepended to the next callback instead of being zero-padded.
    pub(crate) fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.pending.extend_from_slice(input);
        let mut converted = Vec::new();
        loop {
            let needed = self.inner.input_frames_next();
            if self.pending.len() < needed {
                break;
            }
            let input_channels = [&self.pending[..needed]];
            let (consumed, produced) = self
                .inner
                .process_into_buffer(&input_channels, &mut self.output, None)
                .map_err(|error| error.to_string())?;
            converted.extend_from_slice(&self.output[0][..produced]);
            self.pending.drain(..consumed);
        }
        Ok(converted)
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let t = index as f32 / 44_100.0;
                (2.0 * std::f32::consts::PI * 997.0 * t).sin() * 0.2
            })
            .collect()
    }

    #[test]
    fn callback_chunking_does_not_change_the_result() {
        let input = signal(44_100 + 731);
        let mut one_shot = StreamingResampler::new(44_100, 48_000).unwrap().unwrap();
        let expected = one_shot.process(&input).unwrap();

        let mut chunked = StreamingResampler::new(44_100, 48_000).unwrap().unwrap();
        let mut actual = Vec::new();
        let mut start = 0;
        for size in [97_usize, 480, 333, 1024, 211, 997].into_iter().cycle() {
            if start >= input.len() {
                break;
            }
            let end = (start + size).min(input.len());
            actual.extend(chunked.process(&input[start..end]).unwrap());
            start = end;
        }

        assert_eq!(actual, expected);
        assert_eq!(chunked.pending_len(), one_shot.pending_len());
    }

    #[test]
    fn incomplete_callback_is_buffered_without_fake_silence() {
        let mut resampler = StreamingResampler::new(44_100, 48_000).unwrap().unwrap();
        assert!(resampler.process(&vec![0.25; 127]).unwrap().is_empty());
        assert_eq!(resampler.pending_len(), 127);
        assert!(resampler.process(&vec![0.25; 128]).unwrap().is_empty());
        assert_eq!(resampler.pending_len(), 255);
    }
}

//! Qwen's periodic-Hann STFT and Slaney mel projection.
use realfft::{RealFftPlanner, RealToComplex};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mel contract: {0}")]
    Contract(&'static str),
    #[error("FFT: {0}")]
    Fft(#[from] realfft::FftError),
}

#[derive(Deserialize)]
pub struct Filters {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub weights: Vec<Vec<f32>>,
}

pub struct Spectrogram {
    filters: Filters,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
}

impl Spectrogram {
    pub fn new(filters: Filters) -> Result<Self, Error> {
        if filters.n_fft == 0
            || filters.hop_length == 0
            || filters.hop_length > filters.n_fft
            || filters.weights.is_empty()
            || filters
                .weights
                .iter()
                .any(|row| row.len() != filters.n_fft / 2 + 1)
        {
            return Err(Error::Contract("invalid STFT dimensions or mel matrix"));
        }
        let fft = RealFftPlanner::new().plan_fft_forward(filters.n_fft);
        let window = (0..filters.n_fft)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / filters.n_fft as f32).cos())
            .collect();
        Ok(Self {
            filters,
            fft,
            window,
        })
    }
    pub fn sample_rate(&self) -> u32 {
        self.filters.sample_rate
    }
    pub fn channels(&self) -> usize {
        self.filters.weights.len()
    }
    /// Returns [frames, mel_channels] in row-major order, without padding the
    /// time axis of the speaker's global statistical pooling.
    pub fn compute(&self, audio: &[f32]) -> Result<Vec<f32>, Error> {
        let padding = (self.filters.n_fft - self.filters.hop_length) / 2;
        if audio.len() <= padding || audio.len() + 2 * padding < self.filters.n_fft {
            return Err(Error::Contract("audio is too short for reflection padding"));
        }
        if audio.iter().any(|x| !x.is_finite()) {
            return Err(Error::Contract("audio contains nonfinite samples"));
        }
        let frames = (audio.len() + 2 * padding - self.filters.n_fft) / self.filters.hop_length + 1;
        let mut input = self.fft.make_input_vec();
        let mut output = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut magnitude = vec![0.0; output.len()];
        let mut mel = Vec::with_capacity(frames * self.channels());
        for frame in 0..frames {
            for (index, value) in input.iter_mut().enumerate() {
                let position =
                    (frame * self.filters.hop_length + index) as isize - padding as isize;
                let reflected = if position < 0 {
                    -position
                } else if position >= audio.len() as isize {
                    2 * audio.len() as isize - 2 - position
                } else {
                    position
                };
                *value = audio[reflected as usize] * self.window[index];
            }
            self.fft
                .process_with_scratch(&mut input, &mut output, &mut scratch)?;
            for (value, complex) in magnitude.iter_mut().zip(&output) {
                // Qwen's published mel_spectrogram adds 1e-9 before sqrt.
                *value = (complex.norm_sqr() + 1e-9).sqrt();
            }
            for filter in &self.filters.weights {
                let energy: f32 = filter.iter().zip(&magnitude).map(|(a, b)| a * b).sum();
                // Qwen's dynamic_range_compression_torch clamps at 1e-5.
                mel.push(energy.max(1e-5).ln());
            }
        }
        Ok(mel)
    }
}

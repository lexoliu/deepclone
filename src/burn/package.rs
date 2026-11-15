//! Portable package metadata and native file loading.
use serde::Deserialize;
#[derive(Deserialize)]
pub struct Manifest {
    pub format: String,
    pub shards: Vec<Shard>,
}
#[derive(Deserialize)]
pub struct Shard {
    pub path: String,
    pub conditioning: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: crate::sampling::TokenSampler, S: crate::sampling::TokenSampler>
    crate::inference::Engine<super::Burn, T, S>
{
    /// Load a verified model package on its owning inference worker.
    pub async fn from_burn_package(
        path: &std::path::Path,
        capacity: std::num::NonZeroUsize,
        sampling: crate::inference::Sampling<T, S>,
    ) -> Result<Self, crate::inference::Error> {
        use super::{AudioConfig, Burn, SpeechConfig, UploadedWeights};
        use crate::inference::{Assets, AudioContract, Error};
        use burn::tensor::{DType, Device};
        fn json<T: serde::de::DeserializeOwned>(path: std::path::PathBuf) -> Result<T, Error> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        }
        let manifest: Manifest = json(path.join("manifest.json"))?;
        if manifest.format != "deepclone-burn-v1" {
            return Err(Error::Contract("unsupported Burn package format"));
        }
        let config: SpeechConfig = json(path.join("config.json"))?;
        let audio: AudioConfig = json(path.join("codec.json"))?;
        let encoder_contract = AudioContract {
            capacity: usize::MAX,
            sample_rate: audio.input_sample_rate,
            samples_per_frame: audio.encode_downsample_rate,
            code_groups: audio.encoder_valid_num_quantizers,
        };
        let decoder_contract = AudioContract {
            capacity: 325,
            sample_rate: audio.output_sample_rate,
            samples_per_frame: audio.decode_upsample_rate,
            code_groups: audio.encoder_valid_num_quantizers,
        };
        let device = Device::wgpu_options()
            .init_async()
            .await
            .map_err(|error| super::Error::Execution(error.to_string()))?;
        let mut weights = UploadedWeights::new(device, DType::F16);
        for shard in manifest.shards {
            if std::path::Path::new(&shard.path).components().count() != 1 {
                return Err(Error::Contract("shard path must be a filename"));
            }
            weights.upload(
                &std::fs::read(path.join(shard.path))?,
                if shard.conditioning {
                    DType::F32
                } else {
                    DType::F16
                },
            )?;
        }
        let backend = Burn::load(&weights, &weights, &weights, config, audio, capacity)?;
        Self::new(
            backend,
            Assets {
                weight_bits: 16,
                config: json(path.join("config.json"))?,
                encoder_contract,
                decoder_contract,
                filters: json(path.join("mel.json"))?,
                tokenizer: tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
                    .map_err(|e| Error::Tokenizer(e.to_string()))?,
            },
            sampling,
        )
    }
}

#![cfg(feature = "burn")]
use burn::tensor::{DType, Device};
use deepclone::burn::{Decoder, DecoderConfig, Weights, read};
use std::{num::NonZeroUsize, path::PathBuf};

#[test]
#[ignore = "requires pinned official weights and generated PyTorch decoder fixtures"]
fn wgpu_incremental_decoder_matches_official_multilingual_outputs() {
    check_decoder(DType::F32, "talker.model", "config.json", "");
}

#[test]
#[ignore = "requires pinned official weights and generated PyTorch decoder fixtures"]
fn fp16_decoder_is_finite() {
    check_decoder(DType::F16, "talker.model", "config.json", "");
}

#[test]
#[ignore = "requires pinned official weights and generated PyTorch decoder fixtures"]
fn predictor_matches_official_multilingual_outputs() {
    check_decoder(
        DType::F32,
        "talker.code_predictor.model",
        "predictor-config.json",
        "predictor_",
    );
}

#[test]
#[ignore = "requires pinned official weights and generated PyTorch decoder fixtures"]
fn fp16_predictor_is_finite() {
    check_decoder(
        DType::F16,
        "talker.code_predictor.model",
        "predictor-config.json",
        "predictor_",
    );
}

fn check_decoder(dtype: DType, prefix: &str, config_file: &str, fixture_prefix: &str) {
    futures_lite::future::block_on(async {
        let source = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_SOURCE").unwrap());
        let fixture = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_FIXTURES").unwrap());
        let device = Device::wgpu_options().init_async().await.unwrap();
        let bytes = std::fs::read(source.join("model.safetensors")).unwrap();
        let weights = Weights::new(&bytes, device.clone(), dtype).unwrap();
        let config: DecoderConfig =
            serde_json::from_slice(&std::fs::read(fixture.join(config_file)).unwrap()).unwrap();
        let mut decoder =
            Decoder::load(&weights, prefix, config, NonZeroUsize::new(64).unwrap()).unwrap();
        drop(weights);
        drop(bytes);
        let bytes = std::fs::read(fixture.join("decoder-fixtures.safetensors")).unwrap();
        let fixtures = Weights::new(&bytes, device, DType::F32).unwrap();
        for index in 0..4 {
            decoder.reset();
            let input = fixtures
                .tensor::<3>(&format!("{fixture_prefix}input.{index}"))
                .unwrap();
            let expected = read(
                fixtures
                    .tensor::<3>(&format!("{fixture_prefix}output.{index}"))
                    .unwrap(),
            )
            .await
            .unwrap();
            let [b, t, h] = input.dims();
            let mut actual = Vec::new();
            for position in 0..t {
                let hidden = decoder
                    .step(
                        input
                            .clone()
                            .slice([0..b, position..position + 1, 0..h])
                            .cast(dtype),
                    )
                    .unwrap();
                actual.extend(read(decoder.normalize(hidden)).await.unwrap());
            }
            assert!(
                actual.iter().all(|v| v.is_finite()),
                "nonfinite output for prompt {index}"
            );
            let maximum = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert_eq!(actual.len(), expected.len());
            if dtype == DType::F32 {
                assert!(maximum < 0.005, "prompt {index}: maximum error {maximum}");
            } else {
                eprintln!("FP16 prompt {index}: maximum error {maximum}");
            }
        }
    });
}

#[test]
#[ignore = "requires pinned multilingual embedding fixtures"]
fn uploaded_and_file_weights_preserve_fp16_precision() {
    futures_lite::future::block_on(async {
        let fixture =
            std::path::PathBuf::from(std::env::var_os("DEEPCLONE_BURN_FIXTURES").unwrap());
        let bytes = std::fs::read(fixture.join("decoder-fixtures.safetensors")).unwrap();
        let device = Device::wgpu_options().init_async().await.unwrap();
        let source = Weights::new(&bytes, device.clone(), DType::F16).unwrap();
        let mut uploaded = deepclone::burn::UploadedWeights::new(device, DType::F16);
        uploaded.upload(&bytes, DType::F16).unwrap();
        use deepclone::burn::WeightSource;
        let a = source.tensor::<3>("input.0").unwrap();
        let b = uploaded.tensor::<3>("input.0").unwrap();
        assert_eq!(a.dtype(), DType::F16);
        assert_eq!(b.dtype(), DType::F16);
        assert_eq!(read(a).await.unwrap(), read(b).await.unwrap());
    });
}

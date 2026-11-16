#![cfg(feature = "burn")]
use burn::tensor::{DType, Device, Tensor};
use deepclone::burn::{
    CodecDecoder, CodecDecoderConfig, CodecEncoder, EncoderConfig, Speaker, SpeakerConfig, Weights,
    read,
};
use std::path::PathBuf;

async fn compare<const D: usize, const E: usize>(
    actual: Tensor<D>,
    expected: Tensor<E>,
    name: &str,
) {
    let actual = read(actual).await.unwrap();
    let expected = read(expected).await.unwrap();
    assert_eq!(actual.len(), expected.len(), "{name} shape");
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(maximum < 1e-3, "{name}: maximum error {maximum}");
}
#[test]
#[ignore = "requires real reference audio fixtures and pinned official weights"]
fn wgpu_conditioning_and_codec_match_official_fp32() {
    futures_lite::future::block_on(async {
        let source = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_SOURCE").unwrap());
        let fixture = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_FIXTURES").unwrap());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.join("model-config.json")).unwrap())
                .unwrap();
        let codec: serde_json::Value = serde_json::from_slice(
            &std::fs::read(source.join("speech_tokenizer/config.json")).unwrap(),
        )
        .unwrap();
        let device = Device::wgpu_options().init_async().await.unwrap();
        let bytes = std::fs::read(fixture.join("audio-fixtures.safetensors")).unwrap();
        let data = Weights::new(&bytes, device.clone(), DType::F32).unwrap();
        let weights = std::fs::read(source.join("model.safetensors")).unwrap();
        let w = Weights::new(&weights, device.clone(), DType::F32).unwrap();
        let c: SpeakerConfig =
            serde_json::from_value(config["speaker_encoder_config"].clone()).unwrap();
        let speaker = Speaker::load(&w, &c).unwrap();
        drop(w);
        drop(weights);
        compare(
            speaker.forward(data.tensor("mel").unwrap()),
            data.tensor::<2>("speaker").unwrap(),
            "speaker",
        )
        .await;
        drop(speaker);
        let weights = std::fs::read(source.join("speech_tokenizer/model.safetensors")).unwrap();
        let w = Weights::new(&weights, device, DType::F32).unwrap();
        let c: EncoderConfig = serde_json::from_value(codec["encoder_config"].clone()).unwrap();
        let encoder = CodecEncoder::load(
            &w,
            &c,
            codec["encoder_valid_num_quantizers"].as_u64().unwrap() as usize,
            codec["encode_downsample_rate"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let actual = encoder
            .forward(data.tensor("audio").unwrap())
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec::<i32>()
            .unwrap();
        let expected = data
            .indices::<3>("codes")
            .unwrap()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec::<i32>()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        let differences: Vec<_> = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        drop(encoder);
        let c: CodecDecoderConfig =
            serde_json::from_value(codec["decoder_config"].clone()).unwrap();
        let decoder = CodecDecoder::load(&w, &c).unwrap();
        for frames in [8, 80] {
            compare(
                decoder.forward(data.indices(&format!("codes.{frames}")).unwrap()),
                data.tensor::<3>(&format!("waveform.{frames}")).unwrap(),
                "waveform",
            )
            .await;
        }
        assert!(
            differences.is_empty(),
            "reference acoustic code differences: {differences:?}"
        );
    });
}

#![cfg(all(feature = "burn", feature = "cli"))]
use deepclone::{
    inference::{Engine, Sampling},
    sampling::{Multinomial, Parameters, RepetitionPenalty},
};
use rand::{SeedableRng, rngs::StdRng};
use std::{num::NonZeroUsize, path::PathBuf};
#[test]
#[ignore = "requires verified model shards and a private reference recording"]
fn cloned_speech_completes_in_four_languages() {
    futures_lite::future::block_on(async {
        let package = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_PACKAGE").unwrap());
        let reference = PathBuf::from(std::env::var_os("DEEPCLONE_REFERENCE").unwrap());
        let output = PathBuf::from(std::env::var_os("DEEPCLONE_OUTPUT").unwrap());
        std::fs::create_dir_all(&output).unwrap();
        let mut reader = hound::WavReader::open(reference.join("reference.wav")).unwrap();
        let samples = reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let transcript: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference.join("reference.json")).unwrap())
                .unwrap();
        let parameters = Parameters::new(0.9, NonZeroUsize::new(50), 1.0).unwrap();
        let sampling = Sampling {
            talker: Multinomial::new(StdRng::seed_from_u64(42), parameters),
            predictor: Multinomial::new(StdRng::seed_from_u64(43), parameters),
            repetition: RepetitionPenalty::new(1.05).unwrap(),
            minimum_frames: 2,
        };
        let mut engine =
            Engine::from_burn_package(&package, NonZeroUsize::new(2048).unwrap(), sampling)
                .await
                .unwrap();
        let voice = engine
            .reference(
                &samples,
                reader.spec().sample_rate,
                transcript["text"].as_str().unwrap(),
            )
            .await
            .unwrap();
        for (language, text) in [
            ("Chinese", "愿每个人都能安心生活，带着信任走向明天。"),
            (
                "English",
                "Let us take a quiet walk and decide what comes next.",
            ),
            (
                "Japanese",
                "今日は静かな道を歩きながら、明日のことを考えましょう。",
            ),
            ("Korean", "오늘은 조용한 길을 걸으며 내일을 생각해 봅시다."),
        ] {
            let result = engine
                .synthesize(&voice, text, language, 512)
                .await
                .unwrap();
            assert!(!result.waveform.samples.is_empty());
            assert!(result.waveform.samples.iter().all(|x| x.is_finite()));
            assert!(result.waveform.samples.iter().any(|x| x.abs() > 0.01));
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: result.waveform.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer =
                hound::WavWriter::create(output.join(format!("{language}.wav")), spec).unwrap();
            for sample in result.waveform.samples {
                writer.write_sample(sample).unwrap();
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
            std::fs::write(
                output.join(format!("{language}.json")),
                serde_json::to_vec(&result.generation).unwrap(),
            )
            .unwrap();
        }
    });
}

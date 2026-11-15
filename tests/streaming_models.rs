#![cfg(all(feature = "cli", target_vendor = "apple"))]

use deepclone::{
    inference::Sampling,
    sampling::{Greedy, Multinomial, Parameters, RepetitionPenalty},
    streaming::{Load, Options, ReferenceAudio, Voice},
};
use futures_lite::{StreamExt, stream};
use ondevice_coreml::ComputeUnits;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::{
    convert::Infallible,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

fn directory(name: &str) -> PathBuf {
    std::env::var_os(name).expect(name).into()
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
}
fn load() -> Load {
    let reference = directory("DEEPCLONE_REFERENCE");
    let mut reader = hound::WavReader::open(reference.join("reference.wav")).unwrap();
    let sample_rate = reader.spec().sample_rate;
    assert_eq!(reader.spec().channels, 1);
    let samples = reader.samples::<f32>().map(Result::unwrap).collect();
    let transcript: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("reference.json")).unwrap()).unwrap();
    Load {
        transformers: directory("DEEPCLONE_TRANSFORMERS"),
        audio_models: directory("DEEPCLONE_AUDIO_MODELS"),
        compute_units: ComputeUnits::CpuAndNeuralEngine,
        reference: ReferenceAudio {
            samples,
            sample_rate,
            transcript: transcript["text"].as_str().unwrap().to_owned(),
        },
    }
}
fn options() -> Options {
    let mut options = Options::new(NonZeroUsize::new(512).unwrap());
    options.text_chunk_chars = NonZeroUsize::new(1).unwrap();
    options
}
fn save(path: &Path, samples: &[f32], sample_rate: u32) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

#[derive(Deserialize)]
struct Baseline {
    prompt: Prompt,
    file: String,
}
#[derive(Deserialize)]
struct Prompt {
    language: String,
    text: String,
}

#[test]
#[ignore = "requires published models and local real-voice baseline fixtures"]
fn multilingual_streams_match_validated_codes_and_waveforms() {
    trace_run(async {
        let baseline = directory("DEEPCLONE_BASELINE");
        let output = directory("DEEPCLONE_STREAM_OUTPUT");
        std::fs::create_dir_all(&output).unwrap();
        let prompts: Vec<Baseline> =
            serde_json::from_slice(&std::fs::read(baseline.join("samples.json")).unwrap()).unwrap();
        let parameters = Parameters::new(0.9, NonZeroUsize::new(50), 1.0).unwrap();
        let mut voice = Voice::load(
            load(),
            Sampling {
                talker: Multinomial::new(StdRng::seed_from_u64(42), parameters),
                predictor: Multinomial::new(StdRng::seed_from_u64(42), parameters),
                repetition: RepetitionPenalty::new(1.05).unwrap(),
                minimum_frames: 2,
            },
        )
        .await
        .unwrap();
        let mut reports = Vec::new();
        for item in prompts {
            let parts: Vec<_> = item
                .prompt
                .text
                .chars()
                .map(|ch| Ok::<_, Infallible>(ch.to_string()))
                .collect();
            let result = voice
                .synthesize(stream::iter(parts), &item.prompt.language, options())
                .await
                .unwrap()
                .collect_audio()
                .await
                .unwrap();
            let codes_path = Path::new(&item.file).with_extension("codes.json");
            let expected: serde_json::Value =
                serde_json::from_slice(&std::fs::read(baseline.join(codes_path)).unwrap()).unwrap();
            assert_eq!(
                serde_json::to_value(&result.generation.codes).unwrap(),
                expected["codes"],
                "{} acoustic codes",
                item.prompt.language
            );
            let mut reader = hound::WavReader::open(baseline.join(&item.file)).unwrap();
            let reference: Vec<_> = reader.samples::<f32>().map(Result::unwrap).collect();
            assert_eq!(reference.len(), result.waveform.samples.len());
            let maximum = reference
                .iter()
                .zip(&result.waveform.samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let mse = reference
                .iter()
                .zip(&result.waveform.samples)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / reference.len() as f64;
            reports.push(serde_json::json!({"language":item.prompt.language,"identical_codes":true,"samples":reference.len(),"maximum_absolute_error":maximum,"rmse":mse.sqrt()}));
            std::fs::write(
                output.join("comparison.json"),
                serde_json::to_vec_pretty(&reports).unwrap(),
            )
            .unwrap();
            save(
                &output.join(&item.file),
                &result.waveform.samples,
                result.waveform.sample_rate,
            );
            // Same Core ML graphs; allow only sub-milliscale float16 rounding.
            assert!(maximum < 1e-3, "waveform prefix changed by {maximum}");
        }
    });
}

#[test]
#[ignore = "requires published models and local real-voice reference audio"]
fn audio_before_eof_matches_whole_text_prefix_and_cancels_cleanly() {
    trace_run(async {
        let mut voice = Voice::load(
            load(),
            Sampling {
                talker: Greedy,
                predictor: Greedy,
                repetition: RepetitionPenalty::new(1.05).unwrap(),
                minimum_frames: 2,
            },
        )
        .await
        .unwrap();
        // Enough stable text to fill the reference-conditioned prefill while
        // the producer deliberately keeps the input stream open.
        let text = streaming_text();
        let (tx, rx) = async_channel::bounded(1);
        tx.send(Ok::<_, Infallible>(text.to_owned())).await.unwrap();
        let mut audio = voice.synthesize(rx, "English", options()).await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(60), audio.next())
            .await
            .expect("first audio while input remains open")
            .unwrap()
            .unwrap();
        assert_eq!(first.start_sample, 0);
        assert!(!first.samples.is_empty());
        // Cancellation must release both a waiting producer and the worker,
        // so the same voice can immediately start another independent request.
        drop(audio);
        drop(tx);
        let mut whole = voice
            .synthesize(
                stream::once(Ok::<_, Infallible>(text.to_owned())),
                "English",
                options(),
            )
            .await
            .unwrap();
        let expected = tokio::time::timeout(Duration::from_secs(60), whole.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first.samples.len(), expected.samples.len());
        let maximum = first
            .samples
            .iter()
            .zip(&expected.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            maximum < 1e-3,
            "EOF timing changed waveform prefix by {maximum}"
        );
        drop(whole);
        let output = directory("DEEPCLONE_STREAM_OUTPUT");
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(output.join("before-eof.json"), serde_json::to_vec_pretty(&serde_json::json!({"audio_before_input_eof":true,"cancelled_voice_reused":true,"maximum_absolute_error":maximum,"prefix_samples":first.samples.len()})).unwrap()).unwrap();
    });
}

fn streaming_text() -> &'static str {
    "We walk; we talk; we pause; we listen; we smile; we stop; we look; we breathe; we learn; we sing; we laugh; we rest. We walk; we talk; we pause; we listen; we smile; we stop; we look; we breathe; we learn; we sing; we laugh; we rest. We walk; we talk; we pause; we listen; we smile; we stop; we look; we breathe; we learn; we sing; we laugh; we rest. We walk; we talk; we pause; we listen; we smile; we stop; we look; we breathe; we learn; we sing; we laugh; we rest."
}

#[test]
#[ignore = "requires the published tokenizer and local reference fixtures"]
fn streaming_prompt_covers_conditioning_prefix() {
    let load = load();
    let tokenizer =
        tokenizers::Tokenizer::from_file(load.transformers.join("tokenizer.json")).unwrap();
    let reference = tokenizer
        .encode(load.reference.transcript, false)
        .unwrap()
        .len();
    let target = tokenizer.encode(streaming_text(), false).unwrap().len();
    let code: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory("DEEPCLONE_BASELINE").join("English.codes.json")).unwrap(),
    )
    .unwrap();
    let frames = code["codes"]["reference_frames"].as_u64().unwrap() as usize;
    let output = directory("DEEPCLONE_STREAM_OUTPUT");
    std::fs::create_dir_all(&output).unwrap();
    std::fs::write(output.join("token-budget.json"), serde_json::to_vec_pretty(&serde_json::json!({"reference_tokens":reference,"target_tokens":target,"reference_frames":frames})).unwrap()).unwrap();
    assert!(
        reference + target > frames + options().audio_chunk_frames.get() + 2,
        "reference {reference} + target {target}, reference frames {frames}"
    );
}

fn trace_run(future: impl std::future::Future<Output = ()>) {
    tracing::subscriber::with_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(std::io::stderr)
            .finish(),
        || runtime().block_on(future),
    );
}

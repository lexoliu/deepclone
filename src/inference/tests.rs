use super::*;
use crate::{generation::Model, sampling::Greedy};

#[cfg(all(feature = "cli", target_vendor = "apple"))]
#[test]
#[ignore = "requires the staged Core ML model and a real recorded reference"]
fn real_coreml_voice_delivery_is_sustained() {
    use crate::{
        sampling::{Multinomial, Parameters},
        streaming::{Load, Options, ReferenceAudio, Voice},
    };
    use futures_lite::{StreamExt, stream};
    use ondevice_coreml::ComputeUnits;
    use rand::{SeedableRng, rngs::StdRng};
    use std::{path::PathBuf, time::Instant};

    #[derive(Deserialize)]
    struct Prompt {
        text: String,
        language: String,
    }

    #[derive(serde::Serialize)]
    struct Delivery {
        load_and_condition_seconds: f64,
        first_audio_seconds: f64,
        steady_rtf: Option<f64>,
        playback_gap_seconds: f64,
        audio_seconds: f64,
        generated_frames: usize,
        arrivals: Vec<(f64, f64)>,
    }

    let directory = |key| PathBuf::from(std::env::var_os(key).expect(key));
    let output = directory("DEEPCLONE_OUTPUT");
    std::fs::create_dir_all(&output).unwrap();
    let (samples, sample_rate, transcript) = recorded_reference(&directory("DEEPCLONE_REFERENCE"));
    let prompts: Vec<Prompt> =
        serde_json::from_slice(&std::fs::read(directory("DEEPCLONE_PROMPTS")).unwrap()).unwrap();
    let (long, auditions) = prompts.split_first().unwrap();
    let max_frames = NonZeroUsize::new(512).unwrap();
    let parameters = Parameters::new(0.9, NonZeroUsize::new(50), 1.).unwrap();

    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
        .block_on(async {
            let load = Instant::now();
            let mut voice = Voice::load(
                Load {
                    transformers: directory("DEEPCLONE_TRANSFORMERS"),
                    audio_models: directory("DEEPCLONE_AUDIO_MODELS"),
                    compute_units: ComputeUnits::CpuAndNeuralEngine,
                    reference: ReferenceAudio {
                        samples,
                        sample_rate,
                        transcript,
                    },
                },
                Sampling {
                    talker: Multinomial::new(StdRng::seed_from_u64(42), parameters),
                    predictor: Multinomial::new(StdRng::seed_from_u64(42), parameters),
                    repetition: RepetitionPenalty::new(1.05).unwrap(),
                    minimum_frames: 2,
                },
            )
            .await
            .unwrap();
            let load_and_condition_seconds = load.elapsed().as_secs_f64();

            let mut measure = async |prompt: &Prompt, stem: &str| {
                let started = Instant::now();
                let input = stream::once(Ok::<_, std::convert::Infallible>(prompt.text.clone()));
                let mut audio = voice
                    .synthesize(input, &prompt.language, Options::new(max_frames))
                    .await
                    .unwrap();
                let mut arrivals = Vec::new();
                let mut samples = Vec::new();
                while let Some(chunk) = audio.next().await {
                    let chunk = chunk.unwrap();
                    assert_eq!(chunk.start_sample, samples.len());
                    assert_eq!(chunk.sample_rate, sample_rate);
                    arrivals.push((
                        started.elapsed().as_secs_f64(),
                        chunk.samples.len() as f64 / f64::from(sample_rate),
                    ));
                    samples.extend(chunk.samples);
                }
                let generation = audio.take_generation().unwrap();
                let first = arrivals.first().unwrap().0;
                let last = arrivals.last().unwrap().0;
                let audio_seconds = samples.len() as f64 / f64::from(sample_rate);
                let mut playback = first;
                let mut gaps = 0.0;
                for &(arrival, duration) in &arrivals {
                    gaps += (arrival - playback).max(0.0);
                    playback = playback.max(arrival) + duration;
                }
                let steady_rtf =
                    (arrivals.len() > 1).then(|| (last - first) / (audio_seconds - arrivals[0].1));
                let report = Delivery {
                    load_and_condition_seconds,
                    first_audio_seconds: first,
                    steady_rtf,
                    playback_gap_seconds: gaps,
                    audio_seconds,
                    generated_frames: generation.generated_frames(),
                    arrivals,
                };
                std::fs::write(
                    output.join(format!("{stem}.json")),
                    serde_json::to_vec_pretty(&report).unwrap(),
                )
                .unwrap();
                std::fs::write(
                    output.join(format!("{stem}.codes.json")),
                    serde_json::to_vec(&generation).unwrap(),
                )
                .unwrap();
                let mut writer = hound::WavWriter::create(
                    output.join(format!("{stem}.wav")),
                    hound::WavSpec {
                        channels: 2,
                        sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .unwrap();
                for sample in samples {
                    writer.write_sample(sample).unwrap();
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
                (report, generation)
            };
            let (report, generation) = measure(long, "native-pipeline").await;
            for (index, prompt) in auditions.iter().enumerate() {
                measure(
                    prompt,
                    &format!("audition-{}-{}", index + 1, prompt.language),
                )
                .await;
            }
            assert!(generation.generated_frames() > max_frames.get());
            assert!(generation.segments.len() > 1);
            let rtf = report.steady_rtf.unwrap();
            let gaps = report.playback_gap_seconds;
            assert!(
                rtf < 1.0 && gaps == 0.0,
                "RTF {rtf:.4}, playback gaps {gaps:.4} s"
            );
        });
}

struct TestModel {
    eos: bool,
    ignore_eos: bool,
    generation_resets: usize,
    frame_inputs: Vec<Vec<(Vec<u32>, u32)>>,
}
impl Model for TestModel {
    type Hidden = u32;
    type Token = u32;
    type Logits = Vec<f32>;
    type Codec = TestCodec;
    fn code_token(&mut self, _: usize, token: &u32) -> Result<u32, Error> {
        Ok(*token)
    }
    async fn read_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>, Error> {
        Ok(tokens.to_vec())
    }
    fn groups(&self) -> usize {
        2
    }
    fn context_capacity(&self) -> usize {
        96
    }
    fn reset_generation(&mut self) {
        self.eos = false;
        self.generation_resets += 1;
        self.frame_inputs.push(Vec::new());
    }
    fn reset_predictor(&mut self) {}
    fn text(&mut self, token: u32) -> Result<u32, Error> {
        Ok(token)
    }
    fn code(&mut self, _: usize, token: u32) -> Result<u32, Error> {
        Ok(token)
    }
    fn frame(&mut self, codes: &[u32], text: &u32) -> Result<u32, Error> {
        self.frame_inputs
            .last_mut()
            .unwrap()
            .push((codes.to_vec(), *text));
        self.eos |= *text == 1000;
        Ok(*text)
    }
    fn add(a: &u32, _: &u32) -> Result<u32, Error> {
        Ok(*a)
    }
    fn talker(&mut self, hidden: u32) -> Result<u32, Error> {
        Ok(hidden)
    }
    async fn talker_head(&mut self, hidden: u32) -> Result<(u32, Vec<f32>), Error> {
        let logits = if self.eos && !self.ignore_eos {
            vec![f32::NEG_INFINITY, 0.]
        } else {
            vec![0., f32::NEG_INFINITY]
        };
        Ok((hidden, logits))
    }
    fn project(&mut self, hidden: u32) -> Result<u32, Error> {
        Ok(hidden)
    }
    fn predictor(&mut self, hidden: u32) -> Result<u32, Error> {
        Ok(hidden)
    }
    async fn predictor_head(&mut self, _: usize, _: u32) -> Result<Vec<f32>, Error> {
        Ok(vec![0., f32::NEG_INFINITY])
    }
    async fn condition(
        &mut self,
        _: &[f32],
        _: &[f32],
        _: usize,
    ) -> Result<(u32, Vec<Vec<u32>>), Error> {
        Err(Error::Contract("test supplies a conditioned reference"))
    }
}
#[derive(Default)]
struct TestCodec {
    waveform_resets: usize,
    waveform_inputs: Vec<Vec<i32>>,
    fail: bool,
}
impl Codec for TestCodec {
    fn reset(&mut self) {
        self.waveform_resets += 1;
    }
    fn input_frames(&self, _valid: usize, capacity: usize) -> usize {
        capacity
    }
    async fn decode_prefix(
        &mut self,
        codes: Vec<i32>,
        _: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        if self.fail {
            return Err(Error::Contract("waveform fixture failure"));
        }
        self.waveform_inputs.push(codes);
        // An absolute index within the supplied causal prefix exposes decoding
        // that restarts at a generation boundary instead of a codec boundary.
        Ok((0..frames).flat_map(|i| [i as f32; 2]).collect())
    }
}
fn engine() -> Engine<TestModel, Greedy, Greedy> {
    // Test-only vocabulary: codec 0, EOS 1, then 1024 reserved IDs.
    Engine::new(
        TestModel {
            eos: false,
            ignore_eos: false,
            generation_resets: 0,
            frame_inputs: Vec::new(),
        },
        TestCodec::default(),
        Assets {
            weight_bits: 16,
            config: Config {
                tts_model_size: "test".into(),
                tts_bos_token_id: 1002,
                tts_eos_token_id: 1000,
                tts_pad_token_id: 1001,
                talker_config: TalkerConfig {
                    hidden_size: 1,
                    vocab_size: 1026,
                    num_code_groups: 2,
                    codec_bos_id: 0,
                    codec_pad_id: 0,
                    codec_eos_token_id: 1,
                    codec_think_id: 0,
                    codec_nothink_id: 0,
                    codec_think_bos_id: 0,
                    codec_think_eos_id: 0,
                    codec_language_id: BTreeMap::new(),
                },
            },
            encoder_contract: AudioContract {
                capacity: 8,
                sample_rate: 25,
                samples_per_frame: 2,
                code_groups: 2,
            },
            decoder_contract: AudioContract {
                capacity: CODEC_CAPACITY,
                sample_rate: 25,
                samples_per_frame: 2,
                code_groups: 2,
            },
            filters: crate::mel::Filters {
                sample_rate: 25,
                n_fft: 2,
                hop_length: 1,
                weights: vec![vec![1., 1.]],
            },
            tokenizer: crate::text::tests::tokenizer(),
        },
        Sampling {
            talker: Greedy,
            predictor: Greedy,
            repetition: RepetitionPenalty::new(1.).unwrap(),
            minimum_frames: 0,
        },
    )
    .unwrap()
}
fn reference() -> Reference<u32> {
    Reference {
        codes: vec![vec![0, 0]; 2],
        speaker: 0,
        text: vec![0],
    }
}

#[cfg(all(feature = "streaming", not(target_arch = "wasm32")))]
pub(crate) fn failing_waveform_fixture() -> (
    Engine<impl Model<Hidden = u32, Codec: Send>, Greedy, Greedy>,
    Reference<u32>,
) {
    let mut engine = engine();
    engine.codec.fail = true;
    (engine, reference())
}

fn options() -> SynthesisOptions {
    SynthesisOptions::new(NonZeroUsize::new(48).unwrap())
}

#[test]
fn long_text_exceeds_one_frame_budget_and_decodes_every_segment() {
    futures_lite::future::block_on(async {
        let mut engine = engine();
        let reference = reference();
        let text = "Hello there. ".repeat(40);
        let result = engine
            .synthesize(&reference, &text, "auto", options())
            .await
            .unwrap();
        assert!(result.generation.segments.len() > 1);
        assert!(result.generation.segments.len() < 40);
        assert!(result.generation.generated_frames() > options().maximum_frames.get());
        assert_eq!(
            result.waveform.samples.len(),
            result.generation.generated_frames() * 2
        );
        assert_eq!(result.generation.codes.reference_frames(), 2);
        assert_eq!(engine.codec.waveform_resets, 1);
        assert_eq!(
            engine.generator.networks.generation_resets,
            result.generation.segments.len()
        );
        for (index, span) in result.generation.segments.iter().enumerate().skip(1) {
            let previous = &result.generation.segments[index - 1];
            let expected =
                &result.generation.codes.values()[previous.start_frame * 2..previous.end_frame * 2];
            let actual: Vec<_> = engine.generator.networks.frame_inputs[index]
                .iter()
                .take(previous.end_frame - previous.start_frame)
                .flat_map(|(codes, _)| codes.iter().copied())
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(span.start_frame, previous.end_frame);
        }
        let decoded = engine.decode(&result.generation.codes).await.unwrap();
        assert_eq!(decoded.samples, result.waveform.samples);
        let exported = serde_json::to_value(&result.generation).unwrap();
        assert_eq!(
            exported["segments"].as_array().unwrap().len(),
            result.generation.segments.len()
        );
    });
}
#[test]
fn frame_budget_delivers_partial_chunk_then_error_and_eos_at_limit_succeeds() {
    futures_lite::future::block_on(async {
        let mut engine = engine();
        let reference = reference();
        let text = "Hello there.";
        let expected = engine
            .generate(&reference, text, "auto", options())
            .await
            .unwrap();
        let frames = expected.generated_frames();
        let mut utterance = engine
            .session(
                &reference,
                "auto",
                SynthesisOptions::new(NonZeroUsize::new(frames).unwrap()),
            )
            .unwrap();
        utterance.push_text(text).unwrap();
        utterance.finish_input().unwrap();
        let chunk = utterance.next_chunk(12, || false).await.unwrap().unwrap();
        assert_eq!(chunk.samples.len(), frames * 2);
        assert!(utterance.next_chunk(12, || false).await.unwrap().is_none());
        assert_eq!(
            utterance.into_generation().unwrap().generated_frames(),
            frames
        );
        engine.generator.networks.ignore_eos = true;
        let mut utterance = engine
            .session(
                &reference,
                "auto",
                SynthesisOptions::new(NonZeroUsize::new(5).unwrap()),
            )
            .unwrap();
        utterance.push_text(text).unwrap();
        utterance.finish_input().unwrap();
        let chunk = utterance.next_chunk(12, || false).await.unwrap().unwrap();
        assert_eq!(chunk.samples.len(), 10);
        assert!(
            matches!(utterance.next_chunk(12, || false).await, Err(Error::FrameBudget { limit: 5, codes }) if codes.generated_frames() == 5)
        );
    });
}

#[cfg(feature = "streaming")]
struct TestClock;
#[cfg(feature = "streaming")]
impl crate::streaming::Clock for TestClock {
    fn now(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
    async fn sleep(&self, _: std::time::Duration) {
        std::future::pending::<()>().await;
    }
}
#[cfg(feature = "streaming")]
#[test]
fn shared_driver_emits_before_input_eof_and_keeps_global_positions() {
    use crate::streaming::{self, Execution, Options};
    futures_lite::future::block_on(async {
        let mut engine = engine();
        let reference = reference();
        let text = "Hello there. ".repeat(40);
        let mut policy = Options::new(options().maximum_frames);
        policy.audio_chunk_frames = NonZeroUsize::new(3).unwrap();
        let mut samples = Vec::new();
        let result = streaming::run(
            &mut engine,
            &reference,
            futures_lite::stream::once(Ok(text)),
            "auto",
            policy.clone(),
            Execution {
                clock: TestClock,
                cancelled: || false,
            },
            async |chunk| {
                assert_eq!(chunk.start_sample, samples.len());
                samples.extend(chunk.samples);
                Ok(())
            },
        )
        .await
        .unwrap();
        assert!(result.generated_frames() > policy.generation.maximum_frames.get());
        assert_eq!(samples.len(), result.generated_frames() * 2);
        let (send, receive) = async_channel::bounded(1);
        send.send(Ok("Hello there. Next.".to_owned()))
            .await
            .unwrap();
        let mut received = false;
        let result = streaming::run(
            &mut engine,
            &reference,
            receive,
            "auto",
            policy,
            Execution {
                clock: TestClock,
                cancelled: || false,
            },
            async |_| {
                received = true;
                Err(Error::Callback("consumer stopped before text EOF".into()))
            },
        )
        .await;
        assert!(received && matches!(result, Err(Error::Callback(_))));
        assert!(send.is_closed());
    });
}

#[test]
fn sentences_share_context_and_paragraphs_roll_over_without_resetting_codec() {
    futures_lite::future::block_on(async {
        let mut engine = engine();
        let reference = reference();
        let text = "Hello. Next!\n\nAgain. Done!";
        let result = engine
            .synthesize(&reference, text, "auto", options())
            .await
            .unwrap();
        assert_eq!(result.generation.segments.len(), 2);
        assert_eq!(engine.generator.networks.generation_resets, 2);
        assert_eq!(engine.codec.waveform_resets, 1);
        let first = &result.generation.segments[0];
        assert_eq!(
            first.text_tokens,
            engine.tokenize("Hello. Next!\n\n").unwrap()
        );
        let second = &result.generation.segments[1];
        assert_eq!(second.text_tokens, engine.tokenize("Again. Done!").unwrap());
        let prompt_frames = first.end_frame - first.start_frame;
        let actual: Vec<_> = engine.generator.networks.frame_inputs[1]
            .iter()
            .take(prompt_frames)
            .map(|(_, text)| *text)
            .collect();
        let expected: Vec<_> = first
            .text_tokens
            .iter()
            .skip(1)
            .chain(&second.text_tokens)
            .copied()
            .chain([1000])
            .chain(std::iter::repeat(1001))
            .take(prompt_frames)
            .collect();
        assert_eq!(actual, expected);
    });
}

#[test]
fn incremental_cjk_and_english_inputs_preserve_context_boundaries_and_pcm() {
    futures_lite::future::block_on(async {
        for text in [
            "早晨好。一起走吧。明天再见。",
            "Good morning. Walk with me! See you soon.",
            "おはよう。一緒に歩こう。また明日。",
            "좋은 아침. 같이 걸어요. 내일 봐요.",
        ] {
            let mut engine = engine();
            let reference = reference();
            let expected = engine
                .synthesize(&reference, text, "auto", options())
                .await
                .unwrap();
            let mut session = engine.session(&reference, "auto", options()).unwrap();
            let mut samples = Vec::new();
            for ch in text.chars() {
                session.push_text(&ch.to_string()).unwrap();
                while let Some(chunk) = session.next_chunk(3, || false).await.unwrap() {
                    assert_eq!(chunk.start_sample, samples.len());
                    samples.extend(chunk.samples);
                }
            }
            session.finish_input().unwrap();
            while let Some(chunk) = session.next_chunk(3, || false).await.unwrap() {
                assert_eq!(chunk.start_sample, samples.len());
                samples.extend(chunk.samples);
            }
            let actual = session.into_generation().unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(expected.generation).unwrap(),
                "{text}"
            );
            assert_eq!(samples, expected.waveform.samples, "{text}");
        }
    });
}

#[cfg(all(feature = "cli", target_vendor = "apple"))]
fn recorded_reference(path: &std::path::Path) -> (Vec<f32>, u32, String) {
    let mut reader = hound::WavReader::open(path.join("reference.wav")).unwrap();
    let samples = reader
        .samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let transcript: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("reference.json")).unwrap()).unwrap();
    (
        samples,
        reader.spec().sample_rate,
        transcript["text"].as_str().unwrap().to_owned(),
    )
}

#[cfg(all(feature = "cli", target_vendor = "apple"))]
#[test]
#[ignore = "requires Core ML models and real growing-prefix decoder inputs"]
fn real_coreml_waveform_replays_prediction_inputs() {
    use crate::sampling::Greedy;
    use ondevice_coreml::ComputeUnits;
    use std::path::PathBuf;
    #[derive(serde::Deserialize)]
    struct Call {
        context: usize,
        frames: usize,
        values: Vec<i32>,
    }
    futures_lite::future::block_on(async {
        let directory = |key| PathBuf::from(std::env::var_os(key).expect(key));
        let output = directory("DEEPCLONE_OUTPUT");
        std::fs::create_dir_all(&output).unwrap();
        let started = std::time::Instant::now();
        let mut engine = Engine::open(
            &directory("DEEPCLONE_TRANSFORMERS"),
            &directory("DEEPCLONE_AUDIO_MODELS"),
            ComputeUnits::CpuAndNeuralEngine,
            Sampling {
                talker: Greedy,
                predictor: Greedy,
                repetition: RepetitionPenalty::new(1.05).unwrap(),
                minimum_frames: 2,
            },
        )
        .unwrap();
        let mut reports = vec![serde_json::json!({
            "stage": "load", "seconds": started.elapsed().as_secs_f64()
        })];
        let reference = if let Some(path) = std::env::var_os("DEEPCLONE_REFERENCE") {
            let (samples, sample_rate, text) = recorded_reference(&PathBuf::from(path));
            let start = std::time::Instant::now();
            let reference = engine
                .reference(&samples, sample_rate, &text)
                .await
                .unwrap();
            reports.push(serde_json::json!({
                "stage": "reference", "seconds": start.elapsed().as_secs_f64()
            }));
            Some(reference)
        } else {
            None
        };
        let calls: Vec<Call> =
            serde_json::from_slice(&std::fs::read(directory("DEEPCLONE_COREML_INPUTS")).unwrap())
                .unwrap();
        let contract = &engine.decoder_contract;
        let mut context = None;
        for (index, call) in calls.into_iter().enumerate() {
            if context != Some(call.context) {
                engine.codec.reset();
                context = Some(call.context);
            }
            let start = std::time::Instant::now();
            let result = engine
                .codec
                .decode_prefix(call.values, contract.code_groups, call.frames)
                .await;
            reports.push(serde_json::json!({
                "index": index,
                "seconds": start.elapsed().as_secs_f64(),
                "samples": result.as_ref().ok().map(Vec::len),
                "error": result.as_ref().err().map(ToString::to_string),
            }));
            std::fs::write(
                output.join("replay.json"),
                serde_json::to_vec(&reports).unwrap(),
            )
            .unwrap();
            let samples = result.unwrap();
            assert_eq!(samples.len(), call.frames * contract.samples_per_frame);
            assert!(samples.iter().all(|v| v.is_finite()));
        }
        drop(reference);
    });
}

#[cfg(all(feature = "cli", target_vendor = "apple"))]
#[test]
#[ignore = "requires Core ML models, a private recording and a long-form prompt"]
fn real_coreml_long_stream_preserves_codes_and_contiguous_pcm() {
    use crate::{
        profiling::{Counter, TimedSampler},
        sampling::{Multinomial, Parameters},
    };
    use ondevice_coreml::ComputeUnits;
    use rand::{SeedableRng, rngs::StdRng};
    use std::path::PathBuf;
    futures_lite::future::block_on(async {
        let directory = |key| PathBuf::from(std::env::var_os(key).expect(key));
        let output = directory("DEEPCLONE_OUTPUT");
        std::fs::create_dir_all(&output).unwrap();
        let started = std::time::Instant::now();
        let progress = |stage: &str, frames: usize, samples: usize| {
            std::fs::write(
                output.join("progress.json"),
                serde_json::to_vec(&serde_json::json!({
                    "stage": stage,
                    "seconds": started.elapsed().as_secs_f64(),
                    "frames": frames,
                    "samples": samples,
                }))
                .unwrap(),
            )
            .unwrap();
        };
        let reference_path = directory("DEEPCLONE_REFERENCE");
        let (samples, sample_rate, text) = recorded_reference(&reference_path);
        let prompts: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(directory("DEEPCLONE_PROMPTS")).unwrap())
                .unwrap();
        let parameters = Parameters::new(0.9, NonZeroUsize::new(50), 1.).unwrap();
        progress("load", 0, 0);
        let mut talker_sampling = Counter::default();
        let mut predictor_sampling = Counter::default();
        let mut engine = Engine::open(
            &directory("DEEPCLONE_TRANSFORMERS"),
            &directory("DEEPCLONE_AUDIO_MODELS"),
            ComputeUnits::CpuAndNeuralEngine,
            Sampling {
                talker: TimedSampler {
                    inner: Multinomial::new(StdRng::seed_from_u64(42), parameters),
                    counter: &mut talker_sampling,
                },
                predictor: TimedSampler {
                    inner: Multinomial::new(StdRng::seed_from_u64(42), parameters),
                    counter: &mut predictor_sampling,
                },
                repetition: RepetitionPenalty::new(1.05).unwrap(),
                minimum_frames: 2,
            },
        )
        .unwrap();
        let load_seconds = started.elapsed().as_secs_f64();
        let condition_started = std::time::Instant::now();
        progress("reference", 0, 0);
        let reference = engine
            .reference(&samples, sample_rate, &text)
            .await
            .unwrap();
        let conditioning_seconds = condition_started.elapsed().as_secs_f64();
        let synthesis_started = std::time::Instant::now();
        let maximum = NonZeroUsize::new(512).unwrap(); // Former total stream frame limit.
        let mut session = engine
            .session(
                &reference,
                prompts[0]["language"].as_str().unwrap(),
                SynthesisOptions::new(maximum),
            )
            .unwrap();
        session
            .push_text(prompts[0]["text"].as_str().unwrap())
            .unwrap();
        session.finish_input().unwrap();
        let mut samples = Vec::new();
        progress("synthesis", session.producer.codes.frames(), 0);
        let mut chunks = Vec::new();
        let mut playback_end = 0.0f64;
        let mut gaps = 0.0;
        let mut first_audio_seconds = None;
        while !session.producer.finished {
            let before = session
                .producer
                .generator
                .networks
                .profile
                .with(&session.codec.profile);
            let chunk_started = std::time::Instant::now();
            match session.next_chunk(12, || false).await {
                Ok(Some(chunk)) => {
                    let production_seconds = chunk_started.elapsed().as_secs_f64();
                    let delivered = synthesis_started.elapsed().as_secs_f64();
                    let duration = chunk.samples.len() as f64
                        / session.producer.decoder_contract.sample_rate as f64;
                    if first_audio_seconds.is_none() {
                        first_audio_seconds = Some(delivered);
                        playback_end = delivered;
                    } else {
                        gaps += (delivered - playback_end).max(0.0);
                    }
                    playback_end = playback_end.max(delivered) + duration;
                    chunks.push(serde_json::json!({
                        "start_sample":chunk.start_sample, "samples":chunk.samples.len(),
                        "production_seconds":production_seconds, "delivered_seconds":delivered,
                        "audio_seconds":duration,
                        "stages":session.producer.generator.networks.profile.with(&session.codec.profile).since(&before),
                    }));
                    assert_eq!(chunk.start_sample, samples.len());
                    samples.extend(chunk.samples);
                    progress("pcm", session.producer.codes.frames(), samples.len());
                }
                Ok(None) => {}
                Err(error) => {
                    progress("error", session.producer.codes.frames(), samples.len());
                    std::fs::write(
                        output.join("failure.codes.json"),
                        serde_json::to_vec(
                            &serde_json::json!({"codes":session.producer.codes,"segments":session.producer.segments}),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    panic!("long-form decoding failed after saving its acoustic codes: {error}");
                }
            }
        }
        let generation = session.into_generation().unwrap();
        let synthesis_seconds = synthesis_started.elapsed().as_secs_f64();
        let stages = engine
            .generator
            .networks
            .profile
            .with(&engine.codec.profile);
        let rate = engine.sample_rate();
        let samples_per_frame = engine.decoder_contract.samples_per_frame;
        drop(engine);
        let steady_production: f64 = chunks
            .iter()
            .skip(1)
            .map(|c| c["production_seconds"].as_f64().unwrap())
            .sum();
        let steady_audio: f64 = chunks
            .iter()
            .skip(1)
            .map(|c| c["audio_seconds"].as_f64().unwrap())
            .sum();
        std::fs::write(output.join("profile.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "load_seconds":load_seconds, "conditioning_seconds":conditioning_seconds,
            "synthesis_seconds":synthesis_seconds, "first_audio_seconds":first_audio_seconds,
            "steady_production_seconds":steady_production, "steady_audio_seconds":steady_audio,
            "steady_rtf":steady_production/steady_audio, "playback_gap_seconds":gaps,
            "stages":stages, "talker_sampling":talker_sampling, "predictor_sampling":predictor_sampling,
            "chunks":chunks,
        })).unwrap()).unwrap();
        std::fs::write(
            output.join("long.codes.json"),
            serde_json::to_vec(&generation).unwrap(),
        )
        .unwrap();
        assert!(generation.generated_frames() > maximum.get());
        assert!(generation.segments.len() > 1);
        assert_eq!(
            generation.generated_frames() * samples_per_frame,
            samples.len()
        );
        let mut writer = hound::WavWriter::create(
            output.join("long.wav"),
            hound::WavSpec {
                channels: 2,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in samples {
            writer.write_sample(sample).unwrap();
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        progress(
            "complete",
            generation.codes.frames(),
            generation.generated_frames() * samples_per_frame,
        );
    });
}

use super::*;
use crate::{generation::Model, sampling::Greedy};

struct TestModel {
    eos: bool,
    ignore_eos: bool,
    generation_resets: usize,
    waveform_resets: usize,
    frame_inputs: Vec<Vec<(Vec<u32>, u32)>>,
    waveform_inputs: Vec<Vec<i32>>,
}
impl Model for TestModel {
    type Hidden = u32;
    type Token = u32;
    type Logits = Vec<f32>;
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
    fn reset_waveform(&mut self) {
        self.waveform_resets += 1;
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
    async fn waveform(
        &mut self,
        codes: Vec<i32>,
        _: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
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
            waveform_resets: 0,
            frame_inputs: Vec::new(),
            waveform_inputs: Vec::new(),
        },
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
        assert_eq!(engine.generator.networks.waveform_resets, 1);
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
        assert_eq!(engine.generator.networks.waveform_resets, 1);
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

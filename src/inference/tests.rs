use super::*;
use crate::{generation::Model, sampling::Greedy};

struct TestModel {
    eos: bool,
    ignore_eos: bool,
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
    fn reset(&mut self) {
        self.eos = false;
    }
    fn reset_predictor(&mut self) {}
    fn text(&mut self, token: u32) -> Result<u32, Error> {
        Ok(token)
    }
    fn code(&mut self, _: usize, token: u32) -> Result<u32, Error> {
        Ok(token)
    }
    fn frame(&mut self, _: &[u32], text: &u32) -> Result<u32, Error> {
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
    async fn waveform(&mut self, _: Vec<i32>, _: usize, frames: usize) -> Result<Vec<f32>, Error> {
        Ok(vec![0.25; frames * 2])
    }
}
fn engine() -> Engine<TestModel, Greedy, Greedy> {
    // Test-only vocabulary: codec 0, EOS 1, then 1024 reserved IDs.
    Engine::new(
        TestModel {
            eos: false,
            ignore_eos: false,
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
    SynthesisOptions::new(NonZeroUsize::new(24).unwrap())
}

#[test]
fn long_text_exceeds_one_frame_budget_and_decodes_every_segment() {
    futures_lite::future::block_on(async {
        let mut engine = engine();
        let reference = reference();
        let text = "Hello there. ".repeat(8);
        let result = engine
            .synthesize(&reference, &text, "auto", options())
            .await
            .unwrap();
        assert_eq!(result.generation.segments.len(), 8);
        assert!(result.generation.generated_frames() > options().maximum_frames.get());
        assert_eq!(
            result.waveform.samples.len(),
            result.generation.generated_frames() * 2
        );
        for codes in &result.generation.segments {
            assert_eq!(codes.reference_frames(), 2);
        }
        let decoded = engine.decode(&result.generation.segments).await.unwrap();
        assert_eq!(decoded.samples, result.waveform.samples);
        let exported = serde_json::to_value(&result.generation).unwrap();
        assert_eq!(exported["segments"].as_array().unwrap().len(), 8);
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
            .utterance(&reference, "auto", frames, usize::MAX)
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
        let mut utterance = engine.utterance(&reference, "auto", 5, usize::MAX).unwrap();
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
        let text = "Hello there. ".repeat(8);
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

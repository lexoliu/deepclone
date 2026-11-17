//! Runtime-independent text coalescing and continuous PCM inference.
use crate::{
    generation::Model,
    inference::{AudioChunk, Engine, Error, Generation, Reference, SynthesisOptions},
    sampling::TokenSampler,
};
use async_channel::Sender;
use futures_lite::{Stream, StreamExt, future};
use std::{num::NonZeroUsize, time::Duration};
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub use native::Load;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{AudioStream, ReferenceAudio, StreamError, Voice};

/// Executor-provided monotonic clock and nonblocking timer.
#[allow(async_fn_in_trait)]
pub trait Clock {
    fn now(&self) -> Duration;
    async fn sleep(&self, duration: Duration);
}
impl<C: Clock> Clock for &C {
    fn now(&self) -> Duration {
        (*self).now()
    }
    async fn sleep(&self, duration: Duration) {
        (*self).sleep(duration).await;
    }
}
/// Scheduling knobs. Durations control batching of stable text, not how long
/// the model may wait for a token boundary or for an upstream producer.
#[derive(Clone, Debug)]
pub struct Options {
    pub generation: SynthesisOptions,
    /// Acoustic-code frames, not PCM samples. Published bundles use 80 ms per frame.
    pub audio_chunk_frames: NonZeroUsize,
    pub input_queue_chunks: NonZeroUsize,
    pub output_queue_chunks: NonZeroUsize,
    pub text_chunk_chars: NonZeroUsize,
    pub debounce: Duration,
    pub maximum_wait: Duration,
}
impl Options {
    pub fn new(maximum_frames: NonZeroUsize) -> Self {
        Self {
            generation: SynthesisOptions::new(maximum_frames),
            // Scheduling defaults, adjustable independently of model contracts.
            audio_chunk_frames: NonZeroUsize::new(12).unwrap(),
            input_queue_chunks: NonZeroUsize::new(2).unwrap(),
            output_queue_chunks: NonZeroUsize::new(2).unwrap(),
            text_chunk_chars: NonZeroUsize::new(1024).unwrap(),
            debounce: Duration::from_millis(25),
            maximum_wait: Duration::from_millis(100),
        }
    }
}

pub struct Execution<C, F> {
    pub clock: C,
    pub cancelled: F,
}

/// Drive input and output concurrently without spawning or requiring a runtime.
/// Awaiting the PCM consumer applies backpressure. Dropping this future cancels
/// both input forwarding and inference, including a pending text source.
pub async fn run<B, T, S, I, C, F>(
    engine: &mut Engine<B, T, S>,
    reference: &Reference<B::Hidden>,
    input: I,
    language: &str,
    options: Options,
    execution: Execution<C, impl Fn() -> bool>,
    consume: F,
) -> Result<Generation, Error>
where
    B: Model,
    T: TokenSampler,
    S: TokenSampler,
    I: Stream<Item = Result<String, Error>>,
    C: Clock,
    F: AsyncFnMut(AudioChunk) -> Result<(), Error>,
{
    let Execution { clock, cancelled } = execution;
    let (sender, receiver) = async_channel::bounded(options.input_queue_chunks.get());
    let producer = feed(input, sender, options.clone(), &clock);
    let worker = drive(
        engine,
        reference,
        &receiver,
        language,
        &options,
        Execution {
            clock: &clock,
            cancelled,
        },
        consume,
    );
    let (input, result) = future::zip(producer, async {
        let result = worker.await;
        receiver.close();
        result
    })
    .await;
    input?;
    result
}
/// Both worker environments share utterance boundaries, PCM positions and EOS handling.
pub(crate) async fn drive<B, T, S, C: Clock, F>(
    engine: &mut Engine<B, T, S>,
    reference: &Reference<B::Hidden>,
    receiver: &async_channel::Receiver<Option<String>>,
    language: &str,
    options: &Options,
    execution: Execution<C, impl Fn() -> bool>,
    mut consume: F,
) -> Result<Generation, Error>
where
    B: Model,
    T: TokenSampler,
    S: TokenSampler,
    F: AsyncFnMut(AudioChunk) -> Result<(), Error>,
{
    let Execution { clock, cancelled } = execution;
    let mut splitter = engine.splitter(reference, language, &options.generation)?;
    let mut input_finished = false;
    let mut pending = std::collections::VecDeque::<String>::new();
    let mut segments = Vec::new();
    let mut position = 0;
    loop {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        if let Some(text) = pending.pop_front() {
            let mut started = clock.now();
            let mut session = engine.utterance(
                reference,
                language,
                options.generation.maximum_frames.get(),
                options.generation.unstable_text_bytes.get(),
            )?;
            session.push_text(&text)?;
            session.finish_input()?;
            while !session.finished {
                if let Some(mut chunk) = session
                    .next_chunk(options.audio_chunk_frames.get(), &cancelled)
                    .await?
                {
                    chunk.start_sample += position;
                    let ready = clock.now();
                    let audio_seconds = chunk.samples.len() as f64 / f64::from(chunk.sample_rate);
                    let start_sample = chunk.start_sample;
                    consume(chunk).await?;
                    tracing::debug!(
                        start_sample,
                        audio_seconds,
                        production_ms = (ready - started).as_secs_f64() * 1000.,
                        consumer_wait_ms = (clock.now() - ready).as_secs_f64() * 1000.,
                        "audio chunk timing"
                    );
                    started = clock.now();
                }
            }
            let generation = session.into_generation()?;
            position += generation.generated_frames() * engine.samples_per_frame();
            segments.extend(generation.segments);
        } else if !input_finished {
            match receiver.recv().await.map_err(|_| Error::InputStopped)? {
                Some(text) => pending.extend(splitter.push(&text, engine.tokenizer())?),
                None => {
                    pending.extend(splitter.finish(engine.tokenizer())?);
                    input_finished = true;
                }
            }
        } else {
            break;
        }
    }
    engine.generation(segments)
}

pub(crate) async fn feed<I, E, C: Clock>(
    input: I,
    output: Sender<Option<String>>,
    options: Options,
    clock: C,
) -> Result<(), E>
where
    I: Stream<Item = Result<String, E>>,
{
    let lifetime = output.clone();
    future::race(feed_open(input, output, options, clock), async {
        lifetime.closed().await;
        Ok(())
    })
    .await
}

async fn feed_open<I, E, C: Clock>(
    input: I,
    output: Sender<Option<String>>,
    options: Options,
    clock: C,
) -> Result<(), E>
where
    I: Stream<Item = Result<String, E>>,
{
    futures_lite::pin!(input);
    let mut buffer = String::new();
    let mut count = 0;
    let mut started = clock.now();
    let mut changed = started;
    loop {
        let event = if buffer.is_empty() {
            Some(input.next().await)
        } else {
            let deadline = (started + options.maximum_wait).min(changed + options.debounce);
            if clock.now() >= deadline {
                None
            } else {
                future::race(async { Some(input.next().await) }, async {
                    clock.sleep(deadline.saturating_sub(clock.now())).await;
                    None
                })
                .await
            }
        };
        match event {
            Some(Some(text)) => {
                let text = text?;
                for ch in text.chars() {
                    if buffer.is_empty() {
                        started = clock.now();
                    }
                    buffer.push(ch);
                    count += 1;
                    changed = clock.now();
                    if count == options.text_chunk_chars.get() {
                        if output
                            .send(Some(std::mem::take(&mut buffer)))
                            .await
                            .is_err()
                        {
                            return Ok(());
                        }
                        count = 0;
                    }
                }
                // Transport coalescing is independent of linguistic utterances;
                // the shared driver applies Unicode and tokenizer boundaries.
                if buffer.ends_with(['。', '！', '？', '.', '!', '?', '\n']) {
                    if output
                        .send(Some(std::mem::take(&mut buffer)))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    count = 0;
                }
            }
            Some(None) => {
                tracing::debug!("text source ended");
                if !buffer.is_empty() && output.send(Some(buffer)).await.is_err() {
                    return Ok(());
                }
                let _ = output.send(None).await;
                return Ok(());
            }
            None => {
                if output
                    .send(Some(std::mem::take(&mut buffer)))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                count = 0;
            }
        }
    }
}

//! Runtime-independent text coalescing and continuous PCM inference.
use crate::{
    generation::Model,
    inference::{AudioChunk, Engine, Error, Generation, Reference},
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
/// Scheduling knobs. Durations control batching of stable text, not how long
/// the model may wait for a token boundary or for an upstream producer.
#[derive(Clone, Debug)]
pub struct Options {
    pub maximum_frames: NonZeroUsize,
    /// Acoustic-code frames, not PCM samples. Published bundles use 80 ms per frame.
    pub audio_chunk_frames: NonZeroUsize,
    pub input_queue_chunks: NonZeroUsize,
    pub output_queue_chunks: NonZeroUsize,
    pub text_chunk_chars: NonZeroUsize,
    pub unstable_text_bytes: NonZeroUsize,
    pub debounce: Duration,
    pub maximum_wait: Duration,
}
impl Options {
    pub fn new(maximum_frames: NonZeroUsize) -> Self {
        Self {
            maximum_frames,
            // Scheduling defaults, adjustable independently of model contracts.
            audio_chunk_frames: NonZeroUsize::new(12).unwrap(),
            input_queue_chunks: NonZeroUsize::new(2).unwrap(),
            output_queue_chunks: NonZeroUsize::new(2).unwrap(),
            text_chunk_chars: NonZeroUsize::new(1024).unwrap(),
            unstable_text_bytes: NonZeroUsize::new(65536).unwrap(),
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
    mut consume: F,
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
    let producer = feed(input, sender, options.clone(), clock);
    let worker = async {
        let mut session = engine.session(
            reference,
            language,
            options.maximum_frames.get(),
            options.unstable_text_bytes.get(),
        )?;
        while !session.finished {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            if let Some(chunk) = session
                .next_chunk(options.audio_chunk_frames.get(), &cancelled)
                .await?
            {
                consume(chunk).await?;
            } else if !session.finished {
                match receiver.recv().await.map_err(|_| Error::InputStopped)? {
                    Some(text) => session.push_text(&text)?,
                    None => session.finish_input()?,
                }
            }
        }
        session.into_generation()
    };
    let (input, result) = future::zip(producer, async {
        let result = worker.await;
        receiver.close();
        result
    })
    .await;
    input?;
    result
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
                // A sentence boundary can advance immediately. It still goes
                // through incremental tokenization; no utterance is restarted.
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

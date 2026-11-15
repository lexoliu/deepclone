//! Text streams to continuous PCM, with a Tokio input task and an owned Core ML worker.
//! SSE clients pass their decoded text deltas; this module does not parse HTTP.
//!
//! ```no_run
//! use deepclone::{inference::AudioChunk, streaming::{Options, StreamError, Voice}};
//! use futures_lite::{Stream, StreamExt};
//!
//! async fn forward<S>(
//!     voice: &mut Voice,
//!     text: S,
//!     options: Options,
//!     mut consume_pcm: impl AsyncFnMut(AudioChunk),
//! ) -> Result<(), StreamError<std::io::Error>>
//! where S: Stream<Item = Result<String, std::io::Error>> + Send + 'static {
//!     let mut audio = voice.synthesize(text, "English", options).await?;
//!     while let Some(chunk) = audio.next().await {
//!         consume_pcm(chunk?).await;
//!     }
//!     Ok(())
//! }
//! ```
use super::{Clock, Options};
use crate::{
    inference::{AudioChunk, Engine, Error, Generation, Reference, Synthesis, Waveform},
    sampling::TokenSampler,
};
use async_channel::{Receiver, Sender};
use futures_lite::{Stream, StreamExt, future};
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
use ondevice_coreml::ComputeUnits;
use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{runtime::Handle, task::JoinHandle, time::Instant};

pub struct ReferenceAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub transcript: String,
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub struct Load {
    pub transformers: std::path::PathBuf,
    pub audio_models: std::path::PathBuf,
    pub compute_units: ComputeUnits,
    pub reference: ReferenceAudio,
}

/// A reusable voice with one model owner. A stream borrows the voice exclusively,
/// so requests cannot accidentally interleave their KV or decoder state.
pub struct Voice {
    requests: Sender<Request>,
    sample_rate: u32,
}
struct Request {
    language: String,
    options: Options,
    text: Receiver<Option<String>>,
    audio: Sender<Result<AudioChunk, Error>>,
    completion: Sender<Generation>,
}

impl Voice {
    /// Construct and condition a backend on its dedicated inference worker.
    pub async fn load_with<B, T, S, F, Fut>(
        initialize: F,
        reference: ReferenceAudio,
    ) -> Result<Self, Error>
    where
        B: crate::generation::Model + 'static,
        T: TokenSampler + 'static,
        S: TokenSampler + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Engine<B, T, S>, Error>>,
    {
        let (ready, receive) = async_channel::bounded(1);
        let (requests, incoming) = async_channel::bounded(1);
        let dispatcher = tracing::dispatcher::get_default(Clone::clone);
        std::thread::Builder::new()
            .name("deepclone-inference".into())
            .spawn(move || {
                tracing::dispatcher::with_default(&dispatcher, || {
                    let initialized = futures_lite::future::block_on(async {
                        let mut engine = initialize().await?;
                        let reference = engine
                            .reference(
                                &reference.samples,
                                reference.sample_rate,
                                &reference.transcript,
                            )
                            .await?;
                        Ok::<_, Error>((engine, reference))
                    });
                    match initialized {
                        Ok((engine, reference)) => {
                            if ready.send_blocking(Ok(engine.sample_rate())).is_ok() {
                                serve(engine, reference, incoming);
                            }
                        }
                        Err(error) => {
                            let _ = ready.send_blocking(Err(error));
                        }
                    }
                })
            })?;
        let sample_rate = receive.recv().await.map_err(|_| Error::WorkerStopped)??;
        Ok(Self {
            requests,
            sample_rate,
        })
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    pub async fn load<T: TokenSampler + Send + 'static, S: TokenSampler + Send + 'static>(
        load: Load,
        sampling: crate::inference::Sampling<T, S>,
    ) -> Result<Self, Error> {
        Self::load_with(
            move || async move {
                Engine::open(
                    &load.transformers,
                    &load.audio_models,
                    load.compute_units,
                    sampling,
                )
            },
            load.reference,
        )
        .await
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Input is polled by a separate task on the current Tokio runtime. Dropping
    /// the returned stream cancels that task and closes the worker's channels.
    pub async fn synthesize<I, E>(
        &mut self,
        text: I,
        language: impl Into<String>,
        options: Options,
    ) -> Result<AudioStream<'_, E>, Error>
    where
        I: Stream<Item = Result<String, E>> + Send + 'static,
        E: Send + 'static,
    {
        let runtime = Handle::try_current()
            .map_err(|_| Error::Contract("streaming requires a Tokio runtime with time enabled"))?;
        let (input, receive) = async_channel::bounded(options.input_queue_chunks.get());
        let (audio, output) = async_channel::bounded(options.output_queue_chunks.get());
        let (completion, completed) = async_channel::bounded(1);
        let (input_error, input_errors) = async_channel::bounded(1);
        let cancellation = output.clone();
        let policy = options.clone();
        let feeder = runtime.spawn(async move {
            // Keep a sender alive until the error is published and cancellation
            // is signalled. A worker cannot mistake this gap for normal EOF.
            let input_lifetime = input.clone();
            let result = future::race(feed(text, input, policy), async {
                input_lifetime.closed().await;
                Ok(())
            })
            .await;
            tracing::debug!("text forwarding ended");
            if let Err(error) = result {
                let _ = input_error.send(error).await;
                cancellation.close();
            }
        });
        let stream = AudioStream {
            output: Box::pin(output),
            completed: Box::pin(completed),
            feeder: Some(feeder),
            input_errors: Box::pin(input_errors),
            sample_rate: self.sample_rate,
            generation: None,
            terminal: false,
            voice: PhantomData,
        };
        self.requests
            .send(Request {
                language: language.into(),
                options,
                text: receive,
                audio,
                completion,
            })
            .await
            .map_err(|_| Error::WorkerStopped)?;
        Ok(stream)
    }
}

fn serve<B: crate::generation::Model, T: TokenSampler, S: TokenSampler>(
    mut engine: Engine<B, T, S>,
    reference: Reference<B::Hidden>,
    requests: Receiver<Request>,
) {
    while let Ok(request) = requests.recv_blocking() {
        if request.audio.is_closed() {
            continue;
        }
        tracing::debug!("session started");
        let result = futures_lite::future::block_on(async {
            let mut session = engine.session(
                &reference,
                &request.language,
                request.options.maximum_frames.get(),
                request.options.unstable_text_bytes.get(),
            )?;
            while !session.finished {
                if request.audio.is_closed() {
                    return Ok(());
                }
                if let Some(chunk) = session
                    .next_chunk(request.options.audio_chunk_frames.get(), || {
                        request.audio.is_closed()
                    })
                    .await?
                {
                    tracing::debug!(
                        start_sample = chunk.start_sample,
                        samples = chunk.samples.len(),
                        "audio ready"
                    );
                    if request.audio.send_blocking(Ok(chunk)).is_err() {
                        return Ok(());
                    }
                } else if !session.finished {
                    tracing::trace!("waiting for text");
                    match request.text.recv_blocking() {
                        Ok(Some(text)) => {
                            tracing::trace!(bytes = text.len(), "text received");
                            session.push_text(&text)?;
                        }
                        Ok(None) => {
                            tracing::debug!("finishing text input");
                            session.finish_input()?;
                        }
                        Err(_) => {
                            if request.audio.is_closed() {
                                return Ok(());
                            }
                            return Err(Error::InputStopped);
                        }
                    }
                }
            }
            let _ = request.completion.send_blocking(session.into_generation()?);
            Ok::<_, Error>(())
        });
        if let Err(error) = result {
            let _ = request.audio.send_blocking(Err(error));
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StreamError<E> {
    #[error("text source: {0}")]
    Input(E),
    #[error(transparent)]
    Inference(#[from] Error),
}

/// Read through `None` to finish and obtain `generation()`. An error is terminal;
/// already emitted PCM remains valid. Dropping this value cancels the request.
pub struct AudioStream<'a, E> {
    output: Pin<Box<Receiver<Result<AudioChunk, Error>>>>,
    completed: Pin<Box<Receiver<Generation>>>,
    feeder: Option<JoinHandle<()>>,
    input_errors: Pin<Box<Receiver<E>>>,
    sample_rate: u32,
    generation: Option<Generation>,
    terminal: bool,
    voice: PhantomData<&'a mut Voice>,
}
impl<E> AudioStream<'_, E> {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn generation(&self) -> Option<&Generation> {
        self.generation.as_ref()
    }

    pub fn take_generation(&mut self) -> Option<Generation> {
        self.generation.take()
    }

    /// Consume audio and completion using the same streaming path.
    pub async fn collect_audio(mut self) -> Result<Synthesis, StreamError<E>> {
        let mut samples = Vec::new();
        while let Some(chunk) = self.next().await {
            let chunk = chunk?;
            samples.extend(chunk.samples);
        }
        let generation = self.generation.take().ok_or(Error::WorkerStopped)?;
        Ok(Synthesis {
            generation,
            waveform: Waveform {
                samples,
                sample_rate: self.sample_rate,
            },
        })
    }
    fn cancel(&mut self) {
        self.output.close();
        if let Some(feeder) = self.feeder.take() {
            feeder.abort();
        }
    }
}
impl<E> Drop for AudioStream<'_, E> {
    fn drop(&mut self) {
        self.cancel();
    }
}
impl<E> Stream for AudioStream<'_, E> {
    type Item = Result<AudioChunk, StreamError<E>>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminal {
            return Poll::Ready(None);
        }
        if let Poll::Ready(Some(error)) = this.input_errors.as_mut().poll_next(cx) {
            this.cancel();
            this.terminal = true;
            return Poll::Ready(Some(Err(StreamError::Input(error))));
        }
        if let Some(feeder) = &mut this.feeder
            && let Poll::Ready(result) = Pin::new(feeder).poll(cx)
        {
            this.feeder = None;
            if result.is_err() {
                this.cancel();
                this.terminal = true;
                return Poll::Ready(Some(Err(StreamError::Inference(Error::InputStopped))));
            }
            // The producer may have completed between the earlier poll and
            // this one. Its error always takes precedence over worker EOF.
            if let Poll::Ready(Some(error)) = this.input_errors.as_mut().poll_next(cx) {
                this.cancel();
                this.terminal = true;
                return Poll::Ready(Some(Err(StreamError::Input(error))));
            }
        }
        match this.output.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(audio))) => Poll::Ready(Some(Ok(audio))),
            Poll::Ready(Some(Err(error))) => {
                this.cancel();
                this.terminal = true;
                Poll::Ready(Some(Err(StreamError::Inference(error))))
            }
            Poll::Ready(None) => match this.completed.as_mut().poll_next(cx) {
                Poll::Ready(Some(generation)) => {
                    this.generation = Some(generation);
                    this.terminal = true;
                    this.cancel();
                    Poll::Ready(None)
                }
                Poll::Ready(None) => {
                    this.terminal = true;
                    this.cancel();
                    Poll::Ready(Some(Err(StreamError::Inference(Error::WorkerStopped))))
                }
                Poll::Pending => Poll::Pending,
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

struct TokioClock(Instant);
impl Clock for TokioClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}
async fn feed<I, E>(input: I, output: Sender<Option<String>>, options: Options) -> Result<(), E>
where
    I: Stream<Item = Result<String, E>>,
{
    super::feed(input, output, options, TokioClock(Instant::now())).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use std::{
        convert::Infallible,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
    }
    fn options() -> Options {
        Options::new(NonZeroUsize::new(128).unwrap())
    }

    #[test]
    fn coalescing_preserves_text_and_bounds_queued_chunks() {
        runtime().block_on(async {
            let mut policy = options();
            policy.text_chunk_chars = NonZeroUsize::new(4).unwrap();
            let (tx, rx) = async_channel::bounded(1);
            let parts = ["法", "律。", "こんにちは。", "안녕하세요.", " Hello!"];
            let task = tokio::spawn(feed(
                futures_lite::stream::iter(parts.map(|s| Ok::<_, Infallible>(s.to_owned()))),
                tx,
                policy,
            ));
            tokio::task::yield_now().await;
            assert_eq!(rx.len(), 1);
            let mut combined = String::new();
            while let Ok(Some(part)) = rx.recv().await {
                assert!(part.chars().count() <= 4);
                combined.push_str(&part);
            }
            task.await.unwrap().unwrap();
            assert_eq!(combined, parts.concat());
        });
    }

    #[test]
    fn maximum_wait_flushes_before_continuous_input_ends() {
        runtime().block_on(async {
            let mut policy = options();
            policy.debounce = Duration::from_secs(1);
            policy.maximum_wait = Duration::from_millis(20);
            let input = futures_lite::stream::unfold(0, |n| async move {
                if n == 100 {
                    return None;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
                Some((Ok::<_, Infallible>("a".to_owned()), n + 1))
            });
            let (tx, rx) = async_channel::bounded(1);
            let task = tokio::spawn(feed(input, tx, policy));
            let text = tokio::time::timeout(Duration::from_secs(1), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(!text.is_empty());
            assert!(text.len() < 100);
            rx.close();
            task.await.unwrap().unwrap();
        });
    }

    fn pending_input(
        dropped: Arc<AtomicBool>,
    ) -> impl Stream<Item = Result<String, Infallible>> + Send {
        struct Guard(Arc<AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        futures_lite::stream::unfold(Guard(dropped), |guard| async {
            future::pending::<()>().await;
            Some((Ok::<_, Infallible>(String::new()), guard))
        })
    }

    #[test]
    fn output_drop_cancels_an_idle_input_source() {
        runtime().block_on(async {
            let dropped = Arc::new(AtomicBool::new(false));
            let input = pending_input(dropped.clone());
            let (requests, incoming) = async_channel::bounded(1);
            let mut voice = Voice {
                requests,
                sample_rate: 24000,
            };
            let stream = voice.synthesize(input, "English", options()).await.unwrap();
            let request = incoming.recv().await.unwrap();
            drop(stream);
            tokio::task::yield_now().await;
            assert!(dropped.load(Ordering::SeqCst));
            assert!(request.audio.is_closed());
            assert!(request.text.recv().await.is_err());
        });
    }

    #[test]
    fn upstream_error_is_not_successful_eof() {
        runtime().block_on(async {
            let (requests, incoming) = async_channel::bounded(1);
            let mut voice = Voice {
                requests,
                sample_rate: 24000,
            };
            let mut stream = voice
                .synthesize(
                    futures_lite::stream::once(Err::<String, _>("SSE disconnected")),
                    "English",
                    options(),
                )
                .await
                .unwrap();
            let request = incoming.recv().await.unwrap();
            assert!(matches!(
                stream.next().await,
                Some(Err(StreamError::Input("SSE disconnected")))
            ));
            assert!(stream.next().await.is_none());
            assert!(stream.generation().is_none());
            assert!(request.audio.is_closed());
        });
    }

    #[test]
    fn worker_exit_cancels_input_without_polling_audio() {
        runtime().block_on(async {
            let dropped = Arc::new(AtomicBool::new(false));
            let (requests, incoming) = async_channel::bounded(1);
            let mut voice = Voice {
                requests,
                sample_rate: 24000,
            };
            let audio = voice
                .synthesize(pending_input(dropped.clone()), "English", options())
                .await
                .unwrap();
            let request = incoming.recv().await.unwrap();
            drop(request);
            tokio::task::yield_now().await;
            assert!(dropped.load(Ordering::SeqCst));
            drop(audio);
        });
    }

    #[test]
    fn producer_panic_does_not_send_eof() {
        runtime().block_on(async {
            let (requests, incoming) = async_channel::bounded(1);
            let mut voice = Voice {
                requests,
                sample_rate: 24000,
            };
            let input =
                futures_lite::stream::poll_fn(|_| -> Poll<Option<Result<String, Infallible>>> {
                    panic!("producer fixture failure")
                });
            let mut audio = voice.synthesize(input, "English", options()).await.unwrap();
            let request = incoming.recv().await.unwrap();
            assert!(request.text.recv().await.is_err());
            assert!(matches!(
                audio.next().await,
                Some(Err(StreamError::Inference(Error::InputStopped)))
            ));
            assert!(audio.next().await.is_none());
            assert!(audio.generation().is_none());
        });
    }
}

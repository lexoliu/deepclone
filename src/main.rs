mod cli_io;

use clap::{Parser, Subcommand};
use deepclone::{
    inference::Sampling,
    models::{Bundle, Installed, Store, Zenwave},
    sampling::{Greedy, Multinomial, Parameters, RepetitionPenalty, TokenSampler},
    streaming::{Load, Options, ReferenceAudio, Voice},
};
use futures_lite::{Stream, StreamExt};
use ondevice_coreml::ComputeUnits;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, os::fd::AsFd, time::Duration};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Write},
    num::NonZeroUsize,
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Local Qwen3-TTS voice cloning through Core ML")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Clone a voice from reference audio and stream synthesized audio.
    Synthesize(Box<Synthesize>),
    /// Download and verify a published model without loading it.
    Download(Download),
}

#[derive(clap::Args)]
struct Download {
    #[arg(long, value_parser = parse_model, help = "Published model: 0.6b or 1.7b")]
    model: Bundle<'static>,
    #[arg(long, help = "Content-addressed model cache directory")]
    cache_dir: PathBuf,
}

fn parse_model(value: &str) -> Result<Bundle<'static>, String> {
    match value {
        "0.6b" => Ok(deepclone::models::qwen3_tts_0_6b()),
        "1.7b" => Ok(deepclone::models::qwen3_tts_1_7b()),
        _ => Err("model must be 0.6b or 1.7b".into()),
    }
}

#[derive(clap::Args)]
struct Synthesize {
    #[arg(long, required_unless_present = "model", conflicts_with = "model")]
    transformers: Option<PathBuf>,
    #[arg(long, required_unless_present = "model", conflicts_with = "model")]
    audio_models: Option<PathBuf>,
    #[arg(long, value_parser = parse_model, requires = "cache_dir", help = "Download or reuse published model: 0.6b or 1.7b")]
    model: Option<Bundle<'static>>,
    #[arg(long, requires = "model")]
    cache_dir: Option<PathBuf>,
    #[arg(long, help = "Mono or multichannel WAV at the model sample rate")]
    reference: PathBuf,
    #[arg(long, help = "JSON object containing the exact reference text")]
    transcript: PathBuf,
    #[arg(
        long,
        required_unless_present = "stdin",
        conflicts_with = "stdin",
        help = "JSON array of language/text objects"
    )]
    prompts: Option<PathBuf>,
    #[arg(
        long,
        help = "Read incremental UTF-8 text from stdin; EOF finishes the utterance"
    )]
    stdin: bool,
    #[arg(long, requires = "stdin", default_value = "auto")]
    language: String,
    #[arg(
        long,
        help = "PCM frames per audio chunk (library default when omitted)"
    )]
    audio_chunk_frames: Option<NonZeroUsize>,
    #[arg(long)]
    debounce_ms: Option<u64>,
    #[arg(long)]
    maximum_wait_ms: Option<u64>,
    #[arg(long)]
    text_chunk_chars: Option<NonZeroUsize>,
    #[arg(long)]
    unstable_text_bytes: Option<NonZeroUsize>,
    #[arg(long)]
    input_queue_chunks: Option<NonZeroUsize>,
    #[arg(long)]
    output_queue_chunks: Option<NonZeroUsize>,
    #[arg(
        long,
        help = "Batch output directory, or with --stdin a WAV file; - streams mono f32le PCM to stdout"
    )]
    output: PathBuf,
    #[arg(long)]
    maximum_frames: NonZeroUsize,
    #[arg(
        long,
        conflicts_with = "stdin",
        help = "Save local acoustic codes for decoder comparison or reuse"
    )]
    export_codes: bool,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    greedy: bool,
    #[arg(long,default_value="cpu-and-ne",value_parser=["cpu","cpu-and-ne","all"])]
    compute_units: String,
    // Sampling defaults from Qwen3TTSModel._merge_generate_kwargs.
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    #[arg(long, default_value = "50")]
    top_k: NonZeroUsize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f32,
}

#[cfg(test)]
mod tests {
    use super::Args;
    use clap::{CommandFactory, Parser};

    #[test]
    fn command_contracts() {
        Args::command().debug_assert();
        assert!(
            Args::try_parse_from([
                "deepclone",
                "download",
                "--model",
                "0.6b",
                "--cache-dir",
                "cache"
            ])
            .is_ok()
        );
        assert!(
            Args::try_parse_from([
                "deepclone",
                "download",
                "--model",
                "unknown",
                "--cache-dir",
                "cache"
            ])
            .is_err()
        );
        assert!(Args::try_parse_from(["deepclone", "download", "--model", "1.7b"]).is_err());
        let shared = [
            "deepclone",
            "synthesize",
            "--reference",
            "ref.wav",
            "--transcript",
            "ref.json",
            "--prompts",
            "prompts.json",
            "--output",
            "out",
            "--maximum-frames",
            "256",
            "--seed",
            "42",
        ];
        let parse = |source: &[&str]| {
            Args::try_parse_from(shared.iter().copied().chain(source.iter().copied()))
        };
        assert!(parse(&["--model", "1.7b", "--cache-dir", "cache"]).is_ok());
        assert!(parse(&["--transformers", "transformers", "--audio-models", "audio"]).is_ok());
        assert!(parse(&["--model", "0.6b"]).is_err());
        let streaming = [
            "deepclone",
            "synthesize",
            "--stdin",
            "--language",
            "English",
            "--model",
            "0.6b",
            "--cache-dir",
            "cache",
            "--reference",
            "ref.wav",
            "--transcript",
            "ref.json",
            "--output",
            "-",
            "--maximum-frames",
            "128",
            "--seed",
            "42",
        ];
        assert!(Args::try_parse_from(streaming).is_ok());
        assert!(Args::try_parse_from(streaming.into_iter().chain(["--export-codes"])).is_err());
        assert!(
            Args::try_parse_from(streaming.into_iter().chain(["--prompts", "prompts.json"]))
                .is_err()
        );

        assert!(parse(&["--transformers", "transformers"]).is_err());
        assert!(
            parse(&[
                "--model",
                "0.6b",
                "--cache-dir",
                "cache",
                "--transformers",
                "local"
            ])
            .is_err()
        );
    }
}
#[derive(Deserialize)]
struct Transcript {
    text: String,
}
#[derive(Deserialize, Serialize)]
struct Prompt {
    language: String,
    text: String,
}
#[derive(Serialize)]
struct Sample<'a> {
    model_size: String,
    transformer_weight_bits: u8,
    prompt: &'a Prompt,
    file: String,
    backend: &'static str,
    requested_compute_units: &'a str,
    seed: u64,
    greedy: bool,
    sample_rate: u32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    samples: usize,
    generated_frames: usize,
    synthesis_seconds: f64,
}

async fn run<T: TokenSampler + Send + 'static, S: TokenSampler + Send + 'static>(
    args: &Synthesize,
    talker: T,
    predictor: S,
) -> Result<(), Box<dyn Error>> {
    let units = match args.compute_units.as_str() {
        "cpu" => ComputeUnits::CpuOnly,
        "cpu-and-ne" => ComputeUnits::CpuAndNeuralEngine,
        "all" => ComputeUnits::All,
        _ => unreachable!("clap validates compute units"),
    };
    let (transformers, audio) = match (&args.model, &args.cache_dir) {
        (Some(model), Some(cache)) => {
            let installed = download(model, cache).await?;
            (installed.transformers(), installed.audio())
        }
        _ => (
            args.transformers.clone().ok_or("missing --transformers")?,
            args.audio_models.clone().ok_or("missing --audio-models")?,
        ),
    };
    let mut reader = hound::WavReader::open(&args.reference)?;
    let spec = reader.spec();
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let samples = interleaved
        .chunks_exact(usize::from(spec.channels))
        .map(|frame| frame.iter().sum::<f32>() / f32::from(spec.channels))
        .collect::<Vec<_>>();
    let transcript: Transcript =
        serde_json::from_reader(BufReader::new(File::open(&args.transcript)?))?;
    tracing::info!("loading models and encoding reference");
    let mut voice = Voice::load(
        Load {
            transformers,
            audio_models: audio,
            compute_units: units,
            reference: ReferenceAudio {
                samples,
                sample_rate: spec.sample_rate,
                transcript: transcript.text,
            },
        },
        Sampling {
            talker,
            predictor,
            repetition: RepetitionPenalty::new(args.repetition_penalty)?,
            minimum_frames: 2,
        },
    )
    .await?;
    tracing::info!(sample_rate = voice.sample_rate(), "voice ready");
    let mut options = Options::new(args.maximum_frames);
    if let Some(value) = args.audio_chunk_frames {
        options.audio_chunk_frames = value;
    }
    if let Some(value) = args.debounce_ms {
        options.debounce = Duration::from_millis(value);
    }
    if let Some(value) = args.maximum_wait_ms {
        options.maximum_wait = Duration::from_millis(value);
    }
    if let Some(value) = args.text_chunk_chars {
        options.text_chunk_chars = value;
    }
    if let Some(value) = args.unstable_text_bytes {
        options.unstable_text_bytes = value;
    }
    if let Some(value) = args.input_queue_chunks {
        options.input_queue_chunks = value;
    }
    if let Some(value) = args.output_queue_chunks {
        options.output_queue_chunks = value;
    }
    if args.stdin {
        let input = File::from(std::io::stdin().as_fd().try_clone_to_owned()?);
        if input.metadata()?.is_file() {
            run_stdin(&mut voice, cli_io::file_input(input), args, options).await?;
        } else {
            run_stdin(&mut voice, cli_io::pipe_input(input.into())?, args, options).await?;
        }
        return Ok(());
    }
    let prompts: Vec<Prompt> = serde_json::from_reader(BufReader::new(File::open(
        args.prompts.as_ref().ok_or("missing --prompts")?,
    )?))?;
    std::fs::create_dir_all(&args.output)?;
    let mut results = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        tracing::info!(language=%prompt.language,"synthesizing");
        let started = Instant::now();
        let file = format!("{index:04}.wav");
        let text = futures_lite::stream::once(Ok::<_, Infallible>(prompt.text.clone()));
        let (generation, samples) = write_audio(
            &mut voice,
            text,
            &prompt.language,
            options.clone(),
            &args.output.join(&file),
        )
        .await?;
        let seconds = started.elapsed().as_secs_f64();
        if args.export_codes {
            let path = args.output.join(format!("{index:04}.codes.json"));
            let mut writer = BufWriter::new(File::create_new(path)?);
            serde_json::to_writer(&mut writer, &generation)?;
            writer.flush()?;
        }
        results.push(Sample {
            model_size: generation.model_size,
            transformer_weight_bits: generation.transformer_weight_bits,
            prompt,
            file,
            backend: "Rust/Core ML",
            requested_compute_units: &args.compute_units,
            seed: args.seed,
            greedy: args.greedy,
            temperature: args.temperature,
            top_k: args.top_k.get(),
            top_p: args.top_p,
            repetition_penalty: args.repetition_penalty,
            sample_rate: voice.sample_rate(),
            samples,
            generated_frames: generation.codes.generated_frames(),
            synthesis_seconds: seconds,
        });
        let mut metadata = BufWriter::new(File::create(args.output.join("samples.json"))?);
        serde_json::to_writer_pretty(&mut metadata, &results)?;
        metadata.flush()?;
        tracing::info!(language=%prompt.language,frames=generation.codes.generated_frames(),seconds,"saved audio");
    }
    Ok(())
}

async fn run_stdin<I>(
    voice: &mut Voice,
    bytes: I,
    args: &Synthesize,
    options: Options,
) -> Result<(), Box<dyn Error>>
where
    I: Stream<Item = std::io::Result<Vec<u8>>> + Send + 'static,
{
    let text = cli_io::text_stream(Box::pin(bytes));
    if args.output != std::path::Path::new("-") {
        write_audio(voice, text, &args.language, options, &args.output).await?;
        return Ok(());
    }
    let output = File::from(std::io::stdout().as_fd().try_clone_to_owned()?);
    let audio = voice.synthesize(text, &args.language, options).await?;
    if output.metadata()?.is_file() {
        let mut output = tokio::fs::File::from_std(output);
        write_pcm(audio, async |bytes| {
            use tokio::io::AsyncWriteExt;
            output.write_all(&bytes).await?;
            output.flush().await
        })
        .await?;
    } else {
        let output = async_io::Async::new(std::os::fd::OwnedFd::from(output))?;
        write_pcm(audio, async |bytes| {
            let mut remaining = bytes.as_slice();
            while !remaining.is_empty() {
                let count = output
                    .write_with(|fd| rustix::io::write(fd, remaining).map_err(Into::into))
                    .await?;
                if count == 0 {
                    return Err(std::io::ErrorKind::WriteZero.into());
                }
                remaining = &remaining[count..];
            }
            Ok(())
        })
        .await?;
    }
    Ok(())
}

async fn write_pcm<E: Error + 'static>(
    mut audio: deepclone::streaming::AudioStream<'_, E>,
    mut write: impl AsyncFnMut(Vec<u8>) -> std::io::Result<()>,
) -> Result<(), Box<dyn Error>> {
    while let Some(chunk) = audio.next().await {
        let bytes = chunk?
            .samples
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        write(bytes).await?;
    }
    Ok(())
}

async fn write_audio<I, E>(
    voice: &mut Voice,
    text: I,
    language: &str,
    options: Options,
    path: &std::path::Path,
) -> Result<(deepclone::inference::Generation, usize), Box<dyn Error>>
where
    I: Stream<Item = Result<String, E>> + Send + 'static,
    E: Error + Send + 'static,
{
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: voice.sample_rate(),
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let path = path.to_owned();
    let mut writer =
        tokio::task::spawn_blocking(move || hound::WavWriter::create(path, spec)).await??;
    let mut audio = voice.synthesize(text, language, options).await?;
    let mut samples = 0;
    while let Some(chunk) = audio.next().await {
        let chunk = chunk?;
        samples += chunk.samples.len();
        writer = tokio::task::spawn_blocking(move || {
            for sample in chunk.samples {
                writer.write_sample(sample)?;
            }
            writer.flush()?;
            Ok::<_, hound::Error>(writer)
        })
        .await??;
    }
    tokio::task::spawn_blocking(move || writer.finalize()).await??;
    let generation = audio
        .take_generation()
        .ok_or("missing synthesis completion")?;
    Ok((generation, samples))
}

async fn download(
    model: &Bundle<'_>,
    cache: &std::path::Path,
) -> Result<Installed, Box<dyn Error>> {
    let store = Store::new(cache, Zenwave);
    let mut last_report = Instant::now();
    let installed = model
        .download(&store, |progress| {
            if last_report.elapsed().as_secs() >= 1 || progress.received == progress.total {
                tracing::info!(
                    received = progress.received,
                    total = progress.total,
                    "downloading model"
                );
                last_report = Instant::now();
            }
        })
        .await?;
    tracing::info!(path = %installed.root().display(), "model available");
    Ok(installed)
}

async fn execute(command: Command) -> Result<(), Box<dyn Error>> {
    let args = match command {
        Command::Download(args) => {
            download(&args.model, &args.cache_dir).await?;
            return Ok(());
        }
        Command::Synthesize(args) => args,
    };
    if args.greedy {
        run(&args, Greedy, Greedy).await
    } else {
        let parameters = Parameters::new(args.temperature, Some(args.top_k), args.top_p)?;
        run(
            &args,
            Multinomial::new(StdRng::seed_from_u64(args.seed), parameters),
            Multinomial::new(StdRng::seed_from_u64(args.seed), parameters),
        )
        .await
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let result = (|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .max_blocking_threads(std::thread::available_parallelism()?.get())
                .build()?;
            runtime.block_on(execute(args.command))
        })();
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                tracing::error!(%error,"command failed");
                ExitCode::FAILURE
            }
        }
    })
}

mod cli_io;

use clap::{Parser, Subcommand, ValueEnum};
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
use deepclone::streaming::Load;
use deepclone::{
    inference::Sampling,
    models::{Bundle, Installed, Store, Zenwave},
    sampling::{Greedy, Multinomial, Parameters, RepetitionPenalty, TokenSampler},
    streaming::{Options, ReferenceAudio, Voice},
};
use futures_lite::{Stream, StreamExt};
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
use ondevice_coreml::ComputeUnits;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, os::fd::AsFd, time::Duration};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Seek, SeekFrom, Write},
    num::NonZeroUsize,
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Local streaming voice cloning")]
struct Args {
    #[arg(
        short,
        long,
        global = true,
        help = "Include detailed streaming diagnostics"
    )]
    verbose: bool,
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
    #[arg(long, value_enum, help = "Published model")]
    model: ModelSize,
    #[arg(long, value_enum, default_value = "cpu-and-ne")]
    compute_units: Compute,
    #[arg(long, help = "Content-addressed model cache directory")]
    cache_dir: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelSize {
    #[value(name = "0.6b")]
    Small,
    #[value(name = "1.7b")]
    Large,
}

impl ModelSize {
    fn bundle(self, compute: Compute) -> Bundle<'static> {
        match (self, compute) {
            (Self::Small, Compute::CpuAndNe) => deepclone::models::qwen3_tts_0_6b(),
            (Self::Small, _) => deepclone::models::qwen3_tts_0_6b_fixed_shape(),
            (Self::Large, _) => deepclone::models::qwen3_tts_1_7b(),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Compute {
    Cpu,
    CpuAndNe,
    All,
}

impl Compute {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CpuAndNe => "cpu-and-ne",
            Self::All => "all",
        }
    }
}

#[derive(clap::Args)]
struct Synthesize {
    #[arg(long, conflicts_with_all=["model","transformers","audio_models"], help="Load a local Burn/WebGPU model package")]
    burn_model: Option<PathBuf>,
    #[arg(long, default_value = "2048", help = "KV context capacity for Burn")]
    context: NonZeroUsize,
    #[arg(long, required_unless_present_any = ["model", "burn_model"], conflicts_with = "model")]
    transformers: Option<PathBuf>,
    #[arg(long, required_unless_present_any = ["model", "burn_model"], conflicts_with = "model")]
    audio_models: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        requires = "cache_dir",
        help = "Download or reuse a published model"
    )]
    model: Option<ModelSize>,
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
        help = "Read incremental UTF-8 text from stdin; EOF finishes the text stream"
    )]
    stdin: bool,
    #[arg(long, requires = "stdin", default_value = "auto")]
    language: String,
    #[arg(
        long,
        help = "Acoustic-code frames per audio chunk (library default when omitted)"
    )]
    audio_chunk_frames: Option<NonZeroUsize>,
    #[arg(long)]
    debounce_ms: Option<u64>,
    #[arg(long)]
    maximum_wait_ms: Option<u64>,
    #[arg(long)]
    text_chunk_chars: Option<NonZeroUsize>,
    #[arg(long, help = "Maximum text tokens per continuous generation context")]
    text_segment_tokens: Option<NonZeroUsize>,
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
    #[arg(
        long,
        help = "Maximum acoustic frames per generation context; does not limit total audio duration"
    )]
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
    #[arg(long, value_enum, default_value = "cpu-and-ne")]
    compute_units: Compute,
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
    use super::{Args, Compute, ModelSize};
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

    #[test]
    fn ieee_float_wav_uses_standard_header_and_roundtrips() {
        let path = std::env::temp_dir().join(format!(
            "deepclone-wav-test-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let expected = [0.0f32, 0.25, -0.5, 0.75, -1.0];
        let mut writer = super::IeeeFloatWavWriter::create(&path, 24_000).unwrap();
        writer.write_samples(&expected[..2]).unwrap();
        writer.write_samples(&expected[2..]).unwrap();
        writer.finalize().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(bytes.len(), 44 + expected.len() * 4);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 16);
        // Standard WAVE_FORMAT_IEEE_FLOAT (0x0003) instead of WAVE_FORMAT_EXTENSIBLE (0xFFFE).
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 3);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            24_000
        );
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 32);
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize,
            expected.len() * 4
        );

        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 24_000);
        assert_eq!(spec.bits_per_sample, 32);
        assert_eq!(spec.sample_format, hound::SampleFormat::Float);
        let decoded: Vec<f32> = reader.samples::<f32>().map(Result::unwrap).collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn published_06b_uses_fixed_shapes_for_cpu() {
        let realtime = deepclone::models::qwen3_tts_0_6b();
        let fixed_shape = deepclone::models::qwen3_tts_0_6b_fixed_shape();
        assert_eq!(
            ModelSize::Small.bundle(Compute::CpuAndNe).pin().url,
            realtime.pin().url
        );
        assert_eq!(
            ModelSize::Small.bundle(Compute::Cpu).pin().url,
            fixed_shape.pin().url
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
    #[serde(skip_serializing_if = "Option::is_none")]
    requested_compute_units: Option<&'a str>,
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
    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    let units = match args.compute_units {
        Compute::Cpu => ComputeUnits::CpuOnly,
        Compute::CpuAndNe => ComputeUnits::CpuAndNeuralEngine,
        Compute::All => ComputeUnits::All,
    };
    let paths = if args.burn_model.is_some() {
        None
    } else {
        Some(match (&args.model, &args.cache_dir) {
            (Some(model), Some(cache)) => {
                let installed = download(&model.bundle(args.compute_units), cache).await?;
                (installed.transformers(), installed.audio())
            }
            _ => (
                args.transformers.clone().ok_or("missing --transformers")?,
                args.audio_models.clone().ok_or("missing --audio-models")?,
            ),
        })
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
    let reference = ReferenceAudio {
        samples,
        sample_rate: spec.sample_rate,
        transcript: transcript.text,
    };
    let sampling = Sampling {
        talker,
        predictor,
        repetition: RepetitionPenalty::new(args.repetition_penalty)?,
        minimum_frames: 2,
    };
    let mut voice = if let Some(path) = &args.burn_model {
        #[cfg(feature = "burn")]
        {
            let path = path.clone();
            let context = args.context;
            Voice::load_with(
                move || async move {
                    deepclone::inference::Engine::from_burn_package(&path, context, sampling).await
                },
                reference,
            )
            .await?
        }
        #[cfg(not(feature = "burn"))]
        {
            let _ = path;
            return Err("this binary was built without the burn feature".into());
        }
    } else {
        #[cfg(all(feature = "coreml", target_vendor = "apple"))]
        {
            let (transformers, audio) = paths.ok_or("missing Core ML paths")?;
            Voice::load(
                Load {
                    transformers,
                    audio_models: audio,
                    compute_units: units,
                    reference,
                },
                sampling,
            )
            .await?
        }
        #[cfg(not(all(feature = "coreml", target_vendor = "apple")))]
        {
            let _ = paths;
            return Err("Core ML is unavailable on this platform; pass --burn-model".into());
        }
    };
    tracing::info!(sample_rate = voice.sample_rate(), "voice ready");
    let mut options = Options::new(args.maximum_frames);
    if let Some(tokens) = args.text_segment_tokens {
        options.generation.text_segment_tokens = tokens;
    }
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
        options.generation.unstable_text_bytes = value;
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
        let generated_frames = generation.generated_frames();
        results.push(Sample {
            model_size: generation.model_size,
            transformer_weight_bits: generation.transformer_weight_bits,
            prompt,
            file,
            backend: if args.burn_model.is_some() {
                "Rust/Burn WebGPU"
            } else {
                "Rust/Core ML"
            },
            requested_compute_units: args
                .burn_model
                .is_none()
                .then_some(args.compute_units.name()),
            seed: args.seed,
            greedy: args.greedy,
            temperature: args.temperature,
            top_k: args.top_k.get(),
            top_p: args.top_p,
            repetition_penalty: args.repetition_penalty,
            sample_rate: voice.sample_rate(),
            samples,
            generated_frames,
            synthesis_seconds: seconds,
        });
        let mut metadata = BufWriter::new(File::create(args.output.join("samples.json"))?);
        serde_json::to_writer_pretty(&mut metadata, &results)?;
        metadata.flush()?;
        tracing::info!(language=%prompt.language,frames=generated_frames,seconds,"saved audio");
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

/// Standard 44-byte mono IEEE 754 float WAV writer (`WAVE_FORMAT_IEEE_FLOAT`).
///
/// Avoids `hound`'s `WAVE_FORMAT_EXTENSIBLE` header with `SPEAKER_FRONT_LEFT`,
/// which macOS CoreAudio routes exclusively to the left channel.
struct IeeeFloatWavWriter {
    writer: BufWriter<File>,
    data_bytes: u32,
}

impl IeeeFloatWavWriter {
    fn create(path: &std::path::Path, sample_rate: u32) -> std::io::Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        let byte_rate = sample_rate.checked_mul(4).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "sample rate overflow")
        })?;
        let mut header = [0u8; 44];
        header[0..4].copy_from_slice(b"RIFF");
        header[8..12].copy_from_slice(b"WAVE");
        header[12..16].copy_from_slice(b"fmt ");
        header[16..20].copy_from_slice(&16u32.to_le_bytes());
        // RFC 2361 WAVE_FORMAT_IEEE_FLOAT = 0x0003.
        header[20..22].copy_from_slice(&3u16.to_le_bytes());
        header[22..24].copy_from_slice(&1u16.to_le_bytes());
        header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
        header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
        header[32..34].copy_from_slice(&4u16.to_le_bytes());
        header[34..36].copy_from_slice(&32u16.to_le_bytes());
        header[36..40].copy_from_slice(b"data");
        writer.write_all(&header)?;
        Ok(Self {
            writer,
            data_bytes: 0,
        })
    }

    fn write_samples(&mut self, samples: &[f32]) -> std::io::Result<()> {
        let added = u32::try_from(samples.len().saturating_mul(4))
            .ok()
            .and_then(|bytes| self.data_bytes.checked_add(bytes))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "WAV exceeds 4 GiB RIFF limit",
                )
            })?;
        for &sample in samples {
            self.writer.write_all(&sample.to_le_bytes())?;
        }
        self.writer.flush()?;
        self.data_bytes = added;
        Ok(())
    }

    fn finalize(mut self) -> std::io::Result<()> {
        let riff_size = self.data_bytes.checked_add(36).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "WAV exceeds 4 GiB RIFF limit",
            )
        })?;
        self.writer.flush()?;
        self.writer.seek(SeekFrom::Start(4))?;
        self.writer.write_all(&riff_size.to_le_bytes())?;
        self.writer.seek(SeekFrom::Start(40))?;
        self.writer.write_all(&self.data_bytes.to_le_bytes())?;
        self.writer.flush()
    }
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
    let sample_rate = voice.sample_rate();
    let path = path.to_owned();
    let mut writer =
        tokio::task::spawn_blocking(move || IeeeFloatWavWriter::create(&path, sample_rate))
            .await??;
    let mut audio = voice.synthesize(text, language, options).await?;
    let mut samples = 0;
    while let Some(chunk) = audio.next().await {
        let chunk = chunk?;
        samples += chunk.samples.len();
        writer = tokio::task::spawn_blocking(move || {
            writer.write_samples(&chunk.samples)?;
            Ok::<_, std::io::Error>(writer)
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
            download(&args.model.bundle(args.compute_units), &args.cache_dir).await?;
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
        .with_max_level(if args.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
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

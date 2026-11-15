use clap::{Parser, Subcommand};
use deepclone::{
    inference::{Engine, Sampling},
    models::{Bundle, Installed, Store, Zenwave},
    sampling::{Greedy, Multinomial, Parameters, RepetitionPenalty, TokenSampler},
};
use ondevice_coreml::ComputeUnits;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
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
    /// Clone a voice from reference audio and synthesize complete texts.
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
    #[arg(long, help = "JSON array of language/text objects")]
    prompts: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    maximum_frames: usize,
    #[arg(
        long,
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
    reached_eos: bool,
    synthesis_seconds: f64,
}

fn run<T: TokenSampler, S: TokenSampler>(
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
            let installed = download(model, cache)?;
            (installed.transformers(), installed.audio())
        }
        _ => (
            args.transformers.clone().ok_or("missing --transformers")?,
            args.audio_models.clone().ok_or("missing --audio-models")?,
        ),
    };
    tracing::info!("loading models");
    let loading_started = Instant::now();
    let mut engine = Engine::open(
        &transformers,
        &audio,
        units,
        Sampling {
            talker,
            predictor,
            repetition: RepetitionPenalty::new(args.repetition_penalty)?,
            // The published generator suppresses EOS for its first two tokens.
            minimum_frames: 2,
        },
    )?;
    tracing::info!(
        seconds = loading_started.elapsed().as_secs_f64(),
        "models loaded"
    );
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
    tracing::info!("encoding reference");
    let reference = engine.reference(&samples, spec.sample_rate, &transcript.text)?;
    tracing::info!("reference encoded");
    let prompts: Vec<Prompt> = serde_json::from_reader(BufReader::new(File::open(&args.prompts)?))?;
    std::fs::create_dir_all(&args.output)?;
    let mut results = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        tracing::info!(language=%prompt.language,"synthesizing");
        let started = Instant::now();
        let output = engine.synthesize(
            &reference,
            &prompt.text,
            &prompt.language,
            args.maximum_frames,
        )?;
        let seconds = started.elapsed().as_secs_f64();
        if args.export_codes {
            let path = args.output.join(format!("{index:04}.codes.json"));
            let mut writer = BufWriter::new(File::create_new(path)?);
            serde_json::to_writer(&mut writer, &output.generation)?;
            writer.flush()?;
        }
        let file = format!("{index:04}.wav");
        let mut writer = hound::WavWriter::create(
            args.output.join(&file),
            hound::WavSpec {
                channels: 1,
                sample_rate: output.waveform.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for &sample in &output.waveform.samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        results.push(Sample {
            model_size: output.generation.model_size,
            transformer_weight_bits: output.generation.transformer_weight_bits,
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
            sample_rate: output.waveform.sample_rate,
            samples: output.waveform.samples.len(),
            generated_frames: output.generation.codes.generated_frames(),
            reached_eos: output.generation.reached_eos,
            synthesis_seconds: seconds,
        });
        let mut metadata = BufWriter::new(File::create(args.output.join("samples.json"))?);
        serde_json::to_writer_pretty(&mut metadata, &results)?;
        metadata.flush()?;
        tracing::info!(language=%prompt.language,frames=output.generation.codes.generated_frames(),reached_eos=output.generation.reached_eos,seconds,"saved audio");
    }
    Ok(())
}

fn download(model: &Bundle<'_>, cache: &std::path::Path) -> Result<Installed, Box<dyn Error>> {
    let store = Store::new(cache, Zenwave);
    let mut last_report = Instant::now();
    let installed = futures_lite::future::block_on(model.download(&store, |progress| {
        if last_report.elapsed().as_secs() >= 1 || progress.received == progress.total {
            tracing::info!(
                received = progress.received,
                total = progress.total,
                "downloading model"
            );
            last_report = Instant::now();
        }
    }))?;
    tracing::info!(path = %installed.root().display(), "model available");
    Ok(installed)
}

fn execute(command: Command) -> Result<(), Box<dyn Error>> {
    let args = match command {
        Command::Download(args) => {
            download(&args.model, &args.cache_dir)?;
            return Ok(());
        }
        Command::Synthesize(args) => args,
    };
    if args.greedy {
        run(&args, Greedy, Greedy)
    } else {
        let parameters = Parameters::new(args.temperature, Some(args.top_k), args.top_p)?;
        run(
            &args,
            Multinomial::new(StdRng::seed_from_u64(args.seed), parameters),
            Multinomial::new(StdRng::seed_from_u64(args.seed), parameters),
        )
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let result = execute(args.command);
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                tracing::error!(%error,"command failed");
                ExitCode::FAILURE
            }
        }
    })
}

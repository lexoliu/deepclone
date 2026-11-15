use clap::Parser;
use deepclone::{
    inference::{Engine, Sampling},
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
    #[arg(long)]
    transformers: PathBuf,
    #[arg(long)]
    audio_models: PathBuf,
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
    samples: usize,
    generated_frames: usize,
    reached_eos: bool,
    synthesis_seconds: f64,
}

fn run<T: TokenSampler, S: TokenSampler>(
    args: &Args,
    talker: T,
    predictor: S,
) -> Result<(), Box<dyn Error>> {
    let units = match args.compute_units.as_str() {
        "cpu" => ComputeUnits::CpuOnly,
        "cpu-and-ne" => ComputeUnits::CpuAndNeuralEngine,
        "all" => ComputeUnits::All,
        _ => unreachable!("clap validates compute units"),
    };
    tracing::info!("loading models");
    let loading_started = Instant::now();
    let mut engine = Engine::open(
        &args.transformers,
        &args.audio_models,
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
    for prompt in &prompts {
        tracing::info!(language=%prompt.language,"synthesizing");
        let started = Instant::now();
        let output = engine.synthesize(
            &reference,
            &prompt.text,
            &prompt.language,
            args.maximum_frames,
        )?;
        let seconds = started.elapsed().as_secs_f64();
        let file = format!("{}.wav", prompt.language);
        let mut writer = hound::WavWriter::create(
            args.output.join(&file),
            hound::WavSpec {
                channels: 1,
                sample_rate: output.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for &sample in &output.samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        results.push(Sample {
            model_size: output.model_size,
            transformer_weight_bits: output.transformer_weight_bits,
            prompt,
            file,
            backend: "Rust/Core ML",
            requested_compute_units: &args.compute_units,
            seed: args.seed,
            greedy: args.greedy,
            sample_rate: output.sample_rate,
            samples: output.samples.len(),
            generated_frames: output.generated_frames,
            reached_eos: output.reached_eos,
            synthesis_seconds: seconds,
        });
        let mut metadata = BufWriter::new(File::create(args.output.join("samples.json"))?);
        serde_json::to_writer_pretty(&mut metadata, &results)?;
        metadata.flush()?;
        tracing::info!(language=%prompt.language,frames=output.generated_frames,reached_eos=output.reached_eos,seconds,"saved audio");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let result = if args.greedy {
            run(&args, Greedy, Greedy)
        } else {
            Parameters::new(args.temperature, Some(args.top_k), args.top_p)
                .map_err(|e| Box::new(e) as Box<dyn Error>)
                .and_then(|parameters| {
                    run(
                        &args,
                        Multinomial::new(StdRng::seed_from_u64(args.seed), parameters),
                        Multinomial::new(StdRng::seed_from_u64(args.seed), parameters),
                    )
                })
        };
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                tracing::error!(%error,"voice cloning failed");
                ExitCode::FAILURE
            }
        }
    })
}

//! Core ML model loading and native tensor execution.
use crate::{
    decoder::Decoder,
    generation::Model as Backend,
    inference::{Assets, AudioContract, Config, Engine, Error, Sampling},
    sampling::TokenSampler,
};
use ondevice_coreml::{ComputeUnits, DataType, Function, Inputs, Model, Session, Tensor, f16};
use serde::Deserialize;
use std::{fs::File, io::BufReader, path::Path};
#[derive(Deserialize)]
struct Manifest {
    weight_bits: u8,
    decoders: Decoders,
    text_embedding: String,
    codec_embeddings: Vec<String>,
    codec_frame: String,
    predictor_projection: Option<String>,
}
#[derive(Deserialize)]
struct Decoders {
    talker: DecoderFiles,
    predictor: DecoderFiles,
}
#[derive(Deserialize)]
struct DecoderFiles {
    chunks: Vec<String>,
    heads: Vec<String>,
    rope_theta: f32,
}

pub struct CoreMl {
    talker: Decoder,
    predictor: Decoder,
    talker_head: Function,
    predictor_heads: Vec<Function>,
    predictor_projection: Option<Function>,
    codec_embeddings: Vec<Function>,
    codec_frame: Function,
    text_embedding: Function,
    speaker: Function,
    encoder: Function,
    waveform: Waveform,
}

/// One official codec block, retaining only its causal execution history.
struct Waveform {
    session: Session,
    frequencies: Vec<f32>,
    rotary_scaling: f32,
    window: usize,
    groups: usize,
    samples_per_frame: usize,
    packed: Vec<i32>,
    samples: Vec<f32>,
    frames: usize,
}

impl Waveform {
    fn open(path: &Path, units: ComputeUnits, samples_per_frame: usize) -> Result<Self, Error> {
        #[derive(Deserialize)]
        struct Contract {
            execution: String,
            attention_window: usize,
            rotary_frequencies: Vec<f32>,
            rotary_scaling: f32,
            code_groups: usize,
        }
        let contract: Contract = json(path.with_extension("contract.json"))?;
        if contract.execution != "stateful_frame" {
            return Err(Error::Contract(
                "waveform decoder requires stateful_frame execution",
            ));
        }
        let function = open(path, units)?;
        let description = function.description();
        let Some(cosine) = description.inputs.get("cosine") else {
            return Err(Error::Contract("waveform decoder requires rotary inputs"));
        };
        let [1, 1, 1, head_dim] = cosine.default_shape.as_slice() else {
            return Err(Error::Contract("invalid waveform rotary shape"));
        };
        let head_dim = *head_dim;
        if head_dim == 0
            || head_dim % 2 != 0
            || contract.attention_window == 0
            || contract.rotary_frequencies.len() * 2 != head_dim
        {
            return Err(Error::Contract("invalid waveform attention contract"));
        }
        for (name, shape, dtype) in [
            ("codes", vec![1, contract.code_groups, 1], DataType::Int32),
            ("cosine", vec![1, 1, 1, head_dim], DataType::Float16),
            ("sine", vec![1, 1, 1, head_dim], DataType::Float16),
            (
                "mask",
                vec![1, 1, 1, contract.attention_window],
                DataType::Float16,
            ),
        ] {
            if !description
                .inputs
                .get(name)
                .is_some_and(|input| input.default_shape == shape && input.data_type == dtype)
            {
                return Err(Error::Contract("waveform input shapes or dtypes disagree"));
            }
        }
        if description.states.is_empty()
            || !description.outputs.get("audio").is_some_and(|output| {
                output.default_shape == [1, 1, samples_per_frame]
                    && output.data_type == DataType::Float32
            })
        {
            return Err(Error::Contract(
                "waveform decoder requires state and one frame of PCM",
            ));
        }
        Ok(Self {
            session: function.into_session(),
            frequencies: contract.rotary_frequencies,
            rotary_scaling: contract.rotary_scaling,
            window: contract.attention_window,
            groups: contract.code_groups,
            samples_per_frame,
            packed: Vec::new(),
            samples: Vec::new(),
            frames: 0,
        })
    }

    fn reset(&mut self) {
        self.session.reset();
        self.packed.clear();
        self.samples.clear();
        self.frames = 0;
    }

    fn prefix(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        if groups != self.groups || packed.len() != groups * frames {
            return Err(Error::Contract("waveform code dimensions disagree"));
        }
        let extends = frames >= self.frames
            && (0..groups).all(|group| {
                packed[group * frames..group * frames + self.frames]
                    == self.packed[group * self.frames..(group + 1) * self.frames]
            });
        if !extends {
            self.reset();
        }
        for index in self.frames..frames {
            let result = self.step(&packed, frames, index);
            match result {
                Ok(samples) => self.samples.extend(samples),
                Err(error) => {
                    self.reset();
                    return Err(error);
                }
            }
        }
        self.frames = frames;
        self.packed = packed;
        Ok(self.samples.clone())
    }

    fn step(&mut self, packed: &[i32], frames: usize, index: usize) -> Result<Vec<f32>, Error> {
        let codes: Vec<_> = (0..self.groups)
            .map(|group| packed[group * frames + index])
            .collect();
        let codes = Tensor::from_slice(&[1, self.groups, 1], &codes)?;
        let mut cosines = Vec::with_capacity(self.frequencies.len() * 2);
        let mut sines = Vec::with_capacity(self.frequencies.len() * 2);
        for frequency in &self.frequencies {
            let (sine, cosine) = (frequency * index as f32).sin_cos();
            cosines.push(f16::from_f32(cosine * self.rotary_scaling));
            sines.push(f16::from_f32(sine * self.rotary_scaling));
        }
        cosines.extend_from_within(..);
        sines.extend_from_within(..);
        let cosine = Tensor::from_slice(&[1, 1, 1, cosines.len()], &cosines)?;
        let sine = Tensor::from_slice(&[1, 1, 1, sines.len()], &sines)?;
        let mut mask = vec![f16::MIN; self.window];
        mask[self.window - (index + 1).min(self.window)..].fill(f16::ZERO);
        let mask = Tensor::from_slice(&[1, 1, 1, self.window], &mask)?;
        let mut inputs = Inputs::new();
        inputs.insert("codes", &codes)?;
        inputs.insert("cosine", &cosine)?;
        inputs.insert("sine", &sine)?;
        inputs.insert("mask", &mask)?;
        let audio = self
            .session
            .predict(&inputs)?
            .tensor::<f32>("audio")?
            .read();
        if audio.data().len() != self.samples_per_frame {
            return Err(Error::Contract(
                "waveform decoder returned an incomplete frame",
            ));
        }
        Ok(audio.into_data())
    }
}
fn json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Error> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}
fn open(path: impl AsRef<Path>, units: ComputeUnits) -> Result<Function, Error> {
    Ok(Model::open(path)?.function(None, units)?)
}
impl<T: TokenSampler, S: TokenSampler> Engine<CoreMl, T, S> {
    pub fn open(
        transformers: &Path,
        audio: &Path,
        units: ComputeUnits,
        sampling: Sampling<T, S>,
    ) -> Result<Self, Error> {
        let manifest: Manifest = json(transformers.join("transformers.json"))?;
        let config: Config = json(transformers.join("config.json"))?;
        let encoder_contract: AudioContract = json(audio.join("encoder.contract.json"))?;
        let decoder_contract: AudioContract = json(audio.join("decoder.contract.json"))?;
        if encoder_contract.code_groups != config.talker_config.num_code_groups
            || decoder_contract.code_groups != encoder_contract.code_groups
        {
            return Err(Error::Contract(
                "audio and speech model code groups disagree",
            ));
        }
        let filters = json(audio.join("mel.json"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(transformers.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let load_decoder = |files: &DecoderFiles| {
            Decoder::open(
                files.chunks.iter().map(|p| transformers.join(p)),
                files.rope_theta,
                units,
            )
        };
        if manifest.decoders.talker.heads.len() != 1 {
            return Err(Error::Contract("Talker requires exactly one output head"));
        }
        let backend = CoreMl {
            talker: load_decoder(&manifest.decoders.talker)?,
            talker_head: open(transformers.join(&manifest.decoders.talker.heads[0]), units)?,
            predictor: load_decoder(&manifest.decoders.predictor)?,
            predictor_heads: manifest
                .decoders
                .predictor
                .heads
                .iter()
                .map(|p| open(transformers.join(p), units))
                .collect::<Result<_, _>>()?,
            predictor_projection: manifest
                .predictor_projection
                .map(|p| open(transformers.join(p), units))
                .transpose()?,
            codec_embeddings: manifest
                .codec_embeddings
                .iter()
                .map(|p| open(transformers.join(p), units))
                .collect::<Result<_, _>>()?,
            codec_frame: open(transformers.join(manifest.codec_frame), units)?,
            text_embedding: open(transformers.join(manifest.text_embedding), units)?,
            speaker: open(audio.join("speaker.mlmodelc"), units)?,
            encoder: open(audio.join("encoder.mlmodelc"), units)?,
            waveform: Waveform::open(
                &audio.join("decoder.mlmodelc"),
                units,
                decoder_contract.samples_per_frame,
            )?,
        };
        Self::new(
            backend,
            Assets {
                weight_bits: manifest.weight_bits,
                config,
                encoder_contract,
                decoder_contract,
                filters,
                tokenizer,
            },
            sampling,
        )
    }
}
impl Backend for CoreMl {
    type Hidden = Tensor<f16>;
    type Token = u32;
    type Logits = Vec<f32>;
    fn code_token(&mut self, group: usize, token: &u32) -> Result<Self::Hidden, Error> {
        self.code(group, *token)
    }
    async fn read_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>, Error> {
        Ok(tokens.to_vec())
    }
    fn groups(&self) -> usize {
        self.codec_embeddings.len()
    }
    fn reset(&mut self) {
        self.talker.reset();
        self.predictor.reset();
        self.waveform.reset();
    }
    fn waveform_frames(&self, valid: usize, _capacity: usize) -> usize {
        valid
    }
    fn reset_predictor(&mut self) {
        self.predictor.reset();
    }
    fn text(&mut self, token: u32) -> Result<Self::Hidden, Error> {
        embed(&mut self.text_embedding, token)
    }
    fn code(&mut self, group: usize, token: u32) -> Result<Self::Hidden, Error> {
        embed(&mut self.codec_embeddings[group], token)
    }
    fn frame(&mut self, codes: &[u32], text: &Self::Hidden) -> Result<Self::Hidden, Error> {
        frame(&mut self.codec_frame, codes, text)
    }
    fn add(a: &Self::Hidden, b: &Self::Hidden) -> Result<Self::Hidden, Error> {
        add(a, b)
    }
    fn talker(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.talker.step(hidden)?)
    }
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Vec<f32>), Error> {
        head(&mut self.talker_head, &hidden)
    }
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        if let Some(function) = &mut self.predictor_projection {
            let mut inputs = Inputs::new();
            inputs.insert("hidden", &hidden)?;
            Ok(function.predict(&inputs)?.tensor("hidden_out")?)
        } else {
            Ok(hidden)
        }
    }
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.predictor.step(hidden)?)
    }
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Vec<f32>, Error> {
        Ok(head(&mut self.predictor_heads[group], &hidden)?.1)
    }
    async fn condition(
        &mut self,
        samples: &[f32],
        mel: &[f32],
        channels: usize,
    ) -> Result<(Self::Hidden, Vec<Vec<u32>>), Error> {
        let mel = Tensor::from_slice(&[1, mel.len() / channels, channels], mel)?;
        let mut inputs = Inputs::new();
        inputs.insert("mel", &mel)?;
        let speaker = self
            .speaker
            .predict(&inputs)?
            .tensor::<f16>("speaker")?
            .read()
            .into_data();
        if speaker.len() != self.talker.hidden_size() {
            return Err(Error::Contract(
                "speaker embedding dimension differs from Talker",
            ));
        }
        let speaker = Tensor::from_slice(&[1, speaker.len(), 1, 1], &speaker)?;
        let waveform = Tensor::from_slice(&[1, 1, samples.len()], samples)?;
        let mut inputs = Inputs::new();
        inputs.insert("audio", &waveform)?;
        let prediction = self
            .encoder
            .predict(&inputs)?
            .tensor::<i32>("codes")?
            .read();
        let shape = prediction.shape();
        let groups = self.codec_embeddings.len();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != groups {
            return Err(Error::Contract("audio encoder returned invalid code shape"));
        }
        let frames = shape[2];
        let mut codes = Vec::with_capacity(frames);
        for frame in 0..frames {
            codes.push(
                (0..groups)
                    .map(|group| {
                        u32::try_from(prediction.data()[group * frames + frame])
                            .map_err(|_| Error::Contract("negative audio code"))
                    })
                    .collect::<Result<_, _>>()?,
            );
        }
        Ok((speaker, codes))
    }
    async fn waveform(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        self.waveform.prefix(packed, groups, frames)
    }
}

fn frame(function: &mut Function, codes: &[u32], text: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
    let ids = codes
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| Error::Contract("codec token exceeds int32")))
        .collect::<Result<Vec<_>, _>>()?;
    let ids = Tensor::from_slice(&[1, ids.len()], &ids)?;
    let mut inputs = Inputs::new();
    inputs.insert("codes", &ids)?;
    inputs.insert("text", text)?;
    Ok(function
        .predict(&inputs)
        .map_err(Error::CoreMl)?
        .tensor("hidden")?)
}

pub fn embed(function: &mut Function, token: u32) -> Result<Tensor<f16>, Error> {
    let token = i32::try_from(token).map_err(|_| Error::Contract("token exceeds int32"))?;
    let tensor = Tensor::from_slice(&[1, 1], &[token])?;
    let mut inputs = Inputs::new();
    inputs.insert("token", &tensor)?;
    Ok(function
        .predict(&inputs)
        .map_err(Error::CoreMl)?
        .tensor("hidden")?)
}

fn head(function: &mut Function, hidden: &Tensor<f16>) -> Result<(Tensor<f16>, Vec<f32>), Error> {
    let mut inputs = Inputs::new();
    inputs.insert("hidden", hidden)?;
    let prediction = function.predict(&inputs).map_err(Error::CoreMl)?;
    let logits = prediction
        .tensor::<f16>("logits")?
        .read()
        .into_data()
        .into_iter()
        .map(f16::to_f32)
        .collect();
    Ok((prediction.tensor("normalized")?, logits))
}

fn add(a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
    if a.shape() != b.shape() {
        return Err(Error::Contract("prompt embedding shapes disagree"));
    }
    let values = a
        .read()
        .data()
        .iter()
        .zip(b.read().data())
        .map(|(x, y)| f16::from_f32(x.to_f32() + y.to_f32()))
        .collect::<Vec<_>>();
    Ok(Tensor::from_slice(a.shape(), &values)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs, io::Write, path::PathBuf, time::Instant};

    #[test]
    #[ignore = "requires a compiled incremental decoder and recorded real acoustic codes"]
    fn real_coreml_incremental_waveform_preserves_chunk_boundaries() {
        #[derive(Deserialize)]
        struct Codes {
            groups: usize,
            values: Vec<i32>,
            reference_frames: usize,
        }
        #[derive(Deserialize)]
        struct Export {
            codes: Codes,
        }
        let path = |key| PathBuf::from(env::var(key).expect(key));
        let decoder = path("DEEPCLONE_DECODER");
        let contract: AudioContract = json(decoder.with_extension("contract.json")).unwrap();
        let codes: Export = json(path("DEEPCLONE_CODES")).unwrap();
        let codes = codes.codes;
        let frames = codes.values.len() / codes.groups;
        assert_eq!(codes.values.len(), frames * codes.groups);
        assert!(
            frames > 300,
            "fixture must cross an official codec block boundary"
        );
        let mut waveform = Waveform::open(
            &decoder,
            ComputeUnits::CpuAndNeuralEngine,
            contract.samples_per_frame,
        )
        .unwrap();
        let mut actual = Vec::new();
        let mut cursor = codes.reference_frames;
        let mut calls = Vec::new();
        while cursor < frames {
            // Qwen's official block and left-context boundaries, including reference frames.
            let block = cursor / 300 * 300;
            let context = block.saturating_sub(25);
            let stop = (cursor + 12).min(frames).min(block + 300);
            let count = stop - context;
            let packed: Vec<_> = (0..codes.groups)
                .flat_map(|group| {
                    (context..stop)
                        .map(|frame| codes.values[frame * codes.groups + group])
                        .collect::<Vec<_>>()
                })
                .collect();
            let started = Instant::now();
            let output = waveform.prefix(packed, codes.groups, count).unwrap();
            calls.push(serde_json::json!({"start": cursor, "end": stop, "seconds": started.elapsed().as_secs_f64()}));
            actual.extend_from_slice(&output[(cursor - context) * contract.samples_per_frame..]);
            cursor = stop;
        }
        let expected = fs::read(path("DEEPCLONE_EXPECTED")).unwrap();
        assert_eq!(expected.len(), actual.len() * 4);
        let mut difference = 0.0f64;
        let mut energy = 0.0f64;
        for (&value, bytes) in actual.iter().zip(expected.as_chunks::<4>().0) {
            assert!(value.is_finite());
            let reference = f32::from_le_bytes(*bytes) as f64;
            difference += (value as f64 - reference).powi(2);
            energy += reference.powi(2);
        }
        let relative_error = (difference / energy).sqrt();
        let output = path("DEEPCLONE_OUTPUT");
        fs::create_dir_all(&output).unwrap();
        let mut samples = File::create(output.join("audio.f32le")).unwrap();
        for value in &actual {
            samples.write_all(&value.to_le_bytes()).unwrap();
        }
        serde_json::to_writer(
            File::create(output.join("report.json")).unwrap(),
            &serde_json::json!({
                "frames": frames, "reference_frames": codes.reference_frames,
                "relative_error": relative_error, "calls": calls,
            }),
        )
        .unwrap();
        assert!(
            relative_error < 0.03,
            "relative waveform error {relative_error}"
        );
    }
}

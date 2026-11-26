//! Core ML model loading and native tensor execution.
use crate::{
    coreml_batch::BatchSizes,
    decoder::Decoder,
    generation::{Codec, Model as Backend},
    inference::{Assets, AudioContract, Config, Engine, Error, Sampling},
    sampling::TokenSampler,
};
use ondevice_coreml::{ComputeUnits, DataType, Function, Inputs, Model, Session, Tensor, f16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::{fs, fs::File, io::BufReader, num::NonZeroUsize, path::Path};
#[derive(Deserialize)]
struct Manifest {
    weight_bits: u8,
    decoders: Decoders,
    text_embedding: String,
    #[serde(default)]
    tables: Option<String>,
    #[serde(default)]
    codec_embeddings: Vec<String>,
    #[serde(default)]
    codec_frame: Option<String>,
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
    #[serde(default)]
    heads: Vec<String>,
    rope_theta: f32,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    sequence_lengths: BatchSizes,
}

struct MatrixF16 {
    rows: usize,
    cols: usize,
    data: Vec<f16>,
}

impl MatrixF16 {
    fn row(&self, index: usize) -> Option<&[f16]> {
        let start = index.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.data.get(start..end)
    }

    fn gemv(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        #[cfg(target_arch = "aarch64")]
        if self.cols.is_multiple_of(8) {
            return gemv_f16_neon(&self.data, self.rows, self.cols, x);
        }
        let mut out = Vec::with_capacity(self.rows);
        for row in self.data.chunks_exact(self.cols) {
            let mut sum = 0.0f32;
            for (&w, &v) in row.iter().zip(x) {
                sum += w.to_f32() * v;
            }
            out.push(sum);
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
fn gemv_f16_neon(weights: &[f16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    use std::arch::aarch64::*;
    const { assert!(std::mem::size_of::<f16>() == 2 && std::mem::align_of::<f16>() == 2) };
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert!(cols.is_multiple_of(8));
    let mut out = vec![0.0f32; rows];
    let w_base = weights.as_ptr().cast::<u16>();
    let x_ptr = x.as_ptr();
    // SAFETY: Caller ensures weights.len() == rows * cols, x.len() == cols, and cols % 8 == 0.
    // f16 has identical size and alignment to u16; fcvtl/fcvtl2/vfmaq_f32/vaddvq_f32 are baseline aarch64 NEON.
    unsafe {
        let mut r = 0;
        while r + 4 <= rows {
            let w0 = w_base.add(r * cols);
            let w1 = w_base.add((r + 1) * cols);
            let w2 = w_base.add((r + 2) * cols);
            let w3 = w_base.add((r + 3) * cols);
            let (mut a0, mut b0) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut a1, mut b1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut a2, mut b2) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let (mut a3, mut b3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut k = 0;
            while k < cols {
                let xlo = vld1q_f32(x_ptr.add(k));
                let xhi = vld1q_f32(x_ptr.add(k + 4));
                macro_rules! fma_row {
                    ($w:expr, $a:ident, $b:ident) => {{
                        let (lo, hi): (float32x4_t, float32x4_t);
                        std::arch::asm!(
                            "ldr {tmp:q}, [{ptr}]",
                            "fcvtl {lo:v}.4s, {tmp:v}.4h",
                            "fcvtl2 {hi:v}.4s, {tmp:v}.8h",
                            ptr = in(reg) $w.add(k),
                            tmp = out(vreg) _,
                            lo = out(vreg) lo,
                            hi = out(vreg) hi,
                            options(pure, nostack, readonly),
                        );
                        $a = vfmaq_f32($a, lo, xlo);
                        $b = vfmaq_f32($b, hi, xhi);
                    }};
                }
                fma_row!(w0, a0, b0);
                fma_row!(w1, a1, b1);
                fma_row!(w2, a2, b2);
                fma_row!(w3, a3, b3);
                k += 8;
            }
            out[r] = vaddvq_f32(vaddq_f32(a0, b0));
            out[r + 1] = vaddvq_f32(vaddq_f32(a1, b1));
            out[r + 2] = vaddvq_f32(vaddq_f32(a2, b2));
            out[r + 3] = vaddvq_f32(vaddq_f32(a3, b3));
            r += 4;
        }
        while r < rows {
            let w0 = w_base.add(r * cols);
            let (mut a0, mut b0) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut k = 0;
            while k < cols {
                let xlo = vld1q_f32(x_ptr.add(k));
                let xhi = vld1q_f32(x_ptr.add(k + 4));
                let (lo, hi): (float32x4_t, float32x4_t);
                std::arch::asm!(
                    "ldr {tmp:q}, [{ptr}]",
                    "fcvtl {lo:v}.4s, {tmp:v}.4h",
                    "fcvtl2 {hi:v}.4s, {tmp:v}.8h",
                    ptr = in(reg) w0.add(k),
                    tmp = out(vreg) _,
                    lo = out(vreg) lo,
                    hi = out(vreg) hi,
                    options(pure, nostack, readonly),
                );
                a0 = vfmaq_f32(a0, lo, xlo);
                b0 = vfmaq_f32(b0, hi, xhi);
                k += 8;
            }
            out[r] = vaddvq_f32(vaddq_f32(a0, b0));
            r += 1;
        }
    }
    out
}

struct HostTables {
    codec_embeddings: Vec<MatrixF16>,
    talker_norm: Vec<f32>,
    talker_eps: f32,
    talker_head: MatrixF16,
    predictor_norm: Vec<f32>,
    predictor_eps: f32,
    predictor_heads: Vec<MatrixF16>,
}

impl HostTables {
    fn open(
        path: &Path,
        groups: usize,
        talker_hidden: usize,
        talker_vocab: usize,
        talker_eps: f32,
        predictor_hidden: usize,
        predictor_eps: f32,
    ) -> Result<Self, Error> {
        if groups < 2
            || !talker_eps.is_finite()
            || talker_eps <= 0.0
            || !predictor_eps.is_finite()
            || predictor_eps <= 0.0
        {
            return Err(Error::Contract("invalid host table configuration"));
        }
        let bytes = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|_| Error::Contract("invalid host tables safetensors"))?;
        let read_matrix = |name: &str,
                           expected_rows: Option<usize>,
                           expected_cols: usize|
         -> Result<MatrixF16, Error> {
            let view = tensors
                .tensor(name)
                .map_err(|_| Error::Contract("missing tensor in host tables"))?;
            if view.dtype() != Dtype::F16 {
                return Err(Error::Contract("host table tensor must be float16"));
            }
            let [rows, cols] = *view.shape() else {
                return Err(Error::Contract("host table matrix must be rank 2"));
            };
            if rows == 0
                || cols != expected_cols
                || expected_rows.is_some_and(|expected| rows != expected)
                || view.data().len() != rows * cols * 2
            {
                return Err(Error::Contract("host table matrix shape disagrees"));
            }
            let data = view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(f16::from_le_bytes)
                .collect();
            Ok(MatrixF16 { rows, cols, data })
        };
        let read_norm = |name: &str, expected_len: usize| -> Result<Vec<f32>, Error> {
            let view = tensors
                .tensor(name)
                .map_err(|_| Error::Contract("missing norm tensor in host tables"))?;
            if view.dtype() != Dtype::F16
                || view.shape() != [expected_len]
                || view.data().len() != expected_len * 2
            {
                return Err(Error::Contract("host table norm shape or dtype disagrees"));
            }
            Ok(view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(|pair| f16::from_le_bytes(pair).to_f32())
                .collect())
        };
        let codec_embeddings = (0..groups)
            .map(|group| read_matrix(&format!("codec_embeddings.{group}"), None, talker_hidden))
            .collect::<Result<Vec<_>, _>>()?;
        let talker_norm = read_norm("talker_norm", talker_hidden)?;
        let talker_head = read_matrix("talker_head", Some(talker_vocab), talker_hidden)?;
        let predictor_norm = read_norm("predictor_norm", predictor_hidden)?;
        let predictor_heads = (0..groups - 1)
            .map(|group| read_matrix(&format!("predictor_heads.{group}"), None, predictor_hidden))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            codec_embeddings,
            talker_norm,
            talker_eps,
            talker_head,
            predictor_norm,
            predictor_eps,
            predictor_heads,
        })
    }

    fn code(&self, group: usize, token: u32) -> Result<Tensor<f16>, Error> {
        let table = self
            .codec_embeddings
            .get(group)
            .ok_or(Error::Contract("codec embedding group out of range"))?;
        let row = table
            .row(token as usize)
            .ok_or(Error::Contract("codec token out of range"))?;
        Ok(Tensor::from_slice(&[1, table.cols, 1, 1], row)?)
    }

    fn frame(&self, codes: &[u32], text: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
        if codes.len() != self.codec_embeddings.len() {
            return Err(Error::Contract("codec frame group count disagrees"));
        }
        let hidden = self.codec_embeddings[0].cols;
        let text_view = text.read();
        if text_view.shape() != [1, hidden, 1, 1] {
            return Err(Error::Contract("text embedding shape disagrees"));
        }
        let mut acc: Vec<f32> = text_view.data().iter().map(|v| v.to_f32()).collect();
        for (table, &code) in self.codec_embeddings.iter().zip(codes) {
            let row = table
                .row(code as usize)
                .ok_or(Error::Contract("codec token out of range"))?;
            for (sum, &value) in acc.iter_mut().zip(row) {
                *sum += value.to_f32();
            }
        }
        let values: Vec<f16> = acc.into_iter().map(f16::from_f32).collect();
        Ok(Tensor::from_slice(&[1, hidden, 1, 1], &values)?)
    }

    fn talker_head(&self, hidden: &Tensor<f16>) -> Result<(Tensor<f16>, Vec<f32>), Error> {
        let normalized = rms_norm(hidden, &self.talker_norm, self.talker_eps)?;
        let logits = self.talker_head.gemv(&normalized);
        let normalized_f16: Vec<f16> = normalized.into_iter().map(f16::from_f32).collect();
        Ok((
            Tensor::from_slice(&[1, self.talker_norm.len(), 1, 1], &normalized_f16)?,
            logits,
        ))
    }

    fn predictor_head(&self, group: usize, hidden: &Tensor<f16>) -> Result<Vec<f32>, Error> {
        let weight = self
            .predictor_heads
            .get(group)
            .ok_or(Error::Contract("predictor head group out of range"))?;
        let normalized = rms_norm(hidden, &self.predictor_norm, self.predictor_eps)?;
        Ok(weight.gemv(&normalized))
    }
}

fn rms_norm(hidden: &Tensor<f16>, weight: &[f32], eps: f32) -> Result<Vec<f32>, Error> {
    let view = hidden.read();
    let cols = weight.len();
    if view.shape() != [1, cols, 1, 1] {
        return Err(Error::Contract("output head hidden shape disagrees"));
    }
    let data = view.data();
    let mut values = Vec::with_capacity(cols);
    let mut sum_sq = 0.0f32;
    for &x in data {
        let v = x.to_f32();
        sum_sq += v * v;
        values.push(v);
    }
    let inv_rms = (sum_sq / cols as f32 + eps).sqrt().recip();
    for (v, &w) in values.iter_mut().zip(weight) {
        *v *= inv_rms * w;
    }
    Ok(values)
}

enum CodeTables {
    CoreMl {
        codec_embeddings: Vec<Function>,
        codec_frame: Function,
        talker_head: Function,
        predictor_heads: Vec<Function>,
    },
    Host(HostTables),
}

impl CodeTables {
    fn groups(&self) -> usize {
        match self {
            Self::CoreMl {
                codec_embeddings, ..
            } => codec_embeddings.len(),
            Self::Host(host) => host.codec_embeddings.len(),
        }
    }
}

pub struct CoreMl {
    #[cfg(all(test, feature = "cli"))]
    pub(crate) profile: crate::profiling::Profile,
    talker: Decoder,
    predictor: Decoder,
    tables: CodeTables,
    predictor_projection: Option<Function>,
    text_embedding: Function,
    speaker: Function,
    encoder: Function,
}

/// One official codec block, retaining only its causal execution history.
pub struct CoreMlCodec {
    #[cfg(all(test, feature = "cli"))]
    pub(crate) profile: crate::profiling::Profile,
    #[cfg(feature = "streaming")]
    pub(crate) units: ComputeUnits,
    session: Session,
    cos_table: Vec<f16>,
    sin_table: Vec<f16>,
    head_dim: usize,
    capacity: usize,
    window: usize,
    groups: usize,
    samples_per_frame: usize,
    batch_frames: BatchSizes,
    packed: Vec<i32>,
    samples: Vec<f32>,
    frames: usize,
}

impl CoreMlCodec {
    fn open(
        path: &Path,
        units: impl FnOnce(bool) -> ComputeUnits,
        samples_per_frame: usize,
    ) -> Result<Self, Error> {
        #[derive(Deserialize)]
        struct Contract {
            execution: String,
            attention_window: usize,
            rotary_frequencies: Vec<f32>,
            rotary_scaling: f32,
            code_groups: usize,
            capacity: usize,
            #[serde(default)]
            batch_frames: BatchSizes,
        }
        let contract: Contract = json(path.with_extension("contract.json"))?;
        if contract.execution != "stateful_frame" || !contract.batch_frames.fits(contract.capacity)
        {
            return Err(Error::Contract(
                "invalid stateful waveform execution contract",
            ));
        }
        let units = units(contract.batch_frames.flexible());
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
                output.data_type == DataType::Float32
                    && (output.default_shape == [1, 1, samples_per_frame]
                        || (contract.batch_frames.flexible() && output.default_shape.is_empty()))
            })
        {
            return Err(Error::Contract(
                "waveform decoder requires state and PCM output",
            ));
        }
        let mut cos_table = Vec::with_capacity(contract.capacity * head_dim);
        let mut sin_table = Vec::with_capacity(contract.capacity * head_dim);
        for position in 0..contract.capacity {
            let start = cos_table.len();
            for &frequency in &contract.rotary_frequencies {
                let (sine, cosine) = (frequency * position as f32).sin_cos();
                cos_table.push(f16::from_f32(cosine * contract.rotary_scaling));
                sin_table.push(f16::from_f32(sine * contract.rotary_scaling));
            }
            cos_table.extend_from_within(start..);
            sin_table.extend_from_within(start..);
        }
        Ok(Self {
            #[cfg(all(test, feature = "cli"))]
            profile: crate::profiling::Profile::default(),
            #[cfg(feature = "streaming")]
            units,
            session: function.into_session(),
            cos_table,
            sin_table,
            head_dim,
            capacity: contract.capacity,
            window: contract.attention_window,
            groups: contract.code_groups,
            samples_per_frame,
            batch_frames: contract.batch_frames,
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

    pub(crate) fn prefix(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        if groups != self.groups || packed.len() != groups * frames || frames > self.capacity {
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
        let mut index = self.frames;
        while let Some(remaining) = NonZeroUsize::new(frames - index) {
            let length = self.batch_frames.batch(remaining);
            let result = self.step(&packed, frames, index, length);
            match result {
                Ok(samples) => self.samples.extend(samples),
                Err(error) => {
                    self.reset();
                    return Err(error);
                }
            }
            index += length;
        }
        self.frames = frames;
        self.packed = packed;
        Ok(self.samples.clone())
    }

    fn step(
        &mut self,
        packed: &[i32],
        frames: usize,
        index: usize,
        length: usize,
    ) -> Result<Vec<f32>, Error> {
        let codes: Vec<_> = (0..self.groups)
            .flat_map(|group| {
                packed[group * frames + index..group * frames + index + length]
                    .iter()
                    .copied()
            })
            .collect();
        let codes = Tensor::from_slice(&[1, self.groups, length], &codes)?;
        let head_dim = self.head_dim;
        let rotary_range = index * head_dim..(index + length) * head_dim;
        let cosine = Tensor::from_slice(
            &[1, 1, length, head_dim],
            &self.cos_table[rotary_range.clone()],
        )?;
        let sine = Tensor::from_slice(&[1, 1, length, head_dim], &self.sin_table[rotary_range])?;
        let columns = self.window + length - 1;
        let mut mask = vec![f16::MIN; length * columns];
        // The graph joins the previous window minus its oldest frame with this batch.
        let first_present = (self.window - 1).saturating_sub(index);
        for frame in 0..length {
            let offset = frame * columns;
            mask[offset + first_present.max(frame)..offset + self.window + frame].fill(f16::ZERO);
        }
        let mask = Tensor::from_slice(&[1, 1, length, columns], &mask)?;
        let mut inputs = Inputs::new();
        inputs.insert("codes", &codes)?;
        inputs.insert("cosine", &cosine)?;
        inputs.insert("sine", &sine)?;
        inputs.insert("mask", &mask)?;
        let audio = self
            .session
            .predict(&inputs)
            .map_err(|source| Error::CoreMlOperation {
                operation: "waveform decoder",
                source,
            })?
            .tensor::<f32>("audio")?;
        if audio.shape() != [1, 1, length * self.samples_per_frame] {
            return Err(Error::Contract(
                "waveform decoder returned an unexpected PCM span",
            ));
        }
        Ok(audio.read().into_data())
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
        Self::open_with_waveform(transformers, audio, units, sampling, |_, _, _| units)
    }
    pub(crate) fn open_with_waveform(
        transformers: &Path,
        audio: &Path,
        units: ComputeUnits,
        sampling: Sampling<T, S>,
        waveform_units: impl FnOnce(&str, ComputeUnits, bool) -> ComputeUnits,
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
            Decoder::open_with_lengths(
                files.chunks.iter().map(|p| transformers.join(p)),
                files.rope_theta,
                files.sequence_lengths.clone(),
                units,
            )
        };
        let talker = load_decoder(&manifest.decoders.talker)?;
        let predictor = load_decoder(&manifest.decoders.predictor)?;
        let tables = if let Some(tables_file) = &manifest.tables {
            if !manifest.codec_embeddings.is_empty()
                || manifest.codec_frame.is_some()
                || !manifest.decoders.talker.heads.is_empty()
                || !manifest.decoders.predictor.heads.is_empty()
            {
                return Err(Error::Contract(
                    "host tables manifest must not also declare Core ML embedding/head models",
                ));
            }
            let talker_eps = manifest
                .decoders
                .talker
                .rms_norm_eps
                .ok_or(Error::Contract("host tables require Talker rms_norm_eps"))?;
            let predictor_eps = manifest
                .decoders
                .predictor
                .rms_norm_eps
                .ok_or(Error::Contract(
                    "host tables require Predictor rms_norm_eps",
                ))?;
            CodeTables::Host(HostTables::open(
                &transformers.join(tables_file),
                config.talker_config.num_code_groups,
                talker.hidden_size(),
                config.talker_config.vocab_size as usize,
                talker_eps,
                predictor.hidden_size(),
                predictor_eps,
            )?)
        } else {
            if manifest.decoders.talker.heads.len() != 1 {
                return Err(Error::Contract("Talker requires exactly one output head"));
            }
            let codec_frame = manifest
                .codec_frame
                .as_deref()
                .ok_or(Error::Contract("legacy manifest requires codec_frame"))?;
            CodeTables::CoreMl {
                talker_head: open(transformers.join(&manifest.decoders.talker.heads[0]), units)?,
                predictor_heads: manifest
                    .decoders
                    .predictor
                    .heads
                    .iter()
                    .map(|p| open(transformers.join(p), units))
                    .collect::<Result<_, _>>()?,
                codec_embeddings: manifest
                    .codec_embeddings
                    .iter()
                    .map(|p| open(transformers.join(p), units))
                    .collect::<Result<_, _>>()?,
                codec_frame: open(transformers.join(codec_frame), units)?,
            }
        };
        let backend = CoreMl {
            #[cfg(all(test, feature = "cli"))]
            profile: crate::profiling::Profile::default(),
            talker,
            predictor,
            tables,
            predictor_projection: manifest
                .predictor_projection
                .map(|p| open(transformers.join(p), units))
                .transpose()?,
            text_embedding: open(transformers.join(manifest.text_embedding), units)?,
            speaker: open(audio.join("speaker.mlmodelc"), units)?,
            encoder: open(audio.join("encoder.mlmodelc"), units)?,
        };
        let codec = CoreMlCodec::open(
            &audio.join("decoder.mlmodelc"),
            |batched| waveform_units(&config.tts_model_size, units, batched),
            decoder_contract.samples_per_frame,
        )?;
        Self::new(
            backend,
            codec,
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
    type Codec = CoreMlCodec;
    fn code_token(&mut self, group: usize, token: &u32) -> Result<Self::Hidden, Error> {
        self.code(group, *token)
    }
    async fn read_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>, Error> {
        Ok(tokens.to_vec())
    }
    fn groups(&self) -> usize {
        self.tables.groups()
    }
    fn context_capacity(&self) -> usize {
        self.talker.capacity()
    }
    fn reset_generation(&mut self) {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("generation_reset");
        self.talker.reset();
        self.predictor.reset();
    }
    fn reset_predictor(&mut self) {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("predictor_reset");
        self.predictor.reset();
    }
    fn text(&mut self, token: u32) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("text_embedding");
        embed(&mut self.text_embedding, token, "text embedding")
    }
    fn code(&mut self, group: usize, token: u32) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("code_embedding");
        match &mut self.tables {
            CodeTables::CoreMl {
                codec_embeddings, ..
            } => embed(&mut codec_embeddings[group], token, "codec embedding"),
            CodeTables::Host(host) => host.code(group, token),
        }
    }
    fn frame(&mut self, codes: &[u32], text: &Self::Hidden) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("frame_embedding");
        match &mut self.tables {
            CodeTables::CoreMl { codec_frame, .. } => frame(codec_frame, codes, text),
            CodeTables::Host(host) => host.frame(codes, text),
        }
    }
    fn add(a: &Self::Hidden, b: &Self::Hidden) -> Result<Self::Hidden, Error> {
        add(a, b)
    }
    fn talker(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("talker");
        Ok(self.talker.step(hidden)?)
    }
    fn prefill(
        &mut self,
        first: Self::Hidden,
        remaining: impl IntoIterator<Item = Self::Hidden>,
    ) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("prefill");
        Ok(self.talker.prefill(first, remaining)?)
    }
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Vec<f32>), Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("talker_head");
        match &mut self.tables {
            CodeTables::CoreMl { talker_head, .. } => head(talker_head, &hidden, "Talker head"),
            CodeTables::Host(host) => host.talker_head(&hidden),
        }
    }
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("projection");
        if let Some(function) = &mut self.predictor_projection {
            let mut inputs = Inputs::new();
            inputs.insert("hidden", &hidden)?;
            Ok(function
                .predict(&inputs)
                .map_err(|source| Error::CoreMlOperation {
                    operation: "detail predictor projection",
                    source,
                })?
                .tensor("hidden_out")?)
        } else {
            Ok(hidden)
        }
    }
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("predictor");
        Ok(self.predictor.step(hidden)?)
    }
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Vec<f32>, Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("predictor_head");
        match &mut self.tables {
            CodeTables::CoreMl {
                predictor_heads, ..
            } => Ok(head(
                &mut predictor_heads[group],
                &hidden,
                "detail predictor head",
            )?
            .1),
            CodeTables::Host(host) => host.predictor_head(group, &hidden),
        }
    }
    async fn condition(
        &mut self,
        samples: &[f32],
        mel: &[f32],
        channels: usize,
    ) -> Result<(Self::Hidden, Vec<Vec<u32>>), Error> {
        #[cfg(all(test, feature = "cli"))]
        let _measurement = self.profile.measure("conditioning_models");
        let mel = Tensor::from_slice(&[1, mel.len() / channels, channels], mel)?;
        let mut inputs = Inputs::new();
        inputs.insert("mel", &mel)?;
        let speaker = self
            .speaker
            .predict(&inputs)
            .map_err(|source| Error::CoreMlOperation {
                operation: "speaker encoder",
                source,
            })?
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
            .predict(&inputs)
            .map_err(|source| Error::CoreMlOperation {
                operation: "reference codec encoder",
                source,
            })?
            .tensor::<i32>("codes")?
            .read();
        let shape = prediction.shape();
        let groups = self.tables.groups();
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
}

impl Codec for CoreMlCodec {
    fn reset(&mut self) {
        CoreMlCodec::reset(self);
    }
    async fn decode_prefix(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        #[cfg(all(test, feature = "cli"))]
        let started = std::time::Instant::now();
        let result = self.prefix(packed, groups, frames);
        #[cfg(all(test, feature = "cli"))]
        self.profile.record("waveform", started.elapsed());
        result
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
        .map_err(|source| Error::CoreMlOperation {
            operation: "codec frame embedding",
            source,
        })?
        .tensor("hidden")?)
}

fn embed(
    function: &mut Function,
    token: u32,
    operation: &'static str,
) -> Result<Tensor<f16>, Error> {
    let token = i32::try_from(token).map_err(|_| Error::Contract("token exceeds int32"))?;
    let tensor = Tensor::from_slice(&[1, 1], &[token])?;
    let mut inputs = Inputs::new();
    inputs.insert("token", &tensor)?;
    Ok(function
        .predict(&inputs)
        .map_err(|source| Error::CoreMlOperation { operation, source })?
        .tensor("hidden")?)
}

fn head(
    function: &mut Function,
    hidden: &Tensor<f16>,
    operation: &'static str,
) -> Result<(Tensor<f16>, Vec<f32>), Error> {
    let mut inputs = Inputs::new();
    inputs.insert("hidden", hidden)?;
    let prediction = function
        .predict(&inputs)
        .map_err(|source| Error::CoreMlOperation { operation, source })?;
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
    use std::{
        env, fs,
        io::{BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };

    #[test]
    fn host_tables_gemv_norm_and_embeddings_match_fp32_reference() {
        let cols = 16;
        let rows = 10;
        let data: Vec<f16> = (0..rows * cols)
            .map(|i| f16::from_f32(((i % 13) as f32 - 6.0) * 0.125))
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 - 7.5) * 0.25).collect();
        let matrix = MatrixF16 {
            rows,
            cols,
            data: data.clone(),
        };
        let actual = matrix.gemv(&x);
        for r in 0..rows {
            let expected: f32 = (0..cols).map(|c| data[r * cols + c].to_f32() * x[c]).sum();
            assert!(
                (actual[r] - expected).abs() < 1e-5,
                "row {r}: actual={} expected={expected}",
                actual[r]
            );
        }

        let norm_weight: Vec<f32> = (0..cols).map(|i| 0.5 + i as f32 * 0.05).collect();
        let hidden_f16: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
        let hidden = Tensor::from_slice(&[1, cols, 1, 1], &hidden_f16).unwrap();
        let eps = 1e-6;
        let host = HostTables {
            codec_embeddings: vec![
                MatrixF16 {
                    rows,
                    cols,
                    data: data.clone(),
                },
                MatrixF16 {
                    rows,
                    cols,
                    data: data.iter().map(|v| f16::from_f32(-v.to_f32())).collect(),
                },
            ],
            talker_norm: norm_weight.clone(),
            talker_eps: eps,
            talker_head: MatrixF16 {
                rows,
                cols,
                data: data.clone(),
            },
            predictor_norm: norm_weight.clone(),
            predictor_eps: eps,
            predictor_heads: vec![MatrixF16 {
                rows,
                cols,
                data: data.clone(),
            }],
        };

        let emb0 = host.code(0, 3).unwrap();
        assert_eq!(emb0.shape(), [1, cols, 1, 1]);
        assert_eq!(emb0.read().data(), &data[3 * cols..4 * cols]);

        // Opposite embeddings for token 2 cancel, leaving `hidden` intact.
        let frame = host.frame(&[2, 2], &hidden).unwrap();
        assert_eq!(frame.read().data(), &hidden_f16);

        let (norm_tensor, talker_logits) = host.talker_head(&hidden).unwrap();
        let pred_logits = host.predictor_head(0, &hidden).unwrap();
        assert_eq!(talker_logits, pred_logits);
        let x_rounded: Vec<f32> = hidden_f16.iter().map(|v| v.to_f32()).collect();
        let mean_sq = x_rounded.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let expected_norm: Vec<f32> = x_rounded
            .iter()
            .zip(&norm_weight)
            .map(|(&v, &w)| v * inv_rms * w)
            .collect();
        for (a, e) in norm_tensor.read().data().iter().zip(&expected_norm) {
            assert!((a.to_f32() - e).abs() < 2e-3);
        }
        for r in 0..rows {
            let expected: f32 = (0..cols)
                .map(|c| data[r * cols + c].to_f32() * expected_norm[c])
                .sum();
            assert!((talker_logits[r] - expected).abs() < 1e-5);
        }
    }

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
        let mut waveform = CoreMlCodec::open(
            &decoder,
            |_| ComputeUnits::CpuAndNeuralEngine,
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
        let mut samples = BufWriter::new(File::create(output.join("audio.f32le")).unwrap());
        for value in &actual {
            samples.write_all(&value.to_le_bytes()).unwrap();
        }
        samples.flush().unwrap();
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

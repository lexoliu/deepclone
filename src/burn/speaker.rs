use super::{
    Error, Weights,
    layers::{Conv, ReflectSame},
};
use burn::tensor::{Tensor, activation};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct SpeakerConfig {
    pub enc_channels: Vec<usize>,
    pub enc_kernel_sizes: Vec<usize>,
    pub enc_dilations: Vec<usize>,
    pub enc_res2net_scale: usize,
}
struct Residual {
    first: Conv<ReflectSame>,
    blocks: Vec<Conv<ReflectSame>>,
    last: Conv<ReflectSame>,
    squeeze: Conv<ReflectSame>,
    excite: Conv<ReflectSame>,
}
fn conv(w: &Weights<'_>, prefix: &str, dilation: usize) -> Result<Conv<ReflectSame>, Error> {
    Conv::load(w, prefix, 1, dilation, 1, ReflectSame)
}
impl Residual {
    fn load(w: &Weights<'_>, p: &str, scale: usize, dilation: usize) -> Result<Self, Error> {
        Ok(Self {
            first: conv(w, &format!("{p}.tdnn1.conv"), 1)?,
            blocks: (0..scale - 1)
                .map(|i| conv(w, &format!("{p}.res2net_block.blocks.{i}.conv"), dilation))
                .collect::<Result<_, _>>()?,
            last: conv(w, &format!("{p}.tdnn2.conv"), 1)?,
            squeeze: conv(w, &format!("{p}.se_block.conv1"), 1)?,
            excite: conv(w, &format!("{p}.se_block.conv2"), 1)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = activation::relu(self.first.forward(x.clone()));
        let [batch, channels, time] = hidden.dims();
        let width = channels / (self.blocks.len() + 1);
        let mut parts = vec![hidden.clone().slice([0..batch, 0..width, 0..time])];
        for (i, block) in self.blocks.iter().enumerate() {
            let part = hidden
                .clone()
                .slice([0..batch, (i + 1) * width..(i + 2) * width, 0..time]);
            let part = if i == 0 {
                part
            } else {
                part + parts[i].clone()
            };
            parts.push(activation::relu(block.forward(part)));
        }
        let hidden = activation::relu(self.last.forward(Tensor::cat(parts, 1)));
        let gate = activation::sigmoid(self.excite.forward(activation::relu(
            self.squeeze.forward(hidden.clone().mean_dim(2)),
        )));
        x + hidden * gate
    }
}
/// FP32 ECAPA-TDNN speaker conditioning, including attentive statistics pooling.
pub struct Speaker {
    first: Conv<ReflectSame>,
    blocks: Vec<Residual>,
    aggregate: Conv<ReflectSame>,
    attention: Conv<ReflectSame>,
    attention_output: Conv<ReflectSame>,
    output: Conv<ReflectSame>,
}
impl Speaker {
    pub fn load(w: &Weights<'_>, c: &SpeakerConfig) -> Result<Self, Error> {
        if c.enc_channels.len() < 3
            || c.enc_channels.len() != c.enc_dilations.len()
            || c.enc_channels.len() != c.enc_kernel_sizes.len()
            || c.enc_res2net_scale == 0
        {
            return Err(Error::Tensor {
                name: "speaker configuration".into(),
                reason: "invalid channel, dilation, kernel or Res2Net dimensions".into(),
            });
        }
        let p = "speaker_encoder";
        Ok(Self {
            first: conv(w, &format!("{p}.blocks.0.conv"), c.enc_dilations[0])?,
            blocks: (1..c.enc_channels.len() - 1)
                .map(|i| {
                    Residual::load(
                        w,
                        &format!("{p}.blocks.{i}"),
                        c.enc_res2net_scale,
                        c.enc_dilations[i],
                    )
                })
                .collect::<Result<_, _>>()?,
            aggregate: conv(
                w,
                &format!("{p}.mfa.conv"),
                *c.enc_dilations.last().unwrap(),
            )?,
            attention: conv(w, &format!("{p}.asp.tdnn.conv"), 1)?,
            attention_output: conv(w, &format!("{p}.asp.conv"), 1)?,
            output: conv(w, &format!("{p}.fc"), 1)?,
        })
    }
    pub fn forward(&self, mel: Tensor<3>) -> Tensor<3> {
        let mut hidden = activation::relu(self.first.forward(mel.swap_dims(1, 2)));
        let mut parts = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            hidden = block.forward(hidden);
            parts.push(hidden.clone());
        }
        let hidden = activation::relu(self.aggregate.forward(Tensor::cat(parts, 1)));
        let time = hidden.dims()[2];
        let mean = hidden.clone().mean_dim(2);
        let delta = hidden.clone() - mean.clone();
        // Epsilon is specified by Qwen's AttentiveStatisticsPooling.
        let std = (delta.clone() * delta).mean_dim(2).clamp_min(1e-12).sqrt();
        let context = Tensor::cat(
            vec![
                hidden.clone(),
                mean.repeat_dim(2, time),
                std.repeat_dim(2, time),
            ],
            1,
        );
        let attention = activation::softmax(
            self.attention_output
                .forward(activation::relu(self.attention.forward(context)).tanh()),
            2,
        );
        let mean = (attention.clone() * hidden.clone()).sum_dim(2);
        let delta = hidden - mean.clone();
        let std = (attention * delta.clone() * delta)
            .sum_dim(2)
            .clamp_min(1e-12)
            .sqrt();
        self.output.forward(Tensor::cat(vec![mean, std], 1))
    }
}

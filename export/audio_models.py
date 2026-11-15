"""Traceable audio codec graphs with explicit causal attention masks."""

import copy

import torch
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2CausalTransConvNet,
)
from torch import nn
from torch.nn import functional as F
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiEuclideanCodebook


class CausalTranspose(nn.Module):
    """Exact polyphase form of a causal, right-trimmed transposed convolution."""

    def __init__(self, original):
        super().__init__()
        conv = original.conv
        self.stride = conv.stride[0]
        if (
            conv.groups != 1
            or conv.dilation != (1,)
            or conv.padding != (0,)
            or conv.output_padding != (0,)
        ):
            raise ValueError(
                "Causal transpose requires ungrouped, undilated, unpadded convolution"
            )
        if (
            original.left_pad != 0
            or original.right_pad != conv.kernel_size[0] - self.stride
        ):
            raise ValueError(
                "Causal transpose requires the upstream right-trim boundary"
            )
        self.taps = (conv.kernel_size[0] + self.stride - 1) // self.stride
        weight = F.pad(
            conv.weight.detach(), (0, self.taps * self.stride - conv.kernel_size[0])
        )
        weight = weight.reshape(
            conv.in_channels, conv.out_channels, self.taps, self.stride
        )
        weight = (
            weight.permute(1, 3, 0, 2)
            .flip(-1)
            .reshape(conv.out_channels * self.stride, conv.in_channels, 1, self.taps)
        )
        self.weight = nn.Parameter(weight.contiguous())
        self.bias = (
            None if conv.bias is None else nn.Parameter(conv.bias.detach().clone())
        )
        self.channels = conv.out_channels

    def forward(self, hidden):
        x = F.pad(hidden.unsqueeze(2), (self.taps - 1, 0))
        output = F.conv2d(x, self.weight)
        output = output.reshape(
            hidden.shape[0], self.channels, self.stride, hidden.shape[-1]
        )
        output = output.transpose(2, 3).reshape(hidden.shape[0], self.channels, -1)
        return output if self.bias is None else output + self.bias.reshape(1, -1, 1)


def lower_decoder(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Qwen3TTSTokenizerV2CausalTransConvNet):
            setattr(module, name, CausalTranspose(child))
        else:
            lower_decoder(child)


class CausalConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not original.causal:
            raise ValueError("Audio encoder requires causal convolutions")
        self.conv = original.conv
        stride = self.conv.stride[0]
        kernel = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        # A maximum right pad gives exactly ceil(length / stride) outputs.
        # Unused padding never enters a convolution window.
        self.padding = (kernel - stride, stride - 1)
        self.mode = original.pad_mode

    def forward(self, hidden):
        return self.conv(F.pad(hidden, self.padding, mode=self.mode))


class NearestCode(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("embed", original.embed.detach().clone())

    def forward(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        weights = self.embed.float()
        x_norm = (flat * flat).sum(dim=1, keepdim=True)
        w_norm = (weights * weights).sum(dim=1, keepdim=True)
        x_augmented = torch.cat((-2 * flat, x_norm, torch.ones_like(x_norm)), dim=1)
        w_augmented = torch.cat((weights, torch.ones_like(w_norm), w_norm), dim=1)
        distances = torch.matmul(x_augmented, w_augmented.T)
        return distances.clamp_min(0.0).sqrt().argmin(dim=-1).reshape(hidden.shape[:-1])

    def encode(self, hidden):
        return self(hidden)

    def decode(self, codes):
        return F.embedding(codes, self.embed)


def lower_encoder(module):
    for name, child in list(module.named_children()):
        if isinstance(child, MimiConv1d):
            setattr(module, name, CausalConv(child))
        elif isinstance(child, MimiEuclideanCodebook):
            setattr(module, name, NearestCode(child))
        else:
            lower_encoder(child)


def causal_mask(length, window=None):
    positions = torch.arange(length)
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if window is not None:
        allowed = allowed & (distance < window)
    mask = torch.where(
        allowed,
        torch.zeros_like(allowed, dtype=torch.float32),
        torch.full_like(allowed, torch.finfo(torch.float32).min, dtype=torch.float32),
    )
    return mask.unsqueeze(0).unsqueeze(0)


class AudioEncoder(nn.Module):
    def __init__(self, model, samples):
        super().__init__()
        self.model = copy.deepcopy(model.encoder)
        lower_encoder(self.model)
        self.quantizers = model.encoder_valid_num_quantizers

    def forward(self, audio):
        hidden = self.model.encoder(audio).transpose(1, 2)
        mask = causal_mask(hidden.shape[1])
        positions = torch.arange(hidden.shape[1]).reshape(1, -1)
        for layer in self.model.encoder_transformer.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            )[0]
        hidden = self.model.downsample(hidden.transpose(1, 2))
        return (
            self.model.quantizer.encode(hidden, self.quantizers)
            .transpose(0, 1)
            .to(torch.int32)
        )


class AudioDecoder(nn.Module):
    def __init__(self, decoder, frames):
        super().__init__()
        self.decoder = copy.deepcopy(decoder)
        lower_decoder(self.decoder)
        self.register_buffer("full_mask", causal_mask(frames))
        self.register_buffer(
            "sliding_mask", causal_mask(frames, decoder.config.sliding_window)
        )

    def forward(self, codes):
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        hidden = self.decoder.pre_transformer(
            inputs_embeds=hidden,
            attention_mask={
                "full_attention": self.full_mask,
                "sliding_attention": self.sliding_mask,
            },
            use_cache=False,
        ).last_hidden_state.transpose(1, 2)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in self.decoder.decoder:
            hidden = block(hidden)
        return hidden.clamp(-1, 1)

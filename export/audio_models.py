"""Traceable audio codec graphs with explicit causal attention masks."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiEuclideanCodebook


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
        allowed &= distance < window
    return torch.where(allowed, 0.0, torch.finfo(torch.float32).min).reshape(
        1, 1, length, length
    )


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
        self.decoder = decoder
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

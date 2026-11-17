"""Stateful, causal waveform reconstruction with one codec frame per call."""

import copy

import torch
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    SnakeBeta,
)
from torch import nn
from torch.nn import functional as F

from audio_models import CausalTranspose


def metadata(frequencies, scaling, window, position, dtype=torch.float32):
    phase = torch.as_tensor(frequencies, dtype=torch.float32) * position
    phase = torch.cat((phase, phase)).reshape(1, 1, 1, -1)
    cosine = (phase.cos() * scaling).to(dtype)
    sine = (phase.sin() * scaling).to(dtype)
    mask = torch.full((1, 1, 1, window), torch.finfo(dtype).min, dtype=dtype)
    mask[..., -min(position + 1, window) :] = 0
    return cosine, sine, mask


class StreamingConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.conv = original.conv
        if self.conv.stride != (1,):
            raise ValueError(
                "Waveform decoder requires unit-stride causal convolutions"
            )
        self.context = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        self.depthwise = (
            self.conv.groups == self.conv.in_channels == self.conv.out_channels
        )
        if self.conv.groups != 1 and not self.depthwise:
            raise ValueError("Decoder convolutions must be dense or depthwise")
        if self.context:
            self.register_buffer(
                "history", torch.zeros(1, 1, self.context, self.conv.in_channels)
            )

    def forward(self, hidden):
        values = hidden.permute(0, 2, 3, 1)
        joined = torch.cat((self.history, values), dim=2) if self.context else values
        windows = [
            joined[
                :,
                :,
                tap * self.conv.dilation[0] : tap * self.conv.dilation[0]
                + hidden.shape[-1],
                :,
            ]
            for tap in range(self.conv.kernel_size[0])
        ]
        if self.depthwise:
            output = windows[0] * self.conv.weight[:, 0, 0].reshape(1, 1, 1, -1)
            for tap in range(1, self.conv.kernel_size[0]):
                output = output + windows[tap] * self.conv.weight[:, 0, tap].reshape(
                    1, 1, 1, -1
                )
            if self.conv.bias is not None:
                output = output + self.conv.bias.reshape(1, 1, 1, -1)
        else:
            # Concatenate taps in temporal order; permute the kernel to the
            # same [tap, input-channel] order before its dense projection.
            weight = self.conv.weight.permute(0, 2, 1).reshape(
                self.conv.out_channels, -1
            )
            output = F.linear(torch.cat(windows, dim=-1), weight, self.conv.bias)
        if self.context:
            self.history[:, :, :, :] = joined[:, :, -self.context :, :]
        return output.permute(0, 3, 1, 2)


class StreamingTranspose(CausalTranspose):
    def __init__(self, original):
        super().__init__(original)
        if self.taps > 1:
            self.register_buffer(
                "history", torch.zeros(1, 1, self.taps - 1, original.conv.in_channels)
            )

    def forward(self, hidden):
        values = hidden.permute(0, 2, 3, 1)
        if self.taps > 1:
            joined = torch.cat((self.history, values), dim=2)
            self.history[:, :, :, :] = joined[:, :, -(self.taps - 1) :, :]
        else:
            joined = values
        windows = [
            joined[:, :, tap : tap + hidden.shape[-1], :] for tap in range(self.taps)
        ]
        weight = (
            self.weight[:, :, 0, :]
            .permute(0, 2, 1)
            .reshape(self.channels * self.stride, -1)
        )
        output = F.linear(torch.cat(windows, dim=-1), weight)
        output = output.reshape(
            hidden.shape[0], 1, hidden.shape[-1], self.channels, self.stride
        )
        output = output.transpose(3, 4).reshape(hidden.shape[0], 1, -1, self.channels)
        if self.bias is not None:
            output = output + self.bias.reshape(1, 1, 1, -1)
        return output.permute(0, 3, 1, 2)


class StreamingConvNext(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.dwconv = StreamingConv(original.dwconv)
        self.norm = original.norm
        self.first = original.pwconv1
        self.activation = original.act
        self.last = original.pwconv2
        self.gamma = original.gamma

    def forward(self, hidden):
        output = self.dwconv(hidden).permute(0, 2, 3, 1)
        output = self.norm(output)
        output = self.last(self.activation(self.first(output))) * self.gamma
        return hidden + output.permute(0, 3, 1, 2)


class StreamingSnake(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.alpha = nn.Parameter(original.alpha.detach().clone().reshape(1, 1, 1, -1))
        self.beta = nn.Parameter(original.beta.detach().clone().reshape(1, 1, 1, -1))
        self.epsilon = original.no_div_by_zero

    def forward(self, hidden):
        values = hidden.permute(0, 2, 3, 1)
        alpha = self.alpha.exp()
        beta = self.beta.exp()
        output = values + (1.0 / (beta + self.epsilon)) * (values * alpha).sin().pow(2)
        return output.permute(0, 3, 1, 2)


class StreamingQuantizer(nn.Module):
    def __init__(self, original):
        super().__init__()
        first, rest = original.rvq_first, original.rvq_rest
        layers = [*first.vq.layers, *rest.vq.layers]
        # Normalize in source precision before casting the frozen embeddings.
        books = [
            layer._codebook.embedding_sum.detach()
            / layer._codebook.cluster_usage.detach().clamp_min(layer._codebook.epsilon)[
                :, None
            ]
            for layer in layers
        ]
        self.embedding = nn.Parameter(torch.cat(books))
        self.offsets = tuple(
            sum(book.shape[0] for book in books[:index]) for index in range(len(books))
        )
        self.projections = nn.ModuleList(layer.project_out for layer in layers)
        self.semantic_groups = len(first.vq.layers)
        self.semantic_output = first.output_proj
        self.acoustic_output = rest.output_proj

    def forward(self, codes):
        # A single indexed lookup retains every group's independent codebook.
        indices = codes.reshape(1, len(self.offsets)) + torch.tensor(
            self.offsets, dtype=codes.dtype
        )
        decoded = F.embedding(indices, self.embedding)
        groups = [
            projection(decoded[:, index : index + 1, :]).transpose(1, 2)
            for index, projection in enumerate(self.projections)
        ]
        semantic = groups[0]
        for group in groups[1 : self.semantic_groups]:
            semantic = semantic + group
        acoustic = groups[self.semantic_groups]
        for group in groups[self.semantic_groups + 1 :]:
            acoustic = acoustic + group
        return self.semantic_output(semantic) + self.acoustic_output(acoustic)


def lower(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Qwen3TTSTokenizerV2ConvNeXtBlock):
            setattr(module, name, StreamingConvNext(child))
        elif isinstance(child, SnakeBeta):
            setattr(module, name, StreamingSnake(child))
        elif isinstance(child, Qwen3TTSTokenizerV2CausalConvNet):
            setattr(module, name, StreamingConv(child))
        elif isinstance(child, Qwen3TTSTokenizerV2CausalTransConvNet):
            setattr(module, name, StreamingTranspose(child))
        else:
            lower(child)


def rotate(hidden, cosine, sine):
    first, second = hidden.chunk(2, dim=-1)
    return hidden * cosine + torch.cat((-second, first), dim=-1) * sine


class StreamingLayer(nn.Module):
    def __init__(self, original, window):
        super().__init__()
        self.layer = original
        attention = original.self_attn
        self.heads = attention.config.num_attention_heads
        self.kv_heads = attention.config.num_key_value_heads
        self.head_dim = attention.head_dim
        self.groups = attention.num_key_value_groups
        self.scaling = attention.scaling
        for name in ["keys", "values"]:
            self.register_buffer(
                name, torch.zeros(1, self.kv_heads, window, self.head_dim)
            )

    def forward(self, hidden, cosine, sine, mask):
        attention = self.layer.self_attn
        normalized = self.layer.input_layernorm(hidden)
        query = attention.q_norm(
            attention.q_proj(normalized).reshape(1, 1, self.heads, self.head_dim)
        ).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(normalized).reshape(1, 1, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        value = attention.v_proj(normalized).reshape(1, 1, self.kv_heads, self.head_dim)
        value = value.transpose(1, 2)
        query = rotate(query, cosine, sine)
        key = rotate(key, cosine, sine)
        keys = torch.cat((self.keys[:, :, 1:, :], key), dim=2)
        values = torch.cat((self.values[:, :, 1:, :], value), dim=2)
        scores = (
            torch.matmul(
                query, keys.repeat_interleave(self.groups, dim=1).transpose(2, 3)
            )
            * self.scaling
        )
        # Match the source eager attention's FP32 softmax accumulation.
        probabilities = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        mixed = (
            torch.matmul(probabilities, values.repeat_interleave(self.groups, dim=1))
            .transpose(1, 2)
            .reshape(1, 1, self.heads * self.head_dim)
        )
        hidden = hidden + self.layer.self_attn_layer_scale(attention.o_proj(mixed))
        hidden = hidden + self.layer.mlp_layer_scale(
            self.layer.mlp(self.layer.post_attention_layernorm(hidden))
        )
        self.keys[:, :, :, :] = keys
        self.values[:, :, :, :] = values
        return hidden


class StreamingAudioDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        decoder = copy.deepcopy(decoder)
        lower(decoder)
        self.quantizer = StreamingQuantizer(decoder.quantizer)
        self.pre_conv = decoder.pre_conv
        transformer = decoder.pre_transformer
        self.input_proj = transformer.input_proj
        self.layers = nn.ModuleList(
            StreamingLayer(layer, decoder.config.sliding_window)
            for layer in transformer.layers
        )
        self.norm = transformer.norm
        self.output_proj = transformer.output_proj
        self.upsample = decoder.upsample
        self.decoder = decoder.decoder

    def forward(self, codes, cosine, sine, mask):
        hidden = (
            self.pre_conv(self.quantizer(codes).unsqueeze(2)).squeeze(2).transpose(1, 2)
        )
        hidden = self.input_proj(hidden)
        for layer in self.layers:
            hidden = layer(hidden, cosine, sine, mask)
        hidden = self.output_proj(self.norm(hidden)).transpose(1, 2).unsqueeze(2)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in self.decoder:
            hidden = block(hidden)
        return hidden.clamp(-1, 1).squeeze(2)

    def reset(self):
        for buffer in self.buffers():
            buffer.zero_()

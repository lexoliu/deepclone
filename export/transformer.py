"""Qwen decoder equations in channel-first form for Core ML conversion.

Original implementation, MIT OR Apache-2.0. Weights retain their model license.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone()[:, :, None, None])
        self.bias = (
            None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        )

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias)

    @classmethod
    def join(cls, *layers):
        weight = torch.cat([layer.weight.detach() for layer in layers], dim=0)
        bias = None
        if any(layer.bias is not None for layer in layers):
            bias = torch.cat(
                [
                    layer.bias.detach()
                    if layer.bias is not None
                    else layer.weight.new_zeros(layer.out_features)
                    for layer in layers
                ]
            )
        return cls(SimpleNamespace(weight=weight, bias=bias))


class RmsNorm(nn.Module):
    def __init__(self, norm, axis=1):
        super().__init__()
        self.axis = axis
        self.eps = norm.variance_epsilon
        shape = [1, 1, 1, 1]
        shape[axis] = norm.weight.numel()
        self.weight = nn.Parameter(norm.weight.detach().clone().reshape(shape))

    def forward(self, x):
        # Scaling cancels algebraically. It bounds the squared values by one
        # without promoting the complete decoder graph to float32.
        scale = x.abs().amax(dim=self.axis, keepdim=True).clamp_min(math.sqrt(self.eps))
        scaled = x / scale
        variance = (scaled * scaled).mean(dim=self.axis, keepdim=True)
        # sqrt(epsilon) is representable as a normal float16 number. Keeping
        # epsilon out of the graph's scalar constants avoids subnormal rounding.
        epsilon_root = math.sqrt(self.eps) / scale
        return (
            scaled * torch.rsqrt(variance + epsilon_root * epsilon_root) * self.weight
        )


def gated_silu(gate, up):
    return (gate * up) * torch.sigmoid(gate)


def rotary(x, cosine, sine):
    first, second = x.chunk(2, dim=2)
    return x * cosine + torch.cat((-second, first), dim=2) * sine


class DecoderLayer(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.input_norm = RmsNorm(layer.input_layernorm)
        self.post_norm = RmsNorm(layer.post_attention_layernorm)
        attention = layer.self_attn
        self.qkv = Projection.join(attention.q_proj, attention.k_proj, attention.v_proj)
        self.output = Projection(attention.o_proj)
        self.query_norm = RmsNorm(attention.q_norm, axis=2)
        self.key_norm = RmsNorm(attention.k_norm, axis=2)
        self.gate_up = Projection.join(layer.mlp.gate_proj, layer.mlp.up_proj)
        self.down = Projection(layer.mlp.down_proj)
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = self.heads // self.kv_heads
        self.scaling = self.head_dim**-0.5
        self.widths = [
            self.heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.kv_heads * self.head_dim,
        ]

    def forward(self, hidden, cosine, sine, mask, keys, values):
        length = hidden.shape[-1]
        normalized = self.input_norm(hidden)
        query, key, value = self.qkv(normalized).split(self.widths, dim=1)
        query = query.reshape(1, self.heads, self.head_dim, length)
        key = key.reshape(1, self.kv_heads, self.head_dim, length)
        value = value.reshape(1, self.kv_heads, self.head_dim, length)
        query = rotary(self.query_norm(query), cosine, sine)
        key = rotary(self.key_norm(key), cosine, sine)
        # Fixed-shape cache rotation keeps cache updates inside the execution
        # graph. The caller rejects context exhaustion instead of dropping tokens.
        keys = torch.cat((keys[:, :, length:, :], key.transpose(2, 3)), dim=2)
        values = torch.cat((values[:, :, length:, :], value.transpose(2, 3)), dim=2)
        # Queries sharing a KV head occupy separate rows of the same matmul.
        # Keep the cache resident once per KV head, including during prefill.
        queries = query.transpose(2, 3).reshape(
            1, self.kv_heads, self.groups * length, self.head_dim
        )
        scores = torch.matmul(queries * self.scaling, keys.transpose(2, 3))
        capacity = mask.shape[-1]
        scores = scores.reshape(1, self.heads, length, capacity) + mask
        attention = torch.softmax(scores, dim=-1).reshape(
            1, self.kv_heads, self.groups * length, capacity
        )
        mixed = (
            torch.matmul(attention, values)
            .reshape(1, self.heads, length, self.head_dim)
            .transpose(2, 3)
        )
        hidden = hidden + self.output(
            mixed.reshape(1, self.heads * self.head_dim, 1, length)
        )
        normalized = self.post_norm(hidden)
        gate, up = self.gate_up(normalized).chunk(2, dim=1)
        hidden = hidden + self.down(gated_silu(gate, up))
        return hidden, keys, values


class DecoderChunk(nn.Module):
    def __init__(self, layers, config, context):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(layer, config) for layer in layers)
        for index in range(len(layers)):
            self.register_buffer(
                f"key_{index}",
                torch.zeros(
                    1,
                    config.num_key_value_heads,
                    context,
                    config.head_dim,
                    dtype=torch.float16,
                ),
            )
            self.register_buffer(
                f"value_{index}",
                torch.zeros(
                    1,
                    config.num_key_value_heads,
                    context,
                    config.head_dim,
                    dtype=torch.float16,
                ),
            )

    def forward(self, hidden, cosine, sine, mask):
        for index, layer in enumerate(self.layers):
            key = getattr(self, f"key_{index}")
            value = getattr(self, f"value_{index}")
            hidden, next_key, next_value = layer(hidden, cosine, sine, mask, key, value)
            key[:, :, :, :] = next_key
            value[:, :, :, :] = next_value
        return hidden

    def reset(self):
        for buffer in self.buffers():
            buffer.zero_()


class OutputHead(nn.Module):
    def __init__(self, norm, head):
        super().__init__()
        self.norm = RmsNorm(norm)
        self.head = Projection(head)

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return normalized, self.head(normalized)


class TextEmbedding(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embedding = talker.get_text_embeddings()
        self.projection = talker.text_projection

    def forward(self, tokens):
        return self.projection(self.embedding(tokens)).transpose(1, 2).unsqueeze(2)


class CodecEmbedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, token):
        return self.embedding(token).transpose(1, 2).unsqueeze(2)


class CodecFrame(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                talker.get_input_embeddings(),
                *talker.code_predictor.get_input_embeddings(),
            ]
        )

    def forward(self, codes, text):
        embedded = torch.stack(
            [
                embedding(codes[:, index : index + 1])
                for index, embedding in enumerate(self.embeddings)
            ],
            dim=0,
        ).sum(dim=0)
        return embedded.transpose(1, 2).unsqueeze(2) + text

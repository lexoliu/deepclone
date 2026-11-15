"""Qwen decoder equations in channel-first form for Core ML conversion.

Original implementation, MIT OR Apache-2.0. Weights retain their model license.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, linear, accumulation_width=128):
        super().__init__()
        self.accumulation_width = accumulation_width
        self.weight = nn.Parameter(linear.weight.detach().clone()[:, :, None, None])
        self.bias = (
            None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        )

    def forward(self, x):
        # Bound each half-precision reduction, then combine partial sums as a
        # balanced tree rather than accumulating every channel sequentially.
        width = self.accumulation_width
        parts = [
            F.conv2d(x[:, start : start + width], self.weight[:, start : start + width])
            for start in range(0, self.weight.shape[1], width)
        ]
        while len(parts) > 1:
            parts = [
                parts[index] + parts[index + 1]
                if index + 1 < len(parts)
                else parts[index]
                for index in range(0, len(parts), 2)
            ]
        output = parts[0]
        return output if self.bias is None else output + self.bias.reshape(1, -1, 1, 1)


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
        self.query = Projection(attention.q_proj)
        self.key = Projection(attention.k_proj)
        self.value = Projection(attention.v_proj)
        self.output = Projection(attention.o_proj)
        self.query_norm = RmsNorm(attention.q_norm, axis=2)
        self.key_norm = RmsNorm(attention.k_norm, axis=2)
        self.gate = Projection(layer.mlp.gate_proj)
        self.up = Projection(layer.mlp.up_proj)
        self.down = Projection(layer.mlp.down_proj)
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = self.heads // self.kv_heads
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden, cosine, sine, mask, keys, values):
        normalized = self.input_norm(hidden)
        query = self.query(normalized).reshape(1, self.heads, self.head_dim, 1)
        key = self.key(normalized).reshape(1, self.kv_heads, self.head_dim, 1)
        value = self.value(normalized).reshape(1, self.kv_heads, self.head_dim, 1)
        query = rotary(self.query_norm(query), cosine, sine)
        key = rotary(self.key_norm(key), cosine, sine)
        # Fixed-shape cache rotation keeps cache updates inside the execution
        # graph. The caller rejects context exhaustion instead of dropping tokens.
        keys = torch.cat((keys[:, :, :, 1:], key), dim=3)
        values = torch.cat((values[:, :, 1:, :], value.transpose(2, 3)), dim=2)
        expanded_keys = keys.repeat_interleave(self.groups, dim=1)
        expanded_values = values.repeat_interleave(self.groups, dim=1)
        scores = (
            torch.matmul(query.transpose(2, 3) * self.scaling, expanded_keys) + mask
        )
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention, expanded_values).transpose(2, 3)
        hidden = hidden + self.output(
            mixed.reshape(1, self.heads * self.head_dim, 1, 1)
        )
        normalized = self.post_norm(hidden)
        hidden = hidden + self.down(
            gated_silu(self.gate(normalized), self.up(normalized))
        )
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
                    config.head_dim,
                    context,
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

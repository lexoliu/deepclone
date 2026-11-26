"""TextEmbedding stays FP16 when exporting quantized Core ML transformer bundles."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerDecoderLayer

import convert_transformers


class ConvertTransformers(unittest.TestCase):
    def test_text_embedding_stays_fp16_for_quantized_export(self):
        talker_config = Qwen3TTSTalkerConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_hidden_layers=1,
            vocab_size=32,
            num_code_groups=2,
        )
        talker_config._attn_implementation = "eager"
        predictor_config = Qwen3TTSTalkerConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_hidden_layers=1,
            vocab_size=32,
        )
        predictor_config._attn_implementation = "eager"

        talker_emb = torch.nn.Embedding(32, 16)
        pred_emb = torch.nn.Embedding(32, 16)
        text_emb = torch.nn.Embedding(32, 16)
        norm = torch.nn.RMSNorm(16)
        norm.variance_epsilon = 1e-6

        predictor = SimpleNamespace(
            config=predictor_config,
            small_to_mtp_projection=None,
            get_input_embeddings=lambda: [pred_emb],
            model=SimpleNamespace(
                layers=[Qwen3TTSTalkerDecoderLayer(predictor_config, 0)],
                norm=norm,
            ),
            lm_head=[torch.nn.Linear(16, 32, bias=False)],
        )
        talker = SimpleNamespace(
            config=talker_config,
            code_predictor=predictor,
            text_projection=torch.nn.Identity(),
            get_text_embeddings=lambda: text_emb,
            get_input_embeddings=lambda: talker_emb,
            model=SimpleNamespace(
                layers=[Qwen3TTSTalkerDecoderLayer(talker_config, 0)],
                norm=norm,
            ),
            codec_head=torch.nn.Linear(16, 32, bias=False),
        )
        reference = SimpleNamespace(
            model=SimpleNamespace(
                talker=talker,
                config=SimpleNamespace(
                    to_json_string=lambda use_diff=False: json.dumps({"ok": True})
                ),
            ),
            processor=SimpleNamespace(
                tokenizer=SimpleNamespace(
                    backend_tokenizer=SimpleNamespace(
                        save=lambda path: Path(path).write_text("{}")
                    )
                )
            ),
        )

        recorded = {}

        def fake_convert(
            module, inputs, names, outputs, path, weight_bits, states=(), shapes=None
        ):
            recorded[Path(path).name] = weight_bits
            Path(path).mkdir(parents=True, exist_ok=True)
            Path(path).with_suffix(".json").write_text(
                json.dumps({"weight_bits": weight_bits})
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                convert_transformers, "convert", side_effect=fake_convert
            ),
        ):
            out = Path(directory)
            convert_transformers.export(
                reference,
                out,
                context=16,
                chunk_layers=1,
                predictor_chunk_layers=1,
                weight_bits=8,
                sequence_lengths=[1, 4],
                resume=False,
                host_tables=True,
            )
            self.assertEqual(recorded["text-embedding.mlmodelc"], 16)
            self.assertEqual(recorded["talker-0.mlmodelc"], 8)
            self.assertEqual(recorded["predictor-0.mlmodelc"], 8)

            # Resume verifies each component against its own expected precision.
            recorded.clear()
            convert_transformers.export(
                reference,
                out,
                context=16,
                chunk_layers=1,
                predictor_chunk_layers=1,
                weight_bits=8,
                sequence_lengths=[1, 4],
                resume=True,
                host_tables=True,
            )
            self.assertEqual(recorded, {})


if __name__ == "__main__":
    unittest.main()

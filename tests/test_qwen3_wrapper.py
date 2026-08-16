import copy
import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)

from models.int_llama_layer import QuantLlamaDecoderLayer
from train_utils import make_causal_mask


class Qwen3WrapperTest(unittest.TestCase):
    def test_fp_wrapper_matches_huggingface_layer(self):
        torch.manual_seed(2)
        config = Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
        )
        config._attn_implementation = "eager"
        original = Qwen3DecoderLayer(config, 0).eval()
        args = SimpleNamespace(
            weight_quant_params={
                "n_bits": 1,
                "group_size": 128,
                "quant_mode": "catq",
                "init_round_thd": 0.5,
                "progressive_ratio": 0.8,
                "s0": 30.0,
            },
            act_quant_params={"n_bits": 16},
            q_quant_params={"n_bits": 16},
            k_quant_params={"n_bits": 16},
            v_quant_params={"n_bits": 16},
            p_quant_params={"n_bits": 16, "metric": "fix0to1"},
            quant_gate=False,
            update_gate=False,
            lora_rank=2,
            abits=16,
            quant_rate=1.0,
            use_down_scale=False,
            quant_mode_layer_list={0: "catq"},
        )
        wrapped = QuantLlamaDecoderLayer(
            config,
            copy.deepcopy(original),
            0,
            args,
            quant_mode="fp16",
            use_lora=True,
            lora_attr={
                "lora_iter_num": 1,
                "lora_quant": True,
                "lora_r": 2,
                "lora_only": False,
            },
        ).eval()
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_()

        hidden = torch.randn(2, 8, config.hidden_size)
        token_mask = torch.ones(2, 8, dtype=torch.bool)
        token_mask[1, 5:] = False
        attention_mask = make_causal_mask(token_mask, hidden.dtype, hidden.device)
        position_ids = torch.arange(8).unsqueeze(0)
        position_embeddings = Qwen3RotaryEmbedding(config)(hidden, position_ids)

        with torch.no_grad():
            expected = original(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            if isinstance(expected, tuple):
                expected = expected[0]
            actual = wrapped(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        wrapped.update_quant_mode("catq", args)
        wrapped.set_catq_progress(0.5)
        quantized = wrapped(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]
        (quantized - expected).square().mean().backward()
        catq_gradients = [
            parameter.grad
            for name, parameter in wrapped.named_parameters()
            if name.endswith(("raw_mu", "raw_scale", "raw_round"))
        ]
        self.assertTrue(catq_gradients)
        self.assertTrue(all(gradient is not None for gradient in catq_gradients))
        lora_gradients = [
            parameter.grad
            for name, parameter in wrapped.named_parameters()
            if ".lora_" in name
        ]
        self.assertTrue(lora_gradients)
        self.assertTrue(all(gradient is not None for gradient in lora_gradients))

        wrapped.set_catq_progress(1.0)
        with torch.no_grad():
            hard = wrapped(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            wrapped.update_quant_mode("weight_merge", args)
            merged = wrapped(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        torch.testing.assert_close(merged, hard)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_bittern_checkpoint import convert_checkpoint  # noqa: E402


class BitTernCheckpointTest(unittest.TestCase):
    def test_maps_catq_factor_names(self):
        checkpoint = convert_checkpoint(
            {
                0: {
                    "self_attn.q_proj.weight_quantizer.raw_scale": torch.tensor([[1.0]]),
                    "self_attn.q_proj.weight_quantizer.raw_mu": torch.tensor([[2.0]]),
                    "self_attn.q_proj.weight_quantizer.raw_round": torch.tensor([[3.0]]),
                    "self_attn.q_proj.lora_A.0": torch.zeros(2, 4),
                    "self_attn.q_proj.lora_B.0": torch.zeros(4, 2),
                }
            }
        )
        self.assertEqual(
            set(checkpoint[0]),
            {
                "self_attn.q_proj.weight_quantizer.generate_scale_factor.bound_factor",
                "self_attn.q_proj.weight_quantizer.generate_mu_factor.bound_factor",
                "self_attn.q_proj.weight_quantizer.generate_round_factor.bound_factor",
                "self_attn.q_proj.lora_A.0",
                "self_attn.q_proj.lora_B.0",
            },
        )

    def test_rejects_incomplete_lora_pair(self):
        with self.assertRaises(ValueError):
            convert_checkpoint(
                {
                    0: {
                        "self_attn.q_proj.weight_quantizer.raw_scale": torch.tensor([[1.0]]),
                        "self_attn.q_proj.weight_quantizer.raw_mu": torch.tensor([[2.0]]),
                        "self_attn.q_proj.weight_quantizer.raw_round": torch.tensor([[3.0]]),
                        "self_attn.q_proj.lora_A.0": torch.zeros(2, 4),
                    }
                }
            )

    def test_lora_requirement_rejects_factor_only_state(self):
        with self.assertRaises(ValueError):
            convert_checkpoint(
                {
                    0: {
                        "self_attn.q_proj.weight_quantizer.raw_scale": torch.tensor([[1.0]]),
                        "self_attn.q_proj.weight_quantizer.raw_mu": torch.tensor([[2.0]]),
                        "self_attn.q_proj.weight_quantizer.raw_round": torch.tensor([[3.0]]),
                    }
                },
                require_lora=True,
            )


if __name__ == "__main__":
    unittest.main()

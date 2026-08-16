import unittest

import torch

from quantize.catq import CATQQuantizer
from quantize.int_linear import QuantLinear
from quantize.int_linear_lora import LoRAQuantLinear
from quantize.sliderquant import masked_reconstruction_loss
from quantize.utils import slider_state_dict
from train_utils import load_catq_state_dict, to_half


class CATQQuantizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.weight = torch.randn(4, 128)
        self.quantizer = CATQQuantizer(self.weight)

    def test_initial_factors(self):
        mu, alpha, threshold = self.quantizer.factors()
        torch.testing.assert_close(mu, self.quantizer.mu0)
        torch.testing.assert_close(alpha, self.quantizer.alpha0)
        grouped = self.weight.float().reshape(-1, 128)
        expected_alpha0 = (
            grouped - grouped.mean(dim=1, keepdim=True)
        ).abs().mean(dim=1, keepdim=True) + 1e-6
        torch.testing.assert_close(self.quantizer.alpha0, expected_alpha0)
        torch.testing.assert_close(
            threshold, torch.full_like(threshold, 0.5)
        )

    def test_hard_symbols_and_materialization(self):
        symbols = self.quantizer.symbols(self.weight, hard=True)
        self.assertTrue(set(symbols.unique().tolist()) <= {-1.0, 0.0, 1.0})
        materialized = self.quantizer.materialize(self.weight).reshape(-1, 128)
        _, alpha, _ = self.quantizer.factors()
        torch.testing.assert_close(materialized, alpha * symbols)

    def test_factors_follow_adapted_weight(self):
        adapted = self.weight + torch.linspace(
            -0.5,
            0.5,
            self.weight.numel(),
        ).reshape_as(self.weight)
        mu, alpha, _ = self.quantizer.factors(adapted)
        grouped = adapted.reshape(-1, 128)
        expected_mu = grouped.mean(dim=1, keepdim=True)
        expected_alpha = (
            grouped - expected_mu
        ).abs().mean(dim=1, keepdim=True) + 1e-6
        torch.testing.assert_close(mu, expected_mu)
        torch.testing.assert_close(alpha, expected_alpha)

    def test_soft_and_hard_stages_update_all_factors(self):
        for progress in (0.01, 0.5, 1.0):
            self.quantizer.zero_grad()
            self.quantizer.set_progress(progress)
            self.quantizer(self.weight).square().mean().backward()
            for parameter in (
                self.quantizer.raw_mu,
                self.quantizer.raw_scale,
                self.quantizer.raw_round,
            ):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_group_size_must_divide_weight(self):
        with self.assertRaises(AssertionError):
            CATQQuantizer(torch.randn(3, 127))

    def test_materialized_linear_matches_on_the_fly_hard_quantization(self):
        linear = torch.nn.Linear(128, 4, bias=False)
        quantized = QuantLinear(
            linear,
            {
                "n_bits": 1,
                "group_size": 128,
                "quant_mode": "catq",
                "init_round_thd": 0.5,
                "progressive_ratio": 0.8,
                "s0": 30.0,
            },
            {"n_bits": 16},
        )
        quantized.set_catq_progress(1.0)
        quantized.set_quant_state(weight_quant=True)
        inputs = torch.randn(3, 128)
        expected = quantized(inputs)

        quantized.materialize_weight()
        quantized.set_quant_state(weight_quant=False)
        actual = quantized(inputs)
        torch.testing.assert_close(actual, expected)

    def test_lora_materialization_matches_on_the_fly_quantization(self):
        linear = torch.nn.Linear(128, 4, bias=False)
        quantized = LoRAQuantLinear(
            linear,
            {
                "n_bits": 1,
                "group_size": 128,
                "quant_mode": "catq",
                "init_round_thd": 0.5,
                "progressive_ratio": 0.8,
                "s0": 30.0,
            },
            {"n_bits": 16},
            r=2,
            lora_attr={
                "lora_iter_num": 1,
                "lora_quant": True,
                "lora_r": 2,
                "lora_only": False,
            },
        )
        with torch.no_grad():
            quantized.lora_B[0].normal_()
        quantized.set_catq_progress(1.0)
        quantized.set_quant_state(weight_quant=True)
        inputs = torch.randn(3, 128)
        expected = quantized(inputs)

        quantized.materialize_weight()
        quantized.set_quant_state(weight_quant=False)
        actual = quantized(inputs)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(quantized.merged)

    def test_catq_state_requires_complete_learned_parameters(self):
        linear = torch.nn.Linear(128, 4, bias=False).bfloat16()
        quantized = LoRAQuantLinear(
            linear,
            {
                "n_bits": 1,
                "group_size": 128,
                "quant_mode": "catq",
                "init_round_thd": 0.5,
                "progressive_ratio": 0.8,
                "s0": 30.0,
            },
            {"n_bits": 16},
            r=2,
            lora_attr={
                "lora_iter_num": 1,
                "lora_quant": True,
                "lora_r": 2,
                "lora_only": False,
            },
        )
        self.assertEqual(quantized.lora_A[0].dtype, torch.float32)
        with torch.no_grad():
            quantized.lora_B[0].fill_(0.123456)
        state = slider_state_dict(quantized)
        restored = LoRAQuantLinear(
            torch.nn.Linear(128, 4, bias=False).bfloat16(),
            {
                "n_bits": 1,
                "group_size": 128,
                "quant_mode": "catq",
                "init_round_thd": 0.5,
                "progressive_ratio": 0.8,
                "s0": 30.0,
            },
            {"n_bits": 16},
            r=2,
            lora_attr={
                "lora_iter_num": 1,
                "lora_quant": True,
                "lora_r": 2,
                "lora_only": False,
            },
        )
        load_catq_state_dict(restored, state)
        restored_state = slider_state_dict(restored)
        for name, value in state.items():
            torch.testing.assert_close(
                restored_state[name],
                value,
                rtol=0,
                atol=0,
            )

        incomplete = state.copy()
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(AssertionError):
            load_catq_state_dict(restored, incomplete)

    def test_catq_layers_stay_fp32_during_offload(self):
        linear = torch.nn.Linear(128, 4, bias=False)
        quantized = QuantLinear(
            linear,
            {
                "n_bits": 1,
                "group_size": 128,
                "quant_mode": "catq",
                "init_round_thd": 0.5,
                "progressive_ratio": 0.8,
                "s0": 30.0,
            },
            {"n_bits": 16},
        )
        quantized.weight_quantizer.raw_scale.data.fill_(0.123456)
        expected = quantized.weight_quantizer.raw_scale.detach().clone()
        to_half([quantized], torch.bfloat16)
        self.assertEqual(quantized.weight.dtype, torch.float32)
        self.assertEqual(
            quantized.weight_quantizer.raw_scale.dtype,
            torch.float32,
        )
        torch.testing.assert_close(
            quantized.weight_quantizer.raw_scale,
            expected,
            rtol=0,
            atol=0,
        )

    def test_masked_huber_loss_is_honored(self):
        output = torch.tensor([[[2.0], [100.0]]])
        target = torch.zeros_like(output)
        token_mask = torch.tensor([[True, False]])
        loss = masked_reconstruction_loss(
            output,
            target,
            token_mask,
            torch.nn.HuberLoss(delta=1.0, reduction="none"),
        )
        torch.testing.assert_close(loss, torch.tensor(1.5))


if __name__ == "__main__":
    unittest.main()

import math

import torch
import torch.nn as nn


class CATQQuantizer(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        group_size: int = 128,
        init_round_thd: float = 0.5,
        progressive_ratio: float = 0.8,
        s0: float = 30.0,
    ):
        super().__init__()
        assert weight.ndim == 2
        assert weight.numel() % group_size == 0

        self.group_size = group_size
        self.init_round_thd = init_round_thd
        self.progressive_ratio = progressive_ratio
        self.s0 = s0

        mu0, alpha0 = self._base_factors(weight.detach())
        self.register_buffer("mu0", mu0)
        self.register_buffer("alpha0", alpha0)
        self.progress = 0.0

        self.raw_mu = nn.Parameter(torch.zeros_like(mu0))
        self.raw_scale = nn.Parameter(torch.zeros_like(alpha0))
        self.raw_round = nn.Parameter(torch.zeros_like(alpha0))

    def set_progress(self, progress: float) -> None:
        assert 0.0 <= progress <= 1.0
        self.progress = progress

    def _base_factors(self, weight: torch.Tensor):
        grouped = weight.float().reshape(-1, self.group_size)
        mu0 = grouped.mean(dim=1, keepdim=True)
        alpha0 = (grouped - mu0).abs().mean(dim=1, keepdim=True) + 1e-6
        return mu0, alpha0

    def factors(self, weight: torch.Tensor | None = None):
        mu0, alpha0 = (
            (self.mu0, self.alpha0)
            if weight is None
            else self._base_factors(weight)
        )
        delta_mu = 2.0 * torch.sigmoid(self.raw_mu) - 1.0
        delta_alpha = 2.0 * torch.sigmoid(self.raw_scale)
        delta_round = 2.0 * torch.sigmoid(self.raw_round)
        mu = mu0 + delta_mu * alpha0
        alpha = delta_alpha * alpha0
        threshold = delta_round * self.init_round_thd
        return mu, alpha, threshold

    @staticmethod
    def _softened(x: torch.Tensor, sharpness: float, threshold: torch.Tensor):
        denominator = 2.0 * math.tanh(sharpness)
        return (
            torch.tanh(sharpness * (x - threshold))
            + torch.tanh(sharpness * (x + threshold))
        ) / denominator

    @staticmethod
    def _hard(x: torch.Tensor, threshold: torch.Tensor):
        return torch.where(
            x > threshold,
            torch.ones_like(x),
            torch.where(x < -threshold, -torch.ones_like(x), torch.zeros_like(x)),
        )

    def _symbols_from_factors(self, grouped, mu, alpha, threshold, hard):
        normalized = (grouped - mu) / alpha

        progress = self.progress
        if hard:
            return self._hard(normalized, threshold)
        if progress == 0.0:
            return normalized
        if progress <= self.progressive_ratio:
            sharpness = progress / self.progressive_ratio * self.s0
            return self._softened(normalized, sharpness, threshold)

        hard_symbols = self._hard(normalized, threshold)
        soft_symbols = self._softened(normalized, self.s0, threshold)
        return hard_symbols.detach() + soft_symbols - soft_symbols.detach()

    def symbols(self, weight: torch.Tensor, hard: bool = False):
        grouped = weight.float().reshape(-1, self.group_size)
        return self._symbols_from_factors(
            grouped,
            *self.factors(weight),
            hard,
        )

    def forward(self, weight: torch.Tensor, quant_rate: float = 1.0):
        grouped = weight.float().reshape(-1, self.group_size)
        mu, alpha, threshold = self.factors(weight)
        reconstructed = alpha * self._symbols_from_factors(
            grouped,
            mu,
            alpha,
            threshold,
            hard=False,
        )

        if quant_rate < 1.0:
            quantized_groups = math.ceil(grouped.shape[0] * quant_rate)
            reconstructed = torch.cat(
                (reconstructed[:quantized_groups], grouped[quantized_groups:]), dim=0
            )

        return reconstructed.reshape_as(weight).to(weight.dtype)

    @torch.no_grad()
    def materialize(self, weight: torch.Tensor):
        grouped = weight.float().reshape(-1, self.group_size)
        mu, alpha, threshold = self.factors(weight)
        symbols = self._symbols_from_factors(
            grouped,
            mu,
            alpha,
            threshold,
            hard=True,
        )
        return (alpha * symbols).reshape_as(weight).to(weight.dtype)

    @torch.no_grad()
    def occupancy(self, weight: torch.Tensor):
        symbols = self.symbols(weight, hard=True)
        total = symbols.numel()
        return {
            "negative": int((symbols < 0).sum()) / total,
            "zero": int((symbols == 0).sum()) / total,
            "positive": int((symbols > 0).sum()) / total,
        }

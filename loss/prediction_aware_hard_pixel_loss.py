from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionAwareHardPixelLoss(nn.Module):
    """
    Prediction-aware hard-pixel correction for Tiny/Small samples.

    Hard positive:
        GT foreground pixel whose detached base foreground
        probability is below positive_threshold.

    Hard negative:
        Pixel in the dilated GT ring whose detached base
        foreground probability is above negative_threshold.

    The loss is computed on final logits, while the base
    prediction used for gating is detached.
    """

    def __init__(
        self,
        positive_threshold: float = 0.40,
        negative_threshold: float = 0.60,
        positive_weight: float = 0.02,
        negative_weight: float = 0.02,
        residual_reg_weight: float = 0.002,
        ring_radius: int = 6,
        tiny_max_ratio: float = 0.001,
        small_max_ratio: float = 0.005,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not 0.0 < positive_threshold < 1.0:
            raise ValueError("positive_threshold must be in (0, 1)")
        if not 0.0 < negative_threshold < 1.0:
            raise ValueError("negative_threshold must be in (0, 1)")
        if positive_threshold >= negative_threshold:
            raise ValueError(
                "positive_threshold must be smaller than "
                "negative_threshold"
            )
        if ring_radius < 1:
            raise ValueError("ring_radius must be >= 1")

        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.residual_reg_weight = float(residual_reg_weight)
        self.ring_radius = int(ring_radius)
        self.tiny_max_ratio = float(tiny_max_ratio)
        self.small_max_ratio = float(small_max_ratio)
        self.eps = float(eps)

    def _weighted_sample_mean(
        self,
        loss_map: torch.Tensor,
        weight_map: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (loss_map * weight_map).flatten(1).sum(dim=1)
        denominator = weight_map.flatten(1).sum(dim=1)
        valid = denominator > self.eps

        if not torch.any(valid):
            return loss_map.sum() * 0.0

        per_sample = numerator / denominator.clamp_min(self.eps)
        return per_sample[valid].mean()

    def forward(
        self,
        base_logits: torch.Tensor,
        final_logits: torch.Tensor,
        residual_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if target.ndim == 4:
            if target.shape[1] != 1:
                raise ValueError(
                    "4D target must have one channel"
                )
            target = target[:, 0]

        if target.ndim != 3:
            raise ValueError(
                "target must have shape [B,H,W] or [B,1,H,W]"
            )

        if final_logits.ndim != 4 or final_logits.shape[1] != 2:
            raise ValueError(
                "final_logits must have shape [B,2,H,W]"
            )

        if base_logits.shape != final_logits.shape:
            raise ValueError("base/final logits shape mismatch")

        if residual_logits.shape != final_logits.shape:
            raise ValueError("residual/final logits shape mismatch")

        target = target.long()
        foreground = target == 1

        area_ratio = foreground.float().flatten(1).mean(dim=1)
        nonempty = area_ratio > 0.0
        active = nonempty & (area_ratio <= self.small_max_ratio)

        tiny = active & (area_ratio <= self.tiny_max_ratio)
        small = active & (area_ratio > self.tiny_max_ratio)

        active_map = active[:, None, None]

        with torch.no_grad():
            base_fg_probability = torch.softmax(
                base_logits.detach(),
                dim=1,
            )[:, 1]

            kernel_size = 2 * self.ring_radius + 1
            dilated = F.max_pool2d(
                foreground[:, None].float(),
                kernel_size=kernel_size,
                stride=1,
                padding=self.ring_radius,
            )[:, 0] > 0.5

            ring = dilated & (~foreground)

            hard_positive = (
                foreground
                & active_map
                & (
                    base_fg_probability
                    < self.positive_threshold
                )
            )

            hard_negative = (
                ring
                & active_map
                & (
                    base_fg_probability
                    > self.negative_threshold
                )
            )

            positive_confidence = (
                (
                    self.positive_threshold
                    - base_fg_probability
                )
                / self.positive_threshold
            ).clamp(min=0.0, max=1.0)

            negative_confidence = (
                (
                    base_fg_probability
                    - self.negative_threshold
                )
                / (1.0 - self.negative_threshold)
            ).clamp(min=0.0, max=1.0)

            positive_weights = (
                hard_positive.float()
                * positive_confidence
            )

            negative_weights = (
                hard_negative.float()
                * negative_confidence
            )

        log_probability = F.log_softmax(final_logits, dim=1)

        positive_loss = self._weighted_sample_mean(
            -log_probability[:, 1],
            positive_weights,
        )

        negative_loss = self._weighted_sample_mean(
            -log_probability[:, 0],
            negative_weights,
        )

        # Two-class residual is shift-invariant through its margin.
        residual_margin = (
            residual_logits[:, 1]
            - residual_logits[:, 0]
        )

        protected_region = (
            active_map
            & (~hard_positive)
            & (~hard_negative)
        ).float()

        residual_reg_loss = self._weighted_sample_mean(
            residual_margin.abs(),
            protected_region,
        )

        weighted_loss = (
            self.positive_weight * positive_loss
            + self.negative_weight * negative_loss
            + self.residual_reg_weight * residual_reg_loss
        )

        stats = {
            "active_count": float(active.sum().item()),
            "tiny_count": float(tiny.sum().item()),
            "small_count": float(small.sum().item()),
            "hard_positive_pixels": float(
                hard_positive.sum().item()
            ),
            "hard_negative_pixels": float(
                hard_negative.sum().item()
            ),
            "positive_loss": float(
                positive_loss.detach().item()
            ),
            "negative_loss": float(
                negative_loss.detach().item()
            ),
            "residual_reg_loss": float(
                residual_reg_loss.detach().item()
            ),
            "weighted_loss": float(
                weighted_loss.detach().item()
            ),
        }

        return weighted_loss, stats

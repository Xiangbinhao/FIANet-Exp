import torch
import torch.nn.functional as F
from torch import nn


class LocalPositiveRingLoss(nn.Module):
    """
    S3-A local positive and hard-negative ring supervision.

    Only nonempty Tiny/Small samples are activated.

    Positive region:
        GT foreground.

    Hard-negative region:
        Dilate(GT, radius) - GT.

    The positive and ring terms are normalized independently
    for every sample, preventing the negative ring from
    overwhelming the small foreground region.
    """

    def __init__(
        self,
        positive_weight=0.05,
        ring_weight=0.05,
        ring_radius=8,
        tiny_max_ratio=0.001,
        small_max_ratio=0.005,
        warmup_epochs=5,
        ramp_epochs=5,
        eps=1.0e-6,
    ):
        super().__init__()

        if positive_weight < 0:
            raise ValueError(
                "positive_weight must be nonnegative"
            )

        if ring_weight < 0:
            raise ValueError(
                "ring_weight must be nonnegative"
            )

        if ring_radius < 1:
            raise ValueError(
                "ring_radius must be at least 1"
            )

        if not (
            0.0
            < tiny_max_ratio
            < small_max_ratio
        ):
            raise ValueError(
                "Expected 0 < tiny_max_ratio "
                "< small_max_ratio"
            )

        self.positive_weight = float(
            positive_weight
        )
        self.ring_weight = float(
            ring_weight
        )
        self.ring_radius = int(
            ring_radius
        )

        self.tiny_max_ratio = float(
            tiny_max_ratio
        )
        self.small_max_ratio = float(
            small_max_ratio
        )

        self.warmup_epochs = int(
            warmup_epochs
        )
        self.ramp_epochs = int(
            ramp_epochs
        )

        self.eps = float(eps)

    def schedule_factor(self, epoch):
        epoch = int(epoch)

        if epoch < self.warmup_epochs:
            return 0.0

        if self.ramp_epochs <= 0:
            return 1.0

        factor = (
            epoch
            - self.warmup_epochs
            + 1
        ) / float(self.ramp_epochs)

        return max(
            0.0,
            min(1.0, factor),
        )

    def _prepare_target(self, target):
        if target.ndim == 4:
            if target.shape[1] != 1:
                raise ValueError(
                    "Four-dimensional target must "
                    "have one channel"
                )

            target = target[:, 0]

        if target.ndim != 3:
            raise ValueError(
                "Expected target shape [B,H,W], "
                "received {}".format(
                    tuple(target.shape)
                )
            )

        return (
            target > 0
        ).float()

    def forward(
        self,
        logits,
        target,
        epoch,
    ):
        if logits.ndim != 4:
            raise ValueError(
                "Expected logits [B,C,H,W], "
                "received {}".format(
                    tuple(logits.shape)
                )
            )

        if logits.shape[1] != 2:
            raise ValueError(
                "S3-A currently expects two-class logits"
            )

        target = self._prepare_target(
            target
        )

        if (
            logits.shape[-2:]
            != target.shape[-2:]
        ):
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        batch_size = target.shape[0]

        foreground_ratios = (
            target.reshape(batch_size, -1)
            .mean(dim=1)
        )

        tiny_mask = (
            (foreground_ratios > 0.0)
            & (
                foreground_ratios
                <= self.tiny_max_ratio
            )
        )

        small_mask = (
            (
                foreground_ratios
                > self.tiny_max_ratio
            )
            & (
                foreground_ratios
                <= self.small_max_ratio
            )
        )

        active_mask = (
            tiny_mask | small_mask
        )

        active_count = int(
            active_mask.sum().item()
        )

        factor = self.schedule_factor(
            epoch
        )

        zero = logits.sum() * 0.0

        empty_stats = {
            "factor": float(factor),
            "active_count": active_count,
            "tiny_count": int(
                tiny_mask.sum().item()
            ),
            "small_count": int(
                small_mask.sum().item()
            ),
            "positive_loss": 0.0,
            "ring_loss": 0.0,
            "weighted_loss": 0.0,
        }

        if (
            active_count == 0
            or factor <= 0.0
            or (
                self.positive_weight <= 0.0
                and self.ring_weight <= 0.0
            )
        ):
            return zero, empty_stats

        # Explicit FP32 log-probabilities improve numerical
        # stability under automatic mixed precision.
        log_probability = F.log_softmax(
            logits.float(),
            dim=1,
        )

        foreground_nll = (
            -log_probability[:, 1]
        )

        background_nll = (
            -log_probability[:, 0]
        )

        kernel_size = (
            2 * self.ring_radius + 1
        )

        dilated = F.max_pool2d(
            target.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=self.ring_radius,
        ).squeeze(1)

        positive_region = target

        ring_region = (
            (dilated > 0.5)
            & (target < 0.5)
        ).float()

        positive_pixels = (
            positive_region
            .reshape(batch_size, -1)
            .sum(dim=1)
        )

        ring_pixels = (
            ring_region
            .reshape(batch_size, -1)
            .sum(dim=1)
        )

        positive_per_sample = (
            (
                foreground_nll
                * positive_region
            )
            .reshape(batch_size, -1)
            .sum(dim=1)
            / (
                positive_pixels
                + self.eps
            )
        )

        ring_per_sample = (
            (
                background_nll
                * ring_region
            )
            .reshape(batch_size, -1)
            .sum(dim=1)
            / (
                ring_pixels
                + self.eps
            )
        )

        active_float = active_mask.float()

        positive_loss = (
            (
                positive_per_sample
                * active_float
            ).sum()
            / (
                active_float.sum()
                + self.eps
            )
        )

        valid_ring = (
            active_mask
            & (ring_pixels > 0)
        ).float()

        if valid_ring.sum().item() > 0:
            ring_loss = (
                (
                    ring_per_sample
                    * valid_ring
                ).sum()
                / (
                    valid_ring.sum()
                    + self.eps
                )
            )
        else:
            ring_loss = zero

        weighted_loss = factor * (
            self.positive_weight
            * positive_loss
            + self.ring_weight
            * ring_loss
        )

        stats = {
            "factor": float(factor),
            "active_count": active_count,
            "tiny_count": int(
                tiny_mask.sum().item()
            ),
            "small_count": int(
                small_mask.sum().item()
            ),
            "positive_loss": float(
                positive_loss.detach().item()
            ),
            "ring_loss": float(
                ring_loss.detach().item()
            ),
            "weighted_loss": float(
                weighted_loss.detach().item()
            ),
        }

        return weighted_loss, stats

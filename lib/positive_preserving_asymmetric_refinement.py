import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositivePreservingAsymmetricResidualWrapper(nn.Module):
    """
    S3-C: positive-preserving asymmetric bounded residual refinement.

    The wrapped S2 head predicts a two-channel residual. We first convert
    it into a foreground-background logit-difference correction:

        raw_delta = 0.5 * (r_fg - r_bg)

    The positive and negative directions are then treated asymmetrically:

        positive_delta = relu(tanh(raw_delta))
        negative_delta = relu(-tanh(raw_delta))

        alpha_negative
            = alpha_positive
            * negative_ratio_max
            * sigmoid(negative_ratio_logit)

        delta
            = gate * (
                alpha_positive * positive_delta
                - alpha_negative * negative_delta
            )

        residual = [-delta, +delta]

    Positive corrections retain the full bounded strength, while negative
    foreground-suppressing corrections are explicitly limited.
    """

    def __init__(
        self,
        base_head,
        alpha_init=0.10,
        gate_init=0.20,
        negative_ratio_init=0.25,
        negative_ratio_max=0.50,
    ):
        super().__init__()

        if base_head is None:
            raise ValueError("base_head must not be None")

        if not 0.0 < alpha_init < 1.0:
            raise ValueError(
                "alpha_init must be in (0, 1), got {}".format(
                    alpha_init
                )
            )

        if not 0.0 < gate_init < 1.0:
            raise ValueError(
                "gate_init must be in (0, 1), got {}".format(
                    gate_init
                )
            )

        if not 0.0 < negative_ratio_max <= 1.0:
            raise ValueError(
                "negative_ratio_max must be in (0, 1], got {}".format(
                    negative_ratio_max
                )
            )

        if not 0.0 < negative_ratio_init < negative_ratio_max:
            raise ValueError(
                "negative_ratio_init must satisfy "
                "0 < negative_ratio_init < negative_ratio_max, "
                "got init={} max={}".format(
                    negative_ratio_init,
                    negative_ratio_max,
                )
            )

        self.base_head = base_head
        self.negative_ratio_max = float(
            negative_ratio_max
        )

        # Spatial gate generated from the raw two-channel S2 residual.
        self.gate_predictor = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Conservative spatially uniform initialization.
        nn.init.zeros_(
            self.gate_predictor.weight
        )

        gate_bias = math.log(
            gate_init / (1.0 - gate_init)
        )
        nn.init.constant_(
            self.gate_predictor.bias,
            gate_bias,
        )

        # Positive correction strength.
        alpha_logit = math.log(
            alpha_init / (1.0 - alpha_init)
        )
        self.alpha_positive_logit = nn.Parameter(
            torch.tensor(
                alpha_logit,
                dtype=torch.float32,
            )
        )

        # Learnable negative-to-positive strength ratio.
        normalized_ratio = (
            negative_ratio_init /
            negative_ratio_max
        )
        negative_ratio_logit = math.log(
            normalized_ratio /
            (1.0 - normalized_ratio)
        )

        self.negative_ratio_logit = nn.Parameter(
            torch.tensor(
                negative_ratio_logit,
                dtype=torch.float32,
            )
        )

    def get_control_values(self):
        """
        Return differentiable scalar control values.
        """
        alpha_positive = torch.sigmoid(
            self.alpha_positive_logit
        )

        negative_ratio = (
            self.negative_ratio_max
            * torch.sigmoid(
                self.negative_ratio_logit
            )
        )

        alpha_negative = (
            alpha_positive
            * negative_ratio
        )

        return (
            alpha_positive,
            alpha_negative,
            negative_ratio,
        )

    def forward(self, x_c1, x_c2):
        raw_residual = self.base_head(
            x_c1,
            x_c2,
        )

        if raw_residual.ndim != 4:
            raise RuntimeError(
                "Expected residual shape [B,2,H,W], got {}".format(
                    tuple(raw_residual.shape)
                )
            )

        if raw_residual.shape[1] != 2:
            raise RuntimeError(
                "Expected two residual channels, got {}".format(
                    raw_residual.shape[1]
                )
            )

        raw_delta = 0.5 * (
            raw_residual[:, 1:2]
            - raw_residual[:, 0:1]
        )

        bounded_raw_delta = torch.tanh(
            raw_delta
        )

        positive_delta = F.relu(
            bounded_raw_delta
        )
        negative_delta = F.relu(
            -bounded_raw_delta
        )

        gate = torch.sigmoid(
            self.gate_predictor(
                raw_residual
            )
        )

        (
            alpha_positive,
            alpha_negative,
            negative_ratio,
        ) = self.get_control_values()

        asymmetric_delta = gate * (
            alpha_positive * positive_delta
            - alpha_negative * negative_delta
        )

        asymmetric_residual = torch.cat(
            [
                -asymmetric_delta,
                asymmetric_delta,
            ],
            dim=1,
        )

        return asymmetric_residual

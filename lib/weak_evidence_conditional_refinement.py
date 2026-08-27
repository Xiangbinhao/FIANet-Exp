import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakEvidenceConditionalResidualWrapper(nn.Module):
    """
    S3-D: weak-evidence conditional asymmetric residual refinement.

    The original S2 head predicts a two-channel residual. Its binary
    foreground-background correction is

        raw_delta = 0.5 * (r_fg - r_bg)

    Positive corrections are preserved. Negative corrections are relaxed
    only when their bounded residual evidence is weak:

        bounded_delta = tanh(raw_delta)

        protection =
            sigmoid(
                (ambiguity_center - abs(bounded_delta))
                / ambiguity_temperature
            )

        negative_scale =
            1 - relax_strength * protection

        delta =
            alpha * gate * (
                relu(bounded_delta)
                - negative_scale * relu(-bounded_delta)
            )

    Therefore:
    - ambiguous weak negative evidence receives reduced suppression;
    - strong negative evidence remains close to symmetric S3-B;
    - positive foreground correction is not weakened.
    """

    def __init__(
        self,
        base_head,
        alpha_init=0.10,
        gate_init=0.20,
        relax_init=0.75,
        ambiguity_center=0.20,
        ambiguity_temperature=0.05,
    ):
        super().__init__()

        if base_head is None:
            raise ValueError("base_head must not be None")

        if not 0.0 < alpha_init < 1.0:
            raise ValueError(
                "alpha_init must be in (0,1), got {}".format(
                    alpha_init
                )
            )

        if not 0.0 < gate_init < 1.0:
            raise ValueError(
                "gate_init must be in (0,1), got {}".format(
                    gate_init
                )
            )

        if not 0.0 < relax_init < 1.0:
            raise ValueError(
                "relax_init must be in (0,1), got {}".format(
                    relax_init
                )
            )

        if not 0.0 < ambiguity_center < 1.0:
            raise ValueError(
                "ambiguity_center must be in (0,1), got {}".format(
                    ambiguity_center
                )
            )

        if ambiguity_temperature <= 0.0:
            raise ValueError(
                "ambiguity_temperature must be positive, got {}".format(
                    ambiguity_temperature
                )
            )

        self.base_head = base_head

        self.ambiguity_center = float(
            ambiguity_center
        )
        self.ambiguity_temperature = float(
            ambiguity_temperature
        )

        self.gate_predictor = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Conservative uniform initial gate.
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

        alpha_logit = math.log(
            alpha_init / (1.0 - alpha_init)
        )

        self.alpha_logit = nn.Parameter(
            torch.tensor(
                alpha_logit,
                dtype=torch.float32,
            )
        )

        relax_logit = math.log(
            relax_init / (1.0 - relax_init)
        )

        self.relax_logit = nn.Parameter(
            torch.tensor(
                relax_logit,
                dtype=torch.float32,
            )
        )

    def get_control_values(self):
        alpha = torch.sigmoid(
            self.alpha_logit
        )

        relax_strength = torch.sigmoid(
            self.relax_logit
        )

        return alpha, relax_strength

    def negative_scale_from_bounded_abs(
        self,
        bounded_abs,
    ):
        _, relax_strength = (
            self.get_control_values()
        )

        protection = torch.sigmoid(
            (
                self.ambiguity_center
                - bounded_abs
            )
            / self.ambiguity_temperature
        )

        negative_scale = (
            1.0
            - relax_strength * protection
        )

        return negative_scale, protection

    def forward(self, x_c1, x_c2):
        raw_residual = self.base_head(
            x_c1,
            x_c2,
        )

        if raw_residual.ndim != 4:
            raise RuntimeError(
                "Expected [B,2,H,W], got {}".format(
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

        bounded_delta = torch.tanh(
            raw_delta
        )

        positive_delta = F.relu(
            bounded_delta
        )

        negative_delta = F.relu(
            -bounded_delta
        )

        gate = torch.sigmoid(
            self.gate_predictor(
                raw_residual
            )
        )

        alpha, _ = self.get_control_values()

        negative_scale, _ = (
            self.negative_scale_from_bounded_abs(
                bounded_delta.abs()
            )
        )

        conditional_delta = (
            alpha
            * gate
            * (
                positive_delta
                - negative_scale * negative_delta
            )
        )

        residual = torch.cat(
            [
                -conditional_delta,
                conditional_delta,
            ],
            dim=1,
        )

        return residual

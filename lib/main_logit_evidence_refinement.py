import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MainLogitEvidenceConditionalResidualWrapper(nn.Module):
    """
    S3-E: main-logit-evidence-conditioned bounded residual refinement.

    The wrapped S2 head predicts a two-channel residual:

        raw_delta = 0.5 * (r_fg - r_bg)

    The original main-branch foreground probability is used as read-only
    evidence. Negative foreground correction is relaxed only where the
    main branch provides weak-positive evidence:

        p_fg = Softmax(main_logits.detach())_fg

        protection =
            sigmoid((p_fg - weak_positive_low) / temperature)
            *
            sigmoid((weak_positive_high - p_fg) / temperature)

        negative_scale =
            1 - relax_strength * protection

        delta =
            alpha * gate * (
                relu(tanh(raw_delta))
                -
                negative_scale * relu(-tanh(raw_delta))
            )

    Strong-background and strong-foreground regions remain close to the
    symmetric S3-B correction, while weak-positive foreground regions
    receive selective protection against negative suppression.
    """

    requires_main_logits = True

    def __init__(
        self,
        base_head,
        alpha_init=0.10,
        gate_init=0.20,
        relax_init=0.75,
        weak_positive_low=0.45,
        weak_positive_high=0.70,
        evidence_temperature=0.05,
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

        if not 0.0 < weak_positive_low < weak_positive_high < 1.0:
            raise ValueError(
                "Expected 0 < low < high < 1, got low={} high={}".format(
                    weak_positive_low,
                    weak_positive_high,
                )
            )

        if evidence_temperature <= 0.0:
            raise ValueError(
                "evidence_temperature must be positive, got {}".format(
                    evidence_temperature
                )
            )

        self.base_head = base_head

        self.weak_positive_low = float(
            weak_positive_low
        )
        self.weak_positive_high = float(
            weak_positive_high
        )
        self.evidence_temperature = float(
            evidence_temperature
        )

        self.gate_predictor = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

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

    def evidence_protection_from_probability(
        self,
        foreground_probability,
    ):
        """
        Produce a smooth band-pass protection score.

        Low foreground probability:
            confident background, protection approaches zero.

        Intermediate foreground probability:
            weak-positive evidence, protection becomes high.

        High foreground probability:
            confident foreground, protection decreases again.
        """
        _, relax_strength = (
            self.get_control_values()
        )

        lower_activation = torch.sigmoid(
            (
                foreground_probability
                - self.weak_positive_low
            )
            / self.evidence_temperature
        )

        upper_activation = torch.sigmoid(
            (
                self.weak_positive_high
                - foreground_probability
            )
            / self.evidence_temperature
        )

        protection = (
            lower_activation
            * upper_activation
        )

        negative_scale = (
            1.0
            - relax_strength * protection
        )

        return negative_scale, protection

    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):
        if main_logits is None:
            raise RuntimeError(
                "S3-E requires main_logits."
            )

        raw_residual = self.base_head(
            x_c1,
            x_c2,
        )

        if raw_residual.ndim != 4:
            raise RuntimeError(
                "Expected residual [B,2,H,W], got {}".format(
                    tuple(raw_residual.shape)
                )
            )

        if raw_residual.shape[1] != 2:
            raise RuntimeError(
                "Expected two residual channels, got {}".format(
                    raw_residual.shape[1]
                )
            )

        if main_logits.ndim != 4 or main_logits.shape[1] != 2:
            raise RuntimeError(
                "Expected main_logits [B,2,H,W], got {}".format(
                    tuple(main_logits.shape)
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

        # Read-only evidence prevents the main decoder from manipulating
        # its logits solely to change the conditional gate.
        foreground_probability = torch.softmax(
            main_logits.detach(),
            dim=1,
        )[:, 1:2]

        foreground_probability = F.interpolate(
            foreground_probability,
            size=raw_delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        negative_scale, _ = (
            self.evidence_protection_from_probability(
                foreground_probability
            )
        )

        alpha, _ = self.get_control_values()

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

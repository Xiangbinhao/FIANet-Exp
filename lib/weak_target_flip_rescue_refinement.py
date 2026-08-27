import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)


class WeakTargetFlipRescueWrapper(
    BoundedGatedResidualWrapper
):
    """
    S3-F: weak-target flip-risk rescue.

    S3-B remains the primary refinement mechanism.

    S3-F only intervenes when:
        1. E0/main logits originally prefer foreground.
        2. S3-B negative correction suppresses that positive
           margin close to or across the decision boundary.

    Rescue never turns an E0-background pixel into foreground.

    Let:
        m       = 0.5 * (L_fg - L_bg)
        delta_B = S3-B bounded correction
        m_B     = m + delta_B

    For m > 0:
        floor = preserve_ratio * m

        rescue_needed =
            relu(floor - m_B)

    Then:
        delta_F =
            delta_B
            + rescue_gate * rescue_needed

    rescue_gate is learned from:
        [delta_B, main_margin]

    and initialized conservatively.
    """

    requires_main_logits = True

    def __init__(
        self,
        *args,
        rescue_init=0.10,
        preserve_ratio=0.05,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        if not (
            0.0 < rescue_init < 1.0
        ):
            raise ValueError(
                "rescue_init must be in (0, 1)"
            )

        if not (
            0.0 <= preserve_ratio <= 1.0
        ):
            raise ValueError(
                "preserve_ratio must be in [0, 1]"
            )

        self.rescue_init = float(
            rescue_init
        )

        self.preserve_ratio = float(
            preserve_ratio
        )

        # Inputs:
        #   channel 0: S3-B delta
        #   channel 1: E0/main decision margin
        #
        # Only 2 weights + 1 bias = 3 parameters.
        self.rescue_gate = nn.Conv2d(
            2,
            1,
            kernel_size=1,
            bias=True,
        )

        # Spatially uniform conservative initialization.
        nn.init.zeros_(
            self.rescue_gate.weight
        )

        rescue_bias = math.log(
            self.rescue_init
            / (1.0 - self.rescue_init)
        )

        nn.init.constant_(
            self.rescue_gate.bias,
            rescue_bias,
        )

    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):
        # --------------------------------------------------
        # Original S3-B bounded gated residual.
        # Expected form:
        #     [-delta_B, +delta_B]
        # --------------------------------------------------
        base_residual = super().forward(
            x_c1,
            x_c2,
        )

        if (
            base_residual.ndim != 4
            or base_residual.shape[1] != 2
        ):
            raise RuntimeError(
                "S3-F expects S3-B to return "
                "[B, 2, H, W], got {}".format(
                    tuple(base_residual.shape)
                )
            )

        base_delta = 0.5 * (
            base_residual[:, 1:2]
            - base_residual[:, 0:1]
        )

        # --------------------------------------------------
        # E0/main prediction margin.
        # Read-only evidence.
        # --------------------------------------------------
        if (
            main_logits.ndim != 4
            or main_logits.shape[1] != 2
        ):
            raise RuntimeError(
                "S3-F requires two-class main logits, "
                "got {}".format(
                    tuple(main_logits.shape)
                )
            )

        main_logits_local = F.interpolate(
            main_logits.detach(),
            size=base_delta.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        main_margin = 0.5 * (
            main_logits_local[:, 1:2]
            - main_logits_local[:, 0:1]
        )

        # Margin after original S3-B correction.
        corrected_margin = (
            main_margin
            + base_delta
        )

        # --------------------------------------------------
        # Weak-target rescue.
        #
        # Only E0-positive locations are eligible.
        # --------------------------------------------------
        main_foreground = (
            main_margin > 0.0
        ).to(
            dtype=base_delta.dtype
        )

        safe_floor = (
            self.preserve_ratio
            * torch.relu(main_margin)
        )

        rescue_needed = torch.relu(
            safe_floor
            - corrected_margin
        )

        rescue_needed = (
            rescue_needed
            * main_foreground
        )

        # Gate uses detached diagnostic evidence so the
        # gate cannot manipulate S3-B merely to make its
        # own gating task easier.
        gate_input = torch.cat(
            [
                base_delta.detach(),
                main_margin.detach(),
            ],
            dim=1,
        )

        rescue_gate = torch.sigmoid(
            self.rescue_gate(
                gate_input
            )
        )

        rescued_delta = (
            base_delta
            + rescue_gate
            * rescue_needed
        )

        # Preserve the original symmetric margin correction.
        return torch.cat(
            [
                -rescued_delta,
                rescued_delta,
            ],
            dim=1,
        )

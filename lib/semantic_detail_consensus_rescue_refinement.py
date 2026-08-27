import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)


class SemanticDetailConsensusRescueWrapper(
    BoundedGatedResidualWrapper
):
    """
    S3-G: Semantic-Detail Consensus Weak-Target Rescue.

    Base mechanism:
        Keep S3-B bounded gated residual unchanged.

    Problem found in S3-F:
        E0-positive evidence alone is not sufficiently reliable.
        Rescue reduces Empty / Under50, but also restores some
        false-positive foreground regions.

    S3-G:
        Preserve the S3-F flip-risk rescue condition, but introduce
        a high-resolution consensus rejector using x_c1 and x_c2.

    Definitions:
        m       = 0.5 * (L_fg - L_bg)
        delta_B = original S3-B correction
        m_B     = m + delta_B

    Rescue eligibility:
        m > 0
        delta_B < 0
        m_B < preserve_ratio * m

    rescue_needed =
        relu(preserve_ratio * m - m_B)

    High-resolution rejection:
        reject = sigmoid(
            Conv1x1([x_c1, upsample(x_c2)])
        )

    Final rescue:
        rescue =
            rescue_scale
            * (1 - reject)
            * rescue_needed

    Important guarantees:
      1. E0-background pixels can never be rescued.
      2. S3-B positive corrections are never rescued.
      3. Consensus branch can only attenuate rescue.
      4. It cannot produce stronger rescue than rescue_scale.
      5. x_c1/x_c2 are detached for this auxiliary decision branch.
    """

    requires_main_logits = True

    def __init__(
        self,
        *args,
        rescue_scale=0.10,
        preserve_ratio=0.05,
        reject_init=0.05,
        support_in_channels=384,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        if not (
            0.0 < rescue_scale <= 1.0
        ):
            raise ValueError(
                "rescue_scale must be in (0, 1]"
            )

        if not (
            0.0 <= preserve_ratio <= 1.0
        ):
            raise ValueError(
                "preserve_ratio must be in [0, 1]"
            )

        if not (
            0.0 < reject_init < 1.0
        ):
            raise ValueError(
                "reject_init must be in (0, 1)"
            )

        self.rescue_scale = float(
            rescue_scale
        )

        self.preserve_ratio = float(
            preserve_ratio
        )

        self.reject_init = float(
            reject_init
        )

        self.support_in_channels = int(
            support_in_channels
        )

        # --------------------------------------------------
        # High-resolution rescue rejector.
        #
        # FIANet:
        #   x_c1 = 128 channels
        #   x_c2 = 256 channels
        #
        # concat = 384 channels
        #
        # Only 384 weights + 1 bias = 385 parameters.
        # --------------------------------------------------
        self.consensus_rejector = nn.Conv2d(
            self.support_in_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        # Start almost identical to S3-F:
        #
        # reject_init = 0.05
        # support = 0.95
        #
        # Effective initial rescue:
        # 0.10 * 0.95 = 0.095
        #
        # S3-F learned rescue was ~0.102, so initialization
        # remains deliberately close to S3-F.
        nn.init.zeros_(
            self.consensus_rejector.weight
        )

        reject_bias = math.log(
            self.reject_init
            / (1.0 - self.reject_init)
        )

        nn.init.constant_(
            self.consensus_rejector.bias,
            reject_bias,
        )

    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):
        # --------------------------------------------------
        # 1. Original S3-B correction.
        #
        # Expected output:
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
                "S3-G expects S3-B output "
                "[B, 2, H, W], got {}".format(
                    tuple(base_residual.shape)
                )
            )

        base_delta = 0.5 * (
            base_residual[:, 1:2]
            - base_residual[:, 0:1]
        )

        # --------------------------------------------------
        # 2. Read-only E0 semantic margin.
        # --------------------------------------------------
        if (
            main_logits.ndim != 4
            or main_logits.shape[1] != 2
        ):
            raise RuntimeError(
                "S3-G requires two-class main logits, "
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

        corrected_margin = (
            main_margin
            + base_delta
        )

        # --------------------------------------------------
        # 3. Same flip-risk condition as S3-F.
        #
        # Only protect:
        #   - E0-positive pixels
        #   - that S3-B is suppressing
        # --------------------------------------------------
        main_foreground = (
            main_margin > 0.0
        ).to(
            dtype=base_delta.dtype
        )

        negative_correction = (
            base_delta < 0.0
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
            * negative_correction
        )

        # --------------------------------------------------
        # 4. High-resolution semantic-detail consensus.
        #
        # x_c1/x_c2 are already multimodal FIANet features.
        # They are read-only here: the auxiliary rejector
        # must learn from them, not manipulate the backbone.
        # --------------------------------------------------
        detail_c1 = x_c1.detach()
        detail_c2 = x_c2.detach()

        detail_c2 = F.interpolate(
            detail_c2,
            size=detail_c1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        consensus_feature = torch.cat(
            [
                detail_c1,
                detail_c2,
            ],
            dim=1,
        )

        if (
            consensus_feature.shape[1]
            != self.support_in_channels
        ):
            raise RuntimeError(
                "S3-G consensus feature has {} channels, "
                "expected {}".format(
                    consensus_feature.shape[1],
                    self.support_in_channels,
                )
            )

        # Resize if S3-B residual resolution differs.
        if (
            consensus_feature.shape[-2:]
            != base_delta.shape[-2:]
        ):
            consensus_feature = F.interpolate(
                consensus_feature,
                size=base_delta.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        reject_probability = torch.sigmoid(
            self.consensus_rejector(
                consensus_feature
            )
        )

        rescue_support = (
            1.0
            - reject_probability
        )

        # --------------------------------------------------
        # 5. Selective bounded rescue.
        #
        # Consensus can only REDUCE rescue.
        # It can never create stronger rescue.
        # --------------------------------------------------
        rescue = (
            self.rescue_scale
            * rescue_support
            * rescue_needed
        )

        rescued_delta = (
            base_delta
            + rescue
        )

        # Preserve symmetric two-class margin correction.
        return torch.cat(
            [
                -rescued_delta,
                rescued_delta,
            ],
            dim=1,
        )

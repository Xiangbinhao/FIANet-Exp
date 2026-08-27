import torch
import torch.nn.functional as F

from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)


class BudgetedWeakTargetRescueWrapper(
    BoundedGatedResidualWrapper
):
    """
    S3-H: Budgeted Weak-Target Rescue.

    Base:
        Original S3-B bounded gated residual.

    Motivation:
        S3-F proved that flip-risk rescue can reduce
        Empty / Under50 predictions, but unrestricted
        per-image rescue can reintroduce Tiny foreground
        expansion.

        S3-G showed that learning another pixel-wise
        rejector does not reliably solve this problem.

    S3-H therefore keeps the useful S3-F rescue rule,
    but constrains the TOTAL rescue mass independently
    for each image.

    Let:
        m       = 0.5 * (L_fg - L_bg)
        delta_B = original S3-B correction
        m_B     = m + delta_B

    Flip-risk rescue is allowed only when:
        m > 0
        delta_B < 0
        m_B < preserve_ratio * m

    rescue_needed =
        relu(preserve_ratio * m - m_B)

    raw_rescue =
        rescue_scale * rescue_needed

    Per-image negative suppression mass:
        negative_mass =
            sum(
                I(m > 0) * relu(-delta_B)
            )

    Per-image rescue budget:
        budget =
            budget_ratio * negative_mass

    Budget scale:
        scale =
            min(
                1,
                budget / (sum(raw_rescue) + eps)
            )

    Final:
        rescue =
            scale * raw_rescue

        delta_H =
            delta_B + rescue

    Important guarantees:
      1. E0-background pixels can never be rescued.
      2. S3-B positive corrections are never rescued.
      3. Rescue is always non-negative.
      4. Rescue never exceeds the original S3-F-like
         raw rescue.
      5. Total rescue per image is explicitly bounded.
      6. No additional learnable parameters are added.
    """

    requires_main_logits = True

    def __init__(
        self,
        *args,
        rescue_scale=0.10,
        preserve_ratio=0.05,
        budget_ratio=0.06,
        budget_eps=1e-6,
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
            0.0 < budget_ratio <= 1.0
        ):
            raise ValueError(
                "budget_ratio must be in (0, 1]"
            )

        if not (
            budget_eps > 0.0
        ):
            raise ValueError(
                "budget_eps must be > 0"
            )

        self.rescue_scale = float(
            rescue_scale
        )

        self.preserve_ratio = float(
            preserve_ratio
        )

        self.budget_ratio = float(
            budget_ratio
        )

        self.budget_eps = float(
            budget_eps
        )

    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):
        # --------------------------------------------------
        # 1. Original S3-B bounded gated correction.
        #
        # Output is expected to be:
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
                "S3-H expects S3-B output "
                "[B, 2, H, W], got {}".format(
                    tuple(base_residual.shape)
                )
            )

        base_delta = 0.5 * (
            base_residual[:, 1:2]
            - base_residual[:, 0:1]
        )

        # --------------------------------------------------
        # 2. E0/main decision margin.
        # Read-only evidence.
        # --------------------------------------------------
        if (
            main_logits.ndim != 4
            or main_logits.shape[1] != 2
        ):
            raise RuntimeError(
                "S3-H requires two-class main logits, "
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
        # 3. Flip-risk eligibility.
        #
        # Only protect pixels that:
        #   A. E0 originally regards as foreground;
        #   B. S3-B is actively suppressing.
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

        eligible = (
            main_foreground
            * negative_correction
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
            * eligible
        )

        # --------------------------------------------------
        # 4. S3-F-like raw rescue.
        # --------------------------------------------------
        raw_rescue = (
            self.rescue_scale
            * rescue_needed
        )

        # --------------------------------------------------
        # 5. Per-image rescue budget.
        #
        # Measure how much negative correction S3-B applies
        # to E0-positive pixels.
        #
        # Shape after summation:
        #     [B, 1, 1, 1]
        # --------------------------------------------------
        negative_suppression = (
            torch.relu(-base_delta)
            * main_foreground
        )

        negative_mass = (
            negative_suppression.sum(
                dim=(1, 2, 3),
                keepdim=True,
            )
        )

        rescue_mass = (
            raw_rescue.sum(
                dim=(1, 2, 3),
                keepdim=True,
            )
        )

        rescue_budget = (
            self.budget_ratio
            * negative_mass
        )

        # --------------------------------------------------
        # The budget scale is used as a control quantity.
        #
        # Detaching it prevents the network from gaming the
        # denominator / budget itself merely to obtain a
        # larger rescue allowance.
        # --------------------------------------------------
        with torch.no_grad():
            budget_scale = (
                rescue_budget
                / (
                    rescue_mass
                    + self.budget_eps
                )
            )

            budget_scale = torch.clamp(
                budget_scale,
                min=0.0,
                max=1.0,
            )

        # If rescue_mass == 0:
        # raw_rescue is also zero, so any finite scale gives
        # exactly zero final rescue.
        final_rescue = (
            raw_rescue
            * budget_scale
        )

        # --------------------------------------------------
        # 6. Final S3-H correction.
        # --------------------------------------------------
        rescued_delta = (
            base_delta
            + final_rescue
        )

        return torch.cat(
            [
                -rescued_delta,
                rescued_delta,
            ],
            dim=1,
        )

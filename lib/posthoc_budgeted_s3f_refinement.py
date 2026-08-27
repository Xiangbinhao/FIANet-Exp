import os

import torch

from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)

from lib.weak_target_flip_rescue_refinement import (
    WeakTargetFlipRescueWrapper,
)


class PosthocBudgetedS3FWrapper(
    WeakTargetFlipRescueWrapper
):
    """
    S3-H v2:
        Post-hoc Budget-Constrained S3-F.

    IMPORTANT:
        This wrapper is intended for evaluation/inference only.

    It preserves the trained S3-F model exactly and only
    limits the total S3-F rescue mass per image.

    No new learnable parameters are introduced.
    Therefore an S3-F checkpoint can be loaded with strict=True.

    Environment variable:
        S3H_BUDGET_RATIO

    Example:
        S3H_BUDGET_RATIO=0.02
    """

    requires_main_logits = True

    def __init__(
        self,
        *args,
        budget_ratio=None,
        budget_eps=1e-8,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        if budget_ratio is None:
            budget_ratio = float(
                os.environ.get(
                    "S3H_BUDGET_RATIO",
                    "0.02",
                )
            )

        if budget_ratio <= 0.0:
            raise ValueError(
                "budget_ratio must be > 0"
            )

        self.budget_ratio = float(
            budget_ratio
        )

        self.budget_eps = float(
            budget_eps
        )

        print(
            "S3-H v2 post-hoc budget enabled: "
            "budget_ratio={:.6f}".format(
                self.budget_ratio
            )
        )

    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):
        # --------------------------------------------------
        # S3-B output.
        #
        # Explicit base-class call is intentional.
        # In eval mode it is deterministic.
        # --------------------------------------------------
        base_residual = (
            BoundedGatedResidualWrapper.forward(
                self,
                x_c1,
                x_c2,
            )
        )

        base_delta = 0.5 * (
            base_residual[:, 1:2]
            - base_residual[:, 0:1]
        )

        # --------------------------------------------------
        # Original trained S3-F output.
        # --------------------------------------------------
        s3f_residual = (
            WeakTargetFlipRescueWrapper.forward(
                self,
                x_c1,
                x_c2,
                main_logits,
            )
        )

        s3f_delta = 0.5 * (
            s3f_residual[:, 1:2]
            - s3f_residual[:, 0:1]
        )

        # S3-F only adds non-negative rescue relative to S3-B.
        raw_rescue = torch.clamp(
            s3f_delta - base_delta,
            min=0.0,
        )

        # --------------------------------------------------
        # Reconstruct read-only E0 foreground eligibility.
        # --------------------------------------------------
        main_logits_local = torch.nn.functional.interpolate(
            main_logits.detach(),
            size=base_delta.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        main_margin = 0.5 * (
            main_logits_local[:, 1:2]
            - main_logits_local[:, 0:1]
        )

        main_foreground = (
            main_margin > 0.0
        ).to(
            dtype=base_delta.dtype
        )

        # --------------------------------------------------
        # Amount of negative S3-B suppression on E0-positive
        # pixels. This is the reference mass for the budget.
        # --------------------------------------------------
        negative_suppression = (
            torch.relu(-base_delta)
            * main_foreground
        )

        # Inference-only control statistics in FP32.
        negative_mass = (
            negative_suppression.float().sum(
                dim=(1, 2, 3),
                keepdim=True,
            )
        )

        rescue_mass = (
            raw_rescue.float().sum(
                dim=(1, 2, 3),
                keepdim=True,
            )
        )

        budget = (
            self.budget_ratio
            * negative_mass
        )

        scale = (
            budget
            / (
                rescue_mass
                + self.budget_eps
            )
        )

        scale = torch.nan_to_num(
            scale,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        scale = torch.clamp(
            scale,
            min=0.0,
            max=1.0,
        )

        scale = scale.to(
            dtype=raw_rescue.dtype
        )

        # --------------------------------------------------
        # Optional inference-only budget diagnostics.
        # --------------------------------------------------
        if os.environ.get(
            "S3H_DEBUG_BUDGET",
            "0",
        ) == "1":
            with torch.no_grad():
                ratio_dbg = (
                    rescue_mass
                    / (
                        negative_mass
                        + self.budget_eps
                    )
                ).detach().float().cpu().reshape(-1)

                scale_dbg = (
                    scale.detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )

                neg_dbg = (
                    negative_mass.detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )

                rescue_dbg = (
                    rescue_mass.detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )

                nz_dbg = (
                    (raw_rescue > 0)
                    .sum(
                        dim=(1, 2, 3)
                    )
                    .detach()
                    .cpu()
                    .reshape(-1)
                )

                for i in range(
                    ratio_dbg.numel()
                ):
                    print(
                        "S3HDBG "
                        "ratio={:.10g} "
                        "scale={:.10g} "
                        "negative_mass={:.10g} "
                        "rescue_mass={:.10g} "
                        "rescue_nz={}".format(
                            float(ratio_dbg[i]),
                            float(scale_dbg[i]),
                            float(neg_dbg[i]),
                            float(rescue_dbg[i]),
                            int(nz_dbg[i]),
                        )
                    )

        final_rescue = (
            raw_rescue
            * scale
        )

        final_delta = (
            base_delta
            + final_rescue
        )

        return torch.cat(
            [
                -final_delta,
                final_delta,
            ],
            dim=1,
        )

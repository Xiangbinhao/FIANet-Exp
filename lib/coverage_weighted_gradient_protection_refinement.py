import os

import torch
import torch.nn.functional as F

from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)


def _identity_forward_gradient_scale(x, scale):
    """
    Forward:
        y == x

    Backward:
        dy/dx == scale

    scale is treated as a fixed gradient-routing mask.
    """
    scale = scale.detach().to(
        device=x.device,
        dtype=x.dtype,
    )

    return (
        x.detach()
        + scale * (
            x - x.detach()
        )
    )


class CoverageWeightedGradientProtectionWrapper(
    BoundedGatedResidualWrapper
):
    """
    S3-J: Target-Consistent Gradient Protection.

    Forward is EXACTLY S3-B.

    Training-only modification:
        attenuate the gradient through the S3-B refinement
        residual on true-GT foreground pixels where S3-B
        is suppressing an E0-positive weak target.

    Protection conditions:
        1. main_margin > 0
        2. base_delta < 0
        3. corrected_margin < preserve_ratio * main_margin
        4. GT == foreground

    On protected cells:
        gradient scale = 1 - protect_strength

    Default:
        protect_strength = 0.10
        gradient scale    = 0.90

    Evaluation:
        target is not required.
        Output is exactly the original S3-B residual.

    No additional learnable parameters.
    """

    requires_main_logits = True

    def __init__(
        self,
        *args,
        protect_strength=None,
        preserve_ratio=None,
        coverage_ref=None,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        if protect_strength is None:
            protect_strength = float(
                os.environ.get(
                    "S3J_PROTECT_STRENGTH",
                    "0.10",
                )
            )

        if preserve_ratio is None:
            preserve_ratio = float(
                os.environ.get(
                    "S3J_PRESERVE_RATIO",
                    "0.05",
                )
            )

        if coverage_ref is None:
            coverage_ref = float(
                os.environ.get(
                    "S3J_COVERAGE_REF",
                    "0.25",
                )
            )

        if not (
            0.0 <= protect_strength < 1.0
        ):
            raise ValueError(
                "protect_strength must be in [0, 1)"
            )

        if not (
            0.0 <= preserve_ratio <= 1.0
        ):
            raise ValueError(
                "preserve_ratio must be in [0, 1]"
            )

        self.protect_strength = float(
            protect_strength
        )

        if not (
            0.0 < coverage_ref <= 1.0
        ):
            raise ValueError(
                "coverage_ref must be in (0, 1]"
            )

        self.preserve_ratio = float(
            preserve_ratio
        )

        self.coverage_ref = float(
            coverage_ref
        )

        # Plain Python attribute:
        # intentionally NOT a buffer / parameter.
        self._training_target = None

        # Debug-only Python counter.
        self._s3i_forward_count = 0

        print(
            "S3-J target-consistent gradient protection enabled: "
            "protect_strength={:.4f}, "
            "protected_grad_scale={:.4f}, "
            "preserve_ratio={:.4f}".format(
                self.protect_strength,
                1.0 - self.protect_strength,
                self.preserve_ratio,
            )
        )

    def set_training_target(
        self,
        target,
    ):
        """
        Inject GT immediately before the training forward.

        The target is consumed by the next training forward.
        """
        if target is None:
            raise RuntimeError(
                "S3-J received target=None"
            )

        self._training_target = target

    def _build_local_gt_foreground(
        self,
        target,
        output_size,
        device,
    ):
        # Expected segmentation labels:
        # background = 0
        # foreground = 1
        # possible ignore label != 1 is NOT treated as FG.

        if target.ndim == 3:
            target = target.unsqueeze(1)

        elif (
            target.ndim == 4
            and target.shape[1] == 1
        ):
            pass

        else:
            raise RuntimeError(
                "S3-J expected target [B,H,W] or "
                "[B,1,H,W], got {}".format(
                    tuple(target.shape)
                )
            )

        target = target.to(
            device=device,
        )

        gt_fg = (
            target == 1
        ).float()

        in_h, in_w = gt_fg.shape[-2:]
        out_h, out_w = output_size

        # S3-J:
        # preserve foreground OCCUPANCY rather than binary
        # "any foreground pixel exists" support.
        if (
            in_h >= out_h
            and in_w >= out_w
        ):
            gt_coverage = (
                F.adaptive_avg_pool2d(
                    gt_fg,
                    output_size,
                )
            )
        else:
            gt_coverage = F.interpolate(
                gt_fg,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return torch.clamp(
            gt_coverage,
            min=0.0,
            max=1.0,
        )

    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):
        # --------------------------------------------------
        # 1. Exact original S3-B forward.
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
                "S3-J expected S3-B residual "
                "[B,2,H,W], got {}".format(
                    tuple(base_residual.shape)
                )
            )

        # Evaluation must be EXACTLY S3-B.
        if not self.training:
            return base_residual

        # --------------------------------------------------
        # 2. Training must explicitly provide GT.
        # Never silently fall back to S3-B.
        # --------------------------------------------------
        if self._training_target is None:
            raise RuntimeError(
                "S3-J training forward has no GT target. "
                "Call set_s3j_training_target(model, target) "
                "immediately before model forward."
            )

        target = self._training_target

        # Consume it now to prevent stale-target reuse.
        self._training_target = None

        base_delta = 0.5 * (
            base_residual[:, 1:2]
            - base_residual[:, 0:1]
        )

        # --------------------------------------------------
        # 3. Read-only E0/main evidence.
        # --------------------------------------------------
        if (
            main_logits.ndim != 4
            or main_logits.shape[1] != 2
        ):
            raise RuntimeError(
                "S3-J requires two-class main logits, "
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

        safe_floor = (
            self.preserve_ratio
            * torch.relu(main_margin)
        )

        # --------------------------------------------------
        # 4. GT foreground at local refinement resolution.
        # --------------------------------------------------
        gt_foreground = (
            self._build_local_gt_foreground(
                target,
                base_delta.shape[-2:],
                base_delta.device,
            )
        )

        if (
            gt_foreground.shape[0]
            != base_delta.shape[0]
        ):
            raise RuntimeError(
                "S3-J target batch mismatch: "
                "target B={}, residual B={}".format(
                    gt_foreground.shape[0],
                    base_delta.shape[0],
                )
            )

        # --------------------------------------------------
        # 5. Target-consistent protection mask.
        #
        # These conditions are routing decisions only;
        # they must not themselves carry gradients.
        # --------------------------------------------------
        with torch.no_grad():
            main_foreground = (
                main_margin > 0.0
            )

            negative_correction = (
                base_delta.detach() < 0.0
            )

            weak_after_refine = (
                corrected_margin.detach()
                < safe_floor
            )

            flip_risk = (
                main_foreground
                & negative_correction
                & weak_after_refine
            )

            # Coverage-weighted GT support.
            #
            # coverage_ref=0.25:
            # >=25% foreground occupancy gets full protection;
            # sparse edge/Tiny overlap gets proportionally less.
            coverage_ref = self.coverage_ref

            gt_support = torch.clamp(
                gt_foreground / coverage_ref,
                min=0.0,
                max=1.0,
            )

            protect_weight = (
                flip_risk.to(
                    dtype=base_residual.dtype
                )
                * gt_support.to(
                    dtype=base_residual.dtype
                )
            )

            grad_scale = (
                1.0
                - self.protect_strength
                * protect_weight
            )

        # --------------------------------------------------
        # 6. Straight-through gradient protection.
        #
        # Forward:
        #     protected_residual == base_residual
        #
        # Backward:
        #     protected cells -> ~0.9 gradient
        #     all others       -> 1.0 gradient
        # --------------------------------------------------
        grad_scale_2c = grad_scale.expand_as(
            base_residual
        )

        protected_residual = (
            _identity_forward_gradient_scale(
                base_residual,
                grad_scale_2c,
            )
        )

        # --------------------------------------------------
        # 7. Optional diagnostics.
        # --------------------------------------------------
        self._s3i_forward_count += 1

        if os.environ.get(
            "S3J_DEBUG_GRAD",
            "0",
        ) == "1":
            if (
                self._s3i_forward_count <= 5
                or self._s3i_forward_count % 200 == 0
            ):
                with torch.no_grad():
                    risk_n = int(
                        flip_risk.sum().item()
                    )

                    protected_n = float(
                        protect_weight.sum().item()
                    )

                    gt_n = int(
                        gt_foreground.sum().item()
                    )

                    total_n = int(
                        protect_weight.numel()
                    )

                    print(
                        "S3JDBG "
                        "step={} "
                        "local={}x{} "
                        "gt_fg_cells={} "
                        "flip_risk_cells={} "
                        "protected_mass={:.4f} "
                        "protected_rate={:.8f}".format(
                            self._s3i_forward_count,
                            base_delta.shape[-2],
                            base_delta.shape[-1],
                            gt_n,
                            risk_n,
                            protected_n,
                            (
                                protected_n
                                / max(total_n, 1)
                            ),
                        )
                    )

        return protected_residual


def set_s3j_training_target(
    model,
    target,
):
    """
    Robust helper for normal model or DDP model.
    """
    root = (
        model.module
        if hasattr(model, "module")
        else model
    )

    head = getattr(
        root,
        "small_refinement_head",
        None,
    )

    if head is None:
        raise RuntimeError(
            "S3-J could not find "
            "model.small_refinement_head"
        )

    if not hasattr(
        head,
        "set_training_target",
    ):
        raise RuntimeError(
            "small_refinement_head does not support "
            "S3-J target injection; got {}".format(
                type(head).__name__
            )
        )

    head.set_training_target(
        target
    )

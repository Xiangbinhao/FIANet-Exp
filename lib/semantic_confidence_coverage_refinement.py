import os
import torch
import torch.nn.functional as F

from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)


def _identity_forward_gradient_scale(x, scale):
    return (
        x * scale
        + x.detach() * (1.0 - scale)
    )


class SemanticConfidenceCoverageRefinementWrapper(
    BoundedGatedResidualWrapper
):
    """
    S5-A:
    Semantic Confidence Guided Coverage Refinement.

    Forward:
        identical residual refinement.

    Backward:
        protect unstable foreground corrections.

    Guidance:
        semantic confidence from main logits.

    No GT dependency.
    """

    requires_main_logits = True


    def __init__(
        self,
        *args,
        confidence_floor=0.25,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.confidence_floor = confidence_floor

        # S5-A gradient protection strength
        self.protect_strength = 0.10

        self._s5a_step = 0


        print(
            "S5-A semantic confidence coverage refinement enabled: "
            "confidence_floor={:.3f}".format(
                self.confidence_floor
            )
        )


    def forward(
        self,
        x_c1,
        x_c2,
        main_logits,
    ):

        residual = super().forward(
            x_c1,
            x_c2,
        )


        if (
            residual.ndim != 4
            or residual.shape[1] != 2
        ):
            raise RuntimeError(
                "S5-A residual shape error {}".format(
                    residual.shape
                )
            )


        if (
            main_logits.ndim != 4
            or main_logits.shape[1] != 2
        ):
            raise RuntimeError(
                "S5-A requires two class logits"
            )


        delta = 0.5 * (
            residual[:,1:2]
            -
            residual[:,0:1]
        )


        logits = F.interpolate(
            main_logits.detach(),
            size=delta.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


        confidence = torch.sigmoid(
            logits[:,1:2]
            -
            logits[:,0:1]
        )


        with torch.no_grad():

            main_fg = (
                logits[:,1:2]
                >
                logits[:,0:1]
            )


            negative_update = (
                delta < 0
            )


            weak_region = (
                confidence
                >
                self.confidence_floor
            )


            protect_mask = (
                main_fg
                &
                negative_update
                &
                weak_region
            )


            protect_weight = protect_mask.float()


            grad_scale = (
                1.0
                -
                self.protect_strength
                *
                protect_weight
            )


        protected = _identity_forward_gradient_scale(
            residual,
            grad_scale.expand_as(residual),
        )


        self._s5a_step += 1


        if os.environ.get(
            "S5A_DEBUG",
            "0"
        ) == "1":

            if self._s5a_step <= 5:

                print(
                    "S5ADBG "
                    "step={} "
                    "confidence={:.5f} "
                    "protected={:.6f}".format(
                        self._s5a_step,
                        confidence.mean().item(),
                        protect_weight.mean().item(),
                    )
                )


        return protected

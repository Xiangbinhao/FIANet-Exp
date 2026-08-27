import math

import torch
import torch.nn as nn


class BoundedGatedResidualWrapper(nn.Module):
    """
    Wrap an existing S2 high-resolution refinement head.

    The original S2 head predicts a two-channel residual. This wrapper
    converts it into a bounded, spatially gated, symmetric binary-logit
    correction:

        raw_delta = 0.5 * (r_fg - r_bg)
        delta = alpha * gate * tanh(raw_delta)
        residual = [-delta, +delta]
    """

    def __init__(
        self,
        base_head,
        alpha_init=0.10,
        gate_init=0.20,
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

        self.base_head = base_head

        # Two residual channels -> one spatial gate.
        self.gate_predictor = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Initially produce a spatially uniform conservative gate.
        nn.init.zeros_(self.gate_predictor.weight)

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

    def forward(self, x_c1, x_c2):
        raw_residual = self.base_head(x_c1, x_c2)

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
            raw_residual[:, 1:2] -
            raw_residual[:, 0:1]
        )

        gate = torch.sigmoid(
            self.gate_predictor(raw_residual)
        )
        alpha = torch.sigmoid(
            self.alpha_logit
        )

        bounded_delta = (
            alpha *
            gate *
            torch.tanh(raw_delta)
        )

        bounded_residual = torch.cat(
            [
                -bounded_delta,
                bounded_delta,
            ],
            dim=1,
        )

        return bounded_residual

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentitySemanticModulationRefinementHead(nn.Module):
    """
    S4-B: identity-initialized bounded semantic modulation.

    Detail content:
        x_c1 + x_c2

    Semantic guidance:
        x_c4 only modulates detail features.
        It does NOT directly contribute residual content.

    Formula:
        D' = D * (1 + beta * tanh(S))

        beta = beta_max * tanh(beta_raw)

    beta_raw is initialized to zero, therefore:
        beta = 0
        D' = D

    x_c4 is detached in this auxiliary branch so that
    refinement cannot distort the backbone semantic hierarchy.
    """

    requires_semantic_feature = True

    def __init__(
        self,
        x_c1_channels=128,
        x_c2_channels=256,
        semantic_channels=1024,
        project_channels=16,
        semantic_project_channels=16,
        hidden_channels=32,
        num_classes=2,
        beta_max=0.25,
    ):
        super().__init__()

        self.beta_max = float(beta_max)

        # --------------------------------------------------
        # High-resolution detail branch
        # --------------------------------------------------
        self.x_c1_project = nn.Sequential(
            nn.Conv2d(
                x_c1_channels,
                project_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(project_channels),
            nn.ReLU(inplace=True),
        )

        self.x_c2_project = nn.Sequential(
            nn.Conv2d(
                x_c2_channels,
                project_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(project_channels),
            nn.ReLU(inplace=True),
        )

        detail_input_channels = (
            2 * project_channels
        )

        self.detail_fusion = nn.Sequential(
            nn.Conv2d(
                detail_input_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------
        # Deep semantic modulation branch
        # --------------------------------------------------
        self.semantic_project = nn.Sequential(
            nn.Conv2d(
                semantic_channels,
                semantic_project_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                semantic_project_channels
            ),
            nn.ReLU(inplace=True),
        )

        self.semantic_to_modulation = nn.Conv2d(
            semantic_project_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )

        # Exact identity initialization.
        #
        # beta = beta_max * tanh(beta_raw)
        #
        # beta_raw = 0 -> beta = 0.
        #
        # Do NOT zero initialize semantic_to_modulation:
        # otherwise semantic branch and beta can become
        # jointly gradient-starved.
        self.beta_raw = nn.Parameter(
            torch.zeros(1)
        )

        # --------------------------------------------------
        # Residual predictor
        # --------------------------------------------------
        self.residual_predictor = nn.Conv2d(
            hidden_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Exact E0-equivalent output at initialization.
        nn.init.zeros_(
            self.residual_predictor.weight
        )
        nn.init.zeros_(
            self.residual_predictor.bias
        )

    def get_beta(self):
        return (
            self.beta_max
            * torch.tanh(self.beta_raw)
        )

    def forward(
        self,
        x_c1,
        x_c2,
        semantic_feature,
    ):
        if semantic_feature is None:
            raise RuntimeError(
                "S4-B requires x_c4 semantic feature."
            )

        # ----------------------------------------------
        # Detail content
        # ----------------------------------------------
        x_c1_feature = self.x_c1_project(
            x_c1
        )

        x_c2_feature = self.x_c2_project(
            x_c2
        )

        x_c2_feature = F.interpolate(
            x_c2_feature,
            size=x_c1_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        detail_feature = torch.cat(
            [
                x_c1_feature,
                x_c2_feature,
            ],
            dim=1,
        )

        detail_feature = self.detail_fusion(
            detail_feature
        )

        # ----------------------------------------------
        # Read-only semantic modulation
        # ----------------------------------------------
        semantic_feature = (
            semantic_feature.detach()
        )

        semantic_feature = (
            self.semantic_project(
                semantic_feature
            )
        )

        semantic_feature = F.interpolate(
            semantic_feature,
            size=detail_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        semantic_modulation = (
            self.semantic_to_modulation(
                semantic_feature
            )
        )

        semantic_modulation = torch.tanh(
            semantic_modulation
        )

        beta = self.get_beta()

        modulation_factor = (
            1.0
            + beta * semantic_modulation
        )

        modulated_detail = (
            detail_feature
            * modulation_factor
        )

        residual_logits = (
            self.residual_predictor(
                modulated_detail
            )
        )

        return residual_logits

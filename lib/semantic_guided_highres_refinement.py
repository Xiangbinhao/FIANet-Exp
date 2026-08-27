import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGuidedHighResolutionRefinementHead(nn.Module):
    """
    S4-A: language-aware semantic-guided high-resolution refinement.

    Inputs
    ------
    x_c1:
        High-resolution multimodal feature.
        Swin-B: [B, 128, H/4, W/4]

    x_c2:
        Mid/high-resolution multimodal feature.
        Swin-B: [B, 256, H/8, W/8]

    x_c4:
        Deep language-aware multimodal semantic feature.
        Swin-B: [B, 1024, H/32, W/32]

    Output
    ------
    residual_logits:
        [B, 2, H/4, W/4]

    Design:
        1. Project x_c1 and x_c2 into lightweight detail embeddings.
        2. Project x_c4 into a semantic embedding.
        3. Upsample all features to x_c1 resolution.
        4. Jointly learn semantic-detail fusion.
        5. Predict a two-class residual correction.

    x_c4 is detached in this branch. The semantic hierarchy is still
    optimized normally through the original FIANet decoder, but the
    auxiliary refinement path cannot manipulate x_c4 simply to make
    its own residual prediction easier.

    The final predictor is zero initialized, so the initial S4-A
    output is exactly the original FIANet prediction.
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
    ):
        super().__init__()

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

        fusion_channels = (
            2 * project_channels
            + semantic_project_channels
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(
                fusion_channels,
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

        self.residual_predictor = nn.Conv2d(
            hidden_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Identity initialization:
        # initial S4-A prediction == E0 prediction.
        nn.init.zeros_(
            self.residual_predictor.weight
        )
        nn.init.zeros_(
            self.residual_predictor.bias
        )

    def forward(
        self,
        x_c1,
        x_c2,
        semantic_feature,
    ):
        if semantic_feature is None:
            raise RuntimeError(
                "S4-A requires x_c4 semantic feature."
            )

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

        # x_c4 acts as read-only semantic guidance for this
        # auxiliary branch.
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
            size=x_c1_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused_feature = torch.cat(
            [
                x_c1_feature,
                x_c2_feature,
                semantic_feature,
            ],
            dim=1,
        )

        refined_feature = self.refinement(
            fused_feature
        )

        residual_logits = (
            self.residual_predictor(
                refined_feature
            )
        )

        return residual_logits

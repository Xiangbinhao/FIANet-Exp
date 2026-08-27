import torch
import torch.nn.functional as F
from torch import nn


class HighResolutionSmallObjectRefinementHead(nn.Module):
    """
    S2 high-resolution residual refinement head.

    This branch is independent of E1/E2 coarse localization.

    Inputs:
        x_c1:
            [B, C1, H/4, W/4], normally
            [B, 128, 120, 120].

        x_c2:
            [B, C2, H/8, W/8], normally
            [B, 256, 60, 60].

    Output:
        residual_logits:
            [B, num_classes, H/4, W/4].
    """

    def __init__(
        self,
        x_c1_channels,
        x_c2_channels,
        project_channels=16,
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

        fusion_channels = 2 * project_channels

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
        # the initial S2 prediction equals the E0 prediction.
        nn.init.zeros_(self.residual_predictor.weight)
        nn.init.zeros_(self.residual_predictor.bias)

    def forward(self, x_c1, x_c2):
        x_c1_feature = self.x_c1_project(x_c1)
        x_c2_feature = self.x_c2_project(x_c2)

        x_c2_feature = F.interpolate(
            x_c2_feature,
            size=x_c1_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused_feature = torch.cat(
            [x_c1_feature, x_c2_feature],
            dim=1,
        )

        refined_feature = self.refinement(
            fused_feature
        )

        residual_logits = self.residual_predictor(
            refined_feature
        )

        return residual_logits

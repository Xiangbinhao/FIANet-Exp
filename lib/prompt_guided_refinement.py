import torch
import torch.nn.functional as F
from torch import nn


class PromptGuidedResidualRefinementHead(nn.Module):
    """
    E2 prompt-guided residual refinement head.

    The coarse localization result generated from x_c4 is used as
    a dense internal prompt to refine the higher-resolution x_c2
    feature.

    Inputs:
        feature:
            x_c2, [B, C, H/8, W/8]

        coarse_logits:
            [B, 2, H/32, W/32]

    Output:
        residual_logits:
            [B, 2, H/8, W/8]
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        prompt_channels=16,
        num_classes=2,
    ):
        super().__init__()

        self.feature_project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Two-channel internal prompt:
        # foreground probability and localization uncertainty.
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(
                2,
                prompt_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(inplace=True),
        )

        fusion_channels = hidden_channels + prompt_channels

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

        # Start from exactly the original FIANet/E1 prediction.
        # At initialization residual_logits = 0.
        nn.init.zeros_(self.residual_predictor.weight)
        nn.init.zeros_(self.residual_predictor.bias)

    def forward(self, feature, coarse_logits):
        coarse_probability = torch.softmax(
            coarse_logits,
            dim=1,
        )[:, 1:2]

        uncertainty = (
            4.0
            * coarse_probability
            * (1.0 - coarse_probability)
        )

        prompt = torch.cat(
            [coarse_probability, uncertainty],
            dim=1,
        )

        prompt = F.interpolate(
            prompt,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        feature = self.feature_project(feature)
        prompt = self.prompt_encoder(prompt)

        fused_feature = torch.cat(
            [feature, prompt],
            dim=1,
        )

        refined_feature = self.refinement(fused_feature)
        residual_logits = self.residual_predictor(
            refined_feature
        )

        return residual_logits

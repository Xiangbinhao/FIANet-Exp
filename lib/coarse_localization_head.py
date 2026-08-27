from torch import nn


class CoarseLocalizationHead(nn.Module):
    """
    E1 auxiliary coarse localization head.

    Input:
        Deep multimodal feature x_c4:
        [B, C, H/32, W/32]

    Output:
        Two-class coarse logits:
        [B, 2, H/32, W/32]
    """

    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 8, 64)

        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.localize = nn.Sequential(
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
                2,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, feature):
        feature = self.project(feature)
        return self.localize(feature)

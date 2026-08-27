import torch
from torch import nn
import torch.nn.functional as F


SIZE_GROUP_NAMES = (
    "empty",
    "tiny",
    "small",
    "medium_or_large",
)


class ForegroundSizeAuxiliaryLoss(nn.Module):
    """
    S1-C foreground-only Dice auxiliary loss.

    The original E0 loss is preserved exactly. An additional
    foreground Dice loss is applied only to Tiny and Small
    samples:

        total_loss = E0_loss
                   + mean(lambda_i * foreground_dice_i)

    Medium, Large and empty-GT samples receive zero auxiliary
    coefficient.

    Setting tiny_lambda=small_lambda=0 reproduces the original
    E0 loss exactly.
    """

    def __init__(
        self,
        base_loss,
        tiny_lambda=0.30,
        small_lambda=0.15,
        tiny_max_ratio=0.001,
        small_max_ratio=0.005,
        foreground_class=1,
        smooth=1e-6,
        log_first_batch=False,
    ):
        super().__init__()

        if not (
            0.0 < tiny_max_ratio
            < small_max_ratio
        ):
            raise ValueError(
                "S1-C thresholds must satisfy "
                "0 < tiny < small."
            )

        if tiny_lambda < 0.0:
            raise ValueError(
                "tiny_lambda must be >= 0."
            )

        if small_lambda < 0.0:
            raise ValueError(
                "small_lambda must be >= 0."
            )

        self.base_loss = base_loss

        self.tiny_lambda = float(tiny_lambda)
        self.small_lambda = float(small_lambda)

        self.tiny_max_ratio = float(
            tiny_max_ratio
        )

        self.small_max_ratio = float(
            small_max_ratio
        )

        self.foreground_class = int(
            foreground_class
        )

        self.smooth = float(smooth)

        self.log_first_batch = bool(
            log_first_batch
        )

        self._has_logged_first_batch = False

    def compute_auxiliary_coefficients(
        self,
        target,
    ):
        """
        Args:
            target: integer target [B,H,W].

        Returns:
            coefficients: [B]
            area_ratios: [B]
            group_ids: [B]

        Group IDs:
            0 = empty
            1 = tiny
            2 = small
            3 = medium_or_large
        """
        if target.ndim != 3:
            raise ValueError(
                "S1-C expects target [B,H,W], "
                "but received {}".format(
                    tuple(target.shape)
                )
            )

        batch_size = target.shape[0]

        foreground = (
            target == self.foreground_class
        )

        area_ratios = (
            foreground
            .reshape(batch_size, -1)
            .float()
            .mean(dim=1)
        )

        empty_mask = area_ratios <= 0.0

        tiny_mask = (
            (area_ratios > 0.0)
            & (
                area_ratios
                <= self.tiny_max_ratio
            )
        )

        small_mask = (
            (
                area_ratios
                > self.tiny_max_ratio
            )
            & (
                area_ratios
                <= self.small_max_ratio
            )
        )

        medium_large_mask = (
            area_ratios
            > self.small_max_ratio
        )

        coefficients = torch.zeros_like(
            area_ratios,
            dtype=torch.float32,
        )

        coefficients[tiny_mask] = (
            self.tiny_lambda
        )

        coefficients[small_mask] = (
            self.small_lambda
        )

        group_ids = torch.full(
            (batch_size,),
            fill_value=3,
            dtype=torch.long,
            device=target.device,
        )

        group_ids[empty_mask] = 0
        group_ids[tiny_mask] = 1
        group_ids[small_mask] = 2
        group_ids[medium_large_mask] = 3

        return (
            coefficients,
            area_ratios,
            group_ids,
        )

    def compute_foreground_dice_per_sample(
        self,
        pred,
        target,
    ):
        """
        Foreground-only soft Dice loss for each sample.

        Returns:
            dice_loss_per_sample: [B]
        """
        probabilities = F.softmax(
            pred.float(),
            dim=1,
        )

        foreground_probability = probabilities[
            :,
            self.foreground_class,
            :,
            :,
        ]

        foreground_target = (
            target == self.foreground_class
        ).float()

        batch_size = pred.shape[0]

        foreground_probability = (
            foreground_probability
            .reshape(batch_size, -1)
        )

        foreground_target = (
            foreground_target
            .reshape(batch_size, -1)
        )

        intersection = (
            foreground_probability
            * foreground_target
        ).sum(dim=1)

        denominator = (
            foreground_probability.sum(dim=1)
            + foreground_target.sum(dim=1)
        )

        foreground_dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - foreground_dice

    def forward(self, pred, target):
        target = target.long()

        base_loss = self.base_loss(
            pred,
            target,
        )

        (
            coefficients,
            area_ratios,
            group_ids,
        ) = self.compute_auxiliary_coefficients(
            target
        )

        coefficients = coefficients.to(
            device=pred.device,
            dtype=torch.float32,
        )

        foreground_dice_per_sample = (
            self.compute_foreground_dice_per_sample(
                pred,
                target,
            )
        )

        auxiliary_loss = (
            coefficients
            * foreground_dice_per_sample
        ).mean()

        total_loss = (
            base_loss
            + auxiliary_loss
        )

        if (
            self.log_first_batch
            and not self._has_logged_first_batch
        ):
            counts = torch.bincount(
                group_ids.detach().cpu(),
                minlength=len(SIZE_GROUP_NAMES),
            )

            print(
                "\n========== S1-C FIRST BATCH =========="
            )

            for index, name in enumerate(
                SIZE_GROUP_NAMES
            ):
                print(
                    "{:<16s} count={}".format(
                        name,
                        int(counts[index].item()),
                    )
                )

            print(
                "Area ratios:",
                [
                    round(float(value), 6)
                    for value in
                    area_ratios.detach().cpu()
                ],
            )

            print(
                "Aux coefficients:",
                [
                    round(float(value), 3)
                    for value in
                    coefficients.detach().cpu()
                ],
            )

            print(
                "Foreground Dice losses:",
                [
                    round(float(value), 6)
                    for value in
                    foreground_dice_per_sample
                    .detach()
                    .cpu()
                ],
            )

            print(
                "Base={:.6f} Aux={:.6f} "
                "Total={:.6f}".format(
                    float(base_loss.detach().cpu()),
                    float(
                        auxiliary_loss
                        .detach()
                        .cpu()
                    ),
                    float(total_loss.detach().cpu()),
                )
            )

            print(
                "======================================\n"
            )

            self._has_logged_first_batch = True

        return total_loss

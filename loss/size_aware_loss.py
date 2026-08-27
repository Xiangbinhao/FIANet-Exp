import torch
from torch import nn
import torch.nn.functional as F


SIZE_GROUP_NAMES = (
    "empty",
    "tiny",
    "small",
    "medium",
    "large",
)


class SizeAwareSegmentationLoss(nn.Module):
    """
    S1-B size-aware sample-level loss.

    The original FIANet loss is preserved:

        total = (1 - dice_mix_weight) * CE
                + dice_mix_weight * Dice

    Only each sample's contribution is multiplied by a
    size-dependent weight.

    When all size weights equal 1.0, this implementation is
    numerically equivalent to the original FIANet loss.
    """

    def __init__(
        self,
        tiny_weight=1.5,
        small_weight=1.25,
        medium_weight=1.0,
        large_weight=1.0,
        empty_weight=1.0,
        tiny_max_ratio=0.001,
        small_max_ratio=0.005,
        medium_max_ratio=0.020,
        dice_mix_weight=0.1,
        foreground_class=1,
        smooth=1e-6,
        log_first_batch=False,
    ):
        super().__init__()

        if not (
            0.0 < tiny_max_ratio
            < small_max_ratio
            < medium_max_ratio
        ):
            raise ValueError(
                "S1-B thresholds must satisfy "
                "0 < tiny < small < medium."
            )

        sample_weights = {
            "empty": float(empty_weight),
            "tiny": float(tiny_weight),
            "small": float(small_weight),
            "medium": float(medium_weight),
            "large": float(large_weight),
        }

        for name, value in sample_weights.items():
            if value <= 0.0:
                raise ValueError(
                    "S1-B weight for {} must be > 0."
                    .format(name)
                )

        if not 0.0 <= dice_mix_weight <= 1.0:
            raise ValueError(
                "dice_mix_weight must be in [0, 1]."
            )

        self.empty_weight = sample_weights["empty"]
        self.tiny_weight = sample_weights["tiny"]
        self.small_weight = sample_weights["small"]
        self.medium_weight = sample_weights["medium"]
        self.large_weight = sample_weights["large"]

        self.tiny_max_ratio = float(tiny_max_ratio)
        self.small_max_ratio = float(small_max_ratio)
        self.medium_max_ratio = float(medium_max_ratio)

        self.dice_mix_weight = float(
            dice_mix_weight
        )

        self.foreground_class = int(
            foreground_class
        )

        self.smooth = float(smooth)
        self.log_first_batch = bool(
            log_first_batch
        )

        self._has_logged_first_batch = False

        # Preserve the original FIANet CE class weights.
        self.register_buffer(
            "ce_class_weights",
            torch.tensor(
                [0.9, 1.1],
                dtype=torch.float32,
            ),
        )

    def compute_sample_weights(self, target):
        """
        Args:
            target: [B, H, W], integer segmentation target.

        Returns:
            sample_weights: [B]
            area_ratios: [B]
            group_ids: [B]
                0=empty, 1=tiny, 2=small,
                3=medium, 4=large.
        """
        if target.ndim != 3:
            raise ValueError(
                "S1-B expects target shape [B,H,W], "
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

        medium_mask = (
            (
                area_ratios
                > self.small_max_ratio
            )
            & (
                area_ratios
                <= self.medium_max_ratio
            )
        )

        large_mask = (
            area_ratios
            > self.medium_max_ratio
        )

        sample_weights = torch.ones_like(
            area_ratios,
            dtype=torch.float32,
        )

        sample_weights[empty_mask] = (
            self.empty_weight
        )

        sample_weights[tiny_mask] = (
            self.tiny_weight
        )

        sample_weights[small_mask] = (
            self.small_weight
        )

        sample_weights[medium_mask] = (
            self.medium_weight
        )

        sample_weights[large_mask] = (
            self.large_weight
        )

        group_ids = torch.full(
            (batch_size,),
            fill_value=4,
            dtype=torch.long,
            device=target.device,
        )

        group_ids[empty_mask] = 0
        group_ids[tiny_mask] = 1
        group_ids[small_mask] = 2
        group_ids[medium_mask] = 3
        group_ids[large_mask] = 4

        return (
            sample_weights,
            area_ratios,
            group_ids,
        )

    def _weighted_cross_entropy(
        self,
        pred,
        target,
        sample_weights,
    ):
        """
        Weighted sample contribution while preserving the
        original class-weighted CE normalization.

        When all sample weights are 1, this is equivalent to
        torch.nn.CrossEntropyLoss(
            weight=[0.9, 1.1]
        ).
        """
        class_weights = self.ce_class_weights.to(
            device=pred.device,
            dtype=pred.dtype,
        )

        ce_map = F.cross_entropy(
            pred,
            target,
            weight=class_weights,
            reduction="none",
        )

        pixel_weight_map = class_weights[target]

        batch_size = pred.shape[0]

        ce_numerator_per_sample = (
            ce_map
            .reshape(batch_size, -1)
            .sum(dim=1)
        )

        ce_denominator_per_sample = (
            pixel_weight_map
            .reshape(batch_size, -1)
            .sum(dim=1)
        )

        weighted_numerator = (
            sample_weights
            * ce_numerator_per_sample
        ).sum()

        weighted_denominator = (
            sample_weights
            * ce_denominator_per_sample
        ).sum().clamp_min(self.smooth)

        return (
            weighted_numerator
            / weighted_denominator
        )

    def _weighted_dice_loss(
        self,
        pred,
        target,
        sample_weights,
    ):
        """
        Preserve the original summed Dice-loss scale.

        Original DiceLoss sums over all samples and classes.
        The B / sum(weights) factor keeps the mean sample
        weight equal to one, preventing an artificial change
        in the total loss scale.
        """
        probabilities = F.softmax(
            pred,
            dim=1,
        )

        num_classes = pred.shape[1]

        one_hot_target = F.one_hot(
            target,
            num_classes=num_classes,
        )

        one_hot_target = (
            one_hot_target
            .permute(0, 3, 1, 2)
            .to(dtype=probabilities.dtype)
        )

        intersection = (
            probabilities
            * one_hot_target
        ).sum(dim=(2, 3))

        union = (
            probabilities
            + one_hot_target
        ).sum(dim=(2, 3))

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        dice_loss_per_sample = (
            1.0 - dice_score
        ).sum(dim=1)

        batch_size = pred.shape[0]

        scale_normalizer = (
            float(batch_size)
            / sample_weights.sum().clamp_min(
                self.smooth
            )
        )

        weighted_dice = (
            sample_weights
            * dice_loss_per_sample
        ).sum()

        return (
            scale_normalizer
            * weighted_dice
        )

    def forward(self, pred, target):
        target = target.long()

        (
            sample_weights,
            area_ratios,
            group_ids,
        ) = self.compute_sample_weights(
            target
        )

        sample_weights = sample_weights.to(
            device=pred.device,
            dtype=torch.float32,
        )

        ce_loss = self._weighted_cross_entropy(
            pred,
            target,
            sample_weights,
        )

        dice_loss = self._weighted_dice_loss(
            pred,
            target,
            sample_weights,
        )

        total_loss = (
            (1.0 - self.dice_mix_weight)
            * ce_loss
            + self.dice_mix_weight
            * dice_loss
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
                "\n========== S1-B FIRST BATCH =========="
            )

            for index, name in enumerate(
                SIZE_GROUP_NAMES
            ):
                print(
                    "{:<6s} count={}".format(
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
                "Sample weights:",
                [
                    round(float(value), 3)
                    for value in
                    sample_weights.detach().cpu()
                ],
            )

            print(
                "CE={:.6f} Dice={:.6f} "
                "Total={:.6f}".format(
                    float(
                        ce_loss.detach().cpu()
                    ),
                    float(
                        dice_loss.detach().cpu()
                    ),
                    float(
                        total_loss.detach().cpu()
                    ),
                )
            )

            print(
                "======================================\n"
            )

            self._has_logged_first_batch = True

        return total_loss

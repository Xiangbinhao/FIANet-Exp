import math

import numpy as np
import torch


EPSILON = 1.0e-12


def to_binary_tensor(mask):
    """
    Convert a prediction or GT mask to a flat CPU bool tensor.

    Supported shapes:
      H x W
      1 x H x W
      B x H x W, when B=1
      B x 1 x H x W, when B=1
    """
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    mask = mask.detach().cpu()

    while (
        mask.ndim > 2
        and mask.shape[0] == 1
    ):
        mask = mask.squeeze(0)

    if (
        mask.ndim == 3
        and mask.shape[-1] == 1
    ):
        mask = mask[..., 0]

    if mask.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional binary mask, "
            "received shape {}".format(
                tuple(mask.shape)
            )
        )

    return mask.bool().reshape(-1)


def safe_ratio(numerator, denominator, empty_value=0.0):
    denominator = float(denominator)

    if denominator <= 0.0:
        return float(empty_value)

    return float(numerator) / denominator


def compute_binary_sample_metrics(
    prediction,
    target,
):
    """
    Compute sample-level foreground metrics.

    For a nonempty GT with empty prediction:
      precision = recall = dice = 0

    For both-empty masks:
      precision = recall = dice = 1
      The empty-GT sample should normally be assigned to
      the separate 'empty' size group.
    """
    prediction = to_binary_tensor(
        prediction
    )

    target = to_binary_tensor(
        target
    )

    if prediction.numel() != target.numel():
        raise ValueError(
            "Prediction and target have different "
            "pixel counts: {} versus {}".format(
                prediction.numel(),
                target.numel(),
            )
        )

    true_positive = int(
        (prediction & target).sum().item()
    )

    false_positive = int(
        (prediction & ~target).sum().item()
    )

    false_negative = int(
        (~prediction & target).sum().item()
    )

    true_negative = int(
        (~prediction & ~target).sum().item()
    )

    predicted_pixels = int(
        prediction.sum().item()
    )

    target_pixels = int(
        target.sum().item()
    )

    if predicted_pixels == 0 and target_pixels == 0:
        precision = 1.0
        recall = 1.0
        dice = 1.0
        iou = 1.0
        area_ratio = 1.0

    else:
        precision = safe_ratio(
            true_positive,
            true_positive + false_positive,
            empty_value=0.0,
        )

        recall = safe_ratio(
            true_positive,
            true_positive + false_negative,
            empty_value=0.0,
        )

        dice = safe_ratio(
            2 * true_positive,
            (
                2 * true_positive
                + false_positive
                + false_negative
            ),
            empty_value=0.0,
        )

        iou = safe_ratio(
            true_positive,
            (
                true_positive
                + false_positive
                + false_negative
            ),
            empty_value=0.0,
        )

        area_ratio = safe_ratio(
            predicted_pixels,
            target_pixels,
            empty_value=float("inf"),
        )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "pred_pixels": predicted_pixels,
        "gt_pixels": target_pixels,
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "area_ratio": float(area_ratio),
        "empty_prediction": int(
            predicted_pixels == 0
        ),
        "empty_target": int(
            target_pixels == 0
        ),
        "under_050": int(
            target_pixels > 0
            and area_ratio < 0.50
        ),
        "over_150": int(
            target_pixels > 0
            and area_ratio > 1.50
        ),
    }


def summarize_numeric(values):
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if not finite_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }

    array = np.asarray(
        finite_values,
        dtype=np.float64,
    )

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


class GroupMetricAccumulator(object):
    """
    Aggregate sample-level and pixel-level statistics
    for one size group.
    """

    def __init__(self, name):
        self.name = str(name)

        self.sample_count = 0

        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0

        self.total_pred_pixels = 0
        self.total_gt_pixels = 0

        self.empty_prediction_count = 0
        self.empty_target_count = 0

        self.under_050_count = 0
        self.over_150_count = 0

        self.precisions = []
        self.recalls = []
        self.dices = []
        self.ious = []
        self.area_ratios = []

        self.rows = []

    def update(
        self,
        prediction,
        target,
        sample_index=None,
        extra=None,
    ):
        metrics = compute_binary_sample_metrics(
            prediction,
            target,
        )

        self.sample_count += 1

        self.total_tp += metrics["tp"]
        self.total_fp += metrics["fp"]
        self.total_fn += metrics["fn"]
        self.total_tn += metrics["tn"]

        self.total_pred_pixels += (
            metrics["pred_pixels"]
        )

        self.total_gt_pixels += (
            metrics["gt_pixels"]
        )

        self.empty_prediction_count += (
            metrics["empty_prediction"]
        )

        self.empty_target_count += (
            metrics["empty_target"]
        )

        self.under_050_count += (
            metrics["under_050"]
        )

        self.over_150_count += (
            metrics["over_150"]
        )

        self.precisions.append(
            metrics["precision"]
        )

        self.recalls.append(
            metrics["recall"]
        )

        self.dices.append(
            metrics["dice"]
        )

        self.ious.append(
            metrics["iou"]
        )

        self.area_ratios.append(
            metrics["area_ratio"]
        )

        row = {
            "sample_index": sample_index,
            "size_group": self.name,
        }

        row.update(metrics)

        if extra:
            row.update(extra)

        self.rows.append(row)

        return metrics

    def summary(self):
        macro_precision = summarize_numeric(
            self.precisions
        )

        macro_recall = summarize_numeric(
            self.recalls
        )

        macro_dice = summarize_numeric(
            self.dices
        )

        macro_iou = summarize_numeric(
            self.ious
        )

        area_ratio = summarize_numeric(
            self.area_ratios
        )

        micro_precision = safe_ratio(
            self.total_tp,
            self.total_tp + self.total_fp,
            empty_value=0.0,
        )

        micro_recall = safe_ratio(
            self.total_tp,
            self.total_tp + self.total_fn,
            empty_value=0.0,
        )

        micro_dice = safe_ratio(
            2 * self.total_tp,
            (
                2 * self.total_tp
                + self.total_fp
                + self.total_fn
            ),
            empty_value=0.0,
        )

        micro_iou = safe_ratio(
            self.total_tp,
            (
                self.total_tp
                + self.total_fp
                + self.total_fn
            ),
            empty_value=0.0,
        )

        return {
            "group": self.name,
            "sample_count": int(
                self.sample_count
            ),
            "pixel_totals": {
                "tp": int(self.total_tp),
                "fp": int(self.total_fp),
                "fn": int(self.total_fn),
                "tn": int(self.total_tn),
                "pred_pixels": int(
                    self.total_pred_pixels
                ),
                "gt_pixels": int(
                    self.total_gt_pixels
                ),
            },
            "macro_precision": (
                macro_precision["mean"]
            ),
            "macro_recall": (
                macro_recall["mean"]
            ),
            "macro_dice": (
                macro_dice["mean"]
            ),
            "macro_iou": (
                macro_iou["mean"]
            ),
            "micro_precision": float(
                micro_precision
            ),
            "micro_recall": float(
                micro_recall
            ),
            "micro_dice": float(
                micro_dice
            ),
            "micro_iou": float(
                micro_iou
            ),
            "fp_per_gt_pixel": safe_ratio(
                self.total_fp,
                self.total_gt_pixels,
                empty_value=0.0,
            ),
            "fn_per_gt_pixel": safe_ratio(
                self.total_fn,
                self.total_gt_pixels,
                empty_value=0.0,
            ),
            "pred_to_gt_pixel_ratio": (
                safe_ratio(
                    self.total_pred_pixels,
                    self.total_gt_pixels,
                    empty_value=0.0,
                )
            ),
            "area_ratio": area_ratio,
            "empty_prediction_count": int(
                self.empty_prediction_count
            ),
            "empty_prediction_rate": safe_ratio(
                self.empty_prediction_count,
                self.sample_count,
                empty_value=0.0,
            ),
            "under_050_count": int(
                self.under_050_count
            ),
            "under_050_rate": safe_ratio(
                self.under_050_count,
                self.sample_count,
                empty_value=0.0,
            ),
            "over_150_count": int(
                self.over_150_count
            ),
            "over_150_rate": safe_ratio(
                self.over_150_count,
                self.sample_count,
                empty_value=0.0,
            ),
        }


def format_group_summary(summary):
    """
    Compact terminal display.

    Percent-valued metrics are multiplied by 100.
    Area ratio and FP/GT, FN/GT remain ratios.
    """
    return (
        "{group:<7s} "
        "n={n:4d} "
        "mPrec={macro_precision:6.2f} "
        "mRec={macro_recall:6.2f} "
        "mDice={macro_dice:6.2f} "
        "uPrec={micro_precision:6.2f} "
        "uRec={micro_recall:6.2f} "
        "uDice={micro_dice:6.2f} "
        "FP/GT={fp_gt:6.3f} "
        "FN/GT={fn_gt:6.3f} "
        "Area={area:6.3f} "
        "Empty={empty:6.2f} "
        "Under50={under:6.2f} "
        "Over150={over:6.2f}"
    ).format(
        group=summary["group"],
        n=summary["sample_count"],
        macro_precision=(
            100.0
            * summary["macro_precision"]
        ),
        macro_recall=(
            100.0
            * summary["macro_recall"]
        ),
        macro_dice=(
            100.0
            * summary["macro_dice"]
        ),
        micro_precision=(
            100.0
            * summary["micro_precision"]
        ),
        micro_recall=(
            100.0
            * summary["micro_recall"]
        ),
        micro_dice=(
            100.0
            * summary["micro_dice"]
        ),
        fp_gt=summary["fp_per_gt_pixel"],
        fn_gt=summary["fn_per_gt_pixel"],
        area=summary[
            "pred_to_gt_pixel_ratio"
        ],
        empty=(
            100.0
            * summary["empty_prediction_rate"]
        ),
        under=(
            100.0
            * summary["under_050_rate"]
        ),
        over=(
            100.0
            * summary["over_150_rate"]
        ),
    )

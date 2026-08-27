import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "tools" / "s0_size_eval.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tools.a2_metrics import (
    compute_binary_sample_metrics,
    summarize_numeric,
)


# The imported base S0 module parses/modifies sys.argv.
# Preserve the real command line before loading it.
_ORIGINAL_ARGV = list(sys.argv)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            "Base S0 script not found: {}".format(
                BASE_SCRIPT
            )
        )

    spec = importlib.util.spec_from_file_location(
        "s0_size_eval_base",
        str(BASE_SCRIPT),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


base = load_base_module()


_original_compute_iou = base.compute_iou
_original_create_accumulator = base.create_accumulator
_original_update_accumulator = base.update_accumulator
_original_summarize_accumulator = (
    base.summarize_accumulator
)


_CURRENT_SAMPLE_METRICS = None
_A2_PER_SAMPLE_ROWS = []
_A2_SAMPLE_INDEX = 0


def get_cli_value(flag, default=None):
    """
    Read a simple command-line option from the untouched
    command line captured before importing base S0.
    """
    argv = _ORIGINAL_ARGV

    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]

        prefix = flag + "="

        if value.startswith(prefix):
            return value[len(prefix):]

    return default


def mask_numel(mask):
    if hasattr(mask, "numel"):
        return int(mask.numel())

    return int(np.asarray(mask).size)


def safe_divide(numerator, denominator):
    denominator = float(denominator)

    if denominator <= 0.0:
        return 0.0

    return float(numerator) / denominator


def percent_or_none(value):
    if value is None:
        return None

    return 100.0 * float(value)


def create_a2_state():
    return {
        "sample_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "pred_pixels": 0,
        "gt_pixels": 0,
        "empty_predictions": 0,
        "under_050": 0,
        "over_150": 0,
        "precisions": [],
        "recalls": [],
        "dices": [],
        "ious": [],
        "area_ratios": [],
    }


def compute_iou_a2(prediction, target):
    """
    Preserve the original S0 IoU calculation and additionally
    retain the complete foreground confusion statistics.
    """
    global _CURRENT_SAMPLE_METRICS
    global _A2_SAMPLE_INDEX

    original_result = _original_compute_iou(
        prediction,
        target,
    )

    metrics = compute_binary_sample_metrics(
        prediction,
        target,
    )

    total_pixels = mask_numel(target)

    gt_area_ratio = (
        metrics["gt_pixels"] / float(total_pixels)
        if total_pixels > 0
        else 0.0
    )

    size_group = base.get_size_group(
        gt_area_ratio
    )

    _CURRENT_SAMPLE_METRICS = metrics

    _A2_PER_SAMPLE_ROWS.append({
        "sample_index": int(
            _A2_SAMPLE_INDEX
        ),
        "size_group": size_group,
        "gt_area_ratio": float(
            gt_area_ratio
        ),
        "tp_pixels": int(
            metrics["tp"]
        ),
        "fp_pixels": int(
            metrics["fp"]
        ),
        "fn_pixels": int(
            metrics["fn"]
        ),
        "pred_pixels": int(
            metrics["pred_pixels"]
        ),
        "gt_pixels": int(
            metrics["gt_pixels"]
        ),
        "precision_percent": (
            100.0 * metrics["precision"]
        ),
        "recall_percent": (
            100.0 * metrics["recall"]
        ),
        "dice_percent": (
            100.0 * metrics["dice"]
        ),
        "iou_percent": (
            100.0 * metrics["iou"]
        ),
        "pred_to_gt_area_ratio": float(
            metrics["area_ratio"]
        ),
        "empty_prediction": int(
            metrics["empty_prediction"]
        ),
        "under_050": int(
            metrics["under_050"]
        ),
        "over_150": int(
            metrics["over_150"]
        ),
    })

    _A2_SAMPLE_INDEX += 1

    return original_result


def create_accumulator_a2(*args, **kwargs):
    accumulator = _original_create_accumulator(
        *args,
        **kwargs
    )

    accumulator["_a2"] = create_a2_state()

    return accumulator


def update_accumulator_a2(*args, **kwargs):
    """
    The original main loop updates:
      1. all_nonempty accumulator
      2. the corresponding size-group accumulator

    The same latest sample statistics are therefore correctly
    copied into both accumulators.
    """
    result = _original_update_accumulator(
        *args,
        **kwargs
    )

    if args:
        accumulator = args[0]
    else:
        accumulator = kwargs.get(
            "accumulator"
        )

    if accumulator is None:
        raise RuntimeError(
            "Could not locate accumulator argument."
        )

    if _CURRENT_SAMPLE_METRICS is None:
        raise RuntimeError(
            "A2 update was called before compute_iou."
        )

    state = accumulator.setdefault(
        "_a2",
        create_a2_state(),
    )

    metrics = _CURRENT_SAMPLE_METRICS

    state["sample_count"] += 1
    state["tp"] += int(metrics["tp"])
    state["fp"] += int(metrics["fp"])
    state["fn"] += int(metrics["fn"])

    state["pred_pixels"] += int(
        metrics["pred_pixels"]
    )

    state["gt_pixels"] += int(
        metrics["gt_pixels"]
    )

    state["empty_predictions"] += int(
        metrics["empty_prediction"]
    )

    state["under_050"] += int(
        metrics["under_050"]
    )

    state["over_150"] += int(
        metrics["over_150"]
    )

    state["precisions"].append(
        float(metrics["precision"])
    )

    state["recalls"].append(
        float(metrics["recall"])
    )

    state["dices"].append(
        float(metrics["dice"])
    )

    state["ious"].append(
        float(metrics["iou"])
    )

    if math.isfinite(
        float(metrics["area_ratio"])
    ):
        state["area_ratios"].append(
            float(metrics["area_ratio"])
        )

    return result


def summarize_accumulator_a2(accumulator):
    summary = _original_summarize_accumulator(
        accumulator
    )

    state = accumulator.get(
        "_a2",
        create_a2_state(),
    )

    sample_count = int(
        state["sample_count"]
    )

    precision_stats = summarize_numeric(
        state["precisions"]
    )

    recall_stats = summarize_numeric(
        state["recalls"]
    )

    dice_stats = summarize_numeric(
        state["dices"]
    )

    iou_stats = summarize_numeric(
        state["ious"]
    )

    area_stats = summarize_numeric(
        state["area_ratios"]
    )

    micro_precision = safe_divide(
        state["tp"],
        state["tp"] + state["fp"],
    )

    micro_recall = safe_divide(
        state["tp"],
        state["tp"] + state["fn"],
    )

    micro_dice = safe_divide(
        2 * state["tp"],
        (
            2 * state["tp"]
            + state["fp"]
            + state["fn"]
        ),
    )

    summary["a2"] = {
        "sample_count": sample_count,
        "pixel_totals": {
            "tp": int(state["tp"]),
            "fp": int(state["fp"]),
            "fn": int(state["fn"]),
            "pred_pixels": int(
                state["pred_pixels"]
            ),
            "gt_pixels": int(
                state["gt_pixels"]
            ),
        },
        "macro_precision_percent": (
            percent_or_none(
                precision_stats["mean"]
            )
        ),
        "macro_recall_percent": (
            percent_or_none(
                recall_stats["mean"]
            )
        ),
        "macro_dice_percent": (
            percent_or_none(
                dice_stats["mean"]
            )
        ),
        "macro_iou_percent": (
            percent_or_none(
                iou_stats["mean"]
            )
        ),
        "micro_precision_percent": (
            100.0 * micro_precision
        ),
        "micro_recall_percent": (
            100.0 * micro_recall
        ),
        "micro_dice_percent": (
            100.0 * micro_dice
        ),
        "fp_per_gt_pixel": safe_divide(
            state["fp"],
            state["gt_pixels"],
        ),
        "fn_per_gt_pixel": safe_divide(
            state["fn"],
            state["gt_pixels"],
        ),
        "pred_to_gt_pixel_ratio": safe_divide(
            state["pred_pixels"],
            state["gt_pixels"],
        ),
        "area_ratio": {
            "count": area_stats["count"],
            "mean": area_stats["mean"],
            "median": area_stats["median"],
            "p10": area_stats["p10"],
            "p90": area_stats["p90"],
            "min": area_stats["min"],
            "max": area_stats["max"],
        },
        "empty_prediction_count": int(
            state["empty_predictions"]
        ),
        "empty_prediction_rate_percent": (
            100.0
            * safe_divide(
                state["empty_predictions"],
                sample_count,
            )
        ),
        "under_050_count": int(
            state["under_050"]
        ),
        "under_050_rate_percent": (
            100.0
            * safe_divide(
                state["under_050"],
                sample_count,
            )
        ),
        "over_150_count": int(
            state["over_150"]
        ),
        "over_150_rate_percent": (
            100.0
            * safe_divide(
                state["over_150"],
                sample_count,
            )
        ),
    }

    return summary


def write_a2_per_sample(output_dir, tag):
    path = (
        output_dir
        / "{}_a2_per_sample.csv".format(tag)
    )

    if not _A2_PER_SAMPLE_ROWS:
        print(
            "A2 warning: no per-sample rows "
            "were collected."
        )
        return path

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                _A2_PER_SAMPLE_ROWS[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            _A2_PER_SAMPLE_ROWS
        )

    return path


def write_a2_summary(output_dir, tag):
    json_path = (
        output_dir
        / "{}_size_metrics.json".format(tag)
    )

    if not json_path.exists():
        raise FileNotFoundError(
            "Base S0 JSON was not generated: {}"
            .format(json_path)
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        result = json.load(file)

    groups = result.get(
        "groups",
        {}
    )

    preferred_order = [
        "all_nonempty",
        "tiny",
        "small",
        "medium",
        "large",
    ]

    group_order = [
        group
        for group in preferred_order
        if group in groups
    ]

    for group in groups:
        if group not in group_order:
            group_order.append(group)

    rows = []

    for group in group_order:
        group_result = groups[group]
        a2 = group_result.get("a2", {})

        area_ratio = a2.get(
            "area_ratio",
            {},
        )

        rows.append({
            "group": group,
            "count": a2.get(
                "sample_count"
            ),
            "mIoU": group_result.get(
                "mIoU"
            ),
            "oIoU": group_result.get(
                "oIoU"
            ),
            "macro_precision_percent": (
                a2.get(
                    "macro_precision_percent"
                )
            ),
            "macro_recall_percent": (
                a2.get(
                    "macro_recall_percent"
                )
            ),
            "macro_dice_percent": (
                a2.get(
                    "macro_dice_percent"
                )
            ),
            "micro_precision_percent": (
                a2.get(
                    "micro_precision_percent"
                )
            ),
            "micro_recall_percent": (
                a2.get(
                    "micro_recall_percent"
                )
            ),
            "micro_dice_percent": (
                a2.get(
                    "micro_dice_percent"
                )
            ),
            "fp_per_gt_pixel": (
                a2.get(
                    "fp_per_gt_pixel"
                )
            ),
            "fn_per_gt_pixel": (
                a2.get(
                    "fn_per_gt_pixel"
                )
            ),
            "pred_to_gt_pixel_ratio": (
                a2.get(
                    "pred_to_gt_pixel_ratio"
                )
            ),
            "area_ratio_mean": (
                area_ratio.get("mean")
            ),
            "area_ratio_median": (
                area_ratio.get("median")
            ),
            "area_ratio_p10": (
                area_ratio.get("p10")
            ),
            "area_ratio_p90": (
                area_ratio.get("p90")
            ),
            "empty_prediction_rate_percent": (
                a2.get(
                    "empty_prediction_rate_percent"
                )
            ),
            "under_050_rate_percent": (
                a2.get(
                    "under_050_rate_percent"
                )
            ),
            "over_150_rate_percent": (
                a2.get(
                    "over_150_rate_percent"
                )
            ),
        })

    csv_path = (
        output_dir
        / "{}_a2_size_metrics.csv".format(tag)
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    return csv_path, rows


def format_value(value, digits=2):
    if value is None:
        return "NA"

    return (
        "{:." + str(digits) + "f}"
    ).format(float(value))


def print_a2_summary(rows):
    print(
        "\n========== A2 ERROR DIAGNOSIS =========="
    )

    for row in rows:
        print(
            "{:<12s} N={:<5} "
            "MaP={} MaR={} MaDice={} "
            "MiP={} MiR={} MiDice={} "
            "FP/GT={} FN/GT={} Area={} "
            "Empty={} Under50={} Over150={}"
            .format(
                str(row["group"]),
                str(row["count"]),
                format_value(
                    row[
                        "macro_precision_percent"
                    ]
                ),
                format_value(
                    row[
                        "macro_recall_percent"
                    ]
                ),
                format_value(
                    row[
                        "macro_dice_percent"
                    ]
                ),
                format_value(
                    row[
                        "micro_precision_percent"
                    ]
                ),
                format_value(
                    row[
                        "micro_recall_percent"
                    ]
                ),
                format_value(
                    row[
                        "micro_dice_percent"
                    ]
                ),
                format_value(
                    row["fp_per_gt_pixel"],
                    3,
                ),
                format_value(
                    row["fn_per_gt_pixel"],
                    3,
                ),
                format_value(
                    row[
                        "pred_to_gt_pixel_ratio"
                    ],
                    3,
                ),
                format_value(
                    row[
                        "empty_prediction_rate_percent"
                    ]
                ),
                format_value(
                    row[
                        "under_050_rate_percent"
                    ]
                ),
                format_value(
                    row[
                        "over_150_rate_percent"
                    ]
                ),
            )
        )

    print(
        "========================================"
    )


def main():
    # Save the A2 output location before base.main(),
    # because the base S0 parser may modify sys.argv.
    output_dir = Path(
        get_cli_value(
            "--s0-output-dir",
            "experiments/S0",
        )
    )

    tag = get_cli_value(
        "--s0-tag",
        "S0",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "A2 output directory:",
        output_dir,
    )
    print(
        "A2 tag:",
        tag,
    )

    # Monkey-patch only inside this A2 process.
    base.compute_iou = compute_iou_a2
    base.create_accumulator = create_accumulator_a2
    base.update_accumulator = update_accumulator_a2
    base.summarize_accumulator = (
        summarize_accumulator_a2
    )

    base.main()

    per_sample_path = write_a2_per_sample(
        output_dir,
        tag,
    )

    summary_path, rows = write_a2_summary(
        output_dir,
        tag,
    )

    print_a2_summary(rows)

    print("Saved:", per_sample_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()

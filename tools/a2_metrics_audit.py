import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tools.a2_metrics import (
    GroupMetricAccumulator,
    compute_binary_sample_metrics,
    format_group_summary,
)


def assert_close(actual, expected, name):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    ):
        raise AssertionError(
            "{}: actual={}, expected={}"
            .format(
                name,
                actual,
                expected,
            )
        )


def main():
    gt = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    # Perfect prediction.
    perfect = gt.clone()

    metrics = compute_binary_sample_metrics(
        perfect,
        gt,
    )

    assert metrics["tp"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0

    assert_close(
        metrics["precision"],
        1.0,
        "perfect precision",
    )

    assert_close(
        metrics["recall"],
        1.0,
        "perfect recall",
    )

    assert_close(
        metrics["dice"],
        1.0,
        "perfect dice",
    )

    assert_close(
        metrics["area_ratio"],
        1.0,
        "perfect area ratio",
    )

    print("Perfect case: PASS")

    # Underprediction: two of four GT pixels.
    under = torch.tensor([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    metrics = compute_binary_sample_metrics(
        under,
        gt,
    )

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 2

    assert_close(
        metrics["precision"],
        1.0,
        "under precision",
    )

    assert_close(
        metrics["recall"],
        0.5,
        "under recall",
    )

    assert_close(
        metrics["dice"],
        2.0 / 3.0,
        "under dice",
    )

    assert_close(
        metrics["area_ratio"],
        0.5,
        "under area ratio",
    )

    print("Underprediction case: PASS")

    # Overprediction: four GT pixels plus four FP.
    over = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    metrics = compute_binary_sample_metrics(
        over,
        gt,
    )

    assert metrics["tp"] == 4
    assert metrics["fp"] == 4
    assert metrics["fn"] == 0

    assert_close(
        metrics["precision"],
        0.5,
        "over precision",
    )

    assert_close(
        metrics["recall"],
        1.0,
        "over recall",
    )

    assert_close(
        metrics["dice"],
        2.0 / 3.0,
        "over dice",
    )

    assert_close(
        metrics["area_ratio"],
        2.0,
        "over area ratio",
    )

    assert metrics["over_150"] == 1

    print("Overprediction case: PASS")

    # Empty prediction with nonempty GT.
    empty_prediction = torch.zeros_like(gt)

    metrics = compute_binary_sample_metrics(
        empty_prediction,
        gt,
    )

    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 4

    assert_close(
        metrics["precision"],
        0.0,
        "empty prediction precision",
    )

    assert_close(
        metrics["recall"],
        0.0,
        "empty prediction recall",
    )

    assert_close(
        metrics["dice"],
        0.0,
        "empty prediction dice",
    )

    assert metrics["empty_prediction"] == 1
    assert metrics["under_050"] == 1

    print("Empty-prediction case: PASS")

    # Aggregate all four cases.
    accumulator = GroupMetricAccumulator(
        "tiny"
    )

    accumulator.update(
        perfect,
        gt,
        sample_index=0,
    )

    accumulator.update(
        under,
        gt,
        sample_index=1,
    )

    accumulator.update(
        over,
        gt,
        sample_index=2,
    )

    accumulator.update(
        empty_prediction,
        gt,
        sample_index=3,
    )

    summary = accumulator.summary()

    assert summary["sample_count"] == 4
    assert summary["empty_prediction_count"] == 1
    assert summary["over_150_count"] == 1
    assert summary["under_050_count"] == 1

    assert summary["pixel_totals"]["tp"] == 10
    assert summary["pixel_totals"]["fp"] == 4
    assert summary["pixel_totals"]["fn"] == 6

    assert_close(
        summary["micro_precision"],
        10.0 / 14.0,
        "aggregate micro precision",
    )

    assert_close(
        summary["micro_recall"],
        10.0 / 16.0,
        "aggregate micro recall",
    )

    assert_close(
        summary["micro_dice"],
        20.0 / 30.0,
        "aggregate micro dice",
    )

    print("Group aggregation: PASS")
    print()
    print(format_group_summary(summary))
    print()
    print("A2 metric audit: ALL PASS")


if __name__ == "__main__":
    main()

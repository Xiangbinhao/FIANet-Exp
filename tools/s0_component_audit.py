import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


pre_parser = argparse.ArgumentParser(add_help=False)

pre_parser.add_argument(
    "--s0-output-dir",
    default="experiments/S0/component_audit",
)

s0_args, remaining_args = pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args


from args import get_parser
from test import get_dataset, get_transform


AREA_THRESHOLDS = [1, 4, 9, 16, 25]


def to_numpy_mask(target):
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()

    target = np.asarray(target)

    while target.ndim > 2:
        target = target[0]

    return (target > 0).astype(np.uint8)


def get_component_areas(mask):
    component_count, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    return [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, component_count)
    ]


def percentile_summary(values):
    if not values:
        return {}

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def main():
    parser = get_parser()
    args = parser.parse_args(remaining_args)

    output_dir = s0_args.s0_output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset, _ = get_dataset(
        args.split,
        get_transform(args=args),
        args,
    )

    all_areas = []
    total_foreground_area = 0
    empty_samples = []

    samples_with_valid_component = {
        threshold: 0
        for threshold in AREA_THRESHOLDS
    }

    retained_components = {
        threshold: 0
        for threshold in AREA_THRESHOLDS
    }

    retained_area = {
        threshold: 0
        for threshold in AREA_THRESHOLDS
    }

    for index in range(len(dataset)):
        data = dataset[index]

        target = data[1]
        sample_name = str(data[-1])

        mask = to_numpy_mask(target)
        foreground_area = int(mask.sum())

        if foreground_area == 0:
            empty_samples.append({
                "index": index,
                "sample": sample_name,
            })

        total_foreground_area += foreground_area

        component_areas = get_component_areas(mask)
        all_areas.extend(component_areas)

        for threshold in AREA_THRESHOLDS:
            valid_areas = [
                area
                for area in component_areas
                if area >= threshold
            ]

            retained_components[threshold] += len(
                valid_areas
            )

            retained_area[threshold] += sum(
                valid_areas
            )

            if valid_areas:
                samples_with_valid_component[
                    threshold
                ] += 1

        if (index + 1) % 500 == 0:
            print(
                "Processed {}/{}".format(
                    index + 1,
                    len(dataset),
                ),
                flush=True,
            )

    raw_component_count = len(all_areas)

    thresholds = {}

    for threshold in AREA_THRESHOLDS:
        component_count = retained_components[threshold]
        area = retained_area[threshold]

        thresholds[str(threshold)] = {
            "minimum_component_area": threshold,
            "retained_components": component_count,
            "retained_component_percentage": (
                component_count
                * 100.0
                / raw_component_count
                if raw_component_count > 0
                else 0.0
            ),
            "retained_foreground_area": area,
            "retained_foreground_area_percentage": (
                area
                * 100.0
                / total_foreground_area
                if total_foreground_area > 0
                else 0.0
            ),
            "samples_with_valid_component": (
                samples_with_valid_component[threshold]
            ),
        }

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": len(dataset),
        "empty_gt_count": len(empty_samples),
        "empty_gt_samples": empty_samples,
        "raw_component_statistics": percentile_summary(
            all_areas
        ),
        "total_foreground_area": total_foreground_area,
        "threshold_audit": thresholds,
    }

    output_path = os.path.join(
        output_dir,
        "component_audit.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========== COMPONENT AUDIT ==========")
    print("Total samples:", len(dataset))
    print("Empty GT:", len(empty_samples))
    print("Raw components:", raw_component_count)

    for threshold in AREA_THRESHOLDS:
        item = thresholds[str(threshold)]

        print(
            "area >= {:>2d}: "
            "components={:>6d} ({:>6.2f}%), "
            "retained area={:>6.2f}%".format(
                threshold,
                item["retained_components"],
                item["retained_component_percentage"],
                item[
                    "retained_foreground_area_percentage"
                ],
            )
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

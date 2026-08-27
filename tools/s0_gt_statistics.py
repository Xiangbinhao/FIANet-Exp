import argparse
from pathlib import Path
import csv
import json
import os
import sys

# Make the FIANet project root importable when this script is
# launched as: python tools/<script>.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


# ------------------------------------------------------------
# Parse S0-only arguments first and remove them from sys.argv.
# This avoids conflicts with FIANet dataset modules that parse
# the shared command-line arguments during import.
# ------------------------------------------------------------
pre_parser = argparse.ArgumentParser(add_help=False)

pre_parser.add_argument(
    "--s0-output-dir",
    default="experiments/S0/gt_test",
)

s0_args, remaining_args = pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

from args import get_parser
from test import get_dataset, get_transform


SIZE_THRESHOLDS = {
    "tiny_max": 0.001,
    "small_max": 0.005,
    "medium_max": 0.020,
}


def get_size_group(area_ratio):
    if area_ratio <= SIZE_THRESHOLDS["tiny_max"]:
        return "tiny"

    if area_ratio <= SIZE_THRESHOLDS["small_max"]:
        return "small"

    if area_ratio <= SIZE_THRESHOLDS["medium_max"]:
        return "medium"

    return "large"


def connected_component_areas(mask):
    mask = mask.astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if count <= 1:
        return []

    return [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
    ]


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def to_numpy_mask(target):
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()

    target = np.asarray(target)

    while target.ndim > 2:
        target = target[0]

    return (target > 0).astype(np.uint8)


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

    rows = []
    area_ratios = []
    foreground_pixels = []
    all_component_areas = []

    group_counts = {
        "tiny": 0,
        "small": 0,
        "medium": 0,
        "large": 0,
    }

    group_component_counts = {
        "tiny": 0,
        "small": 0,
        "medium": 0,
        "large": 0,
    }

    for index in range(len(dataset)):
        data = dataset[index]

        target = data[1]
        save_prefix = data[-1]

        mask = to_numpy_mask(target)

        height, width = mask.shape
        total_pixels = height * width
        foreground = int(mask.sum())
        area_ratio = foreground / float(total_pixels)

        size_group = get_size_group(area_ratio)
        component_areas = connected_component_areas(mask)

        component_count = len(component_areas)

        if component_count > 0:
            mean_component_area = float(
                np.mean(component_areas)
            )
            min_component_area = int(
                np.min(component_areas)
            )
            max_component_area = int(
                np.max(component_areas)
            )
        else:
            mean_component_area = 0.0
            min_component_area = 0
            max_component_area = 0

        rows.append({
            "index": index,
            "sample": str(save_prefix),
            "height": height,
            "width": width,
            "foreground_pixels": foreground,
            "area_ratio": area_ratio,
            "area_percent": area_ratio * 100.0,
            "size_group": size_group,
            "component_count": component_count,
            "mean_component_area": mean_component_area,
            "min_component_area": min_component_area,
            "max_component_area": max_component_area,
        })

        area_ratios.append(area_ratio)
        foreground_pixels.append(foreground)
        all_component_areas.extend(component_areas)

        group_counts[size_group] += 1
        group_component_counts[size_group] += component_count

        if (index + 1) % 500 == 0:
            print(
                "Processed {}/{}".format(
                    index + 1,
                    len(dataset),
                ),
                flush=True,
            )

    csv_path = os.path.join(
        output_dir,
        "gt_sample_statistics.csv",
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    total_samples = len(rows)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "img_size": args.img_size,
        "total_samples": total_samples,
        "thresholds": SIZE_THRESHOLDS,
        "group_counts": group_counts,
        "group_percentages": {
            group: (
                count * 100.0 / total_samples
                if total_samples > 0
                else 0.0
            )
            for group, count in group_counts.items()
        },
        "group_component_counts": group_component_counts,
        "area_ratio_statistics": summarize(area_ratios),
        "foreground_pixel_statistics": summarize(
            foreground_pixels
        ),
        "component_area_statistics": summarize(
            all_component_areas
        ),
        "total_components": len(all_component_areas),
    }

    json_path = os.path.join(
        output_dir,
        "gt_size_summary.json",
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========== S0-A GT SUMMARY ==========")
    print("Total samples:", total_samples)

    for group in [
        "tiny",
        "small",
        "medium",
        "large",
    ]:
        print(
            "{:<8s}: {:>5d} ({:>6.2f}%), components={}".format(
                group,
                group_counts[group],
                summary["group_percentages"][group],
                group_component_counts[group],
            )
        )

    print(
        "\nArea-ratio quantiles:",
        summary["area_ratio_statistics"],
    )

    print(
        "\nComponent-area quantiles:",
        summary["component_area_statistics"],
    )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()

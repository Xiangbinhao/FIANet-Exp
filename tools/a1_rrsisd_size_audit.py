import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from args import get_parser
from train import get_dataset, get_transform


GROUPS = (
    "empty",
    "tiny",
    "small",
    "medium",
    "large",
)


def classify_ratio(
    ratio,
    tiny_max=0.001,
    small_max=0.005,
    medium_max=0.02,
):
    ratio = float(ratio)

    if ratio <= 0:
        return "empty"

    if ratio <= tiny_max:
        return "tiny"

    if ratio <= small_max:
        return "small"

    if ratio <= medium_max:
        return "medium"

    return "large"


def to_binary_tensor(mask):
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
        and mask.shape[-1] in (1, 3, 4)
    ):
        mask = mask[..., 0]

    if mask.ndim != 2:
        raise ValueError(
            "Unsupported transformed mask shape: {}"
            .format(tuple(mask.shape))
        )

    return (mask > 0).to(torch.uint8)


def to_binary_numpy(mask_result):
    if isinstance(mask_result, dict):
        for key in (
            "mask",
            "segmentation",
            "target_mask",
        ):
            if key in mask_result:
                mask_result = mask_result[key]
                break

    if torch.is_tensor(mask_result):
        array = (
            mask_result.detach()
            .cpu()
            .numpy()
        )
    else:
        array = np.asarray(mask_result)

    while (
        array.ndim > 2
        and array.shape[0] == 1
    ):
        array = array[0]

    if (
        array.ndim == 3
        and array.shape[-1] in (1, 3, 4)
    ):
        array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(
            "Unsupported original mask shape: {}"
            .format(array.shape)
        )

    return (
        array > 0
    ).astype(np.uint8)


def get_original_mask(dataset, index):
    if not hasattr(dataset, "refer"):
        raise AttributeError(
            "Dataset has no 'refer' attribute."
        )

    if not hasattr(dataset, "ref_ids"):
        raise AttributeError(
            "Dataset has no 'ref_ids' attribute."
        )

    ref_id = dataset.ref_ids[index]
    refer = dataset.refer

    try:
        refs = refer.loadRefs([ref_id])
    except Exception:
        refs = refer.loadRefs(ref_id)

    if isinstance(refs, dict):
        ref_record = refs
    elif isinstance(refs, (list, tuple)):
        if len(refs) == 0:
            raise RuntimeError(
                "No ref record for ref_id={}"
                .format(ref_id)
            )

        ref_record = refs[0]
    else:
        ref_record = refs

    mask_result = refer.getMask(ref_record)
    mask = to_binary_numpy(mask_result)

    return mask, ref_id


def resize_nearest(mask, height, width):
    tensor = torch.from_numpy(
        mask.astype(np.float32)
    )[None, None]

    resized = F.interpolate(
        tensor,
        size=(int(height), int(width)),
        mode="nearest",
    )

    return (
        resized[0, 0] > 0.5
    ).to(torch.uint8)


def binary_iou(mask_a, mask_b):
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()

    intersection = (
        mask_a & mask_b
    ).sum().item()

    union = (
        mask_a | mask_b
    ).sum().item()

    if union == 0:
        return 1.0

    return (
        float(intersection)
        / float(union)
    )


def value_summary(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset, _ = get_dataset(
        args.a1_split,
        get_transform(args=args),
        args=args,
    )

    if not hasattr(dataset, "refer"):
        raise RuntimeError(
            "This script requires dataset.refer."
        )

    if not hasattr(dataset, "ref_ids"):
        raise RuntimeError(
            "This script requires dataset.ref_ids."
        )

    output_dir = Path(
        args.a1_output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Optional smoke-test limit without adding
    # another command-line argument.
    max_samples = int(
        os.environ.get(
            "A1_MAX_SAMPLES",
            "0",
        )
    )

    if max_samples > 0:
        process_count = min(
            max_samples,
            len(dataset),
        )
    else:
        process_count = len(dataset)

    print("Dataset size:", len(dataset))
    print("Processed samples:", process_count)

    print(
        "Thresholds: "
        "Tiny<=%.4f%%, "
        "Small<=%.4f%%, "
        "Medium<=%.4f%%"
        % (
            100.0
            * args.a1_tiny_max_ratio,
            100.0
            * args.a1_small_max_ratio,
            100.0
            * args.a1_medium_max_ratio,
        )
    )

    original_counts = Counter()
    resized_counts = Counter()
    migration = Counter()

    vanished_counts = Counter()
    appeared_counts = Counter()

    retention_values = defaultdict(list)
    ratio_difference_values = defaultdict(list)

    mapping_ious = []
    low_mapping_rows = []
    rows = []

    for index in range(process_count):
        sample = dataset[index]

        if not isinstance(
            sample,
            (tuple, list),
        ):
            raise TypeError(
                "Unexpected sample type at {}: {}"
                .format(index, type(sample))
            )

        if len(sample) < 2:
            raise RuntimeError(
                "Sample {} has fewer than "
                "two fields.".format(index)
            )

        transformed_mask = to_binary_tensor(
            sample[1]
        )

        original_mask, ref_id = (
            get_original_mask(
                dataset,
                index,
            )
        )

        original_height = int(
            original_mask.shape[0]
        )
        original_width = int(
            original_mask.shape[1]
        )

        resized_height = int(
            transformed_mask.shape[0]
        )
        resized_width = int(
            transformed_mask.shape[1]
        )

        directly_resized = resize_nearest(
            original_mask,
            resized_height,
            resized_width,
        )

        mapping_iou = binary_iou(
            directly_resized,
            transformed_mask,
        )

        mapping_ious.append(
            mapping_iou
        )

        if (
            mapping_iou
            < args.a1_low_mapping_iou
        ):
            low_mapping_rows.append({
                "index": int(index),
                "ref_id": str(ref_id),
                "mapping_iou": float(
                    mapping_iou
                ),
            })

        original_fg = int(
            original_mask.sum()
        )

        resized_fg = int(
            transformed_mask.sum().item()
        )

        original_total = int(
            original_height
            * original_width
        )

        resized_total = int(
            resized_height
            * resized_width
        )

        original_ratio = (
            original_fg / original_total
            if original_total > 0
            else 0.0
        )

        resized_ratio = (
            resized_fg / resized_total
            if resized_total > 0
            else 0.0
        )

        original_group = classify_ratio(
            original_ratio,
            args.a1_tiny_max_ratio,
            args.a1_small_max_ratio,
            args.a1_medium_max_ratio,
        )

        resized_group = classify_ratio(
            resized_ratio,
            args.a1_tiny_max_ratio,
            args.a1_small_max_ratio,
            args.a1_medium_max_ratio,
        )

        original_counts[
            original_group
        ] += 1

        resized_counts[
            resized_group
        ] += 1

        migration[
            (
                original_group,
                resized_group,
            )
        ] += 1

        vanished = (
            original_fg > 0
            and resized_fg == 0
        )

        appeared = (
            original_fg == 0
            and resized_fg > 0
        )

        if vanished:
            vanished_counts[
                original_group
            ] += 1

        if appeared:
            appeared_counts[
                resized_group
            ] += 1

        expected_resized_fg = (
            original_fg
            * resized_total
            / original_total
            if original_total > 0
            else 0.0
        )

        if (
            original_fg > 0
            and expected_resized_fg > 0
        ):
            normalized_retention = (
                resized_fg
                / expected_resized_fg
            )

            retention_values[
                original_group
            ].append(
                normalized_retention
            )
        else:
            normalized_retention = None

        ratio_difference = (
            resized_ratio
            - original_ratio
        )

        ratio_difference_values[
            original_group
        ].append(
            ratio_difference
        )

        rows.append({
            "index": int(index),
            "ref_id": str(ref_id),
            "mapping_iou": float(
                mapping_iou
            ),
            "original_height": original_height,
            "original_width": original_width,
            "original_fg_pixels": original_fg,
            "original_total_pixels": original_total,
            "original_fg_ratio": float(
                original_ratio
            ),
            "original_group": original_group,
            "resized_height": resized_height,
            "resized_width": resized_width,
            "resized_fg_pixels": resized_fg,
            "resized_total_pixels": resized_total,
            "resized_fg_ratio": float(
                resized_ratio
            ),
            "resized_group": resized_group,
            "ratio_difference": float(
                ratio_difference
            ),
            "expected_resized_fg_pixels": float(
                expected_resized_fg
            ),
            "normalized_area_retention": (
                None
                if normalized_retention is None
                else float(
                    normalized_retention
                )
            ),
            "vanished_after_transform": int(
                vanished
            ),
            "appeared_after_transform": int(
                appeared
            ),
        })

        if (
            args.a1_print_freq > 0
            and (
                (index + 1)
                % args.a1_print_freq
                == 0
                or index + 1
                == process_count
            )
        ):
            print(
                "Processed {}/{} samples"
                .format(
                    index + 1,
                    process_count,
                )
            )

    csv_path = (
        output_dir
        / (
            args.a1_tag
            + "_original_vs_resized.csv"
        )
    )

    summary_path = (
        output_dir
        / (
            args.a1_tag
            + "_summary.json"
        )
    )

    with open(
        str(csv_path),
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

    changed_count = sum(
        count
        for (
            source_group,
            target_group,
        ), count in migration.items()
        if source_group != target_group
    )

    migration_matrix = {
        source_group: {
            target_group: int(
                migration[
                    (
                        source_group,
                        target_group,
                    )
                ]
            )
            for target_group in GROUPS
        }
        for source_group in GROUPS
    }

    summary = {
        "dataset_size": int(
            len(dataset)
        ),
        "processed_samples": int(
            process_count
        ),
        "thresholds": {
            "tiny_max_ratio": float(
                args.a1_tiny_max_ratio
            ),
            "small_max_ratio": float(
                args.a1_small_max_ratio
            ),
            "medium_max_ratio": float(
                args.a1_medium_max_ratio
            ),
        },
        "original_group_counts": {
            group: int(
                original_counts[group]
            )
            for group in GROUPS
        },
        "resized_group_counts": {
            group: int(
                resized_counts[group]
            )
            for group in GROUPS
        },
        "migration_matrix": migration_matrix,
        "changed_group_count": int(
            changed_count
        ),
        "changed_group_rate": (
            changed_count
            / process_count
            if process_count > 0
            else 0.0
        ),
        "vanished_after_transform": {
            "total": int(sum(
                vanished_counts.values()
            )),
            "by_original_group": {
                group: int(
                    vanished_counts[group]
                )
                for group in GROUPS
            },
        },
        "appeared_after_transform": {
            "total": int(sum(
                appeared_counts.values()
            )),
            "by_resized_group": {
                group: int(
                    appeared_counts[group]
                )
                for group in GROUPS
            },
        },
        "mapping_iou": value_summary(
            mapping_ious
        ),
        "low_mapping_iou_threshold": float(
            args.a1_low_mapping_iou
        ),
        "low_mapping_sample_count": int(
            len(low_mapping_rows)
        ),
        "low_mapping_samples": (
            low_mapping_rows[:100]
        ),
        "normalized_area_retention": {
            group: value_summary(
                retention_values[group]
            )
            for group in GROUPS
        },
        "foreground_ratio_difference": {
            group: value_summary(
                ratio_difference_values[group]
            )
            for group in GROUPS
        },
    }

    with open(
        str(summary_path),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n========== A1 ORIGINAL VS RESIZED =========="
    )

    print(
        "Processed samples:",
        process_count,
    )

    print("\nOriginal GT groups:")

    for group in GROUPS:
        print(
            "  {:<8s}: {:>5d}".format(
                group,
                original_counts[group],
            )
        )

    print("\nResized GT groups:")

    for group in GROUPS:
        print(
            "  {:<8s}: {:>5d}".format(
                group,
                resized_counts[group],
            )
        )

    print("\nMigration matrix:")

    print(
        "{:>10s}".format("original"),
        end="",
    )

    for group in GROUPS:
        print(
            "{:>10s}".format(group),
            end="",
        )

    print()

    for source_group in GROUPS:
        print(
            "{:>10s}".format(
                source_group
            ),
            end="",
        )

        for target_group in GROUPS:
            print(
                "{:>10d}".format(
                    migration[
                        (
                            source_group,
                            target_group,
                        )
                    ]
                ),
                end="",
            )

        print()

    print(
        "\nChanged size group: {} / {} "
        "({:.2f}%)".format(
            changed_count,
            process_count,
            (
                100.0
                * changed_count
                / process_count
                if process_count > 0
                else 0.0
            ),
        )
    )

    print(
        "Original nonempty -> resized empty:",
        sum(vanished_counts.values()),
    )

    mapping_summary = value_summary(
        mapping_ious
    )

    print(
        "Low mapping-IoU samples (<{:.2f}): {}"
        .format(
            args.a1_low_mapping_iou,
            len(low_mapping_rows),
        )
    )

    print(
        "Mapping IoU: mean={:.4f}, "
        "median={:.4f}, min={:.4f}"
        .format(
            mapping_summary["mean"],
            mapping_summary["median"],
            mapping_summary["min"],
        )
    )

    print("\nNormalized area retention:")

    for group in GROUPS:
        values = value_summary(
            retention_values[group]
        )

        if values["count"] == 0:
            print(
                "  {:<8s}: no samples"
                .format(group)
            )
            continue

        print(
            "  {:<8s}: n={} mean={:.4f} "
            "median={:.4f} p10={:.4f} "
            "p90={:.4f}".format(
                group,
                values["count"],
                values["mean"],
                values["median"],
                values["p10"],
                values["p90"],
            )
        )

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", summary_path)

    print(
        "============================================"
    )


if __name__ == "__main__":
    main()

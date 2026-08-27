import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Parse S0-only arguments before importing FIANet modules.
# ------------------------------------------------------------
pre_parser = argparse.ArgumentParser(add_help=False)

pre_parser.add_argument(
    "--s0-output-dir",
    required=True,
)

pre_parser.add_argument(
    "--s0-tag",
    required=True,
)

pre_parser.add_argument(
    "--s0-max-samples",
    type=int,
    default=0,
)

pre_parser.add_argument(
    "--s0-component-thresholds",
    type=int,
    nargs="+",
    default=[4, 9, 16],
)

s0_args, remaining_args = pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args


from args import get_parser
from lib import segmentation
from test import get_dataset, get_transform


SIZE_THRESHOLDS = {
    "tiny_max": 0.001,
    "small_max": 0.005,
    "medium_max": 0.020,
}

IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def to_scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def get_size_group(area_ratio):
    if area_ratio <= SIZE_THRESHOLDS["tiny_max"]:
        return "tiny"

    if area_ratio <= SIZE_THRESHOLDS["small_max"]:
        return "small"

    if area_ratio <= SIZE_THRESHOLDS["medium_max"]:
        return "medium"

    return "large"


def collect_tensor_outputs(output):
    tensors = []

    if torch.is_tensor(output):
        tensors.append(output)

    elif isinstance(output, dict):
        preferred_keys = [
            "final_logits",
            "logits",
            "out",
            "prediction",
            "pred",
        ]

        for key in preferred_keys:
            if key in output and torch.is_tensor(output[key]):
                return [output[key]]

        for value in output.values():
            tensors.extend(
                collect_tensor_outputs(value)
            )

    elif isinstance(output, (tuple, list)):
        for value in output:
            tensors.extend(
                collect_tensor_outputs(value)
            )

    return tensors


def extract_final_logits(output):
    candidates = collect_tensor_outputs(output)

    candidates = [
        tensor
        for tensor in candidates
        if tensor.ndim >= 3
    ]

    if not candidates:
        raise TypeError(
            "No segmentation tensor found in model output: {}".format(
                type(output)
            )
        )

    # Final segmentation logits normally have the largest spatial size.
    return max(
        candidates,
        key=lambda tensor: (
            int(tensor.shape[-2])
            * int(tensor.shape[-1])
        ),
    )


def target_to_numpy(target):
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    target = np.asarray(target)

    while target.ndim > 2:
        target = target[0]

    return (target > 0).astype(np.uint8)


def logits_to_prediction(logits, target_shape):
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)

    if logits.ndim != 4:
        raise ValueError(
            "Expected four-dimensional logits, got {}".format(
                tuple(logits.shape)
            )
        )

    if tuple(logits.shape[-2:]) != tuple(target_shape):
        logits = F.interpolate(
            logits,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )

    if logits.shape[1] == 1:
        prediction = (
            torch.sigmoid(logits[:, 0]) >= 0.5
        ).long()

    else:
        prediction = logits.argmax(dim=1)

    return (
        prediction[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def compute_iou(prediction, target):
    intersection = int(
        np.logical_and(
            prediction == 1,
            target == 1,
        ).sum()
    )

    union = int(
        np.logical_or(
            prediction == 1,
            target == 1,
        ).sum()
    )

    iou = (
        intersection / float(union)
        if union > 0
        else None
    )

    return intersection, union, iou


def compute_component_hits(
    prediction,
    target,
    area_thresholds,
):
    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            target.astype(np.uint8),
            connectivity=8,
        )
    )

    result = {
        str(threshold): {
            "total": 0,
            "hit_10": 0,
            "hit_50": 0,
        }
        for threshold in area_thresholds
    }

    for component_id in range(1, component_count):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        component_mask = labels == component_id

        covered_pixels = int(
            (prediction[component_mask] == 1).sum()
        )

        coverage = (
            covered_pixels / float(area)
            if area > 0
            else 0.0
        )

        for threshold in area_thresholds:
            if area < threshold:
                continue

            item = result[str(threshold)]
            item["total"] += 1

            if coverage >= 0.1:
                item["hit_10"] += 1

            if coverage >= 0.5:
                item["hit_50"] += 1

    return result


def create_accumulator(area_thresholds):
    return {
        "count": 0,
        "iou_sum": 0.0,
        "intersection": 0,
        "union": 0,
        "foreground_tp": 0,
        "foreground_gt": 0,
        "empty_predictions": 0,
        "precision_counts": {
            str(threshold): 0
            for threshold in IOU_THRESHOLDS
        },
        "components": {
            str(threshold): {
                "total": 0,
                "hit_10": 0,
                "hit_50": 0,
            }
            for threshold in area_thresholds
        },
    }


def update_accumulator(
    accumulator,
    prediction,
    target,
    intersection,
    union,
    iou,
    component_result,
):
    accumulator["count"] += 1
    accumulator["iou_sum"] += iou
    accumulator["intersection"] += intersection
    accumulator["union"] += union

    foreground_tp = int(
        np.logical_and(
            prediction == 1,
            target == 1,
        ).sum()
    )

    foreground_gt = int(
        (target == 1).sum()
    )

    accumulator["foreground_tp"] += foreground_tp
    accumulator["foreground_gt"] += foreground_gt

    if int((prediction == 1).sum()) == 0:
        accumulator["empty_predictions"] += 1

    for threshold in IOU_THRESHOLDS:
        if iou >= threshold:
            accumulator["precision_counts"][
                str(threshold)
            ] += 1

    for threshold, item in component_result.items():
        destination = accumulator["components"][
            threshold
        ]

        destination["total"] += item["total"]
        destination["hit_10"] += item["hit_10"]
        destination["hit_50"] += item["hit_50"]


def summarize_accumulator(accumulator):
    count = accumulator["count"]

    if count == 0:
        return {
            "count": 0,
            "mIoU": None,
            "oIoU": None,
            "foreground_recall": None,
            "empty_prediction_rate": None,
            "precision": {
                str(threshold): None
                for threshold in IOU_THRESHOLDS
            },
            "components": {
                str(threshold): {
                    "total": 0,
                    "hit_10": 0,
                    "hit_50": 0,
                    "recall_10": None,
                    "recall_50": None,
                }
                for threshold in accumulator["components"]
            },
        }

    union = accumulator["union"]
    foreground_gt = accumulator["foreground_gt"]

    component_summary = {}

    for threshold, item in accumulator[
        "components"
    ].items():
        total = item["total"]

        component_summary[threshold] = {
            "total": total,
            "hit_10": item["hit_10"],
            "hit_50": item["hit_50"],
            "recall_10": (
                item["hit_10"] * 100.0 / total
                if total > 0
                else None
            ),
            "recall_50": (
                item["hit_50"] * 100.0 / total
                if total > 0
                else None
            ),
        }

    return {
        "count": count,
        "mIoU": (
            accumulator["iou_sum"]
            * 100.0
            / count
        ),
        "oIoU": (
            accumulator["intersection"]
            * 100.0
            / union
            if union > 0
            else None
        ),
        "foreground_recall": (
            accumulator["foreground_tp"]
            * 100.0
            / foreground_gt
            if foreground_gt > 0
            else None
        ),
        "empty_prediction_rate": (
            accumulator["empty_predictions"]
            * 100.0
            / count
        ),
        "precision": {
            str(threshold): (
                accumulator["precision_counts"][
                    str(threshold)
                ]
                * 100.0
                / count
            )
            for threshold in IOU_THRESHOLDS
        },
        "components": component_summary,
    }


def normalize_state_dict(state_dict):
    keys = list(state_dict.keys())

    if keys and all(
        key.startswith("module.")
        for key in keys
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def format_sample_name(save_prefix):
    if isinstance(save_prefix, (list, tuple)):
        if len(save_prefix) == 1:
            return str(save_prefix[0])

        return "|".join(
            str(value)
            for value in save_prefix
        )

    return str(save_prefix)


def main():
    parser = get_parser()
    args = parser.parse_args(remaining_args)

    output_dir = Path(s0_args.s0_output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    area_thresholds = sorted(
        set(
            int(value)
            for value in s0_args.s0_component_thresholds
        )
    )

    if any(
        threshold <= 0
        for threshold in area_thresholds
    ):
        raise ValueError(
            "Component thresholds must be positive."
        )

    device = torch.device(args.device)

    dataset, _ = get_dataset(
        args.split,
        get_transform(args=args),
        args,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=getattr(
            args,
            "pin_mem",
            False,
        ),
    )

    print("Building model...", flush=True)

    model = segmentation.__dict__[args.model](
        pretrained="",
        args=args,
    )

    print(
        "Loading checkpoint:",
        args.resume,
        flush=True,
    )

    checkpoint = torch.load(
        args.resume,
        map_location="cpu",
    )

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    state_dict = normalize_state_dict(
        state_dict
    )

    # Strict loading prevents accidentally evaluating the wrong model.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "Checkpoint loaded with strict=True.",
        flush=True,
    )

    model = model.to(device)
    model.eval()

    group_names = [
        "all_nonempty",
        "tiny",
        "small",
        "medium",
        "large",
    ]

    accumulators = {
        group: create_accumulator(
            area_thresholds
        )
        for group in group_names
    }

    empty_gt = {
        "count": 0,
        "empty_prediction_count": 0,
        "predicted_foreground_pixels": 0,
        "samples": [],
    }

    rows = []
    start_time = time.time()

    max_samples = len(dataset)

    if s0_args.s0_max_samples > 0:
        max_samples = min(
            max_samples,
            s0_args.s0_max_samples,
        )

    with torch.no_grad():
        for index, data in enumerate(data_loader):
            if index >= max_samples:
                break

            if (
                not isinstance(data, (tuple, list))
                or len(data) != 7
            ):
                raise RuntimeError(
                    "Expected a 7-item FIANet batch; "
                    "got type={} length={}".format(
                        type(data),
                        (
                            len(data)
                            if isinstance(
                                data,
                                (tuple, list),
                            )
                            else "N/A"
                        ),
                    )
                )

            (
                image,
                target,
                sentences,
                attentions,
                target_masks,
                position_masks,
                save_prefix,
            ) = data

            target_numpy = target_to_numpy(
                target
            )

            foreground_pixels = int(
                target_numpy.sum()
            )

            area_ratio = (
                foreground_pixels
                / float(target_numpy.size)
            )

            image = image.to(
                device,
                non_blocking=True,
            )

            sentences = (
                sentences.to(
                    device,
                    non_blocking=True,
                )
                .squeeze(1)
            )

            attentions = (
                attentions.to(
                    device,
                    non_blocking=True,
                )
                .squeeze(1)
            )

            target_masks = (
                target_masks.to(
                    device,
                    non_blocking=True,
                )
                .squeeze(1)
            )

            position_masks = (
                position_masks.to(
                    device,
                    non_blocking=True,
                )
                .squeeze(1)
            )

            output = model(
                image,
                sentences,
                attentions,
                target_masks,
                position_masks,
            )

            logits = extract_final_logits(
                output
            )

            prediction = logits_to_prediction(
                logits,
                target_numpy.shape,
            )

            sample_name = format_sample_name(
                save_prefix
            )

            predicted_foreground = int(
                (prediction == 1).sum()
            )

            # Empty GT is excluded from size groups.
            if foreground_pixels == 0:
                empty_gt["count"] += 1

                empty_gt[
                    "predicted_foreground_pixels"
                ] += predicted_foreground

                if predicted_foreground == 0:
                    empty_gt[
                        "empty_prediction_count"
                    ] += 1

                empty_gt["samples"].append({
                    "index": index,
                    "sample": sample_name,
                    "predicted_foreground_pixels": (
                        predicted_foreground
                    ),
                })

                rows.append({
                    "index": index,
                    "sample": sample_name,
                    "size_group": "empty_gt",
                    "foreground_pixels": 0,
                    "area_ratio": 0.0,
                    "iou": "",
                    "foreground_recall": "",
                    "empty_prediction": int(
                        predicted_foreground == 0
                    ),
                    "predicted_foreground_pixels": (
                        predicted_foreground
                    ),
                })

                continue

            size_group = get_size_group(
                area_ratio
            )

            intersection, union, iou = (
                compute_iou(
                    prediction,
                    target_numpy,
                )
            )

            component_result = (
                compute_component_hits(
                    prediction,
                    target_numpy,
                    area_thresholds,
                )
            )

            update_accumulator(
                accumulators["all_nonempty"],
                prediction,
                target_numpy,
                intersection,
                union,
                iou,
                component_result,
            )

            update_accumulator(
                accumulators[size_group],
                prediction,
                target_numpy,
                intersection,
                union,
                iou,
                component_result,
            )

            foreground_recall = (
                intersection
                / float(foreground_pixels)
            )

            row = {
                "index": index,
                "sample": sample_name,
                "size_group": size_group,
                "foreground_pixels": (
                    foreground_pixels
                ),
                "area_ratio": area_ratio,
                "iou": iou,
                "foreground_recall": (
                    foreground_recall
                ),
                "empty_prediction": int(
                    predicted_foreground == 0
                ),
                "predicted_foreground_pixels": (
                    predicted_foreground
                ),
            }

            for threshold in area_thresholds:
                item = component_result[
                    str(threshold)
                ]

                row[
                    "components_ge_{}".format(
                        threshold
                    )
                ] = item["total"]

                row[
                    "component_hits10_ge_{}".format(
                        threshold
                    )
                ] = item["hit_10"]

                row[
                    "component_hits50_ge_{}".format(
                        threshold
                    )
                ] = item["hit_50"]

            rows.append(row)

            if (index + 1) % 100 == 0:
                elapsed = time.time() - start_time

                print(
                    "Evaluated {}/{} samples, "
                    "elapsed {:.1f} min".format(
                        index + 1,
                        max_samples,
                        elapsed / 60.0,
                    ),
                    flush=True,
                )

    summaries = {
        group: summarize_accumulator(
            accumulator
        )
        for group, accumulator
        in accumulators.items()
    }

    elapsed = time.time() - start_time

    checkpoint_metadata = {}

    if isinstance(checkpoint, dict):
        for key in [
            "epoch",
            "best_oIoU",
            "best_mIoU",
        ]:
            if key in checkpoint:
                checkpoint_metadata[key] = (
                    to_scalar(checkpoint[key])
                )

    result = {
        "tag": s0_args.s0_tag,
        "checkpoint": str(args.resume),
        "checkpoint_metadata": (
            checkpoint_metadata
        ),
        "dataset": args.dataset,
        "split": args.split,
        "img_size": args.img_size,
        "evaluated_samples": max_samples,
        "size_thresholds": SIZE_THRESHOLDS,
        "component_area_thresholds": (
            area_thresholds
        ),
        "component_coverage_thresholds": [
            0.1,
            0.5,
        ],
        "elapsed_seconds": elapsed,
        "groups": summaries,
        "empty_gt": empty_gt,
    }

    per_sample_path = (
        output_dir
        / "{}_per_sample.csv".format(
            s0_args.s0_tag
        )
    )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with per_sample_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        output_dir
        / "{}_size_metrics.json".format(
            s0_args.s0_tag
        )
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary_rows = []

    for group in group_names:
        item = summaries[group]

        row = {
            "group": group,
            "count": item["count"],
            "mIoU": item["mIoU"],
            "oIoU": item["oIoU"],
            "Pr@0.5": item["precision"]["0.5"],
            "Pr@0.6": item["precision"]["0.6"],
            "Pr@0.7": item["precision"]["0.7"],
            "Pr@0.8": item["precision"]["0.8"],
            "Pr@0.9": item["precision"]["0.9"],
            "foreground_recall": (
                item["foreground_recall"]
            ),
            "empty_prediction_rate": (
                item["empty_prediction_rate"]
            ),
        }

        for threshold in area_thresholds:
            component_item = (
                item["components"][
                    str(threshold)
                ]
            )

            row[
                "components_ge_{}".format(
                    threshold
                )
            ] = component_item["total"]

            row[
                "component_recall10_ge_{}".format(
                    threshold
                )
            ] = component_item["recall_10"]

            row[
                "component_recall50_ge_{}".format(
                    threshold
                )
            ] = component_item["recall_50"]

        summary_rows.append(row)

    summary_path = (
        output_dir
        / "{}_size_metrics.csv".format(
            s0_args.s0_tag
        )
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        "\n========== S0 SIZE EVALUATION =========="
    )

    for row in summary_rows:
        print(
            "{:<12s} N={:<5d} "
            "mIoU={} oIoU={} "
            "Pr@0.5={} FG-Recall={} "
            "Empty={}".format(
                row["group"],
                row["count"],
                (
                    "{:.2f}".format(
                        row["mIoU"]
                    )
                    if row["mIoU"] is not None
                    else "NA"
                ),
                (
                    "{:.2f}".format(
                        row["oIoU"]
                    )
                    if row["oIoU"] is not None
                    else "NA"
                ),
                (
                    "{:.2f}".format(
                        row["Pr@0.5"]
                    )
                    if row["Pr@0.5"] is not None
                    else "NA"
                ),
                (
                    "{:.2f}".format(
                        row["foreground_recall"]
                    )
                    if row[
                        "foreground_recall"
                    ] is not None
                    else "NA"
                ),
                (
                    "{:.2f}".format(
                        row["empty_prediction_rate"]
                    )
                    if row[
                        "empty_prediction_rate"
                    ] is not None
                    else "NA"
                ),
            )
        )

    print(
        "Empty-GT samples: {}, "
        "predicted-empty: {}".format(
            empty_gt["count"],
            empty_gt[
                "empty_prediction_count"
            ],
        )
    )

    print("Saved:", per_sample_path)
    print("Saved:", summary_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()

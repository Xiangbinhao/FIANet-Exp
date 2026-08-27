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


GROUP_NAMES = (
    "empty",
    "tiny",
    "small",
    "medium",
    "large",
)


def classify_ratio(
    ratio,
    tiny_max_ratio=0.001,
    small_max_ratio=0.005,
    medium_max_ratio=0.02,
):
    ratio = float(ratio)

    if ratio <= 0.0:
        return "empty"

    if ratio <= tiny_max_ratio:
        return "tiny"

    if ratio <= small_max_ratio:
        return "small"

    if ratio <= medium_max_ratio:
        return "medium"

    return "large"


def tensor_to_binary_mask(target):
    if not torch.is_tensor(target):
        target = torch.as_tensor(target)

    target = target.detach().cpu()

    while target.ndim > 2 and target.shape[0] == 1:
        target = target.squeeze(0)

    if target.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional target mask, "
            "received shape {}".format(
                tuple(target.shape)
            )
        )

    # Test masks should be binary. Using > 0 also
    # supports masks stored as 0/255.
    return (target > 0).to(torch.uint8)


def find_wrapped_objects(root):
    """
    Traverse common dataset wrappers without recursively
    exploring arbitrary Python objects.
    """
    queue = [root]
    visited = set()
    objects = []

    wrapper_names = (
        "dataset",
        "datasets",
        "refer",
        "ref_dataset",
        "base_dataset",
        "coco",
    )

    while queue:
        obj = queue.pop(0)

        if obj is None:
            continue

        object_id = id(obj)

        if object_id in visited:
            continue

        visited.add(object_id)
        objects.append(obj)

        for name in wrapper_names:
            if not hasattr(obj, name):
                continue

            value = getattr(obj, name)

            if isinstance(value, (list, tuple)):
                for item in value:
                    if not isinstance(
                        item,
                        (
                            str,
                            bytes,
                            int,
                            float,
                            bool,
                        ),
                    ):
                        queue.append(item)
            else:
                if not isinstance(
                    value,
                    (
                        str,
                        bytes,
                        int,
                        float,
                        bool,
                    ),
                ):
                    queue.append(value)

    return objects


def find_coco_object(objects):
    for obj in objects:
        if (
            hasattr(obj, "annToMask")
            and hasattr(obj, "anns")
        ):
            return obj

        if hasattr(obj, "coco"):
            coco = getattr(obj, "coco")

            if (
                hasattr(coco, "annToMask")
                and hasattr(coco, "anns")
            ):
                return coco

    return None


def normalize_ann_ids(value):
    if value is None:
        return []

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, dict):
        for key in (
            "ann_id",
            "annotation_id",
            "annId",
        ):
            if key in value:
                return normalize_ann_ids(
                    value[key]
                )

        for key in (
            "ann_ids",
            "annotation_ids",
            "annIds",
        ):
            if key in value:
                return normalize_ann_ids(
                    value[key]
                )

        return []

    if isinstance(value, (list, tuple)):
        output = []

        for item in value:
            output.extend(
                normalize_ann_ids(item)
            )

        return output

    return []


def get_mapping_records(objects):
    """
    Collect dictionaries that may map ref_id to a ref record.
    """
    mappings = []

    for obj in objects:
        for name in (
            "Refs",
            "refs",
            "ref_dict",
            "ref_id_to_ref",
        ):
            if not hasattr(obj, name):
                continue

            value = getattr(obj, name)

            if isinstance(value, dict):
                mappings.append(value)

    return mappings


def resolve_record_from_mapping(
    key,
    mappings,
):
    candidates = []

    possible_keys = [key]

    try:
        possible_keys.append(int(key))
    except Exception:
        pass

    possible_keys.append(str(key))

    for mapping in mappings:
        for possible_key in possible_keys:
            if possible_key in mapping:
                candidates.append(
                    mapping[possible_key]
                )

    return candidates


def collect_candidate_ann_sets(
    dataset,
    objects,
    coco,
    index,
):
    """
    Collect candidate annotation IDs from common referring-
    segmentation dataset layouts.

    Candidate masks are later checked against the transformed
    target, so incorrect integer IDs are not accepted blindly.
    """
    candidate_sets = []
    mappings = get_mapping_records(objects)
    dataset_length = len(dataset)

    sequence_names = (
        "refs",
        "ref_ids",
        "ref_id_list",
        "ids",
        "index_list",
        "samples",
        "items",
    )

    for obj in objects:
        for name in sequence_names:
            if not hasattr(obj, name):
                continue

            sequence = getattr(obj, name)

            if not isinstance(
                sequence,
                (list, tuple),
            ):
                continue

            if len(sequence) != dataset_length:
                continue

            item = sequence[index]

            direct_ann_ids = normalize_ann_ids(
                item
            )

            direct_ann_ids = [
                ann_id
                for ann_id in direct_ann_ids
                if ann_id in coco.anns
            ]

            if direct_ann_ids:
                candidate_sets.append(
                    tuple(sorted(set(
                        direct_ann_ids
                    )))
                )

            scalar_keys = []

            if isinstance(item, np.generic):
                item = item.item()

            if isinstance(item, (int, str)):
                scalar_keys.append(item)

            elif isinstance(item, (tuple, list)):
                for element in item:
                    if isinstance(
                        element,
                        np.generic,
                    ):
                        element = element.item()

                    if isinstance(
                        element,
                        (int, str),
                    ):
                        scalar_keys.append(
                            element
                        )

                    elif isinstance(
                        element,
                        dict,
                    ):
                        ann_ids = (
                            normalize_ann_ids(
                                element
                            )
                        )

                        ann_ids = [
                            ann_id
                            for ann_id in ann_ids
                            if ann_id in coco.anns
                        ]

                        if ann_ids:
                            candidate_sets.append(
                                tuple(sorted(set(
                                    ann_ids
                                )))
                            )

            for scalar_key in scalar_keys:
                records = (
                    resolve_record_from_mapping(
                        scalar_key,
                        mappings,
                    )
                )

                for record in records:
                    ann_ids = normalize_ann_ids(
                        record
                    )

                    ann_ids = [
                        ann_id
                        for ann_id in ann_ids
                        if ann_id in coco.anns
                    ]

                    if ann_ids:
                        candidate_sets.append(
                            tuple(sorted(set(
                                ann_ids
                            )))
                        )

                # Last-resort candidate: the scalar itself
                # may already be an annotation ID.
                try:
                    integer_key = int(
                        scalar_key
                    )
                except Exception:
                    integer_key = None

                if (
                    integer_key is not None
                    and integer_key in coco.anns
                ):
                    candidate_sets.append(
                        (integer_key,)
                    )

    unique = []
    seen = set()

    for candidate in candidate_sets:
        if not candidate:
            continue

        if candidate in seen:
            continue

        seen.add(candidate)
        unique.append(candidate)

    return unique


def build_original_mask(
    coco,
    ann_ids,
):
    masks = []

    for ann_id in ann_ids:
        annotation = coco.anns[int(ann_id)]
        mask = coco.annToMask(annotation)

        mask = np.asarray(
            mask,
            dtype=np.uint8,
        )

        masks.append(mask)

    if not masks:
        raise ValueError(
            "No annotation masks were generated."
        )

    height, width = masks[0].shape

    union_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for mask in masks:
        if mask.shape != (height, width):
            raise ValueError(
                "Annotations in one sample have "
                "different image sizes."
            )

        union_mask = np.maximum(
            union_mask,
            (mask > 0).astype(np.uint8),
        )

    return union_mask


def resize_binary_mask(
    mask,
    output_height,
    output_width,
):
    tensor = torch.from_numpy(
        mask.astype(np.float32)
    )[None, None]

    resized = F.interpolate(
        tensor,
        size=(
            int(output_height),
            int(output_width),
        ),
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

    return float(intersection) / float(union)


def choose_best_candidate(
    coco,
    candidate_sets,
    transformed_mask,
):
    output_height = int(
        transformed_mask.shape[0]
    )
    output_width = int(
        transformed_mask.shape[1]
    )

    best = None

    for ann_ids in candidate_sets:
        try:
            original_mask = (
                build_original_mask(
                    coco,
                    ann_ids,
                )
            )

            directly_resized = (
                resize_binary_mask(
                    original_mask,
                    output_height,
                    output_width,
                )
            )

            mapping_iou = binary_iou(
                directly_resized,
                transformed_mask,
            )

        except Exception:
            continue

        candidate = {
            "ann_ids": ann_ids,
            "original_mask": original_mask,
            "mapping_iou": mapping_iou,
        }

        if (
            best is None
            or mapping_iou
            > best["mapping_iou"]
        ):
            best = candidate

    return best



def normalize_original_mask_value(
    value,
    args,
):
    """
    Convert a mask object/path into a binary NumPy mask.
    Supports:
      - NumPy arrays
      - PyTorch tensors
      - PIL images
      - dictionaries containing a mask/path
      - mask file paths
    """
    from PIL import Image

    if value is None:
        return None

    if torch.is_tensor(value):
        array = (
            value.detach()
            .cpu()
            .numpy()
        )

    elif isinstance(value, np.ndarray):
        array = value

    elif isinstance(value, Image.Image):
        array = np.asarray(value)

    elif isinstance(value, dict):
        for key in (
            "mask",
            "segmentation_mask",
            "target_mask",
            "binary_mask",
        ):
            if key in value:
                mask = normalize_original_mask_value(
                    value[key],
                    args,
                )

                if mask is not None:
                    return mask

        for key in (
            "mask_path",
            "path",
            "file_name",
            "filename",
        ):
            if key in value:
                mask = normalize_original_mask_value(
                    value[key],
                    args,
                )

                if mask is not None:
                    return mask

        return None

    elif isinstance(value, (list, tuple)):
        # A numeric nested list can itself be a mask.
        try:
            array = np.asarray(value)

            if (
                array.ndim >= 2
                and np.issubdtype(
                    array.dtype,
                    np.number,
                )
            ):
                pass
            else:
                raise ValueError

        except Exception:
            for item in value:
                mask = normalize_original_mask_value(
                    item,
                    args,
                )

                if mask is not None:
                    return mask

            return None

    elif isinstance(value, (str, Path)):
        raw_path = Path(str(value))

        candidates = [
            raw_path,
            PROJECT_ROOT / raw_path,
            Path(args.refer_data_root) / raw_path,
        ]

        # Some datasets store paths beginning with "/"
        # even though they are intended to be relative.
        stripped = Path(
            str(value).lstrip("/")
        )

        candidates.extend([
            PROJECT_ROOT / stripped,
            Path(args.refer_data_root) / stripped,
        ])

        existing = None

        for candidate in candidates:
            if candidate.exists():
                existing = candidate
                break

        if existing is None:
            return None

        with Image.open(str(existing)) as image:
            array = np.asarray(image)

    else:
        return None

    array = np.asarray(array)

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
        return None

    return (
        array > 0
    ).astype(np.uint8)


def get_original_mask_from_custom_refer(
    dataset,
    index,
    args,
):
    """
    Resolve an original-resolution GT mask from the
    RRSIS-D ReferDataset.

    Priority:
      1. dataset.refer.loadRefs + dataset.refer.getMask
      2. dataset.target_masks[index]
      3. dataset.images_to_mask
    """
    diagnostics = []

    refer = getattr(
        dataset,
        "refer",
        None,
    )

    ref_ids = getattr(
        dataset,
        "ref_ids",
        None,
    )

    # --------------------------------------------------------
    # 1. Standard REFER-style API
    # --------------------------------------------------------
    if (
        refer is not None
        and ref_ids is not None
        and index < len(ref_ids)
    ):
        ref_id = ref_ids[index]

        try:
            if hasattr(refer, "loadRefs"):
                try:
                    refs = refer.loadRefs(
                        [ref_id]
                    )
                except Exception:
                    refs = refer.loadRefs(
                        ref_id
                    )

                if isinstance(refs, dict):
                    refs = [refs]

                if (
                    isinstance(refs, (list, tuple))
                    and len(refs) > 0
                ):
                    ref_record = refs[0]
                else:
                    ref_record = refs

            else:
                ref_record = None

            if (
                ref_record is not None
                and hasattr(refer, "getMask")
            ):
                mask_result = refer.getMask(
                    ref_record
                )

                original_mask = (
                    normalize_original_mask_value(
                        mask_result,
                        args,
                    )
                )

                if original_mask is not None:
                    ann_ids = normalize_ann_ids(
                        ref_record
                    )

                    if not ann_ids:
                        try:
                            ann_ids = [
                                int(ref_id)
                            ]
                        except Exception:
                            ann_ids = []

                    return (
                        original_mask,
                        ann_ids,
                        "refer.getMask",
                    )

        except Exception as exc:
            diagnostics.append(
                "refer.getMask: {}".format(exc)
            )

    # --------------------------------------------------------
    # 2. Dataset target_masks
    # --------------------------------------------------------
    target_masks = getattr(
        dataset,
        "target_masks",
        None,
    )

    if (
        target_masks is not None
        and hasattr(target_masks, "__len__")
        and index < len(target_masks)
    ):
        try:
            original_mask = (
                normalize_original_mask_value(
                    target_masks[index],
                    args,
                )
            )

            if original_mask is not None:
                ref_id = (
                    ref_ids[index]
                    if (
                        ref_ids is not None
                        and index < len(ref_ids)
                    )
                    else index
                )

                try:
                    source_ids = [int(ref_id)]
                except Exception:
                    source_ids = []

                return (
                    original_mask,
                    source_ids,
                    "dataset.target_masks",
                )

        except Exception as exc:
            diagnostics.append(
                "target_masks: {}".format(exc)
            )

    # --------------------------------------------------------
    # 3. images_to_mask mapping
    # --------------------------------------------------------
    images_to_mask = getattr(
        dataset,
        "images_to_mask",
        None,
    )

    imgs = getattr(
        dataset,
        "imgs",
        None,
    )

    if (
        isinstance(images_to_mask, dict)
        and imgs is not None
        and index < len(imgs)
    ):
        image_item = imgs[index]

        candidate_keys = [
            image_item,
            str(image_item),
            Path(str(image_item)).name,
            Path(str(image_item)).stem,
        ]

        for key in candidate_keys:
            if key not in images_to_mask:
                continue

            try:
                original_mask = (
                    normalize_original_mask_value(
                        images_to_mask[key],
                        args,
                    )
                )

                if original_mask is not None:
                    return (
                        original_mask,
                        [],
                        "dataset.images_to_mask",
                    )

            except Exception as exc:
                diagnostics.append(
                    "images_to_mask: {}".format(
                        exc
                    )
                )

    raise RuntimeError(
        "Could not resolve original mask at "
        "dataset index {}. Attempts: {}"
        .format(
            index,
            diagnostics,
        )
    )


def percentile(values, q):
    if not values:
        return None

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def summarize_values(values):
    if not values:
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
        values,
        dtype=np.float64,
    )

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    parser = get_parser()

    args = parser.parse_args()

    output_dir = Path(
        args.a1_output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading transformed {} dataset..."
        .format(args.a1_split)
    )

    dataset, _ = get_dataset(
        args.a1_split,
        get_transform(args=args),
        args=args,
    )

    objects = find_wrapped_objects(
        dataset
    )

    coco = find_coco_object(
        objects
    )

    if coco is None:
        print(
            "\nNo standard COCO object was found."
        )
        print(
            "Using the RRSIS-D custom REFER / "
            "target_masks fallback."
        )
        print(
            "Dataset type:",
            type(dataset),
        )
        print(
            "Dataset attributes:",
            sorted(dataset.__dict__.keys()),
        )

    print(
        "Dataset size:",
        len(dataset),
    )

    if coco is not None:
        print(
            "COCO annotations:",
            len(coco.anns),
        )
    else:
        print(
            "Original-mask source: custom REFER"
        )

    print(
        "Thresholds: Tiny<=%.4f%%, "
        "Small<=%.4f%%, Medium<=%.4f%%"
        % (
            100.0
            * args.a1_tiny_max_ratio,
            100.0
            * args.a1_small_max_ratio,
            100.0
            * args.a1_medium_max_ratio,
        )
    )

    csv_path = (
        output_dir
        / "{}_original_vs_resized.csv".format(
            args.a1_tag
        )
    )

    summary_path = (
        output_dir
        / "{}_summary.json".format(
            args.a1_tag
        )
    )

    migration_counter = Counter()
    original_group_counter = Counter()
    resized_group_counter = Counter()

    vanished_counter = Counter()
    appeared_counter = Counter()

    mapping_ious = []
    low_mapping_samples = []

    retention_by_group = defaultdict(list)
    ratio_change_by_group = defaultdict(list)

    rows = []

    for index in range(len(dataset)):
        sample = dataset[index]

        if not isinstance(
            sample,
            (tuple, list),
        ):
            raise RuntimeError(
                "Unexpected dataset sample type: {}"
                .format(type(sample))
            )

        if len(sample) < 2:
            raise RuntimeError(
                "Dataset sample contains fewer "
                "than two fields."
            )

        transformed_mask = (
            tensor_to_binary_mask(
                sample[1]
            )
        )

        if coco is not None:
            candidate_sets = (
                collect_candidate_ann_sets(
                    dataset,
                    objects,
                    coco,
                    index,
                )
            )

            if not candidate_sets:
                raise RuntimeError(
                    "Could not map dataset index {} "
                    "to a COCO annotation."
                    .format(index)
                )

            best = choose_best_candidate(
                coco,
                candidate_sets,
                transformed_mask,
            )

            if best is None:
                raise RuntimeError(
                    "Could not build original mask "
                    "for index {}.".format(index)
                )

            original_mask = best[
                "original_mask"
            ]

            ann_ids = best[
                "ann_ids"
            ]

            source_name = "coco"

            mapping_iou = float(
                best["mapping_iou"]
            )

        else:
            (
                original_mask,
                ann_ids,
                source_name,
            ) = (
                get_original_mask_from_custom_refer(
                    dataset,
                    index,
                    args,
                )
            )

            directly_resized = resize_binary_mask(
                original_mask,
                int(transformed_mask.shape[0]),
                int(transformed_mask.shape[1]),
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
            low_mapping_samples.append({
                "index": int(index),
                "ann_ids": [
                    int(value)
                    for value in ann_ids
                ],
                "mapping_iou": mapping_iou,
            })

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

        original_fg = int(
            (original_mask > 0).sum()
        )

        original_total = int(
            original_height
            * original_width
        )

        resized_fg = int(
            transformed_mask.sum().item()
        )

        resized_total = int(
            resized_height
            * resized_width
        )

        original_ratio = (
            float(original_fg)
            / float(original_total)
            if original_total > 0
            else 0.0
        )

        resized_ratio = (
            float(resized_fg)
            / float(resized_total)
            if resized_total > 0
            else 0.0
        )

        original_group = classify_ratio(
            original_ratio,
            tiny_max_ratio=(
                args.a1_tiny_max_ratio
            ),
            small_max_ratio=(
                args.a1_small_max_ratio
            ),
            medium_max_ratio=(
                args.a1_medium_max_ratio
            ),
        )

        resized_group = classify_ratio(
            resized_ratio,
            tiny_max_ratio=(
                args.a1_tiny_max_ratio
            ),
            small_max_ratio=(
                args.a1_small_max_ratio
            ),
            medium_max_ratio=(
                args.a1_medium_max_ratio
            ),
        )

        original_group_counter[
            original_group
        ] += 1

        resized_group_counter[
            resized_group
        ] += 1

        migration_counter[
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
            vanished_counter[
                original_group
            ] += 1

        if appeared:
            appeared_counter[
                resized_group
            ] += 1

        scale_area = (
            float(resized_total)
            / float(original_total)
            if original_total > 0
            else 0.0
        )

        expected_resized_fg = (
            float(original_fg)
            * scale_area
        )

        if (
            original_fg > 0
            and expected_resized_fg > 0.0
        ):
            normalized_retention = (
                float(resized_fg)
                / expected_resized_fg
            )

            retention_by_group[
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

        ratio_change_by_group[
            original_group
        ].append(
            ratio_difference
        )

        rows.append({
            "index": int(index),
            "ann_ids": ",".join(
                str(value)
                for value in ann_ids
            ),
            "original_mask_source": source_name,
            "mapping_iou": mapping_iou,
            "original_height": original_height,
            "original_width": original_width,
            "original_fg_pixels": original_fg,
            "original_total_pixels": original_total,
            "original_fg_ratio": original_ratio,
            "original_group": original_group,
            "resized_height": resized_height,
            "resized_width": resized_width,
            "resized_fg_pixels": resized_fg,
            "resized_total_pixels": resized_total,
            "resized_fg_ratio": resized_ratio,
            "resized_group": resized_group,
            "ratio_difference": ratio_difference,
            "expected_resized_fg_pixels": (
                expected_resized_fg
            ),
            "normalized_area_retention": (
                normalized_retention
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
                or index + 1 == len(dataset)
            )
        ):
            print(
                "Processed {}/{} samples"
                .format(
                    index + 1,
                    len(dataset),
                )
            )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        str(csv_path),
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

    migration_matrix = {}

    for source_group in GROUP_NAMES:
        migration_matrix[
            source_group
        ] = {}

        for target_group in GROUP_NAMES:
            migration_matrix[
                source_group
            ][target_group] = int(
                migration_counter[
                    (
                        source_group,
                        target_group,
                    )
                ]
            )

    changed_group_count = sum(
        count
        for (
            source_group,
            target_group,
        ), count in migration_counter.items()
        if source_group != target_group
    )

    summary = {
        "tag": args.a1_tag,
        "split": args.a1_split,
        "sample_count": int(
            len(dataset)
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
            name: int(
                original_group_counter[name]
            )
            for name in GROUP_NAMES
        },
        "resized_group_counts": {
            name: int(
                resized_group_counter[name]
            )
            for name in GROUP_NAMES
        },
        "migration_matrix": (
            migration_matrix
        ),
        "changed_group_count": int(
            changed_group_count
        ),
        "changed_group_rate": (
            float(changed_group_count)
            / float(len(dataset))
        ),
        "vanished_after_transform": {
            "total": int(sum(
                vanished_counter.values()
            )),
            "by_original_group": {
                name: int(
                    vanished_counter[name]
                )
                for name in GROUP_NAMES
            },
        },
        "appeared_after_transform": {
            "total": int(sum(
                appeared_counter.values()
            )),
            "by_resized_group": {
                name: int(
                    appeared_counter[name]
                )
                for name in GROUP_NAMES
            },
        },
        "mapping_iou": (
            summarize_values(
                mapping_ious
            )
        ),
        "low_mapping_iou_threshold": float(
            args.a1_low_mapping_iou
        ),
        "low_mapping_samples": (
            low_mapping_samples[:100]
        ),
        "low_mapping_sample_count": int(
            len(low_mapping_samples)
        ),
        "normalized_area_retention": {
            name: summarize_values(
                retention_by_group[name]
            )
            for name in GROUP_NAMES
        },
        "foreground_ratio_difference": {
            name: summarize_values(
                ratio_change_by_group[name]
            )
            for name in GROUP_NAMES
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
        "Samples:",
        len(dataset),
    )

    print("\nOriginal GT groups:")

    for name in GROUP_NAMES:
        print(
            "  {:<8s}: {:>5}".format(
                name,
                original_group_counter[name],
            )
        )

    print("\nResized GT groups:")

    for name in GROUP_NAMES:
        print(
            "  {:<8s}: {:>5}".format(
                name,
                resized_group_counter[name],
            )
        )

    print("\nMigration matrix:")
    print(
        "{:>10s}".format("original"),
        end="",
    )

    for name in GROUP_NAMES:
        print(
            "{:>10s}".format(name),
            end="",
        )

    print()

    for source_group in GROUP_NAMES:
        print(
            "{:>10s}".format(
                source_group
            ),
            end="",
        )

        for target_group in GROUP_NAMES:
            print(
                "{:>10d}".format(
                    migration_counter[
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
            changed_group_count,
            len(dataset),
            100.0
            * changed_group_count
            / max(len(dataset), 1),
        )
    )

    print(
        "Original nonempty -> resized empty:",
        sum(vanished_counter.values()),
    )

    print(
        "Low mapping-IoU samples (<{:.2f}): {}"
        .format(
            args.a1_low_mapping_iou,
            len(low_mapping_samples),
        )
    )

    mapping_summary = summarize_values(
        mapping_ious
    )

    print(
        "Mapping IoU: mean={:.4f}, "
        "median={:.4f}, min={:.4f}".format(
            mapping_summary["mean"],
            mapping_summary["median"],
            mapping_summary["min"],
        )
    )

    print("\nNormalized area retention:")

    for name in GROUP_NAMES:
        values = summarize_values(
            retention_by_group[name]
        )

        if values["count"] == 0:
            print(
                "  {:<8s}: no nonempty samples"
                .format(name)
            )
            continue

        print(
            "  {:<8s}: n={} mean={:.4f} "
            "median={:.4f} p10={:.4f} "
            "p90={:.4f}".format(
                name,
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

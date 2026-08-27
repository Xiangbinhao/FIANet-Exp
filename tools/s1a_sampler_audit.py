import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from args import get_parser
from data.size_aware_sampler import (
    SIZE_GROUPS,
    build_size_aware_sampler,
)
from train_s1a import get_dataset, get_transform


def main():
    parser = get_parser()
    args = parser.parse_args()

    seed = int(args.s1a_sampler_seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset, _ = get_dataset(
        "train",
        get_transform(args=args),
        args=args,
    )

    sampler, summary = build_size_aware_sampler(
        dataset,
        args,
        verbose=True,
    )

    sampled_indices = list(iter(sampler))

    actual_counts = Counter(
        summary["group_names"][index]
        for index in sampled_indices
    )

    unique_indices = set(sampled_indices)

    unique_counts = Counter(
        summary["group_names"][index]
        for index in unique_indices
    )

    actual_percentages = {
        name: (
            100.0
            * actual_counts.get(name, 0)
            / len(sampled_indices)
        )
        for name in SIZE_GROUPS
    }

    print(
        "\n========== S1-A ONE-EPOCH AUDIT =========="
    )

    for name in SIZE_GROUPS:
        print(
            "%-6s drawn=%6d (%6.2f%%) "
            "unique=%6d"
            % (
                name,
                actual_counts.get(name, 0),
                actual_percentages[name],
                unique_counts.get(name, 0),
            )
        )

    print(
        "Total draws: {}".format(
            len(sampled_indices)
        )
    )

    print(
        "Unique samples: {} ({:.2f}%)".format(
            len(unique_indices),
            100.0
            * len(unique_indices)
            / len(dataset),
        )
    )

    print(
        "Repeated draws: {}".format(
            len(sampled_indices)
            - len(unique_indices)
        )
    )

    print(
        "=========================================="
    )

    output_path = args.s1a_audit_output
    output_parent = os.path.dirname(output_path)

    if output_parent:
        os.makedirs(
            output_parent,
            exist_ok=True,
        )

    output = {
        "summary": {
            key: value
            for key, value in summary.items()
            if key not in (
                "group_names",
                "area_ratios",
            )
        },
        "one_epoch_actual_counts": {
            name: int(
                actual_counts.get(name, 0)
            )
            for name in SIZE_GROUPS
        },
        "one_epoch_actual_percentages":
            actual_percentages,
        "one_epoch_unique_counts": {
            name: int(
                unique_counts.get(name, 0)
            )
            for name in SIZE_GROUPS
        },
        "total_draws": len(sampled_indices),
        "unique_samples": len(unique_indices),
        "repeated_draws": (
            len(sampled_indices)
            - len(unique_indices)
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "Saved audit: {}".format(
            output_path
        )
    )


if __name__ == "__main__":
    main()

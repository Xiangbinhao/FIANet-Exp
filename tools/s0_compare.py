import argparse
import csv
import json
from pathlib import Path


def parse_input(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Input must use TAG=path.json format."
        )

    tag, path = value.split("=", 1)
    return tag, Path(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=parse_input,
    )

    parser.add_argument(
        "--output-dir",
        default="experiments/S0/comparison",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for tag, path in args.inputs:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(file)

        groups = result["groups"]

        for group in [
            "all_nonempty",
            "tiny",
            "small",
            "medium",
            "large",
        ]:
            item = groups[group]

            row = {
                "model": tag,
                "group": group,
                "count": item["count"],
                "mIoU": item["mIoU"],
                "oIoU": item["oIoU"],
                "Pr@0.5": item[
                    "precision"
                ]["0.5"],
                "Pr@0.7": item[
                    "precision"
                ]["0.7"],
                "Pr@0.9": item[
                    "precision"
                ]["0.9"],
                "foreground_recall": item[
                    "foreground_recall"
                ],
                "empty_prediction_rate": item[
                    "empty_prediction_rate"
                ],
            }

            for threshold in [4, 9, 16]:
                component = item[
                    "components"
                ][str(threshold)]

                row[
                    "component_recall10_ge_{}".format(
                        threshold
                    )
                ] = component["recall_10"]

                row[
                    "component_recall50_ge_{}".format(
                        threshold
                    )
                ] = component["recall_50"]

            rows.append(row)

    csv_path = (
        output_dir
        / "s0_model_size_comparison.csv"
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

    print("Saved:", csv_path)

    print("\n========== TINY / SMALL ==========")

    for row in rows:
        if row["group"] not in [
            "tiny",
            "small",
        ]:
            continue

        print(
            "{:<4s} {:<5s} "
            "mIoU={:>6.2f} "
            "oIoU={:>6.2f} "
            "FG-Recall={:>6.2f} "
            "Empty={:>6.2f} "
            "CR10>=4={:>6.2f}".format(
                row["model"],
                row["group"],
                row["mIoU"],
                row["oIoU"],
                row["foreground_recall"],
                row["empty_prediction_rate"],
                row[
                    "component_recall10_ge_4"
                ],
            )
        )


if __name__ == "__main__":
    main()

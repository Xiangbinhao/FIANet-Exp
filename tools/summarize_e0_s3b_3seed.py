from pathlib import Path
import csv
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "experiments" / "repro_E0_S3B_3seed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [123, 456, 789]

METHODS = {
    "E0": "FIANet_E0_amp_bs8_full40_seed{seed}",
    "S3B": "FIANet_S3B_bounded_gated_refine_amp_bs8_full40_seed{seed}",
}


def read_text(path):
    if not path.exists():
        raise FileNotFoundError(str(path))

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def parse_test(path):
    text = read_text(path)

    patterns = {
        "mIoU": r"Mean IoU is\s+([0-9.]+)",
        "P@0.5": r"precision@0\.5\s*=\s*([0-9.]+)",
        "P@0.6": r"precision@0\.6\s*=\s*([0-9.]+)",
        "P@0.7": r"precision@0\.7\s*=\s*([0-9.]+)",
        "P@0.8": r"precision@0\.8\s*=\s*([0-9.]+)",
        "P@0.9": r"precision@0\.9\s*=\s*([0-9.]+)",
        "oIoU": r"overall IoU\s*=\s*([0-9.]+)",
    }

    result = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text)

        if match is None:
            raise RuntimeError(
                "Could not parse {} from {}".format(
                    key,
                    path,
                )
            )

        result[key] = float(match.group(1))

    return result


def parse_train(path):
    text = read_text(path)

    miou_matches = re.findall(
        r"Average object IoU\s+([0-9.]+)",
        text,
    )

    oiou_matches = re.findall(
        r"Overall IoU\s+([0-9.]+)",
        text,
    )

    epoch_matches = re.findall(
        r"Better epoch:\s*(\d+)",
        text,
    )

    if not miou_matches:
        raise RuntimeError(
            "No validation mIoU in {}".format(path)
        )

    if not oiou_matches:
        raise RuntimeError(
            "No validation oIoU in {}".format(path)
        )

    if not epoch_matches:
        raise RuntimeError(
            "No Better epoch in {}".format(path)
        )

    best_epoch = int(epoch_matches[-1])

    # Reconstruct validation records in log order so that the
    # validation metrics directly preceding each Better epoch are used.
    latest_miou = None
    latest_oiou = None
    best = None

    miou_re = re.compile(
        r"Average object IoU\s+([0-9.]+)"
    )
    oiou_re = re.compile(
        r"Overall IoU\s+([0-9.]+)"
    )
    better_re = re.compile(
        r"Better epoch:\s*(\d+)"
    )

    for line in text.splitlines():
        match = miou_re.search(line)
        if match:
            latest_miou = float(match.group(1))

        match = oiou_re.search(line)
        if match:
            latest_oiou = float(match.group(1))

        match = better_re.search(line)
        if match:
            best = {
                "best_epoch": int(match.group(1)),
                "val_mIoU": latest_miou,
                "val_oIoU": latest_oiou,
            }

    if best is None:
        raise RuntimeError(
            "Could not reconstruct best validation metrics."
        )

    return best


rows = []

for method, pattern in METHODS.items():
    for seed in SEEDS:
        model_id = pattern.format(seed=seed)

        train_log = LOG_DIR / (
            "train_{}.log".format(model_id)
        )

        test_log = LOG_DIR / (
            "test_{}.log".format(model_id)
        )

        train_metrics = parse_train(train_log)
        test_metrics = parse_test(test_log)

        row = {
            "method": method,
            "seed": seed,
            "best_epoch": train_metrics["best_epoch"],
            "val_mIoU": train_metrics["val_mIoU"],
            "val_oIoU": train_metrics["val_oIoU"],
            "test_mIoU": test_metrics["mIoU"],
            "test_oIoU": test_metrics["oIoU"],
            "P@0.5": test_metrics["P@0.5"],
            "P@0.6": test_metrics["P@0.6"],
            "P@0.7": test_metrics["P@0.7"],
            "P@0.8": test_metrics["P@0.8"],
            "P@0.9": test_metrics["P@0.9"],
        }

        rows.append(row)


per_run_path = OUT_DIR / "per_run_metrics.csv"

fields = [
    "method",
    "seed",
    "best_epoch",
    "val_mIoU",
    "val_oIoU",
    "test_mIoU",
    "test_oIoU",
    "P@0.5",
    "P@0.6",
    "P@0.7",
    "P@0.8",
    "P@0.9",
]

with per_run_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )
    writer.writeheader()
    writer.writerows(rows)


metrics = [
    "test_mIoU",
    "test_oIoU",
    "P@0.5",
    "P@0.6",
    "P@0.7",
    "P@0.8",
    "P@0.9",
]

summary_rows = []

for method in ["E0", "S3B"]:
    selected = [
        row for row in rows
        if row["method"] == method
    ]

    result = {
        "method": method,
    }

    for metric in metrics:
        values = [
            row[metric]
            for row in selected
        ]

        result[metric + "_mean"] = (
            statistics.mean(values)
        )

        result[metric + "_std"] = (
            statistics.stdev(values)
        )

    summary_rows.append(result)


summary_path = OUT_DIR / "summary_metrics.csv"

summary_fields = ["method"]

for metric in metrics:
    summary_fields += [
        metric + "_mean",
        metric + "_std",
    ]

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields,
    )
    writer.writeheader()
    writer.writerows(summary_rows)


paired_rows = []

for seed in SEEDS:
    e0 = next(
        row for row in rows
        if row["method"] == "E0"
        and row["seed"] == seed
    )

    s3b = next(
        row for row in rows
        if row["method"] == "S3B"
        and row["seed"] == seed
    )

    paired_rows.append({
        "seed": seed,
        "delta_mIoU":
            s3b["test_mIoU"] - e0["test_mIoU"],
        "delta_oIoU":
            s3b["test_oIoU"] - e0["test_oIoU"],
        "delta_P@0.5":
            s3b["P@0.5"] - e0["P@0.5"],
        "delta_P@0.7":
            s3b["P@0.7"] - e0["P@0.7"],
        "delta_P@0.9":
            s3b["P@0.9"] - e0["P@0.9"],
    })


delta_fields = [
    "delta_mIoU",
    "delta_oIoU",
    "delta_P@0.5",
    "delta_P@0.7",
    "delta_P@0.9",
]

mean_row = {
    "seed": "mean",
}

std_row = {
    "seed": "std",
}

for field in delta_fields:
    values = [
        row[field]
        for row in paired_rows
    ]

    mean_row[field] = statistics.mean(values)
    std_row[field] = statistics.stdev(values)

paired_rows += [
    mean_row,
    std_row,
]


paired_path = OUT_DIR / "paired_improvements.csv"

with paired_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["seed"] + delta_fields,
    )
    writer.writeheader()
    writer.writerows(paired_rows)


print("========== PER-SEED RESULTS ==========")

for row in rows:
    print(
        "{method:4s} seed={seed}: "
        "mIoU={miou:.2f}, "
        "oIoU={oiou:.2f}, "
        "P@0.5={p50:.2f}, "
        "best_epoch={epoch}".format(
            method=row["method"],
            seed=row["seed"],
            miou=row["test_mIoU"],
            oiou=row["test_oIoU"],
            p50=row["P@0.5"],
            epoch=row["best_epoch"],
        )
    )


print()
print("========== THREE-SEED SUMMARY ==========")

for row in summary_rows:
    print(
        "{method}: "
        "mIoU={miou:.3f}+/-{miou_std:.3f}, "
        "oIoU={oiou:.3f}+/-{oiou_std:.3f}, "
        "P@0.5={p50:.3f}+/-{p50_std:.3f}".format(
            method=row["method"],
            miou=row["test_mIoU_mean"],
            miou_std=row["test_mIoU_std"],
            oiou=row["test_oIoU_mean"],
            oiou_std=row["test_oIoU_std"],
            p50=row["P@0.5_mean"],
            p50_std=row["P@0.5_std"],
        )
    )


print()
print("========== PAIRED S3-B - E0 ==========")

for row in paired_rows[:3]:
    print(
        "seed {seed}: "
        "dmIoU={miou:+.3f}, "
        "doIoU={oiou:+.3f}, "
        "dP@0.5={p50:+.3f}".format(
            seed=row["seed"],
            miou=row["delta_mIoU"],
            oiou=row["delta_oIoU"],
            p50=row["delta_P@0.5"],
        )
    )

print(
    "mean: "
    "dmIoU={:+.3f}, "
    "doIoU={:+.3f}, "
    "dP@0.5={:+.3f}".format(
        mean_row["delta_mIoU"],
        mean_row["delta_oIoU"],
        mean_row["delta_P@0.5"],
    )
)

print()
print("Saved:", per_run_path)
print("Saved:", summary_path)
print("Saved:", paired_path)

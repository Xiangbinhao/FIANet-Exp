from pathlib import Path
import datetime
import re
import shutil


source_path = Path("train_s3b.py")
target_path = Path("train_s3c.py")

if not source_path.exists():
    raise FileNotFoundError(
        str(source_path)
    )

text = source_path.read_text(
    encoding="utf-8"
)

required_source_items = [
    "lib.bounded_gated_small_refinement",
    "BoundedGatedResidualWrapper",
    "S3-B bounded gated residual wrapper enabled",
    "S3-B optimizer added refinement parameters:",
]

missing = [
    item
    for item in required_source_items
    if item not in text
]

if missing:
    raise RuntimeError(
        "train_s3b.py is missing expected items: {}".format(
            missing
        )
    )

if target_path.exists():
    stamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_path = Path(
        "train_s3c.py.bak_{}".format(
            stamp
        )
    )
    shutil.copy2(
        target_path,
        backup_path,
    )
    print(
        "Backup:",
        target_path,
        "->",
        backup_path,
    )


# 1. Replace import module.
text = text.replace(
    "lib.bounded_gated_small_refinement",
    "lib.positive_preserving_asymmetric_refinement",
)

# 2. Replace wrapper class.
text = text.replace(
    "BoundedGatedResidualWrapper",
    "PositivePreservingAsymmetricResidualWrapper",
)

# 3. Insert asymmetric-control arguments after gate_init.
pattern = re.compile(
    r"(?P<indent>[ \t]*)gate_init=0\.20,\n"
)

matches = list(
    pattern.finditer(text)
)

if len(matches) != 1:
    raise RuntimeError(
        "Expected exactly one gate_init=0.20 argument, found {}".format(
            len(matches)
        )
    )

match = matches[0]
indent = match.group("indent")

replacement = (
    indent
    + "gate_init=0.20,\n"
    + indent
    + "negative_ratio_init=0.25,\n"
    + indent
    + "negative_ratio_max=0.50,\n"
)

text = (
    text[:match.start()]
    + replacement
    + text[match.end():]
)

# 4. Update diagnostic strings.
text = text.replace(
    "S3-B bounded gated residual wrapper enabled: ",
    "S3-C positive-preserving asymmetric wrapper enabled: ",
)

text = text.replace(
    '"alpha_init=0.10, gate_init=0.20"',
    '"alpha_init=0.10, gate_init=0.20, "'
    '"negative_ratio_init=0.25, negative_ratio_max=0.50"',
)

text = text.replace(
    "S3-B optimizer added refinement parameters:",
    "S3-C optimizer added refinement parameters:",
)

# 5. Update comments for audit readability.
text = text.replace(
    "# S3-B:",
    "# S3-C:",
)

text = text.replace(
    "S3-B requires --use-small-refine",
    "S3-C requires --use-small-refine",
)


required_target_items = [
    "lib.positive_preserving_asymmetric_refinement",
    "PositivePreservingAsymmetricResidualWrapper",
    "negative_ratio_init=0.25",
    "negative_ratio_max=0.50",
    "S3-C positive-preserving asymmetric wrapper enabled",
    "S3-C optimizer added refinement parameters:",
    "model.small_refinement_head.parameters()",
    "optimizer = torch.optim.AdamW",
]

missing_target = [
    item
    for item in required_target_items
    if item not in text
]

if missing_target:
    raise RuntimeError(
        "Generated train_s3c.py is missing: {}".format(
            missing_target
        )
    )

if "BoundedGatedResidualWrapper" in text:
    raise RuntimeError(
        "Old S3-B class remains in train_s3c.py"
    )

target_path.write_text(
    text,
    encoding="utf-8",
)

print("Created:", target_path)
print("S3-C training script generation: PASS")

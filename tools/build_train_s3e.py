from pathlib import Path
import datetime
import re
import shutil


source_path = Path("train_s3b.py")
target_path = Path("train_s3e.py")

if not source_path.exists():
    raise FileNotFoundError(str(source_path))

text = source_path.read_text(
    encoding="utf-8",
)

required_source_items = [
    "lib.bounded_gated_small_refinement",
    "BoundedGatedResidualWrapper",
    "S3-B bounded gated residual wrapper enabled",
    "S3-B optimizer added refinement parameters:",
]

missing_source_items = [
    item
    for item in required_source_items
    if item not in text
]

if missing_source_items:
    raise RuntimeError(
        "train_s3b.py is missing expected items: {}".format(
            missing_source_items
        )
    )

if target_path.exists():
    stamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_path = Path(
        "train_s3e.py.bak_{}".format(stamp)
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


# Replace module import.
text = text.replace(
    "lib.bounded_gated_small_refinement",
    "lib.main_logit_evidence_refinement",
)

# Replace wrapper class.
text = text.replace(
    "BoundedGatedResidualWrapper",
    "MainLogitEvidenceConditionalResidualWrapper",
)


# Replace the original S3-B constructor.
constructor_pattern = re.compile(
    r"MainLogitEvidenceConditionalResidualWrapper\(\s*"
    r"base_head=_s3b_base_head,\s*"
    r"alpha_init=0\.10,\s*"
    r"gate_init=0\.20,\s*"
    r"\)",
    flags=re.DOTALL,
)

constructor_replacement = """MainLogitEvidenceConditionalResidualWrapper(
            base_head=_s3b_base_head,
            alpha_init=0.10,
            gate_init=0.20,
            relax_init=0.75,
            weak_positive_low=0.45,
            weak_positive_high=0.70,
            evidence_temperature=0.05,
        )"""

text, constructor_count = constructor_pattern.subn(
    constructor_replacement,
    text,
    count=1,
)

if constructor_count != 1:
    raise RuntimeError(
        "Expected exactly one S3-E constructor replacement, "
        "got {}".format(constructor_count)
    )


# Replace diagnostic text.
text = text.replace(
    "S3-B bounded gated residual wrapper enabled: ",
    "S3-E main-logit evidence wrapper enabled: ",
)

text = text.replace(
    '"alpha_init=0.10, gate_init=0.20"',
    '"alpha_init=0.10, gate_init=0.20, "'
    '"relax_init=0.75, weak_positive_low=0.45, "'
    '"weak_positive_high=0.70, "'
    '"evidence_temperature=0.05"',
)

text = text.replace(
    "S3-B optimizer added refinement parameters:",
    "S3-E optimizer added refinement parameters:",
)

text = text.replace(
    "# S3-B:",
    "# S3-E:",
)

text = text.replace(
    "S3-B requires --use-small-refine",
    "S3-E requires --use-small-refine",
)


required_target_items = [
    "lib.main_logit_evidence_refinement",
    "MainLogitEvidenceConditionalResidualWrapper",
    "relax_init=0.75",
    "weak_positive_low=0.45",
    "weak_positive_high=0.70",
    "evidence_temperature=0.05",
    "S3-E main-logit evidence wrapper enabled",
    "S3-E optimizer added refinement parameters:",
    "model.small_refinement_head.parameters()",
    "optimizer = torch.optim.AdamW",
]

missing_target_items = [
    item
    for item in required_target_items
    if item not in text
]

if missing_target_items:
    raise RuntimeError(
        "Generated train_s3e.py is missing: {}".format(
            missing_target_items
        )
    )

# Syntax validation before writing.
compile(
    text,
    str(target_path),
    "exec",
)

target_path.write_text(
    text,
    encoding="utf-8",
)

print("Created:", target_path)
print("S3-E training script generation: PASS")

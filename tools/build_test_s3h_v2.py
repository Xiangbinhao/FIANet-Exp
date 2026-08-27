from pathlib import Path
from datetime import datetime
import shutil


source = Path("test_s3f.py")
target = Path("test_s3h_v2.py")

if not source.exists():
    raise FileNotFoundError(str(source))

text = source.read_text(
    encoding="utf-8"
)

if target.exists():
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = target.with_name(
        target.name
        + ".bak_"
        + stamp
    )

    shutil.copy2(
        target,
        backup,
    )

    print(
        "Backup:",
        backup,
    )


required = [
    "lib.weak_target_flip_rescue_refinement",
    "WeakTargetFlipRescueWrapper",
]

for token in required:
    if token not in text:
        raise RuntimeError(
            "Required S3-F token missing: "
            + token
        )


text = text.replace(
    "lib.weak_target_flip_rescue_refinement",
    "lib.posthoc_budgeted_s3f_refinement",
)

text = text.replace(
    "WeakTargetFlipRescueWrapper",
    "PosthocBudgetedS3FWrapper",
)

text = text.replace(
    "S3-F",
    "S3-H v2",
)


compile(
    text,
    str(target),
    "exec",
)

target.write_text(
    text,
    encoding="utf-8",
)

print(
    "Created:",
    target,
)

print(
    "S3-H v2 test build: PASS"
)

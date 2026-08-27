from pathlib import Path
from datetime import datetime
import shutil


source = Path("test_s3b.py")
target = Path("test_s3f.py")

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
        target.name + ".bak_" + stamp
    )
    shutil.copy2(target, backup)
    print("Backup:", backup)


required = [
    "lib.bounded_gated_small_refinement",
    "BoundedGatedResidualWrapper",
]

for token in required:
    if token not in text:
        raise RuntimeError(
            "Required S3-B token not found: {}".format(
                token
            )
        )


text = text.replace(
    "lib.bounded_gated_small_refinement",
    "lib.weak_target_flip_rescue_refinement",
)

text = text.replace(
    "BoundedGatedResidualWrapper",
    "WeakTargetFlipRescueWrapper",
)

# Display-name replacements are optional.
text = text.replace(
    "S3-B",
    "S3-F",
)

text = text.replace(
    "S3B",
    "S3F",
)


required_output = [
    "WeakTargetFlipRescueWrapper",
    "weak_target_flip_rescue_refinement",
]

missing = [
    token
    for token in required_output
    if token not in text
]

if missing:
    raise RuntimeError(
        "Generated test_s3f.py missing: {}".format(
            missing
        )
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

print("Source:", source)
print("Created:", target)
print("S3-F test entry generation: PASS")

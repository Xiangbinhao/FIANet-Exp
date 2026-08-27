from pathlib import Path
from datetime import datetime
import re
import shutil


source = Path("train_s3b.py")
target = Path("train_s3i.py")

if not source.exists():
    raise FileNotFoundError(
        str(source)
    )

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


old_import = (
    "from lib.bounded_gated_small_refinement "
    "import BoundedGatedResidualWrapper"
)

new_import = (
    "from lib.target_consistent_gradient_protection_refinement "
    "import TargetConsistentGradientProtectionWrapper, "
    "set_s3i_training_target"
)

if old_import not in text:
    raise RuntimeError(
        "Could not find expected S3-B import:\n"
        + old_import
    )

text = text.replace(
    old_import,
    new_import,
    1,
)

text = text.replace(
    "BoundedGatedResidualWrapper",
    "TargetConsistentGradientProtectionWrapper",
)

text = text.replace(
    "S3-B",
    "S3-I",
)

text = text.replace(
    "_s3b",
    "_s3i",
)


# ------------------------------------------------------
# Inject GT immediately before every model forward.
#
# In validation:
#     model.training == False
# so no target injection occurs.
#
# In training:
#     target has already been moved to device in the
# standard FIANet loop before model(...).
# ------------------------------------------------------
pattern = re.compile(
    r"^(\s*)(output|outputs)\s*=\s*model\(",
    flags=re.MULTILINE,
)

matches = list(
    pattern.finditer(text)
)

if not matches:
    raise RuntimeError(
        "No 'output = model(' / 'outputs = model(' "
        "calls found in train_s3b.py"
    )


def inject(match):
    indent = match.group(1)

    original = match.group(0)

    return (
        indent
        + "if model.training:\n"
        + indent
        + "    set_s3i_training_target(model, target)\n"
        + original
    )


text = pattern.sub(
    inject,
    text,
)

if "set_s3i_training_target(model, target)" not in text:
    raise RuntimeError(
        "S3-I target injection failed"
    )

if "FIANET_SEED" not in text:
    raise RuntimeError(
        "FIANET_SEED support missing"
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
    "Model forward calls patched:",
    len(matches),
)

print(
    "Created:",
    target,
)

print(
    "S3-I training entry generation: PASS"
)

from pathlib import Path
from datetime import datetime
import re
import shutil


source = Path("test.py")
target = Path("test_s4a.py")

if not source.exists():
    raise FileNotFoundError(
        "Required base test entry does not exist: {}".format(source)
    )

text = source.read_text(
    encoding="utf-8"
)

if target.exists():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(
        target.name + ".bak_" + stamp
    )
    shutil.copy2(target, backup)
    print("Backup:", backup)


prefix = r'''
# ============================================================
# S4-A test model-construction patch
# ============================================================
import lib.segmentation as _s4a_segmentation

from lib.semantic_guided_highres_refinement import (
    SemanticGuidedHighResolutionRefinementHead,
)


_s4a_original_lavt_one = (
    _s4a_segmentation.lavt_one
)


def _s4a_lavt_one(*call_args, **call_kwargs):
    model = _s4a_original_lavt_one(
        *call_args,
        **call_kwargs
    )

    args_obj = call_kwargs.get(
        "args",
        None,
    )

    if args_obj is None:
        for value in reversed(call_args):
            if hasattr(value, "swin_type"):
                args_obj = value
                break

    if args_obj is not None:
        swin_type = getattr(
            args_obj,
            "swin_type",
            None,
        )

        if swin_type is not None and swin_type != "base":
            raise RuntimeError(
                "S4-A checkpoint was trained with "
                "--swin_type base, got {}".format(
                    swin_type
                )
            )

    # Replace/create the refinement slot directly.
    # We do NOT require test.py to construct the old S2 head first.
    model.small_refinement_head = (
        SemanticGuidedHighResolutionRefinementHead(
            x_c1_channels=128,
            x_c2_channels=256,
            semantic_channels=1024,
            project_channels=16,
            semantic_project_channels=16,
            hidden_channels=32,
            num_classes=2,
        )
    )

    params = sum(
        p.numel()
        for p in model.small_refinement_head.parameters()
    )

    print(
        "S4-A semantic-guided high-resolution "
        "refinement enabled for testing"
    )
    print(
        "S4-A refinement parameters: {}".format(
            params
        )
    )
    print(
        "S4-A config: "
        "x_c1=128, x_c2=256, x_c4=1024, "
        "detail_project=16, semantic_project=16, "
        "hidden=32, x_c4_detach=True"
    )

    if params != 25730:
        raise RuntimeError(
            "Unexpected S4-A refinement parameter count: "
            "{} (expected 25730)".format(params)
        )

    return model


_s4a_segmentation.lavt_one = (
    _s4a_lavt_one
)

# ============================================================

'''


# Preserve __future__ imports if test.py has any.
future_lines = []
remaining_lines = []

for line in text.splitlines(True):
    if re.match(
        r"^\s*from\s+__future__\s+import\s+",
        line,
    ):
        future_lines.append(line)
    else:
        remaining_lines.append(line)

generated = (
    "".join(future_lines)
    + prefix
    + "".join(remaining_lines)
)


required = [
    "SemanticGuidedHighResolutionRefinementHead",
    "_s4a_segmentation.lavt_one",
    "semantic_project_channels=16",
    "S4-A refinement parameters:",
    "25730",
]

missing = [
    item
    for item in required
    if item not in generated
]

if missing:
    raise RuntimeError(
        "Generated test_s4a.py missing: {}".format(
            missing
        )
    )


compile(
    generated,
    str(target),
    "exec",
)

target.write_text(
    generated,
    encoding="utf-8",
)

print("Source:", source)
print("Created:", target)
print("S4-A test entry generation: PASS")

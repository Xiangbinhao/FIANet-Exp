from pathlib import Path
from datetime import datetime
import re
import shutil

source = Path("tools/a2_metrics.py")
target = Path("tools/a2_metrics_s4a.py")

if not source.exists():
    raise FileNotFoundError(str(source))

text = source.read_text(
    encoding="utf-8",
    errors="replace",
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
# S4-A A2 model-construction patch
# ============================================================
import sys as _s4a_sys
from pathlib import Path as _S4A_Path

_s4a_project_root = str(
    _S4A_Path(__file__).resolve().parents[1]
)

if _s4a_project_root not in _s4a_sys.path:
    _s4a_sys.path.insert(
        0,
        _s4a_project_root,
    )

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
        "refinement enabled for A2"
    )
    print(
        "S4-A refinement parameters: {}".format(
            params
        )
    )

    if params != 25730:
        raise RuntimeError(
            "Unexpected S4-A refinement parameter count: "
            "{}".format(params)
        )

    return model


_s4a_segmentation.lavt_one = _s4a_lavt_one

# ============================================================

'''

# Keep possible __future__ imports at the beginning.
lines = text.splitlines(True)

future_lines = []
other_lines = []

for line in lines:
    if re.match(
        r"^\s*from\s+__future__\s+import\s+",
        line,
    ):
        future_lines.append(line)
    else:
        other_lines.append(line)

generated = (
    "".join(future_lines)
    + prefix
    + "".join(other_lines)
)

required = [
    "SemanticGuidedHighResolutionRefinementHead",
    "_s4a_segmentation.lavt_one",
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
        "Generated A2 S4-A entry missing: {}".format(
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
print("S4-A A2 entry generation: PASS")

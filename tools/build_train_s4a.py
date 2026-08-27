from pathlib import Path
from datetime import datetime
import re
import shutil


source = Path("train_s2.py")
target = Path("train_s4a.py")

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

    print("Backup:", backup)


prefix = r'''# ============================================================
# S4-A model-construction patch
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

    if args_obj is None:
        raise RuntimeError(
            "S4-A could not resolve args."
        )

    if getattr(
        args_obj,
        "swin_type",
        None,
    ) != "base":
        raise RuntimeError(
            "S4-A first experiment is fixed to "
            "--swin_type base."
        )

    base_head = getattr(
        model,
        "small_refinement_head",
        None,
    )

    if base_head is None:
        raise RuntimeError(
            "S4-A requires --use-small-refine "
            "to create the refinement slot."
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

    print(
        "S4-A semantic-guided high-resolution "
        "refinement enabled: "
        "x_c1=128, x_c2=256, x_c4=1024, "
        "detail_project=16, semantic_project=16, "
        "hidden=32, x_c4_detach=True"
    )

    return model


_s4a_segmentation.lavt_one = (
    _s4a_lavt_one
)

# ============================================================

'''

text = prefix + text


# ------------------------------------------------------------
# Make seed externally controllable, without touching train_s2.py.
# ------------------------------------------------------------
if 'FIANET_SEED' not in text:
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)"
        r"random\.seed\(seed\)",
        flags=re.MULTILINE,
    )

    matches = list(
        pattern.finditer(text)
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected one random.seed(seed), "
            "found {}".format(
                len(matches)
            )
        )

    match = matches[0]
    indent = match.group("indent")

    replacement = (
        indent
        + 'seed = int(__import__("os").environ.get('
        + '"FIANET_SEED", seed))\n'
        + indent
        + 'print("Effective random seed: {}".format(seed))\n'
        + match.group(0)
    )

    text = (
        text[:match.start()]
        + replacement
        + text[match.end():]
    )


text = text.replace(
    "S2 optimizer added refinement parameters:",
    "S4-A optimizer added refinement parameters:",
)


required = [
    "SemanticGuidedHighResolutionRefinementHead",
    "S4-A semantic-guided high-resolution ",
    "refinement enabled:",
    "FIANET_SEED",
    "model.small_refinement_head.parameters()",
    "S4-A optimizer added refinement parameters:",
]

missing = [
    item
    for item in required
    if item not in text
]

if missing:
    raise RuntimeError(
        "Generated train_s4a.py missing: "
        "{}".format(missing)
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

print("Created:", target)
print("S4-A training entry generation: PASS")

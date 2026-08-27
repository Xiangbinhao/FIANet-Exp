import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lib import segmentation as _s4a_segmentation

from lib.semantic_guided_highres_refinement import (
    SemanticGuidedHighResolutionRefinementHead,
)


_original_lavt_one = _s4a_segmentation.lavt_one


def _s4a_lavt_one(*call_args, **call_kwargs):
    model = _original_lavt_one(
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
        "refinement enabled for A2",
        flush=True,
    )

    print(
        "S4-A refinement parameters: {}".format(
            params
        ),
        flush=True,
    )

    print(
        "S4-A config: "
        "x_c1=128, x_c2=256, x_c4=1024, "
        "detail_project=16, semantic_project=16, "
        "hidden=32, x_c4_detach=True",
        flush=True,
    )

    if params != 25730:
        raise RuntimeError(
            "Unexpected S4-A refinement parameter "
            "count: {} (expected 25730)".format(
                params
            )
        )

    return model


# Patch BEFORE importing the A2 runner.
_s4a_segmentation.lavt_one = _s4a_lavt_one


from tools import s0_size_eval_a2 as _a2


if __name__ == "__main__":
    _a2.main()

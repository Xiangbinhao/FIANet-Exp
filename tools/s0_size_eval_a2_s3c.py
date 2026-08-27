import sys
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lib import segmentation
from lib.positive_preserving_asymmetric_refinement import (
    PositivePreservingAsymmetricResidualWrapper,
)


_original_lavt_one = segmentation.lavt_one


def _s3c_lavt_one(*args, **kwargs):
    model = _original_lavt_one(*args, **kwargs)

    base_head = getattr(
        model,
        "small_refinement_head",
        None,
    )

    if base_head is None:
        raise RuntimeError(
            "S3-C A2 evaluation requires --use-small-refine."
        )

    model.small_refinement_head = (
        PositivePreservingAsymmetricResidualWrapper(
            base_head=base_head,
            alpha_init=0.10,
            gate_init=0.20,
            negative_ratio_init=0.25,
            negative_ratio_max=0.50,
        )
    )

    print(
        "S3-C A2 wrapper enabled: "
        "alpha_init=0.10, gate_init=0.20, "
        "negative_ratio_init=0.25, "
        "negative_ratio_max=0.50"
    )

    return model


segmentation.lavt_one = _s3c_lavt_one

runpy.run_path(
    str(PROJECT_ROOT / "tools" / "s0_size_eval_a2.py"),
    run_name="__main__",
)

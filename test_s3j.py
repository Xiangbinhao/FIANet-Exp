import runpy

from lib import segmentation
from lib.coverage_weighted_gradient_protection_refinement import (
    CoverageWeightedGradientProtectionWrapper,
)


_original_lavt_one = segmentation.lavt_one


def _s3j_lavt_one(*args, **kwargs):
    model = _original_lavt_one(*args, **kwargs)

    base_head = getattr(
        model,
        "small_refinement_head",
        None,
    )

    if base_head is None:
        raise RuntimeError(
            "S3-J evaluation requires --use-small-refine."
        )

    model.small_refinement_head = (
        CoverageWeightedGradientProtectionWrapper(
            base_head=base_head,
            alpha_init=0.10,
            gate_init=0.20,
        )
    )

    print(
        "S3-J evaluation wrapper enabled: "
        "alpha_init=0.10, gate_init=0.20"
    )

    return model


segmentation.lavt_one = _s3j_lavt_one

runpy.run_path(
    "test.py",
    run_name="__main__",
)

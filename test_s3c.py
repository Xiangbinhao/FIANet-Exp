import runpy

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
            "S3-C evaluation requires --use-small-refine."
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
        "S3-C evaluation wrapper enabled: "
        "alpha_init=0.10, gate_init=0.20, "
        "negative_ratio_init=0.25, negative_ratio_max=0.50"
    )

    return model


segmentation.lavt_one = _s3c_lavt_one

runpy.run_path(
    "test.py",
    run_name="__main__",
)

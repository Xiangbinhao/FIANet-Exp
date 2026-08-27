import runpy

from lib import segmentation
from lib.bounded_gated_small_refinement import (
    BoundedGatedResidualWrapper,
)


_original_lavt_one = segmentation.lavt_one


def _s3b_lavt_one(*args, **kwargs):
    model = _original_lavt_one(*args, **kwargs)

    base_head = getattr(
        model,
        "small_refinement_head",
        None,
    )

    if base_head is None:
        raise RuntimeError(
            "S3-B A2 evaluation requires --use-small-refine."
        )

    model.small_refinement_head = (
        BoundedGatedResidualWrapper(
            base_head=base_head,
            alpha_init=0.10,
            gate_init=0.20,
        )
    )

    print(
        "S3-B A2 wrapper enabled: "
        "alpha_init=0.10, gate_init=0.20"
    )

    return model


segmentation.lavt_one = _s3b_lavt_one

runpy.run_path(
    "tools/s0_size_eval_a2.py",
    run_name="__main__",
)

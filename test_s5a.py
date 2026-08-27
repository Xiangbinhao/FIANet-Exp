import runpy

from lib import segmentation

from lib.semantic_confidence_coverage_refinement import (
    SemanticConfidenceCoverageRefinementWrapper,
)


_original_lavt_one = segmentation.lavt_one


def _s5a_lavt_one(*args, **kwargs):

    model = _original_lavt_one(
        *args,
        **kwargs
    )

    base_head = getattr(
        model,
        "small_refinement_head",
        None,
    )

    if base_head is None:
        raise RuntimeError(
            "S5-A requires --use-small-refine"
        )


    model.small_refinement_head = (
        SemanticConfidenceCoverageRefinementWrapper(
            base_head=base_head,
            alpha_init=0.10,
            gate_init=0.20,
        )
    )


    print(
        "S5-A semantic confidence coverage refinement enabled"
    )

    return model


segmentation.lavt_one = _s5a_lavt_one


runpy.run_path(
    "train_s5a.py",
    run_name="__main__",
)

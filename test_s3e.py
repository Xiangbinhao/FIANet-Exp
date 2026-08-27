import sys
import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib import segmentation
from lib.main_logit_evidence_refinement import (
    MainLogitEvidenceConditionalResidualWrapper,
)


_original_lavt_one = segmentation.lavt_one


def _s3e_lavt_one(*args, **kwargs):
    model = _original_lavt_one(*args, **kwargs)

    base_head = getattr(
        model,
        "small_refinement_head",
        None,
    )

    if base_head is None:
        raise RuntimeError(
            "S3-E evaluation requires --use-small-refine."
        )

    model.small_refinement_head = (
        MainLogitEvidenceConditionalResidualWrapper(
            base_head=base_head,
            alpha_init=0.10,
            gate_init=0.20,
            relax_init=0.75,
            weak_positive_low=0.45,
            weak_positive_high=0.70,
            evidence_temperature=0.05,
        )
    )

    print(
        "S3-E evaluation wrapper enabled: "
        "alpha_init=0.10, gate_init=0.20, "
        "relax_init=0.75, weak_positive_low=0.45, "
        "weak_positive_high=0.70, evidence_temperature=0.05"
    )

    return model


segmentation.lavt_one = _s3e_lavt_one

runpy.run_path(
    str(PROJECT_ROOT / "test.py"),
    run_name="__main__",
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# IMPORTANT:
# Import A2 FIRST.
#
# s0_size_eval_a2 -> s0_size_eval will pre-parse and remove:
#   --s0-output-dir
#   --s0-tag
#   --s0-max-samples
#   --s0-component-thresholds
#
# before test_s3i gets a chance to parse sys.argv.
from tools import s0_size_eval_a2 as _a2


# Install the S3-I lavt_one monkey-patch.
#
# In eval mode TargetConsistentGradientProtectionWrapper
# returns the original S3-B residual exactly, so no GT
# injection is required here.
import test_s3i  # noqa: F401


if __name__ == "__main__":
    _a2.main()

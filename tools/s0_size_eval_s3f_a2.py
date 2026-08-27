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
# before test/test_s3f gets a chance to parse sys.argv.
from tools import s0_size_eval_a2 as _a2


# Now sys.argv contains only the normal FIANet/test arguments.
#
# Importing test_s3f installs the S3-F lavt_one monkey-patch
# into the shared lib.segmentation module object.
import test_s3f  # noqa: F401


if __name__ == "__main__":
    _a2.main()

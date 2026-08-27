import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# A2 must be imported first so --s0-* arguments
# are removed before test_s3g/test.py parses argv.
from tools import s0_size_eval_a2 as _a2

# Install S3-G lavt_one monkey-patch afterwards.
import test_s3g  # noqa: F401


if __name__ == "__main__":
    _a2.main()

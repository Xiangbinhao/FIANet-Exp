from pathlib import Path
from datetime import datetime
import shutil


source = Path("train_s3b.py")
target = Path("train_s3g.py")

if not source.exists():
    raise FileNotFoundError(
        str(source)
    )

text = source.read_text(
    encoding="utf-8"
)

if target.exists():
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = target.with_name(
        target.name
        + ".bak_"
        + stamp
    )

    shutil.copy2(
        target,
        backup,
    )

    print(
        "Backup:",
        backup,
    )


required = [
    "lib.bounded_gated_small_refinement",
    "BoundedGatedResidualWrapper",
]

for token in required:
    if token not in text:
        raise RuntimeError(
            "Required S3-B token not found: "
            "{}".format(token)
        )


text = text.replace(
    "lib.bounded_gated_small_refinement",
    "lib.semantic_detail_consensus_rescue_refinement",
)

text = text.replace(
    "BoundedGatedResidualWrapper",
    "SemanticDetailConsensusRescueWrapper",
)


# Optional cosmetic replacements only.
text = text.replace(
    "S3-B",
    "S3-G",
)

text = text.replace(
    "_s3b",
    "_s3g",
)


# If train_s3b.py contains a literal parameter audit,
# update it to the S3-G count.
if "8806" in text:
    text = text.replace(
        "8806",
        "9191",
    )


required_output = [
    "SemanticDetailConsensusRescueWrapper",
    "semantic_detail_consensus_rescue_refinement",
    "FIANET_SEED",
    "small_refinement_head.parameters()",
]

missing = [
    token
    for token in required_output
    if token not in text
]

if missing:
    raise RuntimeError(
        "Generated train_s3g.py missing: "
        "{}".format(missing)
    )


compile(
    text,
    str(target),
    "exec",
)

target.write_text(
    text,
    encoding="utf-8",
)

print(
    "Source:",
    source,
)
print(
    "Created:",
    target,
)
print(
    "S3-G training entry generation: PASS"
)

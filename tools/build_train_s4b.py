from pathlib import Path
from datetime import datetime
import shutil


source = Path("train_s4a.py")
target = Path("train_s4b.py")

if not source.exists():
    raise FileNotFoundError(str(source))

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

    print("Backup:", backup)


replacements = [
    (
        "lib.semantic_guided_highres_refinement",
        "lib.identity_semantic_modulation_refinement",
    ),
    (
        "SemanticGuidedHighResolutionRefinementHead",
        "IdentitySemanticModulationRefinementHead",
    ),
    (
        "S4-A",
        "S4-B",
    ),
    (
        "_s4a",
        "_s4b",
    ),
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError(
            "Expected source token not found: "
            "{}".format(old)
        )

    text = text.replace(
        old,
        new,
    )


# Update architecture description.
text = text.replace(
    '"x_c1=128, x_c2=256, x_c4=1024, "\n'
    '        "detail_project=16, semantic_project=16, "\n'
    '        "hidden=32, x_c4_detach=True"',
    '"x_c1=128, x_c2=256, x_c4=1024, "\n'
    '        "detail_project=16, semantic_project=16, "\n'
    '        "hidden=32, beta_max=0.25, "\n'
    '        "identity_beta=True, x_c4_detach=True"',
)


required = [
    "IdentitySemanticModulationRefinementHead",
    "identity_semantic_modulation_refinement",
    "S4-B",
    "FIANET_SEED",
    "model.small_refinement_head.parameters()",
]

missing = [
    item
    for item in required
    if item not in text
]

if missing:
    raise RuntimeError(
        "Generated train_s4b.py missing: "
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

print("Source:", source)
print("Created:", target)
print("S4-B training entry generation: PASS")

from pathlib import Path
from datetime import datetime
import re
import shutil


path = Path("lib/_utils.py")

if not path.exists():
    raise FileNotFoundError(str(path))

text = path.read_text(
    encoding="utf-8"
)

if "requires_semantic_feature" in text:
    print(
        "S4-A semantic forward support "
        "already installed."
    )
    raise SystemExit(0)


pattern = re.compile(
    r"(?P<indent>^[ \t]*)"
    r"if self\.small_refinement_head is not None:\s*\n"
    r"(?P<body>.*?)"
    r"(?=^[ \t]*small_residual_logits\s*=\s*"
    r"F\.interpolate\()",
    flags=re.MULTILINE | re.DOTALL,
)

matches = list(
    pattern.finditer(text)
)

if len(matches) != 1:
    raise RuntimeError(
        "Expected exactly one small-refinement "
        "forward block, found {}".format(
            len(matches)
        )
    )

match = matches[0]
indent = match.group("indent")

replacement = (
    indent
    + "if self.small_refinement_head is not None:\n"
    + indent
    + "    if getattr(\n"
    + indent
    + "        self.small_refinement_head,\n"
    + indent
    + "        'requires_semantic_feature',\n"
    + indent
    + "        False,\n"
    + indent
    + "    ):\n"
    + indent
    + "        small_residual_logits = (\n"
    + indent
    + "            self.small_refinement_head(\n"
    + indent
    + "                x_c1,\n"
    + indent
    + "                x_c2,\n"
    + indent
    + "                x_c4,\n"
    + indent
    + "            )\n"
    + indent
    + "        )\n"
    + indent
    + "    elif getattr(\n"
    + indent
    + "        self.small_refinement_head,\n"
    + indent
    + "        'requires_main_logits',\n"
    + indent
    + "        False,\n"
    + indent
    + "    ):\n"
    + indent
    + "        small_residual_logits = (\n"
    + indent
    + "            self.small_refinement_head(\n"
    + indent
    + "                x_c1,\n"
    + indent
    + "                x_c2,\n"
    + indent
    + "                final_logits,\n"
    + indent
    + "            )\n"
    + indent
    + "        )\n"
    + indent
    + "    else:\n"
    + indent
    + "        small_residual_logits = (\n"
    + indent
    + "            self.small_refinement_head(\n"
    + indent
    + "                x_c1,\n"
    + indent
    + "                x_c2,\n"
    + indent
    + "            )\n"
    + indent
    + "        )\n\n"
)

patched = (
    text[:match.start()]
    + replacement
    + text[match.end():]
)

checks = {
    "semantic marker":
        "requires_semantic_feature"
        in patched,

    "semantic x_c4 call":
        re.search(
            r"self\.small_refinement_head\(\s*"
            r"x_c1\s*,\s*"
            r"x_c2\s*,\s*"
            r"x_c4\s*,?\s*"
            r"\)",
            patched,
            flags=re.DOTALL,
        ) is not None,

    "S3-E compatibility":
        re.search(
            r"self\.small_refinement_head\(\s*"
            r"x_c1\s*,\s*"
            r"x_c2\s*,\s*"
            r"final_logits\s*,?\s*"
            r"\)",
            patched,
            flags=re.DOTALL,
        ) is not None,

    "S2 compatibility":
        re.search(
            r"self\.small_refinement_head\(\s*"
            r"x_c1\s*,\s*"
            r"x_c2\s*,?\s*"
            r"\)",
            patched,
            flags=re.DOTALL,
        ) is not None,
}

failed = [
    name
    for name, passed in checks.items()
    if not passed
]

if failed:
    raise RuntimeError(
        "S4-A semantic validation failed: "
        "{}".format(failed)
    )

compile(
    patched,
    str(path),
    "exec",
)

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup = path.with_name(
    path.name
    + ".bak_before_s4a_"
    + stamp
)

shutil.copy2(
    path,
    backup,
)

path.write_text(
    patched,
    encoding="utf-8",
)

print("Backup:", backup)
print("Patched:", path)

for name, passed in checks.items():
    print(
        "{}: {}".format(
            name,
            "PASS" if passed else "FAIL",
        )
    )

print(
    "S4-A semantic forward support: PASS"
)

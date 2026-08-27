from pathlib import Path
import datetime
import re
import shutil


path = Path("lib/_utils.py")

if not path.exists():
    raise FileNotFoundError(str(path))

text = path.read_text(
    encoding="utf-8",
)


# ------------------------------------------------------------
# 已正确安装时直接退出，避免重复修改
# ------------------------------------------------------------
already_installed = (
    "requires_main_logits" in text
    and re.search(
        r"self\.small_refinement_head\(\s*"
        r"x_c1\s*,\s*"
        r"x_c2\s*,\s*"
        r"final_logits\s*,?\s*"
        r"\)",
        text,
        flags=re.DOTALL,
    )
)

if already_installed:
    print("S3-E forward support already installed.")
    raise SystemExit(0)


# ------------------------------------------------------------
# 兼容两种常见原始写法：
#
# small_residual_logits = (
#     self.small_refinement_head(
#         x_c1,
#         x_c2,
#     )
# )
#
# 或：
#
# small_residual_logits = self.small_refinement_head(x_c1, x_c2)
# ------------------------------------------------------------
patterns = [
    re.compile(
        r"(?P<indent>^[ \t]*)"
        r"small_residual_logits\s*=\s*"
        r"\(\s*"
        r"self\.small_refinement_head\(\s*"
        r"x_c1\s*,\s*"
        r"x_c2\s*,?\s*"
        r"\)\s*"
        r"\)",
        flags=re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"(?P<indent>^[ \t]*)"
        r"small_residual_logits\s*=\s*"
        r"self\.small_refinement_head\(\s*"
        r"x_c1\s*,\s*"
        r"x_c2\s*,?\s*"
        r"\)",
        flags=re.MULTILINE | re.DOTALL,
    ),
]


match = None

for pattern in patterns:
    candidate = pattern.search(text)

    if candidate is not None:
        match = candidate
        break


if match is None:
    locations = [
        "{}: {}".format(index, line.rstrip())
        for index, line in enumerate(
            text.splitlines(),
            start=1,
        )
        if "small_refinement_head" in line
    ]

    raise RuntimeError(
        "Could not locate the original S2 refinement call.\n"
        "Occurrences found:\n{}".format(
            "\n".join(locations)
        )
    )


indent = match.group("indent")

replacement = (
    indent
    + "if getattr(\n"
    + indent
    + "    self.small_refinement_head,\n"
    + indent
    + "    'requires_main_logits',\n"
    + indent
    + "    False,\n"
    + indent
    + "):\n"
    + indent
    + "    small_residual_logits = (\n"
    + indent
    + "        self.small_refinement_head(\n"
    + indent
    + "            x_c1,\n"
    + indent
    + "            x_c2,\n"
    + indent
    + "            final_logits,\n"
    + indent
    + "        )\n"
    + indent
    + "    )\n"
    + indent
    + "else:\n"
    + indent
    + "    small_residual_logits = (\n"
    + indent
    + "        self.small_refinement_head(\n"
    + indent
    + "            x_c1,\n"
    + indent
    + "            x_c2,\n"
    + indent
    + "        )\n"
    + indent
    + "    )"
)


patched = (
    text[:match.start()]
    + replacement
    + text[match.end():]
)


# ------------------------------------------------------------
# 语义检查，不检查固定缩进
# ------------------------------------------------------------
checks = {
    "conditional marker":
        "requires_main_logits" in patched,

    "three-argument S3-E call":
        re.search(
            r"self\.small_refinement_head\(\s*"
            r"x_c1\s*,\s*"
            r"x_c2\s*,\s*"
            r"final_logits\s*,?\s*"
            r"\)",
            patched,
            flags=re.DOTALL,
        ) is not None,

    "two-argument compatibility call":
        re.search(
            r"else\s*:\s*"
            r"small_residual_logits\s*=\s*"
            r"\(\s*"
            r"self\.small_refinement_head\(\s*"
            r"x_c1\s*,\s*"
            r"x_c2\s*,?\s*"
            r"\)",
            patched,
            flags=re.DOTALL,
        ) is not None,
}


failed_checks = [
    name
    for name, passed in checks.items()
    if not passed
]

if failed_checks:
    raise RuntimeError(
        "Semantic validation failed: {}".format(
            failed_checks
        )
    )


# Python 语法检查
compile(
    patched,
    str(path),
    "exec",
)


stamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup = path.with_name(
    path.name
    + ".bak_before_s3e_"
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

print("S3-E forward support installation: PASS")

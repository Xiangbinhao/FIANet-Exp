from pathlib import Path
from datetime import datetime
import re
import shutil


FILES = [
    Path("train.py"),
    Path("train_s3b.py"),
]


def patch_file(path):
    if not path.exists():
        raise FileNotFoundError(str(path))

    text = path.read_text(
        encoding="utf-8"
    )

    marker = 'environ.get("FIANET_SEED"'

    if marker in text:
        print("{}: already patched".format(path))
        return

    pattern = re.compile(
        r"^(?P<indent>[ \t]*)"
        r"random\.seed\(seed\)"
        r"(?P<trailing>[ \t]*(?:#.*)?)$",
        flags=re.MULTILINE,
    )

    matches = list(pattern.finditer(text))

    if len(matches) != 1:
        locations = [
            "{}: {}".format(index, line)
            for index, line in enumerate(
                text.splitlines(),
                start=1,
            )
            if "random.seed" in line
        ]

        raise RuntimeError(
            "Expected exactly one random.seed(seed) in {}, "
            "found {}.\n{}".format(
                path,
                len(matches),
                "\n".join(locations),
            )
        )

    match = matches[0]
    indent = match.group("indent")
    original_line = match.group(0)

    replacement = (
        indent
        + 'seed = int(__import__("os").environ.get('
        + '"FIANET_SEED", seed))\n'
        + indent
        + 'print("Effective random seed: {}".format(seed))\n'
        + original_line
    )

    patched = (
        text[:match.start()]
        + replacement
        + text[match.end():]
    )

    required = [
        'environ.get("FIANET_SEED"',
        'print("Effective random seed:',
        "random.seed(seed)",
        "np.random.seed(seed)",
        "torch.manual_seed(seed)",
        "torch.cuda.manual_seed(seed)",
        "torch.cuda.manual_seed_all(seed)",
        "torch.backends.cudnn.benchmark = False",
        "torch.backends.cudnn.deterministic = True",
    ]

    missing = [
        item
        for item in required
        if item not in patched
    ]

    if missing:
        raise RuntimeError(
            "{} missing required seed operations: {}".format(
                path,
                missing,
            )
        )

    compile(
        patched,
        str(path),
        "exec",
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = path.with_name(
        path.name
        + ".bak_before_env_seed_"
        + timestamp
    )

    shutil.copy2(path, backup)

    path.write_text(
        patched,
        encoding="utf-8",
    )

    print("Backup:", backup)
    print("Patched:", path)


for file_path in FILES:
    patch_file(file_path)

print("FIANET environment-seed support: PASS")

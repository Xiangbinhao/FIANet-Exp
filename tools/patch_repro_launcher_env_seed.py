from pathlib import Path
from datetime import datetime
import re
import shutil


path = Path("run_repro_e0_s3b_3seed.sh")

if not path.exists():
    raise FileNotFoundError(str(path))

text = path.read_text(
    encoding="utf-8"
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


# Replace the obsolete CLI --seed preflight.
lines = text.splitlines(True)

start_index = None
end_index = None

for index, line in enumerate(lines):
    if (
        "if ! python train.py --help" in line
        and "--seed" in line
    ):
        start_index = index
        break

if start_index is not None:
    for index in range(start_index, len(lines)):
        if 'echo "Seed argument audit: PASS"' in lines[index]:
            end_index = index
            break

    if end_index is None:
        raise RuntimeError(
            "Found old seed preflight start but not its end."
        )

    replacement = [
        'if ! grep -q \'FIANET_SEED\' train.py; then\n',
        '    echo "ERROR: train.py does not support FIANET_SEED."\n',
        '    exit 1\n',
        'fi\n',
        '\n',
        'if ! grep -q \'FIANET_SEED\' train_s3b.py; then\n',
        '    echo "ERROR: train_s3b.py does not support FIANET_SEED."\n',
        '    exit 1\n',
        'fi\n',
        '\n',
        'echo "FIANET environment-seed audit: PASS"\n',
    ]

    lines = (
        lines[:start_index]
        + replacement
        + lines[end_index + 1:]
    )

text = "".join(lines)


# Remove CLI --seed arguments from train and test commands.
text = re.sub(
    r'^[ \t]*--seed "\$\{seed\}" \\\s*\n',
    "",
    text,
    flags=re.MULTILINE,
)


# Add FIANET_SEED to each environment block.
python_hash_line = (
    '        PYTHONHASHSEED="${seed}" \\\n'
)

fianet_seed_line = (
    '        FIANET_SEED="${seed}" \\\n'
)

if fianet_seed_line not in text:
    occurrences = text.count(python_hash_line)

    if occurrences < 2:
        raise RuntimeError(
            "Expected multiple PYTHONHASHSEED environment lines, "
            "found {}".format(occurrences)
        )

    text = text.replace(
        python_hash_line,
        python_hash_line + fianet_seed_line,
    )


required = [
    "FIANET environment-seed audit: PASS",
    'FIANET_SEED="${seed}"',
    "python -u train.py",
    "python -u train_s3b.py",
    "python -u test.py",
    "python -u test_s3b.py",
]

missing = [
    item
    for item in required
    if item not in text
]

if missing:
    raise RuntimeError(
        "Patched launcher missing: {}".format(
            missing
        )
    )

if re.search(
    r'^[ \t]*--seed "\$\{seed\}"',
    text,
    flags=re.MULTILINE,
):
    raise RuntimeError(
        "CLI --seed remains in launcher."
    )

path.write_text(
    text,
    encoding="utf-8",
)

print("Backup:", backup)
print("Patched:", path)
print("Environment-seed launcher patch: PASS")

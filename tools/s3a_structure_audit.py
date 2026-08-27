from pathlib import Path
import re


FILES = [
    Path("args.py"),
    Path("lib/high_resolution_small_refinement.py"),
    Path("lib/_utils.py"),
    Path("lib/segmentation.py"),
    Path("train_s2.py"),
]

OUTPUT = Path("logs/s3a_structure_audit.txt")


def print_header(file, title):
    file.write("\n")
    file.write("=" * 80 + "\n")
    file.write(title + "\n")
    file.write("=" * 80 + "\n")


def numbered_lines(text):
    return [
        "{:5d}: {}".format(index + 1, line)
        for index, line in enumerate(
            text.splitlines()
        )
    ]


def find_context(
    lines,
    patterns,
    before=12,
    after=30,
):
    regexes = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]

    selected = set()

    for index, line in enumerate(lines):
        if any(
            regex.search(line)
            for regex in regexes
        ):
            start = max(0, index - before)
            end = min(
                len(lines),
                index + after + 1,
            )

            selected.update(
                range(start, end)
            )

    return sorted(selected)


def write_context(
    file,
    lines,
    indices,
):
    previous = None

    for index in indices:
        if (
            previous is not None
            and index > previous + 1
        ):
            file.write("\n...\n\n")

        file.write(
            "{:5d}: {}\n".format(
                index + 1,
                lines[index],
            )
        )

        previous = index


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    encoding="utf-8",
) as output:
    for path in FILES:
        print_header(
            output,
            "FILE: {}".format(path),
        )

        if not path.exists():
            output.write(
                "FILE NOT FOUND\n"
            )
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        lines = text.splitlines()

        output.write(
            "TOTAL LINES: {}\n".format(
                len(lines)
            )
        )

        # The refinement module is usually compact enough
        # to include in full.
        if (
            path.name
            == "high_resolution_small_refinement.py"
        ):
            output.write(
                "\n--- FULL FILE ---\n"
            )

            output.write(
                "\n".join(
                    numbered_lines(text)
                )
            )

            output.write("\n")
            continue

        if path.name == "args.py":
            patterns = [
                r"use.*high",
                r"high.*refine",
                r"small.*refine",
                r"project.*channel",
                r"hidden.*channel",
                r"return parser",
            ]

        elif path.name == "_utils.py":
            patterns = [
                r"class .*LAVT",
                r"def forward",
                r"small_refinement_head",
                r"x_c1",
                r"x_c2",
                r"residual",
                r"final_logits",
                r"return ",
            ]

        elif path.name == "segmentation.py":
            patterns = [
                r"high_resolution",
                r"small_refinement",
                r"def _segm_lavt_one",
                r"model = base_model",
                r"use.*high",
                r"return model",
            ]

        elif path.name == "train_s2.py":
            patterns = [
                r"^def train_one_epoch",
                r"model\(",
                r"criterion\(",
                r"loss =",
                r"loss_value",
                r"autocast",
                r"GradScaler",
                r"optimizer",
                r"small_refinement_head",
                r"params_to_optimize",
                r"for epoch in",
                r"evaluate\(",
                r"resume_epoch",
            ]

        else:
            patterns = []

        indices = find_context(
            lines,
            patterns,
            before=15,
            after=45,
        )

        write_context(
            output,
            lines,
            indices,
        )

print("Saved:", OUTPUT)

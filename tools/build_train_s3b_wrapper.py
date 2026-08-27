from pathlib import Path
import re
import shutil
import datetime


source_path = Path("train_s2.py")
target_path = Path("train_s3b.py")

if not source_path.exists():
    raise FileNotFoundError(str(source_path))

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if target_path.exists():
    backup_path = Path(
        "train_s3b.py.bak_{}".format(stamp)
    )
    shutil.copy2(target_path, backup_path)
    print("Backup:", target_path, "->", backup_path)

text = source_path.read_text(encoding="utf-8")


# ============================================================
# 1. 添加包装器导入
# ============================================================
import_line = (
    "from lib.bounded_gated_small_refinement import "
    "BoundedGatedResidualWrapper\n"
)

if import_line not in text:
    torch_import_match = re.search(
        r"^import torch[ \t]*$",
        text,
        flags=re.MULTILINE,
    )

    if torch_import_match is None:
        raise RuntimeError(
            "Could not find 'import torch' in train_s2.py"
        )

    line_end = text.find(
        "\n",
        torch_import_match.end(),
    )

    if line_end == -1:
        raise RuntimeError(
            "Malformed import section in train_s2.py"
        )

    text = (
        text[:line_end + 1]
        + import_line
        + text[line_end + 1:]
    )

print("[1/4] Added S3-B wrapper import")


# ============================================================
# 2. 在 checkpoint 加载和 optimizer 构建之前包装 S2 head
#
# 通过 '# parameters to optimize' 定位 main() 中的优化器区域，
# 再向前寻找最近的 resume checkpoint 判断。
# ============================================================
parameters_marker = "    # parameters to optimize"

parameters_position = text.find(parameters_marker)

if parameters_position == -1:
    raise RuntimeError(
        "Could not find '# parameters to optimize' "
        "in train_s2.py"
    )

resume_position = text.rfind(
    "    if args.resume:",
    0,
    parameters_position,
)

if resume_position == -1:
    raise RuntimeError(
        "Could not find checkpoint-loading resume block "
        "before optimizer parameters"
    )

wrapper_marker = (
    "S3-B bounded gated residual wrapper enabled"
)

if wrapper_marker in text:
    raise RuntimeError(
        "Source train_s2.py unexpectedly already contains S3-B"
    )

wrapper_code = '''    # ========================================================
    # S3-B: wrap the existing S2 high-resolution residual head
    # before checkpoint loading and optimizer construction.
    # ========================================================
    if getattr(
        model,
        'small_refinement_head',
        None,
    ) is None:
        raise RuntimeError(
            "S3-B requires --use-small-refine, but "
            "model.small_refinement_head is None."
        )

    _s3b_base_head = model.small_refinement_head

    try:
        _s3b_device = next(
            _s3b_base_head.parameters()
        ).device
    except StopIteration:
        _s3b_device = device

    model.small_refinement_head = (
        BoundedGatedResidualWrapper(
            base_head=_s3b_base_head,
            alpha_init=0.10,
            gate_init=0.20,
        ).to(_s3b_device)
    )

    print(
        "S3-B bounded gated residual wrapper enabled: "
        "alpha_init=0.10, gate_init=0.20"
    )

'''

text = (
    text[:resume_position]
    + wrapper_code
    + text[resume_position:]
)

print("[2/4] Inserted wrapper before checkpoint/optimizer")


# ============================================================
# 3. 修改优化器日志名称
# 原有多行 optimizer block 不需要修改，它会自动收集包装器全部参数。
# ============================================================
old_message = (
    "S2 optimizer added refinement parameters: "
)

new_message = (
    "S3-B optimizer added refinement parameters: "
)

message_count = text.count(old_message)

if message_count != 1:
    raise RuntimeError(
        "Expected exactly one S2 optimizer message, found {}".format(
            message_count
        )
    )

text = text.replace(
    old_message,
    new_message,
    1,
)

print("[3/4] Updated optimizer diagnostic message")


# ============================================================
# 4. 完整性检查并写入
# ============================================================
required_items = [
    "BoundedGatedResidualWrapper",
    "S3-B bounded gated residual wrapper enabled",
    "model.small_refinement_head.parameters()",
    "S3-B optimizer added refinement parameters:",
    "optimizer = torch.optim.AdamW",
]

missing_items = [
    item
    for item in required_items
    if item not in text
]

if missing_items:
    raise RuntimeError(
        "Generated train_s3b.py is missing: {}".format(
            missing_items
        )
    )

target_path.write_text(
    text,
    encoding="utf-8",
)

print("[4/4] Created:", target_path)
print("S3-B training-script generation: PASS")

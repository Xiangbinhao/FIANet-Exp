from pathlib import Path
import ast
import datetime
import shutil


source = Path("train_s2.py")
target = Path("train_s3b.py")

if not source.exists():
    raise FileNotFoundError(source)

text = source.read_text(encoding="utf-8")
ast.parse(text)

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def replace_once(content, old, new, label):
    count = content.count(old)

    if count != 1:
        raise RuntimeError(
            "{}: expected exactly one match, found {}".format(
                label,
                count,
            )
        )

    return content.replace(old, new, 1)


# ==========================================================
# 1. Import S3-B loss
# ==========================================================
import_line = (
    "from loss.prediction_aware_hard_pixel_loss "
    "import PredictionAwareHardPixelLoss\n"
)

if import_line not in text:
    text = replace_once(
        text,
        "import torch\n",
        "import torch\n" + import_line,
        "insert S3-B loss import",
    )


# ==========================================================
# 2. Add residual-output capture class
# ==========================================================
capture_class = '''
class S3BResidualCapture(object):
    """
    Capture the raw output of the S2 high-resolution
    residual refinement head.
    """

    def __init__(self, module):
        if module is None:
            raise ValueError(
                "S3-B requires small_refinement_head"
            )

        self.output = None
        self.enabled = False
        self.handle = module.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        if self.enabled:
            self.output = output

    def enable(self):
        self.output = None
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.output = None

    def clear(self):
        self.output = None

    def pop(self):
        output = self.output
        self.output = None
        return output

    def close(self):
        self.disable()
        self.handle.remove()


'''

if "class S3BResidualCapture" not in text:
    text = replace_once(
        text,
        "def train_one_epoch(\n",
        capture_class + "def train_one_epoch(\n",
        "insert residual capture class",
    )


# ==========================================================
# 3. Extend train_one_epoch signature
# ==========================================================
old_signature = '''def train_one_epoch(
        model,
        criterion,
        optimizer,
        data_loader,
        lr_scheduler,
        epoch,
        print_freq,
        iterations,
        bert_model,
        scaler):
'''

new_signature = '''def train_one_epoch(
        model,
        criterion,
        optimizer,
        data_loader,
        lr_scheduler,
        epoch,
        print_freq,
        iterations,
        bert_model,
        scaler,
        s3b_criterion,
        s3b_capture):
'''

text = replace_once(
    text,
    old_signature,
    new_signature,
    "extend train_one_epoch signature",
)


# ==========================================================
# 4. Frozen-backbone training mode
# ==========================================================
old_train_mode = '''    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
'''

new_train_mode = '''    if s3b_criterion is None or s3b_capture is None:
        raise RuntimeError(
            "S3-B criterion or residual capture is missing"
        )

    # Keep all frozen modules deterministic.
    model.eval()

    if bert_model is not None:
        bert_model.eval()

    # Only the lightweight S2 refinement head is trainable.
    model.small_refinement_head.train()

    # Preserve the BatchNorm statistics learned by the
    # standalone S2 checkpoint during low-data fine-tuning.
    for module in model.small_refinement_head.modules():
        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()

    s3b_capture.enable()

    metric_logger = utils.MetricLogger(delimiter="  ")
'''

text = replace_once(
    text,
    old_train_mode,
    new_train_mode,
    "replace training mode",
)


# ==========================================================
# 5. Clear captured output before each forward pass
# ==========================================================
old_autocast = '''        # BERT、FIANet 前向传播和损失均使用自动混合精度。
        with torch.cuda.amp.autocast(enabled=True):
'''

new_autocast = '''        s3b_capture.clear()

        # BERT、FIANet 前向传播和损失均使用自动混合精度。
        with torch.cuda.amp.autocast(enabled=True):
'''

text = replace_once(
    text,
    old_autocast,
    new_autocast,
    "clear capture before forward",
)


# ==========================================================
# 6. Replace original S2 loss
# ==========================================================
old_loss = '''            loss = criterion(output, target)
'''

new_loss = '''            if isinstance(output, (tuple, list)):
                final_output = output[0]
            else:
                final_output = output

            raw_residual_logits = s3b_capture.pop()

            if raw_residual_logits is None:
                raise RuntimeError(
                    "S3-B failed to capture the output of "
                    "small_refinement_head"
                )

            # Match the actual S2 forward interpolation.
            residual_logits = torch.nn.functional.interpolate(
                raw_residual_logits,
                size=final_output.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            if residual_logits.shape != final_output.shape:
                raise RuntimeError(
                    "S3-B residual/final shape mismatch: "
                    "{} versus {}".format(
                        tuple(residual_logits.shape),
                        tuple(final_output.shape),
                    )
                )

            # Use the complete current S2 prediction as
            # the detached confidence source for selecting
            # hard-positive and hard-negative pixels.
            gate_logits = final_output.detach()

            base_loss = criterion(
                final_output,
                target,
            )

            (
                s3b_auxiliary_loss,
                s3b_stats,
            ) = s3b_criterion(
                base_logits=gate_logits,
                final_logits=final_output,
                residual_logits=residual_logits,
                target=target,
            )

            loss = base_loss + s3b_auxiliary_loss
            output = final_output
'''

# Replace the loss only inside train_one_epoch.
train_start = text.find("def train_one_epoch(")
main_start = text.find(
    "\ndef main(args):",
    train_start,
)

if train_start < 0 or main_start < 0:
    raise RuntimeError(
        "Cannot locate train_one_epoch/main boundary"
    )

prefix = text[:train_start]
train_block = text[train_start:main_start]
suffix = text[main_start:]

loss_count = train_block.count(old_loss)

if loss_count != 1:
    raise RuntimeError(
        "Expected exactly one training loss match, "
        "found {}".format(loss_count)
    )

train_block = train_block.replace(
    old_loss,
    new_loss,
    1,
)

text = prefix + train_block + suffix


# ==========================================================
# 7. Add diagnostic logging
# ==========================================================
old_loss_value = '''        loss_value = loss.detach().item()
        train_loss += loss_value
        iterations += 1

        metric_logger.update(
            loss=loss_value,
            lr=optimizer.param_groups[0]["lr"]
        )
'''

new_loss_value = '''        loss_value = loss.detach().item()
        base_loss_value = base_loss.detach().item()
        s3b_auxiliary_value = (
            s3b_auxiliary_loss.detach().item()
        )

        train_loss += loss_value
        iterations += 1

        metric_logger.update(
            loss=loss_value,
            base_loss=base_loss_value,
            s3b_aux=s3b_auxiliary_value,
            s3b_positive=s3b_stats["positive_loss"],
            s3b_negative=s3b_stats["negative_loss"],
            s3b_reg=s3b_stats["residual_reg_loss"],
            s3b_active=s3b_stats["active_count"],
            s3b_hard_pos=(
                s3b_stats["hard_positive_pixels"]
            ),
            s3b_hard_neg=(
                s3b_stats["hard_negative_pixels"]
            ),
            lr=optimizer.param_groups[0]["lr"],
        )
'''

text = replace_once(
    text,
    old_loss_value,
    new_loss_value,
    "add S3-B metric logging",
)


# ==========================================================
# 8. Release S3-B tensors each batch
# ==========================================================
old_delete = '''        del loss
        del output
        del data
'''

new_delete = '''        del loss
        del base_loss
        del s3b_auxiliary_loss
        del s3b_stats
        del gate_logits
        del residual_logits
        del raw_residual_logits
        del final_output
        del output
        del data
'''

text = replace_once(
    text,
    old_delete,
    new_delete,
    "add S3-B tensor cleanup",
)


# ==========================================================
# 9. Disable capture at end of train_one_epoch
#
# train_s2.py has no explicit return in train_one_epoch.
# Insert immediately before def main(args).
# ==========================================================
main_anchor = "\ndef main(args):"

if main_anchor not in text:
    raise RuntimeError(
        "Cannot locate `def main(args):`"
    )

train_start = text.index("def train_one_epoch(")
main_start = text.index(main_anchor, train_start)

train_block = text[train_start:main_start]

if "    s3b_capture.disable()\n" not in train_block:
    train_block = (
        train_block.rstrip()
        + "\n\n    # Prevent capture during validation.\n"
        + "    s3b_capture.disable()\n"
    )

text = (
    text[:train_start]
    + train_block
    + text[main_start:]
)


# ==========================================================
# 10. Replace the complete original optimizer-parameter block
# ==========================================================
parameter_start_marker = '''    # parameters to optimize
'''

optimizer_start_marker = '''    optimizer = torch.optim.AdamW(params_to_optimize,
'''

parameter_start = text.find(parameter_start_marker)

if parameter_start < 0:
    raise RuntimeError(
        "Cannot locate optimizer parameter block"
    )

optimizer_start = text.find(
    optimizer_start_marker,
    parameter_start,
)

if optimizer_start < 0:
    raise RuntimeError(
        "Cannot locate AdamW optimizer creation"
    )

s3b_setup = '''    # ======================================================
    # S3-B: initialize from the standalone S2 checkpoint and
    # fine-tune only the 8,802-parameter refinement head.
    # ======================================================
    if args.resume:
        raise ValueError(
            "S3-B pilot uses S3B_INIT_CHECKPOINT and does "
            "not support --resume"
        )

    s3b_checkpoint_path = os.environ.get(
        "S3B_INIT_CHECKPOINT",
        "",
    )

    if not s3b_checkpoint_path:
        raise RuntimeError(
            "Environment variable S3B_INIT_CHECKPOINT "
            "is not set"
        )

    if not os.path.isfile(s3b_checkpoint_path):
        raise FileNotFoundError(
            s3b_checkpoint_path
        )

    s3b_checkpoint = torch.load(
        s3b_checkpoint_path,
        map_location="cpu",
    )

    if "model" not in s3b_checkpoint:
        raise KeyError(
            "S2 checkpoint does not contain key `model`"
        )

    model.load_state_dict(
        s3b_checkpoint["model"],
        strict=True,
    )

    if (
        bert_model is not None
        and "bert_model" in s3b_checkpoint
    ):
        bert_model.load_state_dict(
            s3b_checkpoint["bert_model"],
            strict=True,
        )

    print(
        "S3-B initialized from:",
        s3b_checkpoint_path,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    if bert_model is not None:
        for parameter in bert_model.parameters():
            parameter.requires_grad = False

    if getattr(
        model,
        "small_refinement_head",
        None,
    ) is None:
        raise RuntimeError(
            "S3-B requires --use-small-refine"
        )

    for parameter in (
        model.small_refinement_head.parameters()
    ):
        parameter.requires_grad = True

    small_refinement_parameters = [
        parameter
        for parameter
        in model.small_refinement_head.parameters()
        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in small_refinement_parameters
    )

    if trainable_parameter_count != 8802:
        raise RuntimeError(
            "Unexpected S3-B trainable parameter count: "
            "{}".format(trainable_parameter_count)
        )

    params_to_optimize = [{
        "params": small_refinement_parameters,
    }]

    print(
        "S3-B trainable parameters:",
        trainable_parameter_count,
    )

    s3b_criterion = PredictionAwareHardPixelLoss(
        positive_threshold=float(
            os.environ.get(
                "S3B_POSITIVE_THRESHOLD",
                "0.40",
            )
        ),
        negative_threshold=float(
            os.environ.get(
                "S3B_NEGATIVE_THRESHOLD",
                "0.60",
            )
        ),
        positive_weight=float(
            os.environ.get(
                "S3B_POSITIVE_WEIGHT",
                "0.02",
            )
        ),
        negative_weight=float(
            os.environ.get(
                "S3B_NEGATIVE_WEIGHT",
                "0.02",
            )
        ),
        residual_reg_weight=float(
            os.environ.get(
                "S3B_RESIDUAL_REG_WEIGHT",
                "0.002",
            )
        ),
        ring_radius=int(
            os.environ.get(
                "S3B_RING_RADIUS",
                "6",
            )
        ),
        tiny_max_ratio=0.001,
        small_max_ratio=0.005,
    )

    s3b_capture = S3BResidualCapture(
        model.small_refinement_head
    )

    print(
        "S3-B enabled: "
        "pos_thr={:.2f}, neg_thr={:.2f}, "
        "pos_w={:.4f}, neg_w={:.4f}, "
        "reg_w={:.4f}, radius={}".format(
            s3b_criterion.positive_threshold,
            s3b_criterion.negative_threshold,
            s3b_criterion.positive_weight,
            s3b_criterion.negative_weight,
            s3b_criterion.residual_reg_weight,
            s3b_criterion.ring_radius,
        )
    )

'''

text = (
    text[:parameter_start]
    + s3b_setup
    + text[optimizer_start:]
)


# ==========================================================
# 11. Pass S3-B objects to train_one_epoch
# ==========================================================
old_train_call = '''        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, iterations, bert_model, scaler)
'''

new_train_call = '''        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            lr_scheduler,
            epoch,
            args.print_freq,
            iterations,
            bert_model,
            scaler,
            s3b_criterion,
            s3b_capture,
        )
'''

text = replace_once(
    text,
    old_train_call,
    new_train_call,
    "pass S3-B objects into training loop",
)


# ==========================================================
# 12. Final validation
# ==========================================================
ast.parse(text)

required_tokens = [
    "class S3BResidualCapture",
    "PredictionAwareHardPixelLoss",
    "S3B_INIT_CHECKPOINT",
    "S3-B trainable parameters",
    "trainable_parameter_count != 8802",
    "s3b_capture.disable()",
    "s3b_criterion,",
    "s3b_capture,",
]

missing = [
    token
    for token in required_tokens
    if token not in text
]

if missing:
    raise RuntimeError(
        "S3-B static validation failed: {}".format(
            missing
        )
    )

target.write_text(
    text,
    encoding="utf-8",
)

print("Created:", target)
print("Original train_s2.py unchanged.")
print("S3-B exact-anchor generation: PASS")

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from args import get_parser
from loss.loss import Loss
from loss.foreground_size_aux_loss_v2 import (
    ScheduledForegroundSizeAuxiliaryLoss,
)


def main():
    args = get_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")

    base_loss = Loss(weight=0.1)

    loss_fn = ScheduledForegroundSizeAuxiliaryLoss(
        base_loss=base_loss,
        tiny_max_lambda=(
            args.s1cv2_tiny_max_lambda
        ),
        small_max_lambda=(
            args.s1cv2_small_max_lambda
        ),
        tiny_max_ratio=(
            args.s1cv2_tiny_max_ratio
        ),
        small_max_ratio=(
            args.s1cv2_small_max_ratio
        ),
        warmup_epochs=(
            args.s1cv2_warmup_epochs
        ),
        ramp_epochs=(
            args.s1cv2_ramp_epochs
        ),
        hold_epochs=(
            args.s1cv2_hold_epochs
        ),
        decay_epochs=(
            args.s1cv2_decay_epochs
        ),
    ).cuda()

    expected_total = (
        args.s1cv2_warmup_epochs
        + args.s1cv2_ramp_epochs
        + args.s1cv2_hold_epochs
        + args.s1cv2_decay_epochs
    )

    if expected_total != args.epochs:
        raise RuntimeError(
            "Schedule total does not equal --epochs."
        )

    expected = {
        0: 0.0,
        4: 0.0,
        5: 0.1,
        9: 0.5,
        14: 1.0,
        15: 1.0,
        30: 1.0,
        31: 1.0,
        35: 0.5,
        39: 0.0,
    }

    print(
        "\nEpoch  Factor  TinyLambda  SmallLambda"
    )

    for epoch in range(args.epochs):
        loss_fn.set_epoch(epoch)

        tiny_lambda, small_lambda, factor = (
            loss_fn.get_current_lambdas()
        )

        print(
            "{:>5}  {:>6.3f}  {:>10.4f}  "
            "{:>11.4f}".format(
                epoch,
                factor,
                tiny_lambda,
                small_lambda,
            )
        )

    for epoch, expected_factor in expected.items():
        actual = loss_fn.get_schedule_factor(
            epoch
        )

        if abs(
            actual - expected_factor
        ) > 1e-7:
            raise RuntimeError(
                "Schedule mismatch at epoch {}: "
                "{} != {}".format(
                    epoch,
                    actual,
                    expected_factor,
                )
            )

    print("\nSchedule checkpoints: PASS")

    target = torch.zeros(
        4,
        100,
        100,
        dtype=torch.long,
        device=device,
    )

    target[1, 0, 0:5] = 1
    target[2, 0, 0:30] = 1
    target[3, 0:10, 0:10] = 1

    torch.manual_seed(2401)

    logits = torch.randn(
        4,
        2,
        100,
        100,
        device=device,
    )

    loss_fn.set_epoch(0)

    original_value = base_loss(
        logits,
        target,
    )

    warmup_value = loss_fn(
        logits,
        target,
    )

    difference = abs(
        float(original_value.item())
        - float(warmup_value.item())
    )

    print(
        "Original E0 loss: {:.10f}".format(
            float(original_value.item())
        )
    )

    print(
        "Warm-up loss: {:.10f}".format(
            float(warmup_value.item())
        )
    )

    print(
        "Difference: {:.10e}".format(
            difference
        )
    )

    if difference > 1e-7:
        raise RuntimeError(
            "Warm-up does not reproduce E0."
        )

    print("Warm-up identity: PASS")

    loss_fn.set_epoch(14)

    coefficients, ratios, _ = (
        loss_fn.compute_auxiliary_coefficients(
            target
        )
    )

    expected_coefficients = [
        0.0,
        args.s1cv2_tiny_max_lambda,
        args.s1cv2_small_max_lambda,
        0.0,
    ]

    for index, expected_value in enumerate(
        expected_coefficients
    ):
        actual_value = float(
            coefficients[index].item()
        )

        print(
            "sample={} ratio={:.6f} "
            "coefficient={:.4f}".format(
                index,
                float(ratios[index].item()),
                actual_value,
            )
        )

        if abs(
            actual_value - expected_value
        ) > 1e-7:
            raise RuntimeError(
                "Coefficient mismatch."
            )

    print("Coefficient assignment: PASS")

    loss_fn.set_epoch(9)

    train_logits = (
        logits.clone()
        .detach()
        .requires_grad_(True)
    )

    value = loss_fn(
        train_logits,
        target,
    )

    value.backward()

    if train_logits.grad is None:
        raise RuntimeError("No gradient.")

    if not torch.isfinite(
        train_logits.grad
    ).all():
        raise RuntimeError(
            "Gradient contains NaN/Inf."
        )

    print("Forward/backward: PASS")
    print("All S1-Cv2 audits passed.")


if __name__ == "__main__":
    main()

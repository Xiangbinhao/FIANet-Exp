import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from args import get_parser
from loss.loss import Loss
from loss.foreground_size_aux_loss import (
    SIZE_GROUP_NAMES,
    ForegroundSizeAuxiliaryLoss,
)
from train_s1c import (
    get_dataset,
    get_transform,
)


def build_synthetic_target(device):
    """
    Five synthetic 100x100 samples:

      sample 0: empty
      sample 1: Tiny, 5 pixels = 0.0005
      sample 2: Small, 30 pixels = 0.0030
      sample 3: Medium, 100 pixels = 0.0100
      sample 4: Large, 2500 pixels = 0.2500
    """
    target = torch.zeros(
        5,
        100,
        100,
        dtype=torch.long,
        device=device,
    )

    target[1, 0, 0:5] = 1
    target[2, 0, 0:30] = 1
    target[3, 0:10, 0:10] = 1
    target[4, 0:50, 0:50] = 1

    return target


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Run the S1-C audit on CUDA because "
            "the original E0 Loss uses CUDA."
        )

    device = torch.device("cuda")

    torch.manual_seed(2401)
    torch.cuda.manual_seed_all(2401)

    target = build_synthetic_target(device)

    logits = torch.randn(
        5,
        2,
        100,
        100,
        device=device,
        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # 1. Zero-lambda identity check
    # --------------------------------------------------------
    base_loss_function = Loss(weight=0.1)

    identity_loss_function = (
        ForegroundSizeAuxiliaryLoss(
            base_loss=base_loss_function,
            tiny_lambda=0.0,
            small_lambda=0.0,
            tiny_max_ratio=(
                args.s1c_tiny_max_ratio
            ),
            small_max_ratio=(
                args.s1c_small_max_ratio
            ),
        ).cuda()
    )

    original_loss = base_loss_function(
        logits,
        target,
    )

    identity_loss = identity_loss_function(
        logits,
        target,
    )

    absolute_difference = abs(
        float(original_loss.item())
        - float(identity_loss.item())
    )

    print(
        "\n========== S1-C IDENTITY AUDIT =========="
    )

    print(
        "Original E0 loss: {:.10f}".format(
            float(original_loss.item())
        )
    )

    print(
        "S1-C zero-lambda loss: {:.10f}".format(
            float(identity_loss.item())
        )
    )

    print(
        "Absolute difference: {:.10e}".format(
            absolute_difference
        )
    )

    if absolute_difference > 1e-7:
        raise RuntimeError(
            "S1-C zero-lambda mode does not "
            "reproduce E0."
        )

    print("Identity equivalence: PASS")
    print(
        "=========================================\n"
    )


    # --------------------------------------------------------
    # 2. Group and coefficient audit
    # --------------------------------------------------------
    configured_loss_function = (
        ForegroundSizeAuxiliaryLoss(
            base_loss=base_loss_function,
            tiny_lambda=args.s1c_tiny_lambda,
            small_lambda=args.s1c_small_lambda,
            tiny_max_ratio=(
                args.s1c_tiny_max_ratio
            ),
            small_max_ratio=(
                args.s1c_small_max_ratio
            ),
        ).cuda()
    )

    (
        coefficients,
        area_ratios,
        group_ids,
    ) = (
        configured_loss_function
        .compute_auxiliary_coefficients(
            target
        )
    )

    expected_names = [
        "empty",
        "tiny",
        "small",
        "medium_or_large",
        "medium_or_large",
    ]

    actual_names = [
        SIZE_GROUP_NAMES[int(group_id)]
        for group_id in
        group_ids.detach().cpu()
    ]

    print(
        "========== S1-C GROUP AUDIT =========="
    )

    for index in range(target.shape[0]):
        print(
            "sample={} group={:<16s} "
            "ratio={:.6f} coefficient={:.3f}"
            .format(
                index,
                actual_names[index],
                float(area_ratios[index]),
                float(coefficients[index]),
            )
        )

    if actual_names != expected_names:
        raise RuntimeError(
            "Unexpected group assignments: {}"
            .format(actual_names)
        )

    expected_coefficients = [
        0.0,
        args.s1c_tiny_lambda,
        args.s1c_small_lambda,
        0.0,
        0.0,
    ]

    for actual, expected in zip(
        coefficients.detach().cpu(),
        expected_coefficients,
    ):
        if abs(
            float(actual) - float(expected)
        ) > 1e-7:
            raise RuntimeError(
                "Unexpected S1-C coefficient."
            )

    print("Group assignment: PASS")
    print("Coefficient assignment: PASS")
    print(
        "======================================\n"
    )


    # --------------------------------------------------------
    # 3. Forward/backward audit
    # --------------------------------------------------------
    train_logits = (
        logits.clone()
        .detach()
        .requires_grad_(True)
    )

    base_value = base_loss_function(
        train_logits,
        target,
    )

    foreground_dice = (
        configured_loss_function
        .compute_foreground_dice_per_sample(
            train_logits,
            target,
        )
    )

    expected_auxiliary = (
        coefficients
        * foreground_dice
    ).mean()

    total_value = configured_loss_function(
        train_logits,
        target,
    )

    measured_auxiliary = (
        total_value - base_value
    )

    auxiliary_difference = abs(
        float(measured_auxiliary.item())
        - float(expected_auxiliary.item())
    )

    total_value.backward()

    gradient = train_logits.grad

    if gradient is None:
        raise RuntimeError(
            "S1-C produced no gradient."
        )

    if not torch.isfinite(gradient).all():
        raise RuntimeError(
            "S1-C gradient contains NaN/Inf."
        )

    print(
        "========== S1-C BACKWARD AUDIT =========="
    )

    print(
        "Base loss: {:.10f}".format(
            float(base_value.item())
        )
    )

    print(
        "Expected auxiliary: {:.10f}".format(
            float(expected_auxiliary.item())
        )
    )

    print(
        "Measured auxiliary: {:.10f}".format(
            float(measured_auxiliary.item())
        )
    )

    print(
        "Auxiliary difference: {:.10e}".format(
            auxiliary_difference
        )
    )

    print(
        "Gradient max abs: {:.10e}".format(
            float(
                gradient.abs().max().item()
            )
        )
    )

    if auxiliary_difference > 1e-6:
        raise RuntimeError(
            "S1-C auxiliary calculation mismatch."
        )

    print("Forward/backward: PASS")
    print(
        "=========================================\n"
    )


    # --------------------------------------------------------
    # 4. Real resized-target audit
    # --------------------------------------------------------
    dataset, _ = get_dataset(
        "train",
        get_transform(args=args),
        args=args,
    )

    sampler = torch.utils.data.SequentialSampler(
        dataset
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    cpu_loss_function = (
        ForegroundSizeAuxiliaryLoss(
            base_loss=base_loss_function,
            tiny_lambda=args.s1c_tiny_lambda,
            small_lambda=args.s1c_small_lambda,
            tiny_max_ratio=(
                args.s1c_tiny_max_ratio
            ),
            small_max_ratio=(
                args.s1c_small_max_ratio
            ),
        )
    )

    counter = Counter()
    coefficient_sum = 0.0
    active_count = 0
    sample_count = 0

    max_batches = 20

    for batch_index, data in enumerate(loader):
        target_batch = data[1]

        (
            batch_coefficients,
            _,
            batch_group_ids,
        ) = (
            cpu_loss_function
            .compute_auxiliary_coefficients(
                target_batch
            )
        )

        for group_id in batch_group_ids:
            counter[
                SIZE_GROUP_NAMES[
                    int(group_id.item())
                ]
            ] += 1

        coefficient_sum += float(
            batch_coefficients.sum().item()
        )

        active_count += int(
            (
                batch_coefficients > 0
            ).sum().item()
        )

        sample_count += int(
            target_batch.shape[0]
        )

        if batch_index + 1 >= max_batches:
            break

    print(
        "========== S1-C REAL-TARGET AUDIT =========="
    )

    print(
        "Inspected samples: {}".format(
            sample_count
        )
    )

    for name in SIZE_GROUP_NAMES:
        print(
            "{:<16s} count={}".format(
                name,
                counter.get(name, 0),
            )
        )

    print(
        "Aux-active samples: {} ({:.2f}%)"
        .format(
            active_count,
            100.0
            * active_count
            / max(sample_count, 1),
        )
    )

    print(
        "Mean auxiliary coefficient: {:.6f}"
        .format(
            coefficient_sum
            / max(sample_count, 1)
        )
    )

    print("Real target parsing: PASS")
    print(
        "============================================\n"
    )

    print("All S1-C audits passed.")


if __name__ == "__main__":
    main()

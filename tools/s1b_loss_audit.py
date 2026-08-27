import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from args import get_parser
from loss.loss import Loss
from loss.size_aware_loss import (
    SIZE_GROUP_NAMES,
    SizeAwareSegmentationLoss,
)
from train_s1b import (
    get_dataset,
    get_transform,
)


def build_synthetic_target(device):
    """
    Build one sample for every S0 size group.

    Spatial size = 100 x 100:
      empty  : 0 pixels
      tiny   : 5 pixels   = 0.0005
      small  : 30 pixels  = 0.0030
      medium : 100 pixels = 0.0100
      large  : 2500 pixels = 0.2500
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
            "The original FIANet Loss hardcodes CUDA; "
            "run this audit on the training GPU."
        )

    device = torch.device("cuda")

    torch.manual_seed(2401)
    torch.cuda.manual_seed_all(2401)

    target = build_synthetic_target(
        device
    )

    logits = torch.randn(
        5,
        2,
        100,
        100,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Test 1: all-one identity against original E0 loss.
    # --------------------------------------------------------
    original_loss_function = Loss(
        weight=0.1
    )

    identity_loss_function = (
        SizeAwareSegmentationLoss(
            tiny_weight=1.0,
            small_weight=1.0,
            medium_weight=1.0,
            large_weight=1.0,
            empty_weight=1.0,
            tiny_max_ratio=(
                args.s1b_tiny_max_ratio
            ),
            small_max_ratio=(
                args.s1b_small_max_ratio
            ),
            medium_max_ratio=(
                args.s1b_medium_max_ratio
            ),
            dice_mix_weight=0.1,
            log_first_batch=False,
        ).cuda()
    )

    original_loss = original_loss_function(
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
        "\n========== S1-B IDENTITY AUDIT =========="
    )

    print(
        "Original E0 loss: {:.10f}".format(
            float(original_loss.item())
        )
    )

    print(
        "S1-B all-one loss: {:.10f}".format(
            float(identity_loss.item())
        )
    )

    print(
        "Absolute difference: {:.10e}".format(
            absolute_difference
        )
    )

    if absolute_difference > 1e-5:
        raise RuntimeError(
            "S1-B all-one loss does not reproduce E0."
        )

    print(
        "Identity equivalence: PASS"
    )

    print(
        "=========================================\n"
    )


    # --------------------------------------------------------
    # Test 2: group assignment and configured weights.
    # --------------------------------------------------------
    configured_loss_function = (
        SizeAwareSegmentationLoss(
            tiny_weight=args.s1b_tiny_weight,
            small_weight=args.s1b_small_weight,
            medium_weight=(
                args.s1b_medium_weight
            ),
            large_weight=args.s1b_large_weight,
            empty_weight=args.s1b_empty_weight,
            tiny_max_ratio=(
                args.s1b_tiny_max_ratio
            ),
            small_max_ratio=(
                args.s1b_small_max_ratio
            ),
            medium_max_ratio=(
                args.s1b_medium_max_ratio
            ),
            dice_mix_weight=(
                args.s1b_dice_mix_weight
            ),
            log_first_batch=False,
        ).cuda()
    )

    (
        sample_weights,
        area_ratios,
        group_ids,
    ) = configured_loss_function.compute_sample_weights(
        target
    )

    expected_names = list(
        SIZE_GROUP_NAMES
    )

    actual_names = [
        SIZE_GROUP_NAMES[int(group_id)]
        for group_id in group_ids.detach().cpu()
    ]

    print(
        "========== S1-B GROUP AUDIT =========="
    )

    for index in range(target.shape[0]):
        print(
            "sample={} group={:<6s} "
            "ratio={:.6f} weight={:.3f}".format(
                index,
                actual_names[index],
                float(
                    area_ratios[index].item()
                ),
                float(
                    sample_weights[index].item()
                ),
            )
        )

    if actual_names != expected_names:
        raise RuntimeError(
            "Unexpected S1-B group assignments: {}"
            .format(actual_names)
        )

    print(
        "Group assignment: PASS"
    )

    print(
        "======================================\n"
    )


    # --------------------------------------------------------
    # Test 3: forward and backward.
    # --------------------------------------------------------
    train_logits = (
        logits.clone()
        .detach()
        .requires_grad_(True)
    )

    weighted_loss = configured_loss_function(
        train_logits,
        target,
    )

    weighted_loss.backward()

    gradient = train_logits.grad

    if gradient is None:
        raise RuntimeError(
            "S1-B produced no gradient."
        )

    if not torch.isfinite(gradient).all():
        raise RuntimeError(
            "S1-B gradient contains NaN or Inf."
        )

    gradient_max = float(
        gradient.abs().max().item()
    )

    if gradient_max <= 0.0:
        raise RuntimeError(
            "S1-B gradient is zero."
        )

    print(
        "========== S1-B BACKWARD AUDIT =========="
    )

    print(
        "Configured weighted loss: {:.10f}".format(
            float(weighted_loss.item())
        )
    )

    print(
        "Gradient max abs: {:.10e}".format(
            gradient_max
        )
    )

    print(
        "Forward/backward: PASS"
    )

    print(
        "=========================================\n"
    )


    # --------------------------------------------------------
    # Test 4: inspect real resized training targets.
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

    group_counter = Counter()
    weight_sum = 0.0
    sample_count = 0
    max_batches = 20

    cpu_loss_function = (
        SizeAwareSegmentationLoss(
            tiny_weight=args.s1b_tiny_weight,
            small_weight=args.s1b_small_weight,
            medium_weight=(
                args.s1b_medium_weight
            ),
            large_weight=args.s1b_large_weight,
            empty_weight=args.s1b_empty_weight,
            tiny_max_ratio=(
                args.s1b_tiny_max_ratio
            ),
            small_max_ratio=(
                args.s1b_small_max_ratio
            ),
            medium_max_ratio=(
                args.s1b_medium_max_ratio
            ),
            dice_mix_weight=(
                args.s1b_dice_mix_weight
            ),
            log_first_batch=False,
        )
    )

    for batch_index, data in enumerate(loader):
        target_batch = data[1]

        (
            batch_weights,
            _,
            batch_group_ids,
        ) = cpu_loss_function.compute_sample_weights(
            target_batch
        )

        for group_id in batch_group_ids:
            group_counter[
                SIZE_GROUP_NAMES[
                    int(group_id.item())
                ]
            ] += 1

        weight_sum += float(
            batch_weights.sum().item()
        )

        sample_count += int(
            target_batch.shape[0]
        )

        if batch_index + 1 >= max_batches:
            break

    print(
        "========== S1-B REAL-TARGET AUDIT =========="
    )

    print(
        "Inspected samples: {}".format(
            sample_count
        )
    )

    for name in SIZE_GROUP_NAMES:
        print(
            "{:<6s} count={}".format(
                name,
                group_counter.get(name, 0),
            )
        )

    print(
        "Mean sample weight: {:.6f}".format(
            weight_sum / max(sample_count, 1)
        )
    )

    print(
        "Real target parsing: PASS"
    )

    print(
        "============================================\n"
    )

    print(
        "All S1-B loss audits passed."
    )


if __name__ == "__main__":
    main()

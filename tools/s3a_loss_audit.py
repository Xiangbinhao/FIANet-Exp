import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from loss.local_positive_ring_loss import (
    LocalPositiveRingLoss,
)


def assert_close(
    actual,
    expected,
    name,
    tolerance=1.0e-7,
):
    if not math.isclose(
        float(actual),
        float(expected),
        abs_tol=tolerance,
        rel_tol=tolerance,
    ):
        raise AssertionError(
            "{}: actual={}, expected={}"
            .format(
                name,
                actual,
                expected,
            )
        )


def make_target():
    target = torch.zeros(
        2,
        32,
        32,
        dtype=torch.long,
    )

    # One pixel: 1/1024 = 0.0977%, Tiny.
    target[0, 15, 15] = 1

    # Four pixels: 4/1024 = 0.3906%, Small.
    target[1, 14:16, 14:16] = 1

    return target


def main():
    target = make_target()

    criterion = LocalPositiveRingLoss(
        positive_weight=0.05,
        ring_weight=0.05,
        ring_radius=2,
        tiny_max_ratio=0.001,
        small_max_ratio=0.005,
        warmup_epochs=5,
        ramp_epochs=5,
    )

    expected_schedule = {
        0: 0.0,
        4: 0.0,
        5: 0.2,
        6: 0.4,
        7: 0.6,
        8: 0.8,
        9: 1.0,
        20: 1.0,
    }

    for epoch, expected in expected_schedule.items():
        assert_close(
            criterion.schedule_factor(epoch),
            expected,
            "schedule epoch {}".format(epoch),
        )

    print("Schedule audit: PASS")

    # Warm-up must be exact zero.
    random_logits = torch.randn(
        2,
        2,
        32,
        32,
        requires_grad=True,
    )

    warmup_loss, warmup_stats = criterion(
        random_logits,
        target,
        epoch=0,
    )

    assert_close(
        warmup_loss.item(),
        0.0,
        "warm-up zero loss",
    )

    assert warmup_stats["active_count"] == 2
    assert warmup_stats["tiny_count"] == 1
    assert warmup_stats["small_count"] == 1

    print("Warm-up identity: PASS")

    # Background prediction:
    # high positive-region loss, low ring loss.
    under_logits = torch.zeros(
        2,
        2,
        32,
        32,
        requires_grad=True,
    )

    under_logits.data[:, 0] = 6.0
    under_logits.data[:, 1] = -6.0

    under_loss, under_stats = criterion(
        under_logits,
        target,
        epoch=9,
    )

    if not (
        under_stats["positive_loss"]
        > under_stats["ring_loss"]
    ):
        raise AssertionError(
            "Underprediction should have "
            "larger positive loss"
        )

    print("Positive-region response: PASS")

    # Foreground everywhere:
    # low positive-region loss, high ring loss.
    over_logits = torch.zeros(
        2,
        2,
        32,
        32,
        requires_grad=True,
    )

    over_logits.data[:, 0] = -6.0
    over_logits.data[:, 1] = 6.0

    over_loss, over_stats = criterion(
        over_logits,
        target,
        epoch=9,
    )

    if not (
        over_stats["ring_loss"]
        > over_stats["positive_loss"]
    ):
        raise AssertionError(
            "Overprediction should have "
            "larger ring loss"
        )

    print("Hard-negative-ring response: PASS")

    # Verify local supervision gradient isolation:
    # base logits are detached, residual logits receive gradient.
    base_logits = torch.randn(
        2,
        2,
        32,
        32,
        requires_grad=True,
    )

    residual_logits = torch.randn(
        2,
        2,
        32,
        32,
        requires_grad=True,
    )

    local_logits = (
        base_logits.detach()
        + residual_logits
    )

    isolated_loss, _ = criterion(
        local_logits,
        target,
        epoch=9,
    )

    isolated_loss.backward()

    if base_logits.grad is not None:
        raise AssertionError(
            "Base logits unexpectedly received "
            "S3-A auxiliary gradient"
        )

    if residual_logits.grad is None:
        raise AssertionError(
            "Residual logits received no gradient"
        )

    if not torch.isfinite(
        residual_logits.grad
    ).all():
        raise AssertionError(
            "Residual gradient contains NaN/Inf"
        )

    print("Gradient isolation: PASS")

    # All-one compatibility: zero auxiliary weights.
    zero_criterion = LocalPositiveRingLoss(
        positive_weight=0.0,
        ring_weight=0.0,
        ring_radius=2,
        warmup_epochs=0,
        ramp_epochs=1,
    )

    zero_loss, _ = zero_criterion(
        random_logits,
        target,
        epoch=0,
    )

    assert_close(
        zero_loss.item(),
        0.0,
        "zero-weight identity",
    )

    print("Zero-weight identity: PASS")
    print()
    print("Underprediction stats:", under_stats)
    print("Overprediction stats:", over_stats)
    print()
    print("S3-A loss audit: ALL PASS")


if __name__ == "__main__":
    main()

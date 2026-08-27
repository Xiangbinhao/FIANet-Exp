from loss.foreground_size_aux_loss import (
    ForegroundSizeAuxiliaryLoss,
)


class ScheduledForegroundSizeAuxiliaryLoss(
    ForegroundSizeAuxiliaryLoss
):
    """
    S1-Cv2 scheduled Tiny/Small foreground Dice loss.

    Schedule with the default 40 epochs:
        epoch 0-4:   factor = 0
        epoch 5-14:  linearly increase to 1
        epoch 15-30: factor = 1
        epoch 31-39: linearly decrease to 0
    """

    def __init__(
        self,
        base_loss,
        tiny_max_lambda=0.10,
        small_max_lambda=0.05,
        tiny_max_ratio=0.001,
        small_max_ratio=0.005,
        warmup_epochs=5,
        ramp_epochs=10,
        hold_epochs=16,
        decay_epochs=9,
        foreground_class=1,
        smooth=1e-6,
    ):
        self.tiny_max_lambda = float(
            tiny_max_lambda
        )
        self.small_max_lambda = float(
            small_max_lambda
        )

        self.warmup_epochs = int(
            warmup_epochs
        )
        self.ramp_epochs = int(
            ramp_epochs
        )
        self.hold_epochs = int(
            hold_epochs
        )
        self.decay_epochs = int(
            decay_epochs
        )

        for name, value in [
            ("warmup_epochs", self.warmup_epochs),
            ("ramp_epochs", self.ramp_epochs),
            ("hold_epochs", self.hold_epochs),
            ("decay_epochs", self.decay_epochs),
        ]:
            if value < 0:
                raise ValueError(
                    "{} must be >= 0".format(name)
                )

        super().__init__(
            base_loss=base_loss,
            tiny_lambda=0.0,
            small_lambda=0.0,
            tiny_max_ratio=tiny_max_ratio,
            small_max_ratio=small_max_ratio,
            foreground_class=foreground_class,
            smooth=smooth,
            log_first_batch=False,
        )

        self.current_epoch = 0
        self.set_epoch(0)

    @property
    def configured_total_epochs(self):
        return (
            self.warmup_epochs
            + self.ramp_epochs
            + self.hold_epochs
            + self.decay_epochs
        )

    def get_schedule_factor(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch

        epoch = int(epoch)

        warmup_end = self.warmup_epochs

        ramp_end = (
            warmup_end
            + self.ramp_epochs
        )

        hold_end = (
            ramp_end
            + self.hold_epochs
        )

        decay_end = (
            hold_end
            + self.decay_epochs
        )

        if epoch < warmup_end:
            return 0.0

        if epoch < ramp_end:
            if self.ramp_epochs <= 0:
                return 1.0

            ramp_index = (
                epoch - warmup_end + 1
            )

            return min(
                1.0,
                float(ramp_index)
                / float(self.ramp_epochs),
            )

        if epoch < hold_end:
            return 1.0

        if epoch < decay_end:
            if self.decay_epochs <= 1:
                return 0.0

            decay_index = (
                epoch - hold_end
            )

            factor = (
                float(
                    self.decay_epochs
                    - 1
                    - decay_index
                )
                / float(
                    self.decay_epochs - 1
                )
            )

            return max(0.0, factor)

        return 0.0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

        factor = self.get_schedule_factor(
            self.current_epoch
        )

        self.tiny_lambda = (
            self.tiny_max_lambda
            * factor
        )

        self.small_lambda = (
            self.small_max_lambda
            * factor
        )

    def get_current_lambdas(self):
        factor = self.get_schedule_factor(
            self.current_epoch
        )

        return (
            self.tiny_lambda,
            self.small_lambda,
            factor,
        )

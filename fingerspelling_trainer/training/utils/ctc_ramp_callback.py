from lightning.pytorch.callbacks import Callback


class CTCRampCallback(Callback):
    """
    Manages warm up for weighted ctc loss and focal ctc loss.

    Args:
        ctc_type: either standard | weighted | focal
        start_epoch: start epoch for the ramp-up
        ramp_epochs: number of epochs to go from 0 to 1 loss importance
        gamma_final: final value of gamma for focal loss

    """

    def __init__(
        self,
        *,
        ctc_type: str,
        start_epoch: int = 5,
        ramp_epochs: int = 10,
        gamma_final: float = 1.0,
        verbose_name: str = "",
    ):
        super().__init__()
        self.ctc_type = ctc_type.lower()
        self.start = int(start_epoch)
        self.ramp = max(int(ramp_epochs), 1)
        self.gamma_final = float(gamma_final)
        self.verbose = verbose_name or self.ctc_type

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch < self.start:
            return

        # lineal progress p ∈ [0,1]
        p = float(epoch - self.start + 1) / float(self.ramp)
        p = min(p, 1.0)

        if self.ctc_type == "weighted":
            pl_module.w_ctc.alpha = p  # type: ignore
            if epoch == self.start:
                pl_module.train_loss = pl_module.w_ctc
                if trainer.is_global_zero:
                    trainer.logger.log_metrics(  # type: ignore
                        {f"info/weighted_ctc_activated_{self.verbose}": 1}, step=epoch
                    )

        elif self.ctc_type == "focal":
            base = 1e-2  # o 5e-2
            pl_module.focal_ctc.gamma = base + p * (self.gamma_final - base)  # type: ignore

            if epoch == self.start:
                pl_module.train_loss = pl_module.focal_ctc
                if trainer.is_global_zero:
                    trainer.logger.log_metrics(  # type: ignore
                        {f"info/focal_ctc_activated_{self.verbose}": 1}, step=epoch
                    )
        else:
            # "standard" ctc loss does nothing
            return

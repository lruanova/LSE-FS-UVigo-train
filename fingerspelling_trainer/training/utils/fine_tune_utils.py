import re
import lightning as pl
import torch


def load_backbone(model, ckpt_path: str, head_substr: str = "model.head.4"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    to_drop = [k for k in sd if k.startswith(head_substr)]

    for k in to_drop:
        sd.pop(k)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")

    return model


def _match(name: str, pattern: str) -> bool:
    if pattern == "head":
        return name.startswith("model.head")
    if pattern == "tcn":
        return name.startswith("model.tcn")
    if pattern == "gnn.last":
        return bool(re.match(r"model\.gnn\.layers\.\d+$", name))  # última capa
    if pattern == "gnn":
        return name.startswith("model.gnn")
    return False


class ProgressiveUnfreeze(pl.Callback):
    def __init__(self, schedule):
        self.schedule = sorted(schedule, key=lambda x: x["epoch"])  # type: ignore
        self._applied_idx = -1

    def _apply_phase(self, trainer, pl_module, phase):
        to_train = phase["train"]
        lr_mult = phase.get("lr_mult", 1.0)

        for n, p in pl_module.named_parameters():
            p.requires_grad = any(_match(n, tag) for tag in to_train)

        # --- adjust learning-rate on each param-group
        opt = trainer.optimizers[0]
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_mult

        trainer.print(
            f"\n🟢  FT-phase @epoch {phase['epoch']}  ⇒ "
            f"train={to_train}  |  lr×={lr_mult}\n"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        cur_ep = trainer.current_epoch
        if self._applied_idx + 1 >= len(self.schedule):
            return
        next_phase = self.schedule[self._applied_idx + 1]
        if cur_ep >= next_phase["epoch"]:
            self._apply_phase(trainer, pl_module, next_phase)
            self._applied_idx += 1

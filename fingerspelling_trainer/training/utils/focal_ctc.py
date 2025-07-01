# TODO: Review this implementation
import torch
import torch.nn.functional as F
from torch import nn


class FocalCTCLoss(nn.Module):
    """
    Focal-CTC loss -  Li et al.
    https://onlinelibrary.wiley.com/doi/10.1155/2019/9345861

    If gamma=0, equivalent to normal CTC. If gamma > 0 , focus more
    on errors.

    """

    def __init__(self, *, blank: int = 0, gamma: float = 2.0):
        super().__init__()
        self.blank = int(blank)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lens: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        B, C, T = logits.size()

        if self.gamma == 0.0 or not self.training:
            lp = F.log_softmax(logits, dim=1).permute(2, 0, 1)  # (T, B, C)

            input_lens_clamped = input_lens.clamp(max=T)

            return F.ctc_loss(
                lp,
                targets,
                input_lens_clamped,
                target_lens,
                blank=self.blank,
                reduction="mean",
                zero_infinity=True,
            )

        # Log‐probs and CTC “vanilla” with reduction="sum"
        lp = F.log_softmax(logits, dim=1).permute(2, 0, 1)  # (T, B, C)
        input_lens_clamped = input_lens.clamp(max=T)

        vanilla = F.ctc_loss(
            lp,
            targets,
            input_lens_clamped,
            target_lens,
            blank=self.blank,
            reduction="sum",
            zero_infinity=True,
        )
        (grads,) = torch.autograd.grad(vanilla, logits, retain_graph=True)

        p = lp.permute(1, 2, 0).exp()  # (B, C, T)
        gamma_ct = (p - grads).detach()  # (B, C, T)

        focal_factor = (1.0 - p).clamp(min=1e-5).pow(self.gamma)  # (B, C, T)
        log_probs_BCT = lp.permute(1, 2, 0)  # (B, C, T)
        weighted_xent = -(focal_factor * gamma_ct * log_probs_BCT).sum()

        denom = target_lens.sum().clamp_min(1).float()
        w_xent_mean = weighted_xent / denom
        vanilla_mean = vanilla / denom
        return self.gamma * w_xent_mean + (1.0 - self.gamma) * vanilla_mean

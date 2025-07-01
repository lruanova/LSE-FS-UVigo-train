# TODO: Review this implementation

import torch
import torch.nn.functional as F
from torch import nn


class WeightedCTCLoss(nn.Module):
    """
    Weighted CTC loss from (Li et al., 2019 -- https://arxiv.org/abs/1904.10619 )
    reformulating it as weighted cross-entropy

    Alpha parameter controls importance of weighting. When alpha = 0, equivalent
    to normal CTC, when alpha = 1, only weighted loss.


    Args:
        weights: tensor of weights per class, including blank
        blank : id of blank token
        renorm : if true, renorms gamma after applying weights
        alpha: factor of mixture between losses

    """

    def __init__(
        self,
        *,
        weights: torch.Tensor,
        blank: int = 0,
        renorm: bool = True,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("w", weights.float())  # shape (L,)
        self.blank = int(blank)
        self.renorm = bool(renorm)
        self.alpha = float(alpha)

    def forward(
        self,
        logits: torch.Tensor,  # (B, C, T)
        targets: torch.Tensor,
        input_lens: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:

        # if alpha 0 or not training, use normal ctc
        if self.alpha == 0.0 or not self.training:
            lp = F.log_softmax(logits, dim=1).permute(2, 0, 1)  # (T,B,C)
            return F.ctc_loss(
                lp,
                targets,
                input_lens,
                target_lens,
                blank=self.blank,
                reduction="mean",
                zero_infinity=True,
            )

        # get log probs and vanilla ctc
        lp = F.log_softmax(logits, dim=1).permute(2, 0, 1)  # (T,B,C)
        vanilla = F.ctc_loss(
            lp,
            targets,
            input_lens,
            target_lens,
            blank=self.blank,
            reduction="sum",
            zero_infinity=True,
        )
        (grads,) = torch.autograd.grad(vanilla, logits, retain_graph=True)

        # Pseudo-targets  γ = p − grad
        gamma = (lp.permute(1, 2, 0).exp() - grads).detach()  # (B,C,T)

        # Apply weights and normalize
        if self.renorm:
            gamma = gamma * self.w[None, :, None]  # type: ignore
            gamma = gamma / gamma.sum(1, keepdim=True).clamp_min(1e-8)

        # Weighted cross entropy
        if self.renorm:
            w_xent = -(gamma * lp.permute(1, 2, 0)).sum()
        else:
            w_xent = -(gamma * lp.permute(1, 2, 0) * self.w[None, :, None]).sum()  # type: ignore

        # mean scaling
        denom = target_lens.sum().clamp_min(1).float()
        w_xent = w_xent / denom
        vanilla_mean = vanilla / denom

        # soft mixture
        return self.alpha * w_xent + (1.0 - self.alpha) * vanilla_mean

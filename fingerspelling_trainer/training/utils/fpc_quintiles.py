import numpy as np
import torch.distributed as dist


def log_cer_vs_fpc_quintiles(
    local_fpc,
    local_cer,
    epoch,
    trainer=None,
    print_table=True,
):
    if trainer is not None and getattr(trainer, "sanity_checking", False):
        return

    if trainer is not None and dist.is_initialized() and dist.get_world_size() > 1:
        ws = dist.get_world_size()
        gathered_fpc = [None] * ws
        gathered_cer = [None] * ws
        dist.all_gather_object(gathered_fpc, list(local_fpc))
        dist.all_gather_object(gathered_cer, list(local_cer))
        all_fpc = [x for part in gathered_fpc if part is not None for x in part]  # type: ignore
        all_cer = [x for part in gathered_cer if part is not None for x in part]  # type: ignore
    else:
        all_fpc, all_cer = list(local_fpc), list(local_cer)

    # only rank 0 prints
    if (
        trainer is not None
        and hasattr(trainer, "is_global_zero")
        and not trainer.is_global_zero
    ):
        return

    if not all_fpc or not all_cer:
        return

    all_fpc = np.array(all_fpc)
    all_cer = np.array(all_cer)
    edges = np.percentile(all_fpc, [0, 20, 40, 60, 80, 100])
    cer_quintiles = []
    n_quintiles = []
    for q in range(5):
        mask = (
            (all_fpc >= edges[q]) & (all_fpc < edges[q + 1])
            if q < 4
            else (all_fpc >= edges[q]) & (all_fpc <= edges[q + 1])
        )
        cer_this = all_cer[mask]
        mean_cer = float(cer_this.mean()) if len(cer_this) > 0 else float("nan")
        cer_quintiles.append(mean_cer)
        n_quintiles.append(int(mask.sum()))

    lines = []
    lines.append(f"[CER vs FPC quintiles] epoch {epoch}")
    for q in range(5):
        lines.append(
            f"  q{q}: {edges[q]:.2f}–{edges[q+1]:.2f} | mean CER={cer_quintiles[q]:.3f} (n={n_quintiles[q]})"
        )
    summary_text = "\n".join(lines)

    if print_table:
        print(summary_text)

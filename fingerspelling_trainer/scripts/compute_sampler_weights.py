"""
Computes sampler weights for the WeightedRandomSampler.

Two modes of operation (defined on hydra):
    - original : using predefined weights for groups of tokens (base, rare, motion).
    - freq : weights computed as inverse of number of occurrences.

"""

from __future__ import annotations
from collections import Counter
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import orjson
from fingerspelling_trainer.data.transformations.encode_label import EncodeLabel


def read_raw_label_and_len(path: Path) -> tuple[str, int]:
    with path.open("rb") as f:
        obj = orjson.loads(f.read())
    return obj["metadata"]["label"], len(obj["frames"])


def calc_weight_for_path(
    path: str,
    encode_kwargs: dict,
    mode: str,
    motion_ids: set[int],
    rare_ids: set[int],
    token_boost: dict[int, float] | None,
    boosts: tuple[float, float, float],
    boosts_misc: tuple[float, float],  # base, motion
    fpc_q: list[float],
    fpc_boosts: list[float],
    use_fpc: bool,
    clip_val: float | None,
) -> float:
    p = Path(path)
    label, n_frames = read_raw_label_and_len(p)
    from fingerspelling_trainer.data.transformations.encode_label import EncodeLabel

    enc = EncodeLabel(**encode_kwargs)
    toks = enc.alphabet.encode_label(enc._format_label(label))
    ids = set(toks)

    # ---------------------------
    # --- freq
    # ---------------------------
    if mode == "freq":
        if not ids:
            token_w = 1.0
        else:
            token_w = max(token_boost.get(t, 1.0) for t in ids)  # type: ignore
        base_b, motion_b = boosts_misc
        if ids & motion_ids:
            token_w *= motion_b
        token_w *= base_b

    # ---------------------------
    # --- original
    # ---------------------------
    else:
        base_b, motion_b, rare_b = boosts
        token_w = (
            base_b
            * (motion_b if ids & motion_ids else 1.0)
            * (rare_b if ids & rare_ids else 1.0)
        )

    # FPC
    n_chars = len(toks) or 1
    fpc = n_frames / n_chars
    if use_fpc:
        quint = np.digitize(fpc, fpc_q)  # 0…4
        fpc_w = fpc_boosts[int(quint)]
    else:
        fpc_w = 1.0
    w = token_w * fpc_w
    if clip_val is not None:
        w = min(w, clip_val)
    return float(w)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    train_dir = Path(cfg.dataset.data_path) / "train"
    out_path = Path(cfg.dataset.data_path) / cfg.dataset.sampler.filename

    mode = getattr(cfg.dataset.sampler, "mode", "original")  # por defecto "original"
    print(f"\n=== [generate_sampler_weights] Weighting mode: '{mode}' ===\n")

    enc_tmp = EncodeLabel(
        alphabet=hydra.utils.instantiate(cfg.dataset.sampler.alphabet)
    )
    enc_kwargs = enc_tmp.__dict__ | {"alphabet": enc_tmp.alphabet}
    ALPHABET = enc_tmp.alphabet

    files = sorted(train_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON in {train_dir}")

    # --- FPC global
    fpc_vals = []
    for f in tqdm(files, desc="Scanning labels for FPC"):
        lab, n_frames = read_raw_label_and_len(f)
        ids = ALPHABET.encode_label(enc_tmp._format_label(lab))
        fpc_vals.append(n_frames / (len(ids) or 1))
    fpc_q = np.quantile(fpc_vals, [0.2, 0.4, 0.6, 0.8]).tolist()
    print(f"FPC quintiles = {fpc_q}")

    # --- motion/rare ids
    motion_ids = {ALPHABET.LETTER_TO_NUM[t] for t in cfg.dataset.sampler.motion_tokens}
    rare_ids = {
        ALPHABET.LETTER_TO_NUM[t]
        for t in getattr(cfg.dataset.sampler, "rare_tokens", [])
    }

    # --- boosts
    boosts = (
        cfg.dataset.sampler.base_boost,
        cfg.dataset.sampler.motion_boost,
        cfg.dataset.sampler.rare_boost,
    )
    boosts_misc = (cfg.dataset.sampler.base_boost, cfg.dataset.sampler.motion_boost)
    use_fpc = bool(cfg.dataset.sampler.use_fpc)
    clip_val = cfg.dataset.sampler.get("clip_value", None)

    # --- freq-mode: precompute token_boost
    token_boost = None
    if mode == "freq":
        tok_freq: Counter[int] = Counter()
        for f in tqdm(files, desc="Scanning labels for frequency..."):
            lab, _ = read_raw_label_and_len(f)
            ids = ALPHABET.encode_label(enc_tmp._format_label(lab))
            tok_freq.update(ids)
        if not tok_freq:
            raise RuntimeError("No token frequencies found.")
        median_f = np.median(list(tok_freq.values()))
        print(f"Median frequency = {median_f:.0f} samples")
        token_boost = {
            t: min(max((median_f / freq) ** 0.5, 1.0), clip_val or 99)
            for t, freq in tok_freq.items()
        }

    worker = partial(
        calc_weight_for_path,
        encode_kwargs=enc_kwargs,
        mode=mode,
        motion_ids=motion_ids,
        rare_ids=rare_ids,
        token_boost=token_boost,  # type: ignore
        boosts=boosts,
        boosts_misc=boosts_misc,
        fpc_q=fpc_q,
        fpc_boosts=cfg.dataset.sampler.fpc_boosts,
        use_fpc=use_fpc,
        clip_val=clip_val,
    )

    print(f"Computing weights with {cpu_count()} procesess...")
    with Pool(cpu_count()) as pool:
        w_list = list(tqdm(pool.imap(worker, map(str, files)), total=len(files)))

    w_tensor = torch.tensor(w_list, dtype=torch.double)
    torch.save(w_tensor, out_path)
    print(
        f"\n✅  Weights saved on {out_path}\n    mean={w_tensor.mean():.3f} | "
        f"max={w_tensor.max():.3f}"
    )


if __name__ == "__main__":
    main()

# infer_single.py
import hydra
import torch
from pathlib import Path
from fingerspelling_trainer.data.json_dataset import JSONDataset
from fingerspelling_trainer.data.compose_transforms import ComposeTransforms
from fingerspelling_trainer.training.learners.translation_learner import (
    TranslationLearner,
)
import torch.nn.functional as F
import numpy as np


def build_test_transforms(cfg):
    return hydra.utils.instantiate(cfg.dataset.transformations.test.obj)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    transforms = build_test_transforms(cfg)

    sample_path = Path(cfg.inference.sample_path)
    ds = JSONDataset(
        json_files_dir=sample_path.parent,
        transformations=transforms,
        require_label=False,
    )

    idx = [p.name for p in ds.files].index(sample_path.name)
    sample = ds[idx]

    if "frames_per_char" in sample:
        sample["frames_per_char"] = 17.55

    batch = ds.collate_batch([sample])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learner = (
        TranslationLearner.load_from_checkpoint(cfg.inference.checkpoint_path, cfg=cfg)
        .to(device)
        .eval()
    )

    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        logits, lengths = learner._model_forward(batch)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # T,B,C
        preds, _ = learner.decoder(
            alphabet=learner.alphabet_instance,
            targets_for_decoding=[],
            masked_log_probs=log_probs,
            input_lengths=lengths,
        )
    print("🔤  Pred:", preds[0])


if __name__ == "__main__":
    main()

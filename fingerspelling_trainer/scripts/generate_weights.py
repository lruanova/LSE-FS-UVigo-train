"""
python scripts/generate_weights.py \
       --json_dir /data/LSE/train \
       --transforms_yaml configs/transformations_LSE.yaml \
       --alphabet_yaml configs/alphabets/spanish.yaml \
       --output /data/LSE/letter_weights.pt
"""

from pathlib import Path
from collections import Counter
import argparse
import torch
from omegaconf import OmegaConf
import hydra

from fingerspelling_trainer.data.transformations.encode_label import EncodeLabel
from keypoint_extraction_pipeline.savers.json_saver import JSONSaver
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


def build_encoder(transforms_yaml: Path, alphabet_yaml: Path) -> EncodeLabel:
    t_cfg = OmegaConf.load(transforms_yaml)
    alpha_c = OmegaConf.load(alphabet_yaml)

    enc_cfg = None
    for tr in t_cfg.train.obj.transforms:
        if tr["_target_"].endswith("EncodeLabel"):
            enc_cfg = OmegaConf.merge(tr, {"alphabet": alpha_c})
            break
    if enc_cfg is None:
        raise RuntimeError("EncodeLabel not found on YAML")

    return hydra.utils.instantiate(enc_cfg)


def main(args):
    encoder = build_encoder(Path(args.transforms_yaml), Path(args.alphabet_yaml))
    vocab_size = len(encoder.alphabet) + 1  # + blank
    counter = Counter()

    json_dir = Path(args.json_dir)
    assert json_dir.is_dir(), f"{json_dir} not a directory"
    for js in json_dir.glob("*.json"):
        record: AnnotationRecord = JSONSaver.load_record(js)
        record = encoder(record)
        tokens = record.metadata.custom_properties["encoded_label"]
        counter.update(tokens)

    freqs = torch.zeros(vocab_size)
    for k, v in counter.items():
        freqs[k] = v
    median = freqs[freqs > 0].median()
    weights = median / freqs.clamp(min=1)
    weights[0] = 1.0  # blank
    torch.save(weights, args.output)
    print(f"Weights saved on {args.output}  (vocab = {vocab_size} symbols)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", required=True, help="Folder with .json for TRAIN")
    p.add_argument(
        "--transforms_yaml",
        required=True,
        help="YAML ComposeTransforms (transformations_LSE.yaml)",
    )
    p.add_argument(
        "--alphabet_yaml", required=True, help="YAML with the alphabet (spanish.yaml)"
    )
    p.add_argument("--output", default="letter_weights.pt")
    main(p.parse_args())

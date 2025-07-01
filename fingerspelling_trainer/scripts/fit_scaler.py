"""
Fits scaler for training.

It's done before training if scalers for keypoints and velocity aren't found,
this script just allows to fit them manually.

"""

from pathlib import Path
import hydra
from fingerspelling_trainer.data.json_dataset import JSONDataset
from fingerspelling_trainer.data.transformations.scale_keypoints import ScaleKeypoints


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    train_dir = Path(cfg.dataset.data_path) / "train"
    ds = JSONDataset(
        json_files_dir=train_dir,
        transformations=hydra.utils.instantiate(cfg.dataset.transformations.train.obj),
    )
    scaler = ScaleKeypoints(scaler_dir=cfg.dataset.data_path)
    scaler.fit_on_dataset(ds, disable_augments=True)
    print("✔ Scalers listos:", cfg.dataset.data_path)


if __name__ == "__main__":
    main()

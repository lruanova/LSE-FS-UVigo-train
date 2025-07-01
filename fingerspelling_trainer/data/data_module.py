import os
from pathlib import Path
import lightning as pl
from omegaconf import DictConfig
import torch
from fingerspelling_trainer.data.json_dataset import JSONDataset
from torch.utils.data import DataLoader
from fingerspelling_trainer.data.compose_transforms import ComposeTransforms
from torch.utils.data import WeightedRandomSampler


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transformations: ComposeTransforms,
        transformations_test: ComposeTransforms,
        batch_size: int,
        num_workers: int,
        sampler_cfg: DictConfig,
    ) -> None:

        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformations = transformations
        self.transformations_test = transformations_test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.sampler_cfg = sampler_cfg

    def setup(self, stage=None):
        self.train_dir = os.path.join(self.data_dir, "train")
        self.validation_dir = os.path.join(self.data_dir, "validation")
        self.test_dir = os.path.join(self.data_dir, "test")

        self.train_ds = JSONDataset(
            json_files_dir=self.train_dir,
            transformations=self.transformations,
        )

        # >> Weighted sampler config
        self.train_sampler = None
        if stage in (None, "fit") and self.sampler_cfg.enabled:
            print("\n ❗ using weighted sampler \n")
            weights_path = self.data_dir / self.sampler_cfg.filename
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"[DataModule] File: {self.sampler_cfg.filename} doesn't exist on: {weights_path}.\n"
                    "→ Run script generate_sampler_weights under scripts/ folder."
                )
            weights = torch.load(weights_path, map_location="cpu")
            if len(weights) != len(self.train_ds):
                raise RuntimeError(
                    f"[DataModule] weights under {weights_path} length: ({len(weights)}) and train_ds ({len(self.train_ds)}) have different number of elements."
                )

            self.train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.train_ds) * self.sampler_cfg.dataset_multiplier,
                replacement=True,
            )

        self.val_ds = JSONDataset(
            json_files_dir=self.validation_dir,
            transformations=self.transformations_test,
        )
        self.test_ds = JSONDataset(
            json_files_dir=self.test_dir,
            transformations=self.transformations_test,
        )

        print(f"\nNumber of training samples: {len(self.train_ds)}")
        print(f"\nNumber of dev samples: {len(self.val_ds)}")
        print(f"\nNumber of training test: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_ds.collate_batch,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.val_ds.collate_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.test_ds.collate_batch,
        )

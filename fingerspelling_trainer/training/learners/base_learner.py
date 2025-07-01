import lightning as pl
from omegaconf import DictConfig
import torch


class BaseLearner(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

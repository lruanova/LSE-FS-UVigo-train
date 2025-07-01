import re
from typing import Optional
import hydra
from hydra.utils import get_class
from torch import nn
import torch
from torch.nn import functional as F
from omegaconf import DictConfig
from fingerspelling_trainer.training.learners.base_learner import BaseLearner
from fingerspelling_trainer.training.utils.alphabets import Alphabet
from fingerspelling_trainer.training.utils.cer_metric import cer
from fingerspelling_trainer.training.utils.decoders import (
    batched_beam_search_decoder,
    greedy_decoder,
    greedy_pause_decoder,
)
from fingerspelling_trainer.training.utils.confussion_matrix import log_confusion_matrix
from fingerspelling_trainer.training.utils.fpc_quintiles import log_cer_vs_fpc_quintiles
from fingerspelling_trainer.training.utils.weighted_ctc import WeightedCTCLoss

from fingerspelling_trainer.training.utils.focal_ctc import FocalCTCLoss
import torch.distributed as dist
from functools import partial


class TranslationLearner(BaseLearner):
    """
    PyTorch Lighning learner for the translation task.

    Handles the training/validation/test steps and the computation of metrics and artifacts
    and logging process to WanDB.

    """

    def __init__(self, cfg: DictConfig):
        """Initializes the lighning learner for the translation task."""
        super().__init__(cfg)

        # >> Get alphabet config
        self.alphabet_instance: Alphabet = hydra.utils.instantiate(cfg.dataset.alphabet)

        if not hasattr(self.alphabet_instance, "__len__"):
            raise ValueError(
                "The instantiated alphabet object does not have a __len__ method."
            )
        vocab_size = len(self.alphabet_instance) + 1

        # >> Create model components
        self.model: nn.Module = hydra.utils.instantiate(
            cfg.model.obj, vocab_size=vocab_size
        )

        self._create_loss(cfg)
        self._create_decoder(cfg)
        self.automatic_optimization = True

        # >> CM buffers
        self._test_true_seqs: list[list[int]] = []
        self._test_pred_seqs: list[list[int]] = []
        self._val_true_seqs: list[list[int]] = []
        self._val_pred_seqs: list[list[int]] = []

        # FPC buffers
        self.log_fpc_quintiles = cfg.learner.log_fpc_quintiles

        self._val_fpc: list[float] = []
        self._val_cer_per_sample: list[float] = []

    def _create_loss(self, cfg):
        ctc_cfg = cfg.learner.ctc
        warmup_ep = getattr(ctc_cfg, "warmup_epochs", 0)

        self.std_ctc = nn.CTCLoss(
            blank=ctc_cfg.blank_token,
            zero_infinity=True,
            reduction="mean",
        )

        self.w_ctc = None
        self.focal_ctc = None

        if ctc_cfg.type == "weighted":
            self.w_ctc = WeightedCTCLoss(
                weights=torch.load(ctc_cfg.weights_path),
                blank=ctc_cfg.blank_token,
                renorm=ctc_cfg.renorm,
                alpha=0.0,  # starts on 0, callback manages it
            )
            self.train_loss = self.std_ctc if warmup_ep > 0 else self.w_ctc

        elif ctc_cfg.type == "focal":
            self.focal_ctc = FocalCTCLoss(
                blank=ctc_cfg.blank_token,
                gamma=0.0,  # starts on 0, callback manages it
            )
            self.train_loss = self.std_ctc if warmup_ep > 0 else self.focal_ctc

        else:
            self.train_loss = self.std_ctc

    def _create_decoder(self, cfg):
        dec_cfg = cfg.learner.decoder
        if dec_cfg.type == "beam":
            self.decoder = partial(
                batched_beam_search_decoder,
                beam_size=dec_cfg.beam_size,
                verbose=dec_cfg.verbose,
            )
        elif dec_cfg.type == "greedy":
            self.decoder = partial(
                greedy_decoder,
                verbose=dec_cfg.verbose,
            )
        elif dec_cfg.type == "combined":
            self.decoder = partial(
                greedy_pause_decoder,
                verbose=dec_cfg.verbose,
                min_pause=dec_cfg.min_pause,
            )
        else:
            raise ValueError("Decoder not recognized. Use either beam|greedy|combined")

    def _model_forward(self, batch):
        """Returns (logits [B,C,T], input_lengths [B])."""
        logits, lengths = self.model(batch)
        return logits, lengths

    def _compute_loss(self, batch, logits, lengths, *, criterion):
        """
        Applies the selected criterion.
         - if Focal or Weighted CTCLoss, needs (B,C,T)
         - else if standard CTCLoss, needs (T,B,C)

        """

        targets = batch["labels"]
        target_lens = batch["label_lengths"].to(targets.device)
        input_lens = lengths.to(targets.device)

        # WeightedCTC
        if isinstance(criterion, WeightedCTCLoss):
            logits_ct = logits.permute(0, 2, 1)  # (B,C,T)
            return criterion(logits_ct, targets, input_lens, target_lens)

        # FocalCTC
        if isinstance(criterion, FocalCTCLoss):
            # manages log softmax internally
            logits_ct = logits.permute(0, 2, 1)
            return criterion(logits_ct, targets, input_lens, target_lens)

        # CTC standard
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T,B,C)
        return criterion(log_probs, targets, input_lens, target_lens)

    def _compute_cer(self, batch, logits, lengths):
        """
        Computes Character Error Rate (CER) metric for a batch.
        Decoders expect log_probs with shape (T,B,C).

        """
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        decoded_preds, decoded_targets = self.decoder(
            alphabet=self.alphabet_instance,
            targets_for_decoding=batch["original_labels"],
            masked_log_probs=log_probs,
            input_lengths=lengths,
        )
        return (
            cer(reference=decoded_targets, hypothesis=decoded_preds),
            decoded_preds,
            decoded_targets,
        )

    def _log_metrics(
        self,
        stage_prefix: str,
        batch_size: int,
        loss_val: torch.Tensor,
        cer_val: Optional[float] = None,
        **other_metrics,
    ):
        """Logs metrics to WanDB."""

        # >> Progress bar _(log per step but not to wandb)
        if stage_prefix == "train":
            self.log(
                f"{stage_prefix}/loss_step",
                loss_val,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
                sync_dist=True,
                batch_size=batch_size,
            )

        # >> Logging per epoch

        # loss
        self.log(
            f"{stage_prefix}/loss",
            loss_val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # cer
        if cer_val is not None:
            self.log(
                f"{stage_prefix}/cer",
                cer_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # any other metrics
        for metric_name, metric_value in other_metrics.items():
            if metric_value is not None:
                self.log(
                    f"{stage_prefix}/{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

    def training_step(self, batch, batch_idx):
        """Makes a training step."""

        batch_size = batch["lengths"].size(0)
        logits, lengths = self._model_forward(batch)
        loss = self._compute_loss(batch, logits, lengths, criterion=self.train_loss)
        cer_value, decoded_preds, decoded_targets = self._compute_cer(
            batch, logits, lengths
        )

        self._log_metrics(
            stage_prefix="train",
            batch_size=batch_size,
            loss_val=loss,
            cer_val=cer_value,  # type: ignore
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Makes a validation step.

        Returning metrics makes lighning save them on callback_metrics
        so callbacks like ModelCheckpoint can access them.

        """
        batch_size = batch["lengths"].size(0)
        logits, lengths = self._model_forward(batch)
        loss = self._compute_loss(batch, logits, lengths, criterion=self.std_ctc)

        cer_value, decoded_preds, decoded_targets = self._compute_cer(
            batch, logits, lengths
        )
        self._log_metrics(
            stage_prefix="val", batch_size=batch_size, loss_val=loss, cer_val=cer_value  # type: ignore
        )

        # >>> FPC quintils
        if self.log_fpc_quintiles:
            fpcs = batch["frames_per_char"].cpu().numpy()
            # CER per sample
            for fpc, pred, true in zip(fpcs, decoded_preds, decoded_targets):
                single_cer = cer(reference=true, hypothesis=pred)
                self._val_fpc.append(fpc)
                self._val_cer_per_sample.append(single_cer)  # type: ignore

        # >>> Validation confussion matrix
        for true_str, pred_str in zip(decoded_targets, decoded_preds):
            t_ids = self.alphabet_instance.encode_label(true_str)
            p_ids = self.alphabet_instance.encode_label(pred_str)
            self._val_true_seqs.append(t_ids)
            self._val_pred_seqs.append(p_ids)

        return {"val_loss": loss, "val_cer": cer_value}

    def test_step(self, batch, batch_idx):
        """
        Makes a test step.

        Also accumulates the encoded sequences for computing the
        confussion matrix on on_test_epoch_end() callback.

        """

        batch_size = batch["lengths"].size(0)
        logits, lengths = self._model_forward(batch)
        loss = self._compute_loss(batch, logits, lengths, criterion=self.std_ctc)

        cer_value, decoded_preds, decoded_targets = self._compute_cer(
            batch, logits, lengths
        )
        self._log_metrics(
            stage_prefix="test",
            batch_size=batch_size,
            loss_val=loss,
            cer_val=cer_value,  # type: ignore
        )

        # Encode and accumulate labels (without padding) for CM
        for true_str, pred_str in zip(decoded_targets, decoded_preds):
            t_ids = self.alphabet_instance.encode_label(true_str)
            p_ids = self.alphabet_instance.encode_label(pred_str)
            self._test_true_seqs.append(t_ids)
            self._test_pred_seqs.append(p_ids)

        return {"test_loss": loss, "test_cer": cer_value}

    def _select(self, model, regex):
        return [p for n, p in model.named_parameters() if re.search(regex, n)]

    def configure_optimizers(self):
        """
        Configures the optimizers to use based on given hydra config.
        Returns the optimizer and lr_scheduler so lighning can handle them.

        """

        optimizer_cfg = self.cfg.dataset.training.optimizer
        optimizer_cls = get_class(optimizer_cfg.obj["_target_"])

        # if param groups defined, use them
        if "param_groups" in optimizer_cfg:
            used_params = set()
            pg = []
            for g in optimizer_cfg.param_groups:
                params = self._select(self.model, g.pattern)
                pg.append(
                    {
                        "params": params,
                        "lr": g.lr,
                        "initial_lr": g.lr,
                    }
                )
                used_params |= set(params)
            others = [p for p in self.model.parameters() if p not in used_params]
            if others:
                pg.append(
                    {
                        "params": others,
                        "lr": optimizer_cfg.learning_rate,
                        "initial_lr": optimizer_cfg.learning_rate,
                    }
                )
            optimizer = optimizer_cls(pg, weight_decay=optimizer_cfg.weight_decay)

        else:  # baseline case
            optimizer = optimizer_cls(
                self.model.parameters(),
                lr=optimizer_cfg.learning_rate,
                weight_decay=optimizer_cfg.weight_decay,
            )

        # Scheduler
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.dataset.training.max_epochs,
            eta_min=self.cfg.dataset.training.scheduler.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_obj,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self):
        # >>> CER per quintil logging
        if self.log_fpc_quintiles:
            log_cer_vs_fpc_quintiles(
                local_fpc=self._val_fpc,
                local_cer=self._val_cer_per_sample,
                epoch=self.current_epoch,
                trainer=self.trainer,
                print_table=True,
            )
            self._val_fpc.clear()
        self._val_cer_per_sample.clear()
        # >>> CM matrix logging
        if not self._val_true_seqs or not self._val_pred_seqs:
            return
        # only main node
        if self.trainer.is_global_zero:
            sklearn_labels = list(range(1, len(self.alphabet_instance) + 1))
            class_names = [
                self.alphabet_instance.NUM_TO_LETTER[i] for i in sklearn_labels
            ]
            log_confusion_matrix(
                seqs_true=self._val_true_seqs,
                seqs_pred=self._val_pred_seqs,
                sklearn_labels=sklearn_labels,
                class_names=class_names,
                pad_id=self.cfg.learner.ctc.blank_token,
                title="Validation Confusion Matrix",
            )
        self._val_true_seqs.clear()
        self._val_pred_seqs.clear()

    def on_test_epoch_end(self) -> None:
        """
        Callback that runs when test epoch is finished.

        Used to compute the test confussion matrix and logs it to wandb.
        """
        # Gather from different GPUs/Ray workers all lists in main node
        local_t = list(self._test_true_seqs)
        local_p = list(self._test_pred_seqs)
        if dist.is_initialized() and dist.get_world_size() > 1:
            ws = dist.get_world_size()
            gathered_t = [None] * ws
            gathered_p = [None] * ws
            dist.all_gather_object(gathered_t, local_t)
            dist.all_gather_object(gathered_p, local_p)
            if not self.trainer.is_global_zero:
                return
            seqs_true = [seq for part in gathered_t if part for seq in part]  # type: ignore
            seqs_pred = [seq for part in gathered_p if part for seq in part]  # type: ignore
        else:
            seqs_true, seqs_pred = local_t, local_p

        # logging is only made on rank 0 node
        if self.trainer.is_global_zero:
            # Creates sklearn labels (integer IDs) and class names (corresponding alphabetic
            # names) and calls the log confussion matrix function
            sklearn_labels = list(range(1, len(self.alphabet_instance) + 1))
            class_names = [
                self.alphabet_instance.NUM_TO_LETTER[i] for i in sklearn_labels
            ]
            log_confusion_matrix(
                seqs_true=seqs_true,
                seqs_pred=seqs_pred,
                sklearn_labels=sklearn_labels,
                class_names=class_names,
                pad_id=self.cfg.learner.ctc.blank_token,
                title="Test Confusion Matrix",
            )
        # Clear buffers in case several test epochs
        self._test_true_seqs = []  # type: ignore[assignment]
        self._test_pred_seqs = []  # type: ignore[assignment]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict", {})
        for k in list(sd.keys()):
            if k.startswith("train_loss."):
                sd.pop(k)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict", {})
        for k in list(sd.keys()):
            if k.startswith("train_loss."):
                sd.pop(k)

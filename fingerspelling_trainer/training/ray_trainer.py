import os

import hydra
import lightning as pl
from omegaconf import OmegaConf
from ray.train import Result, ScalingConfig
from ray import train as ray_train
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from lightning.pytorch.loggers import WandbLogger
import torch
from fingerspelling_trainer.data.data_module import DataModule
from lightning.pytorch.callbacks import Callback
from ray.train.lightning import RayTrainReportCallback
from fingerspelling_trainer.training.utils.fine_tune_utils import load_backbone


class RayTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def _configure_ray_resources(self) -> ScalingConfig:
        if self.cfg.ray.use_gpu:
            n = self.cfg.ray.num_workers or torch.cuda.device_count()
            return ScalingConfig(
                num_workers=n, resources_per_worker={"CPU": 1, "GPU": 1}, use_gpu=True
            )
        return ScalingConfig(num_workers=self.cfg.ray.num_workers, use_gpu=False)

    def _prepare_callbacks(self) -> list[Callback]:
        cbs = []
        for name, conf in self.cfg.dataset.training.callbacks.items():
            if not conf.get("active", True):
                continue
            if "obj" not in conf:
                raise ValueError(
                    f"Callback '{name}' is active but missing 'obj' config."
                )
            cbs.append(hydra.utils.instantiate(conf.obj))

        return cbs

    def _train_func(self):
        pl.seed_everything(self.cfg.dataset.training.seed)
        global_rank = int(os.environ.get("RANK", "0"))
        is_chief = global_rank == 0

        # >> Create data module
        data_module_instance = DataModule(
            data_dir=self.cfg.dataset.data_path,
            batch_size=self.cfg.dataset.training.batch_size,
            num_workers=self.cfg.dataset.num_workers_dataloader,
            transformations=hydra.utils.instantiate(
                self.cfg.dataset.transformations.train.obj
            ),
            transformations_test=hydra.utils.instantiate(
                self.cfg.dataset.transformations.test.obj
            ),
            sampler_cfg=self.cfg.dataset.sampler,
        )
        data_module_instance.setup("fit")

        print("-------------\n\n")
        print(data_module_instance.transformations)
        print(data_module_instance.transformations_test)
        print("-------------\n\n")
        # >> Scaler
        if is_chief:
            # fit scaler only on chief worker
            for t in data_module_instance.transformations.transforms:
                if hasattr(t, "fit_on_dataset"):
                    print(
                        f"Attempting to fit scaler for transformation: {t.__class__.__name__}"
                    )
                    t.fit_on_dataset(data_module_instance.train_ds)
                    if hasattr(t, "_fitted") and t._fitted:
                        print(
                            f"Scaler for {t.__class__.__name__} successfully fitted and saved."
                        )
                    else:
                        print(
                            f"WARNING: Scaler for {t.__class__.__name__} was not fitted or saved correctly."
                        )
        else:
            print("Not chief worker, skipping scaler fitting.")

        # >>> WanDB logger
        wandb_logger_pl = False
        if is_chief:
            wandb_logger_pl = WandbLogger(
                project=self.cfg.wandb.project_name,
                name=self.cfg.wandb.run_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
            wandb_logger_pl.experiment.define_metric("*", step_metric="epoch")

        # >> Instantiate model
        obj_cls = hydra.utils.get_class(self.cfg.learner.obj["_target_"])

        # fine tune if enabled
        # finetune_cfg = getattr(self.cfg.dataset, "finetune", None)
        model = obj_cls(cfg=self.cfg)
        # if finetune_cfg and getattr(finetune_cfg, "enabled", False):
        #     ckpt = finetune_cfg.ckpt_path
        #     if not ckpt:
        #         raise ValueError("finetune.enabled=true but ckpt_path is empty")

        #     model = obj_cls(cfg=self.cfg)
        #     model = load_backbone(model, ckpt)
        #     print("\n FINE TUNE ENABLED \n")
        # else:
        #     model = obj_cls(cfg=self.cfg)

        # >> Prepare trainer and fit

        trainer = pl.Trainer(
            default_root_dir=self.cfg.dataset.training.default_root_dir,
            max_epochs=self.cfg.dataset.training.max_epochs,
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(
                find_unused_parameters=True, process_group_backend="nccl"
            ),
            plugins=[RayLightningEnvironment()],
            callbacks=self._prepare_callbacks(),
            enable_checkpointing=True,
            gradient_clip_val=self.cfg.dataset.training.gradient_max_norm,
            logger=wandb_logger_pl,
        )
        trainer = prepare_trainer(trainer)

        # >> Fit
        trainer.fit(
            model,
            datamodule=data_module_instance,
        )
        print("Trainer.fit() completed.")

        # >> Report needed stuff for test
        payload = {}
        if is_chief:
            best_ckpt = trainer.checkpoint_callback.best_model_path
            payload = {
                "best_ckpt": os.path.abspath(best_ckpt),
                "wandb_run_id": wandb_logger_pl.experiment.id,  # type: ignore[arg-type]
            }
        ray_train.report(payload)

    def _evaluate_func(self):
        pl.seed_everything(self.cfg.dataset.training.seed)
        global_rank = int(os.environ.get("RANK", "0"))
        is_chief = global_rank == 0

        dm = DataModule(
            data_dir=self.cfg.dataset.data_path,
            batch_size=self.cfg.dataset.training.batch_size,
            num_workers=self.cfg.dataset.num_workers_dataloader,
            transformations=hydra.utils.instantiate(
                self.cfg.dataset.transformations.train.obj
            ),
            transformations_test=hydra.utils.instantiate(
                self.cfg.dataset.transformations.test.obj
            ),
            sampler_cfg=self.cfg.dataset.sampler,
        )
        dm.setup("test")

        obj_cls = hydra.utils.get_class(self.cfg.learner.obj["_target_"])
        model = obj_cls(cfg=self.cfg)

        # load best checkpoint
        ckpt_path = self.cfg.evaluation.checkpoint_path
        if ckpt_path == "":
            raise ValueError(f"evaluation.checkpoint_path: {ckpt_path} - not found. ")

        # log to wandb only if called at train end
        wandb_logger_pl = None
        if is_chief and self.cfg.evaluation.log_to_wandb:
            wandb_logger_pl = WandbLogger(
                project=self.cfg.wandb.project_name,
                name=self.cfg.wandb.run_name,
                id=self.cfg.evaluation.wandb_run_id,
                resume="allow",
            )

        # trainer
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(
                find_unused_parameters=False, process_group_backend="nccl"
            ),
            plugins=[RayLightningEnvironment()],
            logger=(
                wandb_logger_pl if wandb_logger_pl else None
            ),  # none if not called from train
            callbacks=[RayTrainReportCallback()] if is_chief else None,
            enable_checkpointing=False,
        )
        trainer = prepare_trainer(trainer)

        metrics = trainer.test(
            model=model,
            datamodule=dm,
            ckpt_path=self.cfg.evaluation.checkpoint_path,
        )[0]

        metrics = metrics if is_chief else {}
        ray_train.report(metrics)
        if wandb_logger_pl:
            run = wandb_logger_pl.experiment
            run.log(metrics)
            run.finish()

    def train(self):

        trainer = TorchTrainer(
            self._train_func,
            train_loop_config=self.cfg,
            scaling_config=self._configure_ray_resources(),
        )
        result: Result = trainer.fit()

        # results
        if result.metrics is not None:
            for metric, value in result.metrics.items():
                print(f"{metric}: {value}")

            best_ckpt = result.metrics.get("best_ckpt", "")
            wandb_run_id = result.metrics.get("wandb_run_id", "")
            print(f"Best model saved at: {best_ckpt}")

            # Evaluate on test after train if set
            if self.cfg.dataset.training.run_test_after_train and best_ckpt:
                self.cfg.evaluation.checkpoint_path = best_ckpt
                self.cfg.evaluation.log_to_wandb = True
                self.cfg.evaluation.wandb_run_id = wandb_run_id  # tricky hydra
                self.evaluate()

        else:
            print("No metrics found in result.")
        print("Trial Directory: ", result.path)
        print(sorted(os.listdir(result.path)))

    def evaluate(self):
        print("Running evaluation...")
        ray_eval_trainer = TorchTrainer(
            self._evaluate_func,
            train_loop_config=self.cfg,
            scaling_config=self._configure_ray_resources(),
        )

        result: Result = ray_eval_trainer.fit()
        if result.metrics:
            print("Final Metrics Reported by Ray (Aggregated from test_step logs):")
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value}")

        if result.metrics:
            print("\n📊  (Test) metrics aggregated from all workers:")
            for k, v in result.metrics.items():
                print(f"   • {k}: {v}")
        else:
            print("\n⚠️  No metrics received from ray.")

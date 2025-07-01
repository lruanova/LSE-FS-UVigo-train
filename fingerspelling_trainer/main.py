from omegaconf import DictConfig
import hydra
import logging
from fingerspelling_trainer.training.ray_trainer import RayTrainer

log = logging.getLogger(__name__)


# def get_length(obj):
#     if hasattr(obj, "__len__"):
#         return len(obj)
#     raise TypeError(f"Object of type {type(obj).__name__} has no len()")


# OmegaConf.register_new_resolver("custom_len", get_length)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = cfg.mode
    trainer = RayTrainer(cfg=cfg)
    if mode == "train":
        log.info("🚀 Starting training...")
        trainer.train()
        log.info("✅ Training finished.")

    elif mode == "evaluate":
        log.info("🧪 Evaluating on test...")
        trainer.evaluate()
        log.info("✅ Evaluation finished.")
    else:
        print("Mode not recognized. Use either train | evaluate .")


if __name__ == "__main__":
    """
    ********
    usage :
    ********
    - train: `uv run python fingerspelling_trainer/main.py`
    - evaluate uv run python fingerspelling_trainer/main.py mode=evaluate

    """

    main()

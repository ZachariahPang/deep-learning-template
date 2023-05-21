from typing import List, Optional, Tuple

import hydra
import lightning as L
import lightning.pytorch as pl
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)


def get_callbacks(cfg: DictConfig) -> List[Callback]:
    monitor = "val/acc"
    mode = "max"
    log.info(f"Instantiating callbacks <ModelCheckpoint>, <EarlyStopping>")
    callbacks: List[Callback] = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{cfg.output_dir}/checkpoints",
            filename="epoch_{epoch:03d}",
            monitor=monitor,
            mode=mode,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        pl.callbacks.EarlyStopping(monitor=monitor, patience=100, mode=mode),
        pl.callbacks.RichModelSummary(max_depth=-1),
        pl.callbacks.RichProgressBar(),
    ]
    return callbacks


def get_loggers(cfg: DictConfig) -> List[Logger]:
    log.info(f"Instantiating loggers <TensorBoardLogger>")
    loggers: List[Logger] = [
        pl.loggers.TensorBoardLogger(
            save_dir=f"{cfg.output_dir}/tensorboard",
            name="",
            default_hp_metric=True,
        ),
        pl.loggers.CSVLogger(save_dir=f"{cfg.output_dir}/csv"),
    ]
    return loggers


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[Callback] = get_callbacks(cfg)

    logger: List[Logger] = get_loggers(cfg)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        if cfg.get("train"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
        else:
            ckpt_path = cfg.get("ckpt_path")

        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")


@hydra.main(version_base="1.3", config_path="configs", config_name="default.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()

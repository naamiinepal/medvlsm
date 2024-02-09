import pyrootutils
import copy

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import os
from typing import List, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger

from src import utils

torch.set_float32_matmul_precision("medium")

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.use_ckpt:
        assert cfg.ckpt_path, "You must provide a checkpoint path when `use_ckpt=True`"

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Torch compile() with Pytorch 2.X
    # Comment the line below for torch version < 2.X
    # model = model.compile()

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("task_name") == "eval":
        # Logs eval metrics for testing pipeline
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )

        # Writes output masks to files
        if cfg.get("output_masks_dir"):
            output_masks_dir = cfg.get("output_masks_dir")
            log.info("Generating masks of test dataset")
            pred_outputs = trainer.predict(
                model=model,
                dataloaders=datamodule,
                ckpt_path=cfg.ckpt_path,
            )

            preds, mask_names, heights, widths, datasets = [], [], [], [], []
            for p in pred_outputs:
                preds += list(p["preds"])
                mask_names += list(p["mask_names"])
                heights += list(p["heights"])
                widths += list(p["widths"])
                if "dataset" in p:
                    datasets += list(p["dataset"])

            log.info(f"Saving the generated masks in directory {output_masks_dir}")

            # Create directory if it doesn't exist and if exists clear the directory
            if not os.path.exists(output_masks_dir):
                # Recursively create directory
                os.makedirs(output_masks_dir, exist_ok=True)
            else:
                # Clear the directory
                # for f in os.listdir(output_masks_dir):
                #     os.remove(os.path.join(output_masks_dir, f))
                pass

            if preds[0].shape[0] > 1 and len(datasets) > 0:
                for pred, mask_name, h, w, dataset in zip(
                    preds, mask_names, heights, widths, datasets
                ):
                    for cls, p in zip(utils.CLASS_NAMES, pred):
                        file_path = os.path.join(
                            output_masks_dir, dataset, cls, mask_name
                        )
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        torchvision.utils.save_image(
                            TF.resize(
                                p.unsqueeze(0).double(),
                                size=[h, w],
                                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                            ),
                            file_path,
                        )
            else:
                for pred, mask_name, h, w in zip(preds, mask_names, heights, widths):
                    file_path = os.path.join(output_masks_dir, mask_name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    torchvision.utils.save_image(
                        TF.resize(
                            pred.double(),
                            size=[h, w],
                            interpolation=TF.InterpolationMode.NEAREST_EXACT,
                        ),
                        file_path,
                    )
    # Predict the output masks and save in a directory
    elif cfg.get("task_name") == "pred":
        if cfg.get("output_masks_dir"):
            output_masks_dir = cfg.get("output_masks_dir")

            log.info(f"Generating masks of test dataset")
            
            pred_outputs = trainer.predict(
                model=model,
                dataloaders=datamodule,
                ckpt_path=cfg.ckpt_path,
            )

            preds, mask_names, heights, widths = [], [], [], []
            for p in pred_outputs:
                preds += list(p["preds"])
                mask_names += list(p["mask_names"])
                heights += list(p["heights"])
                widths += list(p["widths"])

            log.info(f"Saving prediction masks in directory {output_masks_dir}")

            # Create directory if it doesn't exist and if exists clear the directory
            if not os.path.exists(output_masks_dir):
                # Recursively create directory
                os.makedirs(output_masks_dir, exist_ok=True)
            else:
                # Clear the directory
                for f in os.listdir(output_masks_dir):
                    os.remove(os.path.join(output_masks_dir, f))

            for pred, mask_name, h, w in zip(preds, mask_names, heights, widths):
                file_path = f"{output_masks_dir}/{mask_name}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                torchvision.utils.save_image(
                    TF.resize(
                        pred.double(),
                        size=[h, w],
                        interpolation=TF.InterpolationMode.NEAREST_EXACT,
                    ),
                    file_path,
                )
        else:
            raise ValueError(
                f"Expected value at output_masks_dir, but got {cfg.get('output_masks_dir')} instead."
            )
    else:
        raise ValueError(
            f"Expected task_name to be either 'eval' or 'pred', but got {cfg.get('task_name')} instead."
        )

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()

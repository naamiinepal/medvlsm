import re
from typing import Any, Callable, Mapping, Optional

import torch
from monai.networks import one_hot
from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F


from ..utils.configs import CLASS_NAMES

_mapping_str_any = Mapping[str, Any]


class BaseModule(LightningModule):
    """Base LightningModule for Binary segmentation.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_monitor: str = "val_loss",
        threshold: float = 0.5,
        multi_class: bool = False,
        log_output_masks: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.loss_fn = loss_fn

    def forward(self, **kwargs):
        return self.net(**kwargs)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def step(self, batch: _mapping_str_any) -> _mapping_str_any:
        masks, mask_names, heights, widths, sentences = (
            batch.pop("mask"),
            batch.pop("mask_name"),
            batch.pop("height"),
            batch.pop("width"),
            batch.pop("sentence")
        )
        dataset = None
        if "dataset" in batch:
            dataset = batch.pop("dataset")
        # Pass batch that contains image, text, and attention_mask (optional) tensors to the model
        pred_masks: torch.Tensor = self.forward(
            **batch
        )  # Logits with shape (B, N, H, W)

        # Convert mask shape: (B, H, W) -> (B, 1, H, W)
        if len(masks.shape) == 3:
            masks = masks[:, None]

        if self.hparams.multi_class:
            # Convert prediction logits to softmax
            pred_masks = pred_masks.softmax(dim=1)
        else:
            # Convert prediction logits to sigmoid
            pred_masks = F.sigmoid(pred_masks)

        # The loss function should accept sigmoid predictions and gt mask of same shape
        loss = self.loss_fn(pred_masks, masks)

        # Convert prediction sigmoids to binary masks
        pred_masks = pred_masks > self.hparams.threshold

        # Convert masks to one-hot encoding if multi-class
        if self.hparams.multi_class:
            one_hot_masks = one_hot(masks, num_classes=pred_masks.shape[1])
        else:
            one_hot_masks = masks

        # Compute metrics
        dice = compute_dice(
            pred_masks, one_hot_masks, ignore_empty=False, include_background=False
        )
        iou = compute_iou(
            pred_masks, one_hot_masks, ignore_empty=False, include_background=False
        )

        step_out = dict(
            loss=loss,
            images=batch["pixel_values"],
            targets=masks,
            preds=pred_masks,
            dice=dice,
            iou=iou,
            mask_names=mask_names,
            heights=heights,
            widths=widths,
        )
        if dataset is not None:
            step_out["dataset"] = dataset
        return step_out

    def training_step(self, batch: _mapping_str_any, batch_idx: int):
        step_out = self.step(batch)
        loss, dice, iou = (
            step_out["loss"],
            step_out["dice"],
            step_out["iou"],
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_dice", dice.mean(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou.mean(), on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_train_epoch_end(self):
        pass

    def compile(self):
        assert re.match(
            r"2.", torch.__version__
        ), "Pytorch version >= 2.X is required to use compile() method."
        return torch.compile(self)

    def validation_step(self, batch: _mapping_str_any, batch_idx: int):
        step_out = self.step(batch)
        loss, images, targets, preds, dice, iou = (
            step_out["loss"],
            step_out["images"],
            step_out["targets"],
            step_out["preds"],
            step_out["dice"],
            step_out["iou"],
        )

        # Log images at the start of validation step
        if (
            batch_idx == 0
            and isinstance(self.logger, WandbLogger)
            and self.hparams.log_output_masks
        )  :
            # Only Log 16 images at max
            max_images_logs = 16
            if len(targets) < max_images_logs:
                max_images_logs = len(targets)

            self.logger.log_image(
                key="val_image", images=list(images.float())[:max_images_logs]
            )
            self.logger.log_image(
                key="val_target_mask", images=list(targets.float())[:max_images_logs]
            )
            self.logger.log_image(
                key="val_pred_mask",
                images=list(((preds > self.hparams.threshold) * 1).float())[
                    :max_images_logs
                ],
            )

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice.mean(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou.mean(), on_step=True, on_epoch=True, prog_bar=True)

        return None

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        pass

    def test_step(self, batch: _mapping_str_any, batch_idx: int):
        step_out = self.step(batch)
        loss, images, targets, preds, dice, iou = (
            step_out["loss"],
            step_out["images"],
            step_out["targets"],
            step_out["preds"],
            step_out["dice"],
            step_out["iou"],
        )

        # Log images at the start of test step
        if (
            batch_idx == 0
            and isinstance(self.logger, WandbLogger)
            and self.hparams.log_output_masks
        ):
            # Only Log 16 images at max
            max_images_logs = 16
            if len(targets) < max_images_logs:
                max_images_logs = len(targets)

            self.logger.log_image(
                key="test_image", images=list(images.float())[:max_images_logs]
            )

            self.logger.log_image(
                key="test_target_mask", images=list(targets.float())[:max_images_logs]
            )
            self.logger.log_image(
                key="test_pred_mask",
                images=list(((preds.sigmoid() > self.hparams.threshold) * 1).float())[
                    :max_images_logs
                ],
            )

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_dice", dice.mean(), on_step=True, on_epoch=True, prog_bar=True)

        self.log("test_iou", iou.mean(), on_step=True, on_epoch=True, prog_bar=True)

        return None

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: _mapping_str_any, batch_idx: int) -> Any:
        step_out = self.step(batch)
        pred_out = dict(
            preds=step_out["preds"],
            mask_names=step_out["mask_names"],
            heights=step_out["heights"],
            widths=step_out["widths"],
        )

        if "dataset" in step_out:
            pred_out["dataset"] = step_out["dataset"]

        return pred_out

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
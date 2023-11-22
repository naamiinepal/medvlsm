from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


class BaseDataModule(LightningDataModule):
    r"""Example of LightningDataModule for VL dataset of coco captions.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def predict_dataloader(self):
            # return prediction dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html

    Args:
        train_dataset (Dataset): Dataset instance or list of Dataset instances.
        val_dataset (Dataset): Dataset instance or list of Dataset instances.
            If None, uses the train_val_split attribute to split train_dataset. Default: None
        test_dataset (Dataset): Dataset instance or list of Dataset instances.
            If None, uses random half split from the val_dataset, redcuing the val_dataset size. Default: None
        pred_dataset (Dataset): Dataset instance or list of Dataset instances.
            If None, uses the test_dataset as pred_dataset. Default: None.
        train_val_split (size, size): Sizes (int or float) of splits from train_dataset for train_dataset and val_dataset.
            If val_dataset is not None, it is not used.
        batch_size (int): Batch size for each step and all (train/val/test/pred) of the Dataloaders.
        num_workder (int): num_workers for torch DataLoaders.
        pin_memory (int): pin_memory for torch DataLoaders.
    """

    def __init__(
        self,
        train_dataset: Dataset | List[Dataset],
        val_dataset: Optional[Dataset | List[Dataset]] = None,
        test_dataset: Optional[Dataset | List[Dataset]] = None,
        pred_dataset: Optional[Dataset | List[Dataset]] = None,
        train_val_split: Tuple[float, float] | Tuple[int, int] = (0.8, 0.2),
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["train_dataset", "val_dataset", "test_dataset", "pred_dataset"],
        )

        # Concatenate if there is a list of multiple datasets.
        if isinstance(train_dataset, Iterable):
            self.data_train = ConcatDataset(train_dataset)
        else:
            self.data_train = train_dataset

        # If val_dataset is not passed, use splits (ie. train_val_split) of the training dataset as training and validation dataset.
        if val_dataset is None:
            self.data_train, self.data_val = random_split(
                dataset=self.data_train, lengths=train_val_split
            )
        # Concatenate if there is a list of multiple datasets.
        elif isinstance(val_dataset, Iterable):
            self.data_val = ConcatDataset(val_dataset)
        else:
            self.data_val = val_dataset

        # If test dataset is not passed, use 50% of the validation dataset as test dataset.
        if test_dataset is None:
            self.data_val, self.data_test = random_split(
                dataset=self.data_val, lengths=[0.5, 0.5]
            )
        # Concatenate if there is a list of multiple datasets.
        elif isinstance(test_dataset, Iterable):
            self.data_test = ConcatDataset(test_dataset)
        else:
            self.data_test = test_dataset

        # If predition dataset is not passed, use test dataset as predition dataset.
        if pred_dataset is None:
            self.data_pred = self.data_test
        # Concatenate if there is a list of multiple datasets.
        elif isinstance(pred_dataset, Iterable):
            self.data_pred = ConcatDataset(pred_dataset)
        else:
            self.data_pred = pred_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)

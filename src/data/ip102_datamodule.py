from typing import Any, Dict, Optional, Tuple

import os
import torch
from components.dataset_manager import DatasetManager
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchvision.transforms import transforms


class IP102DataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
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
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "/home/adhi/large-scale-pest-classification/data/ip102_v1.1/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir

        # data augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.7, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                           
        ])

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 102

    def prepare_data(self):
        """Prepares the dataset by initializing and managing the dataset, creating necessary folders,
        loading class labels, creating subfolders for each class, and moving files to their respective subfolders.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # Directory path for the dataset
        dataset_path = self.data_dir

        # Initialize and manage the dataset
        dataset_manager = DatasetManager(dataset_path)

        # Create necessary folders
        dataset_manager.create_folders()

        # Load class labels
        classes_file = os.path.join(dataset_path, "classes.txt")
        dataset_manager.load_class_labels(classes_file)

        # Create subfolders for each class
        dataset_manager.create_class_subfolders()

        # Move files to respective subfolders
        dataset_manager.move_files("train.txt")
        dataset_manager.move_files("test.txt")
        dataset_manager.move_files("val.txt")


    def setup(self):
        """Loads and splits the datasets if they are not already loaded.

        This method is called by Lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # load and split datasets only if not loaded already
        self.data_train = datasets.ImageFolder('/home/adhi/large-scale-pest-classification/data/ip102_v1.1/images/train', transform=self.augmentation)
        self.data_val = datasets.ImageFolder('/home/adhi/large-scale-pest-classification/data/ip102_v1.1/images/val', transform=self.transforms)
        self.data_test = datasets.ImageFolder('/home/adhi/large-scale-pest-classification/data/ip102_v1.1/images/test', transform=self.transforms)

    def train_dataloader(self):
        """Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader object for the training dataset.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader object for the validation dataset.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Returns a DataLoader for the testing dataset.

        Returns:
            DataLoader: The DataLoader object for the testing dataset.
        """
        return DataLoader(
            dataset=self.data_test,
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
    _ = IP102DataModule()

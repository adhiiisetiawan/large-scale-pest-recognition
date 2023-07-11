from typing import Any

import torch
from .components.cutmix import cutmix
from .components.sparse_regularization import sparse_loss
from .components.insect_pest_net import InsectPestClassifier
# from lightning import LightningModule
import lightning.pytorch as pl
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class InsectPestLitModule(pl.LightningModule):
     """LightningModule implementation for training and evaluating an insect pest classifier.

     Args:
        net (torch.nn.Module, optional): The neural network model. If not provided, a default
            InsectPestClassifier model will be used. Default is None.
        optimizer (torch.optim.Optimizer, optional): The optimizer for training the model. Default
            is torch.optim.Adam.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler for adjusting the
            learning rate during training. Default is torch.optim.lr_scheduler.ReduceLROnPlateau.
        num_classes (int, optional): The number of classes in the classification task. Default is 102.
        freeze (bool, optional): Flag indicating whether to freeze the weights of the model during training.
            Default is True.

     Attributes:
        net (torch.nn.Module): The neural network model.
        criterion (torch.nn.CrossEntropyLoss): The loss function for the classifier.
        train_acc (Accuracy): Metric object for calculating and averaging accuracy during training.
        val_acc (Accuracy): Metric object for calculating and averaging accuracy during validation.
        test_acc (Accuracy): Metric object for calculating and averaging accuracy during testing.
        train_loss (MeanMetric): Metric object for averaging loss during training.
        val_loss (MeanMetric): Metric object for averaging loss during validation.
        test_loss (MeanMetric): Metric object for averaging loss during testing.
        val_acc_best (MaxMetric): Metric object for tracking the best validation accuracy so far.

     Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the network.
        on_train_start():
            Callback function called at the beginning of the training.
        model_step(batch: Any, apply_cutmix: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Performs a single step in the training/validation/test process.
        training_step(batch: Any, batch_idx: int) -> torch.Tensor:
            Performs a training step on a batch of data.
        on_train_epoch_end():
            Callback function called at the end of each training epoch.
        validation_step(batch: Any, batch_idx: int):
            Performs a validation step on a batch of data.
        on_validation_epoch_end():
            Callback function called at the end of each validation epoch.
        test_step(batch: Any, batch_idx: int):
            Performs a test step on a batch of data.
        on_test_epoch_end():
            Callback function called at the end of each testing epoch.
        configure_optimizers() -> Union[Dict, Tuple]:
            Configures the optimizers and learning rate schedulers for the training process.

    """

    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
        num_classes: int = 102,
        freeze: bool = True,
    ):
        super().__init__()

        if net is None:
            net = InsectPestClassifier(output_size=num_classes, freeze=freeze)
            
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any, apply_cutmix: bool = False):
        x, y = batch
        if apply_cutmix:
            x_cutmix, y_a, y_b, lam = cutmix(x, y, alpha=1.0)  # Apply CutMix
            logits = self.forward(x_cutmix)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)  # CutMix loss
        else:
            logits = self.forward(x)
            loss = self.criterion(logits, y)  # Original loss

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch, apply_cutmix=True)

        sparse_reg_loss = sparse_loss(self.net, batch[0])  # Sparse regularization loss
        loss += 0.001 * sparse_reg_loss  # Add sparse regularization loss to the total loss

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch, apply_cutmix=False)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = InsectPestLitModule(None, None, None)

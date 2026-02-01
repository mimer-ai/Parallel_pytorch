#!/usr/bin/env python

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import FSDPStrategy

from torchvision import datasets, transforms, models


# -----------------------------
# DataModule for CIFAR-10
# -----------------------------
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Normalize with ImageNet stats (works well with ResNet50 pretraining)
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        # Students: complete the dataset initialization using torchvision.datasets.CIFAR10
        # Make sure download=False and root=self.data_dir
        # TODO: initialize self.train_set and self.val_set
        # Example:
        # self.train_set = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.train_transforms, download=False)
        # self.val_set = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.val_transforms, download=False)
        raise NotImplementedError(
            "TODO: Initialize CIFAR-10 datasets here (train_set and val_set)."
        )

    def train_dataloader(self):
        # TODO: return DataLoader for self.train_set with shuffle=True
        # Use batch_size=self.batch_size and num_workers=self.num_workers
        raise NotImplementedError("TODO: Implement train_dataloader()")

    def val_dataloader(self):
        # TODO: return DataLoader for self.val_set with shuffle=False
        raise NotImplementedError("TODO: Implement val_dataloader()")


# -----------------------------
# LightningModule for ResNet50 on CIFAR-10
# -----------------------------
class LitResNet50(pl.LightningModule):
    def __init__(
        self,
        num_classes=10,
        lr=0.1,
        weights_path="./model_weights/resnet50_imagenet.pth",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Students: build a torchvision.models.resnet50 backbone
        # Replace the final fc with a Linear layer of out_features=num_classes
        # Tip: model.fc.in_features gives the input dim for the classifier
        # TODO: initialize self.model
        self.model = None  # TODO: replace with a proper ResNet50

        # If weights exist, load them (ignore mismatched classifier with strict=False)
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            try:
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(
                    f"[Info] Loaded weights from {weights_path}. Missing: {missing}, Unexpected: {unexpected}"
                )
            except Exception as e:
                print(f"[Warn] Could not load weights from {weights_path}: {e}")
        else:
            print(
                f"[Info] No external weights found at {weights_path}. Using random initialization."
            )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # TODO: forward pass through self.model
        raise NotImplementedError("TODO: Implement forward()")

    def configure_optimizers(self):
        # TODO: return SGD optimizer with self.hparams.lr
        optimizer = None
        return optimizer

    def training_step(self, batch, batch_idx):
        # TODO: unpack batch, forward pass, compute loss, compute accuracy
        # log training loss and accuracy with self.log
        raise NotImplementedError("TODO: Implement training_step()")

    def validation_step(self, batch, batch_idx):
        # TODO: implement validation pass with accuracy metric and log 'val_loss' and 'val_acc'
        raise NotImplementedError("TODO: Implement validation_step()")


# -----------------------------
# Utility to build strategy
# -----------------------------
def build_strategy(name: str):
    """
    Map string to Lightning's strategy or FSDPStrategy.
    Options (suggested):
      - 'ddp'
      - 'fsdp_full'
      - 'fsdp_shard_grad'
      - 'fsdp_auto_wrap'
    """
    if name == "ddp":
        return "ddp"
    if name.startswith("fsdp"):
        if name == "fsdp_full":
            return FSDPStrategy(sharding_strategy=ShardingStrategy.FULL_SHARD)
        if name == "fsdp_shard_grad":
            return FSDPStrategy(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
    raise ValueError(f"Unknown strategy '{name}'")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on CIFAR-10 with PyTorch Lightning."
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--weights_path", type=str, default="./model_weights/resnet50_imagenet.pth"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)

    # Cluster-related args
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of GPUs to use per node."
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp_full", "fsdp_shard_grad"],
    )
    parser.add_argument("--max_epochs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./lightning_logs")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = LitResNet50(
        num_classes=10,
        lr=args.lr,
        weights_path=args.weights_path,
    )

    strategy = build_strategy(args.strategy)

    callbacks = [
        # TODO: add a ModelCheckpoint that monitors 'val_acc' and saves top-1
        # TODO: add a LearningRateMonitor(logging_interval="epoch")
        # (Bonus) add a RichProgressBar
    ]

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.strategy)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        gradient_clip_val=0.0,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

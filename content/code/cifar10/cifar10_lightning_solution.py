#!/usr/bin/env python
# train_cifar10_pl_solution.py

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import FSDPStrategy

from torchvision import datasets, transforms, models

# Optional FSDP imports (no auto_wrap policy here)
try:
    from torch.distributed.fsdp import ShardingStrategy
except Exception:
    ShardingStrategy = None


# -----------------------------
# DataModule
# -----------------------------
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        self.train_set = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.train_transforms,
            download=False,
        )
        self.val_set = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.val_transforms,
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )


# -----------------------------
# LightningModule
# -----------------------------
class LitResNet50(L.LightningModule):
    def __init__(
        self,
        num_classes=10,
        lr=0.1,
        weights_path="./model_weights/resnet50_imagenet.pth",
    ):
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet50(weights=None)  # do not trigger internet download
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            try:
                # Remove fc layer from ImageNet since I have only 10 classes in the end
                state = {k: v for k, v in state.items() if not k.startswith("fc.")}
                # allow mismatched classifier layer
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

        # Fixed optimization hyperparameters (not exposed via CLI)
        self._momentum = 0.9
        self._weight_decay = 1e-4

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def accuracy(logits, targets):
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean()

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            nesterov=True,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = self.criterion(logits, y)
        val_acc = self.accuracy(logits, y)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


# -----------------------------
# Strategy builder
# -----------------------------
def build_strategy(name: str):
    if name == "ddp":
        return "ddp"
    if name.startswith("fsdp"):
        if FSDPStrategy is None or ShardingStrategy is None:
            raise RuntimeError(
                "FSDP strategy requested but not available in this environment."
            )
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
        description="Train ResNet50 on CIFAR-10 with Lightning."
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
    parser.add_argument("--log_dir", type=str, default="./lightning_logs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Fixed seed (not exposed via CLI)
    L.seed_everything(42, workers=True)

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
        ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, args.strategy, "checkpoints"),
            filename="epoch{epoch:03d}-valacc{val_acc:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.strategy)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        gradient_clip_val=0.0,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

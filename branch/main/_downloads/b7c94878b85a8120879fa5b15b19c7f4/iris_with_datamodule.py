import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback


class TrainingLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # This runs automatically at the end of every epoch
        # We access the logged metrics from the module via trainer.callback_metrics
        loss = trainer.callback_metrics.get("train_loss")
        print(f"Spy Report: Epoch {trainer.current_epoch} ended. Loss: {loss:.4f}")


class IrisClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Input: 4 features (sepal/petal measurements)
        # Hidden: 16 neurons
        # Output: 3 classes (Setosa, Versicolour, Virginica)
        self.layer_1 = nn.Linear(4, 16)
        self.layer_2 = nn.Linear(16, 3)

    def forward(self, x):
        # Standard Forward Pass
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def training_step(self, batch, batch_idx):
        # 1. Unpack batch
        # Tabular data usually comes as (features, labels)
        x, y = batch

        # 2. Forward pass
        logits = self(x)

        # 3. Compute Loss
        # CrossEntropyLoss expects logits for multi-class classification
        loss = F.cross_entropy(logits, y)

        # 4. Log the loss (so our Callback can see it later!)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)


class IrisDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir="./data",
        batch_size=16,
        num_workers=0,
        val_size=0.2,
        random_state=42,
        shuffle=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_file = os.path.join(self.data_dir, "iris.npz")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.random_state = random_state
        self.shuffle = shuffle

        self.train_ds = None
        self.val_ds = None

        self.save_hyperparameters(ignore=["num_workers"])

    def prepare_data(self):
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Expected dataset at {self.data_file}. "
                f"Run the download script provided to create it."
            )

    def setup(self, stage=None):
        data = np.load(self.data_file)
        X = data["X"].astype(np.float32)  # shape (150, 4)
        y = data["y"].astype(np.int64)  # shape (150,)

        # Train/val split (stratified to keep class balance)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state, stratify=y
        )

        # Convert to tensors
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)

        self.train_ds = TensorDataset(X_train, y_train)
        self.val_ds = TensorDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


dm = IrisDataModule(data_dir="./data", batch_size=16, val_size=0.2, random_state=42)
logger_callback = TrainingLogger()
trainer = L.Trainer(max_epochs=10, callbacks=[logger_callback], accelerator="gpu")
model = IrisClassifier()

trainer.fit(model, datamodule=dm)

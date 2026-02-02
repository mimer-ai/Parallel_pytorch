import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as L
from pytorch_ligthning.callbacks import Callback


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


iris = np.load("./data/iris.npz")
X = torch.tensor(iris["X"], dtype=torch.float32)  # Features
y = torch.tensor(iris["y"], dtype=torch.long)  # Labels (0, 1, 2)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = IrisClassifier()

# Initialize Trainer with the Callback
# We plug a "Spy" Callback into the callbacks list
logger_callback = TrainingLogger()
trainer = L.Trainer(max_epochs=10, callbacks=[logger_callback], accelerator="gpu")

# Train
trainer.fit(model, train_loader)

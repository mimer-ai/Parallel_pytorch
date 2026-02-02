# PyTorch Lightning

:::{questions}

- What is PyTorch lightning and what is it used for?
- How do I wrap a PyTorch model in Lightning?
- How do I parallelise a Lightning model on several GPUs?
:::

:::{objectives}

- Learn about PyTorch Lightning and why it is useful
- Learn how to wrap a classical torch model in Lightning
- Train Lightning models on multiple GPUs on a SLURM cluster

:::

## Introduction

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is a lightweight
extension of PyTorch that provides structure, automation, and robustness while
keeping full flexibility. It does not replace PyTorch, in the sense that you
still write PyTorch models, layers, and logic, but it removes much of the
repetitive “glue code” around them. Moreover, it simplifies parallelisation
over multiple GPUs, requiring virtually no changes to the code if using
Lightning types; different parallelisation strategies such as DeepSpeed, DDP
and FSDP are readily available and require minimal configuration.
In order to achieve this, Lightning requires the developer to wrap their
network architecture and logic into Lightning types. This also benefits
development since it improves readability of the code, clearly separating ML
logic from engineering.

Lightning focuses on three core ideas:

1. **Organize code cleanly**  
   Lightning separates the core components of a deep learning project
operations:
   - Models  
   - Data loading  
   - Training loop behavior  
   - Logging & checkpointing  
   - Distributed execution  

   This helps keeping the code modular, readable and more reusable.

2. **Automate engineering, keep architecture flexible**  
   The developer can focus on the model and experiments, while device
placement or multiprocessing are transparently handled by Lightning.
   Lightning handles:
   - Training and validation loops  
   - Mixed precision  
   - Gradient clipping  
   - Device and dtype configuration  
   - Multi‑GPU & multi‑node launch logic  
   - Checkpointing  
   - Logging  

3. **Make code portable across hardware and environments**  
   With Lightning, the same script can run:
   - on CPU  
   - on a single GPU  
   - on multiple GPUs using DDP, FSDP or ZeRO

   Switching execution modes becomes a matter of changing a command‑line flag
   rather than rewriting the whole training script.

### Lightning building blocks

The core API of Lightning rotates around two objects: `LightningModule` and
`Trainer`. `LightningModule`  describes the architecture of the network,
including forward pass, validation and test loops, optimisers and LR
schedulers. Conversely, `Trainer` handles the "engineering" side of things:
running training, validation and test dataloaders, calling callbacks at the
right time (e.g. checkpointing, logging), transparently handling device
placement following the prescribed parallelisation strategy. In particular,
`Callback`s are used to inject custom, non-essential code at appropriate times.
This can be very useful for progress tracking, logging and checkpointing.

:::{demo}

In this example, we will build a simple multilayer perceptron to classify
flowers belonging to the Iris dataset based on a set of measurements
(petal/sepal measurements). We will examine an example script creating this
neural network one snippet at a time.

The whole script can be found at {download}`code/iris/iris_example.py` and the
submit script at {download}`code/iris/job.slurm`. The code assumes the iris
dataset has been downloaded into a `./data` folder. There are scripts to
download it included.
For people running on Leonardo, all the code and data can be found at
`/leonardo/pub/userexternal/ffiusco0/code`.

```python
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import Adam 
import lightning as L 
from lightning.pytorch.callbacks import Callback 
```

Our basic imports, plus `Lighthing` and its `Callback`.

```python
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
```

Here is all that pertains architecture and behaviour of the network:

- In the constructor, we define the network itself (input layer + 16 hidden neurons + output layer)
- The `forward()` method describes the forward pass, which in this case is just a ReLU of the first layer and the output
- The `training_step` method defines the core logic of each training step: get the features and labels from the batch, do the forward pass, compute the loss and log it
- The `configure_optimizers` step schedules an Adam optimiser with a certain learning rate.

```python
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

iris = np.load("./data/iris.npz")
X = torch.tensor(iris["X"], dtype=torch.float32) # Features
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
```

The first part of the snippet loads the Iris dataset from `sklearn` (not
crucial, just an easily accessible source). It then converts it into a format
digestible by PyTorch. The `IrisClassifier` model we created above is then
instantiated.
The Lightning `Trainer` object takes care of the *engineering* of
the flow: sets a number of epochs, which accelerator to use and possibly number
of devices/nodes over which the job can be parallelised with a certain
strategy. Note also the inclusion of a `logger_callback`: this exemplifies the use
of callbacks to trigger the execution of arbitrary code at certain moments of
the training cycle. In this case, our own `TrainingLogger` looks like the
following:

```python
class TrainingLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # This runs automatically at the end of every epoch
        # We access the logged metrics from the module via trainer.callback_metrics
        loss = trainer.callback_metrics.get("train_loss")
        print(f"Spy Report: Epoch {trainer.current_epoch} ended. Loss: {loss:.4f}")
```

Lightning provides some plumbing to create `Callback`s and even some specific
types (learning rate schedulers, gradient accumulation, etc.). In the snippet
above, the training loss is printed after every epoch. A number of trigger
events are available (fit end, fit start, checkpoint loading, test epoch
start...).

:::

### Lightning data modules

When training neural networks in PyTorch, it is usually needed to find/download
the dataset, load it from disk, doing train/validation/test splits and a number
of other preprocessing steps. In the previous example, these steps were
performed in the launching snippet. In order to improve code organisation and
reusability, Lightning introduces a `LightningDataModule`, which takes care of
all these steps and returns the correct torch Dataloaders. A generic
`LightningDataModule` features a number of hooks:

```python
class DataModule(L.LightningDataModule):
  def prepare_data(self):
    ...
  def setup(self, stage=None):
    ...
  def train_dataloader(self):
    ...
  def val_dataloader(self):
    ...

```

- `prepare_data()` runs once and is usually used to download and/or check
files. It runs only on one process, even if multiple GPUs are used.
- `setup(stage)` runs on each rank and loads arrays and creates the datasets.
The `stage` parameter can be used to prepare data for different steps of the
training lifecycle (fit, validate, predict, test)
- `train_dataloader(), val_dataloader(), test_dataloader()` return PyTorch
dataloaders for the different splits.

In the next example, we can see how to refactor the Iris training snippet to
use a Lightning data module:

:::{demo}

```python
class IrisDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir = "./data",
        batch_size = 16,
        num_workers = 0,
        val_size = 0.2,
        random_state = 42,
        shuffle = True,
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
        X = data["X"].astype(np.float32)   # shape (150, 4)
        y = data["y"].astype(np.int64)     # shape (150,)

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
```

Now to train the network:

```python
# Assume that IrisClassifier and TrainingLogger are available 

dm = IrisDataModule(data_dir="./data", batch_size=16, val_size=0.2, random_state=42)
logger_callback = TrainingLogger()
trainer = L.Trainer(max_epochs=10, callbacks=[logger_callback], accelerator="gpu")
model = IrisClassifier()

trainer.fit(model, datamodule=dm)
```

The whole script can be found at {download}`code/iris/iris_with_datamodule.py` and the submit script at {download}`code/iris/job.slurm`.

:::

Lightning works extremely well in SLURM-like environments since the number of
nodes/devices and the parallelisation strategy can be passed as arguments to
the `Trainer` object and can be fed from SLURM environmental variables
(`$SLURM_GPUS_PER_NODE`, `$SLURM_NNODES`). This effectively means that both the
Python script itself and the submit script can stay the same. For example, the number of epochs, nodes, GPUs per node and parallelisation strategy can be parsed as arguments:

```python
from pytorch_lightning.strategies import FSDPStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=2, type=int, metavar='N',
                        help='number of GPUs per node')
parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='maximum number of epochs to run')
parser.add_argument('--accelerator', default='gpu', type=str,
                        help='accelerator to use')
parser.add_argument('--strategy', default='ddp', type=str,
                        help='distributed strategy to use')
args = parser.parse_args()

# Some work needed for FSDP
if args.strategy == "fsdp":
    strategy = FSDPStrategy(
        sharding_strategy="FULL_SHARD",  
        cpu_offload=False
    )
if args.strategy == "fsdp1":
    strategy = FSDPStrategy(
        sharding_strategy="SHARD_GRAD_OP",  
        cpu_offload=False
    )    
if args.strategy == "fsdp2":
    strategy = FSDPStrategy(
        sharding_strategy="NO_SHARD", 
        cpu_offload=False
    )
else:
    strategy = args.strategy 
```

and then passed to the Trainer object:

```python
trainer = L.Trainer(
  devices=args.gpus,
  num_nodes=args.nodes,
  max_epochs=args.epochs,
  accelerator=args.accelerator,
  strategy=strategy,
)
```

## Exercises

::::{exercise} ResNet50 trained on the Cifar10 dataset
In the following example, a ResNet50 is trained on the Cifar10 image dataset to classify images into 10 categories. Fill in the `#TODO` lines and submit using the script at {download}`code/cifar10/job.slurm`. The submit script also allows you to test different parallelisation strategies.

```{literalinclude} code/cifar10/cifar10_lightning_skeleton.py
:language: python
```

:::{solution}

```{literalinclude} code/cifar10/cifar10_lightning_solution.py
:language: python
```

Try to run this example while changing parallelisation strategy/number of
resources from the submit script!
:::

::::

## Summary

PyTorch Lightning can be used to produce more organised, reusable code to train neural network by clearly separating architecture, data and plumbing by taking advantage of `LightningModule`, `LightningDataModule` and `Trainer` respectively. Parallelising over several GPUs/nodes is made transparent and requires virtually no changes to the code.

## See also

- Pytorch Lightning [documentation](https://lightning.ai/docs/pytorch/stable/)

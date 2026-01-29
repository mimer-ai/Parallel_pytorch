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
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset, DataLoader

iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32) # Features
y = torch.tensor(iris.target, dtype=torch.long)  # Labels (0, 1, 2)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = IrisClassifier()

# Initialize Trainer with the Callback
# We plug a "Spy" Callback into the callbacks list
spy_callback = TrainingSpy()
trainer = L.Trainer(max_epochs=10, callbacks=[spy_callback], accelerator="gpu")

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
strategy. Note also the inclusion of a `spy_callback`: this exemplifies the use
of callbacks to trigger the execution of arbitrary code at certain moments of
the training cycle. In this case, our own `TrainingSpy` looks like the
following:

```python
class TrainingSpy(Callback):
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

Lightning works extremely well in SLURM-like environments since the number of
nodes/devices and the parallelisation strategy can be passed as arguments to
the `Trainer` object and can be fed from SLURM environmental variables
(`$SLURM_GPUS_PER_NODE`, `$SLURM_NNODES`). This effectively means that both the
Python script itself and the submit script can stay the same.

## Exercises: description

:::{exercise} Exercise Topic-1: imperative description of exercise
Exercise text here.
:::

:::{solution}
Solution text here
:::

## Summary

A Summary of what you learned and why it might be useful.  Maybe a
hint of what comes next.

## See also

- Pytorch Lightning [documentation](https://lightning.ai/docs/pytorch/stable/)

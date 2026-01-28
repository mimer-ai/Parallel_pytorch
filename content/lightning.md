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
(petal/sepal measurements).

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
-

:::

### Why Lightning is great for HPC and SLURM environments

Clusters come with additional challenges:

- You must handle environment variables for distributed communication.  
- You often need multi-node multi-GPU launches.  
- You want reproducible job scripts that don’t depend on internal PyTorch details.  

Lightning abstracts this away.

Your *training script stays identical* whether you run it:

- interactively  
- via `srun`  
- via `sbatch`  
- on 1 node  
- on 8 nodes  
- with DDP or FSDP  

Your SLURM scripts simply launch the trainer with the appropriate strategy.

This dramatically reduces maintenance and prevents subtle bugs caused by manual multiprocess setup.

---

### How we will use Lightning in this lesson

During this lesson, you will build two full training pipelines:

- **MNIST** (simple MLP)
- **CIFAR10** (CNN or ResNet)

You will run them:

- On CPU  
- On one GPU  
- On multiple GPUs using DDP  
- Using FSDP with different wrapping policies  
- On an HPC cluster using SLURM job scripts  

You will learn how to structure a Lightning project so that:

- Your code remains clean  
- Your experiments are reproducible  
- Scaling up requires minimal changes  
- You understand what Lightning is doing under the hood  

The goal is to give you the tools to write deep learning code that is:

- robust  
- scalable  
- maintainable  
- ready for production or research at scale  

By the end, you should be able to take any PyTorch project and convert it into a Lightning-based design suitable for distributed training and HPC environments.

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

- Other relevant links
- Other link

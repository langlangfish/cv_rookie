# CV Lab 🔬

A lightweight computer vision experiment framework built on PyTorch.
Designed for learning and rapid prototyping on CIFAR-10 and beyond.

## Quick Start

```bash
# Train the default SimpleCNN on CIFAR-10
uv run python -m cv_lab

# Train ResNet-Small with custom hyperparameters
uv run python -m cv_lab --model resnet_small --epochs 20 --lr 0.01 --optimizer sgd

# Run without data augmentation
uv run python -m cv_lab --no-augmentation --experiment-name no_aug
```

## Project Structure

```
cv/
├── pyproject.toml          # Project config & dependencies
├── src/cv_lab/
│   ├── __init__.py         # Package init
│   ├── __main__.py         # CLI entry point
│   ├── config.py           # TrainConfig dataclass
│   ├── data.py             # Dataset loaders & transforms
│   ├── device.py           # Device selection (CPU/GPU/MPS)
│   ├── models.py           # Model registry & architectures
│   ├── train.py            # Training & evaluation loops
│   └── viz.py              # Plotting helpers
├── notebooks/
│   └── 01_quickstart.py    # Getting-started example script
├── data/                   # Auto-downloaded datasets (gitignored)
└── runs/                   # Experiment outputs (gitignored)
```

## Available Models

| Model          | Description                              | CIFAR-10 Target |
| -------------- | ---------------------------------------- | --------------- |
| `simple_cnn`   | 2-block CNN with BatchNorm (baseline)    | ~85%            |
| `resnet_small` | Compact pre-activation ResNet (~20 layers) | ~91%          |

## Adding Your Own Model

```python
from cv_lab.models import register_model
import torch.nn as nn

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... your architecture ...

    def forward(self, x):
        # ... forward pass ...
        return x
```

Then train it:
```bash
uv run python -m cv_lab --model my_model
```

## GPU Note

This project is configured for **CPU-only** locally. For GPU training, use
one of the cloud platforms listed in the project documentation, then install
the CUDA version of PyTorch:

```bash
# On a GPU machine, replace the CPU torch with CUDA torch:
uv remove torch torchvision
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

"""Experiment configuration utilities.

Provides a simple dataclass-based config system for managing
hyperparameters and experiment settings.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class TrainConfig:
    """Training configuration."""

    # --- Data ---
    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 2
    pin_memory: bool = True

    # --- Model ---
    model: str = "simple_cnn"
    num_classes: int = 10
    pretrained: bool = False

    # --- Training ---
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    patience: int = 5  # Early stopping patience (0 to disable)

    # --- Augmentation ---
    use_augmentation: bool = True
    random_crop_padding: int = 4
    horizontal_flip: bool = True

    # --- Experiment ---
    experiment_name: str = "default"
    output_dir: str = "./runs"
    seed: int = 42
    log_interval: int = 50

    # --- Device ---
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def save(self, path: str | Path) -> None:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_args(cls, **overrides) -> "TrainConfig":
        """Create config with overrides."""
        return cls(**overrides)

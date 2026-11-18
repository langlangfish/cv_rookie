"""Quick-start example: Train and evaluate on CIFAR-10.

Run with:
    uv run python notebooks/01_quickstart.py
"""

import sys
sys.path.insert(0, "src")

from cv_lab.config import TrainConfig
from cv_lab.device import device_summary, get_device
from cv_lab.data import get_dataloaders
from cv_lab.models import get_model, list_models
from cv_lab.train import train
from cv_lab.viz import plot_training_history


def main():
    # 1. Check device
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    print(device_summary())
    print(f"\nAvailable models: {list_models()}")
    print()

    # 2. Configure experiment
    cfg = TrainConfig(
        model="simple_cnn",
        epochs=5,              # Quick test; increase for better results
        batch_size=128,
        lr=1e-3,
        experiment_name="quickstart",
    )

    # 3. Train
    result = train(cfg)

    # 4. Plot curves
    from pathlib import Path
    history_path = Path(result["run_dir"]) / "history.json"
    plot_training_history(history_path, save_path=Path(result["run_dir"]) / "curves.png")
    print(f"\nTraining curves saved to {result['run_dir']}/curves.png")


if __name__ == "__main__":
    main()

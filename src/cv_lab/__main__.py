"""CV Lab - Quick start experiment runner.

Usage:
    uv run python -m cv_lab                        # Run with defaults
    uv run python -m cv_lab --model resnet_small   # Use ResNet
    uv run python -m cv_lab --epochs 20 --lr 0.01  # Custom hyperparams
"""

import argparse
from .config import TrainConfig
from .train import train


def main():
    parser = argparse.ArgumentParser(description="CV Lab - Training Runner")
    parser.add_argument("--model", type=str, default="simple_cnn", help="Model name")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0 to disable)")
    parser.add_argument("--experiment-name", type=str, default="default", help="Experiment name")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable augmentation")
    args = parser.parse_args()

    cfg = TrainConfig(
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        patience=args.patience,
        experiment_name=args.experiment_name,
        device=args.device,
        seed=args.seed,
        use_augmentation=not args.no_augmentation,
    )

    print("=" * 60)
    print("CV Lab - Computer Vision Experiment")
    print("=" * 60)
    train(cfg)


if __name__ == "__main__":
    main()

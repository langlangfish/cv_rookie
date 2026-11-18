"""Visualization helpers for experiment results."""

from pathlib import Path
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(history_path: str | Path, save_path: str | Path | None = None):
    """Plot training/validation loss and accuracy curves.

    Args:
        history_path: Path to history.json produced by train().
        save_path:    If given, save the figure instead of showing it.
    """
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train")
    ax1.plot(epochs, val_loss, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train")
    ax2.plot(epochs, val_acc, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


def show_predictions(model, test_loader, device, num_images: int = 16, save_path=None):
    """Show a grid of test images with predictions vs ground truth."""
    classes = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images], labels[:num_images]

    with torch.no_grad():
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu()

    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i].imshow(img)
        color = "green" if preds[i] == labels[i] else "red"
        axes[i].set_title(
            f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}",
            color=color, fontsize=9,
        )
        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()

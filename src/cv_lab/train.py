"""Training and evaluation loops."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .config import TrainConfig
from .device import get_device
from .data import get_dataloaders
from .models import get_model


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """Create optimizer from config."""
    if cfg.optimizer == "adam":
        return Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return SGD(
            model.parameters(), lr=cfg.lr,
            momentum=0.9, weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: TrainConfig):
    """Create LR scheduler from config."""
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "step":
        return StepLR(optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.1)
    elif cfg.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> dict:
    """Train for one epoch and return metrics."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  [Epoch {epoch}] Batch {batch_idx + 1}/{len(loader)}  "
                f"Loss: {loss.item():.4f}  Acc: {100.0 * correct / total:.2f}%"
            )

    return {
        "train_loss": running_loss / total,
        "train_acc": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "val_loss": running_loss / total,
        "val_acc": 100.0 * correct / total,
    }


def train(cfg: TrainConfig) -> dict:
    """Full training pipeline.

    Args:
        cfg: Training configuration.

    Returns:
        Dictionary with final metrics and best checkpoint path.
    """
    # Seed for reproducibility
    torch.manual_seed(cfg.seed)

    # Device
    device = get_device(cfg.device)
    from .device import device_summary
    print("-" * 50)
    print(device_summary(device))
    print("-" * 50)

    # Data
    train_loader, test_loader = get_dataloaders(cfg)
    print(f"Dataset: {cfg.dataset} | Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    model = get_model(cfg.model, num_classes=cfg.num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model} | Parameters: {param_count:,}")

    # Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()

    # Output dir
    run_dir = Path(cfg.output_dir) / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    # Training loop
    best_acc = 0.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, cfg.log_interval,
        )
        val_metrics = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_data = {
            "epoch": epoch, "lr": lr, "time": elapsed,
            **train_metrics, **val_metrics,
        }
        history.append(epoch_data)

        print(
            f"Epoch {epoch}/{cfg.epochs}  "
            f"Train Loss: {train_metrics['train_loss']:.4f}  "
            f"Train Acc: {train_metrics['train_acc']:.2f}%  "
            f"Val Loss: {val_metrics['val_loss']:.4f}  "
            f"Val Acc: {val_metrics['val_acc']:.2f}%  "
            f"LR: {lr:.6f}  Time: {elapsed:.1f}s"
        )

        # Save best
        if val_metrics["val_acc"] > best_acc:
            best_acc = val_metrics["val_acc"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print(f"  → New best! ({best_acc:.2f}%)")

    # Save final
    torch.save(model.state_dict(), run_dir / "last_model.pt")

    # Save history
    import json
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")
    print(f"Artifacts saved to: {run_dir}")

    return {"best_acc": best_acc, "history": history, "run_dir": str(run_dir)}

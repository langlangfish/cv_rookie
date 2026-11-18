"""Dataset loaders for computer vision experiments.

Currently supports CIFAR-10. Easily extensible to CIFAR-100, ImageNet, etc.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig


# CIFAR-10 channel-wise mean and std (precomputed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms(cfg: TrainConfig):
    """Build train/test transforms for CIFAR-10."""
    train_transforms = []
    if cfg.use_augmentation:
        train_transforms.extend([
            transforms.RandomCrop(32, padding=cfg.random_crop_padding),
        ])
        if cfg.horizontal_flip:
            train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def get_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders based on config.

    Args:
        cfg: Training configuration.

    Returns:
        (train_loader, test_loader)
    """
    if cfg.dataset == "cifar10":
        train_tf, test_tf = get_cifar10_transforms(cfg)
        train_ds = datasets.CIFAR10(
            root=cfg.data_dir, train=True, download=True, transform=train_tf
        )
        test_ds = datasets.CIFAR10(
            root=cfg.data_dir, train=False, download=True, transform=test_tf
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, test_loader

"""Model registry and built-in architectures.

Add new models here and register them so the training loop can
instantiate them by name from the config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return names of all registered models."""
    return list(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------

@register_model("simple_cnn")
class SimpleCNN(nn.Module):
    """A minimal 2-conv-block CNN for CIFAR-10 (baseline).

    Architecture:
        Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout
        Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout
        FC(64*8*8 → 256) → ReLU → Dropout → FC(256 → num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model("resnet_small")
class ResNetSmall(nn.Module):
    """A compact ResNet-style network for 32×32 images.

    Uses BasicBlock residual connections with pre-activation (BN → ReLU → Conv).
    Total depth: ~20 layers.
    """

    class BasicBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

        def forward(self, x):
            out = F.relu(self.bn1(x), inplace=True)
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = F.relu(self.bn2(out), inplace=True)
            out = self.conv2(out)
            return out + shortcut

    def __init__(self, num_classes: int = 10, num_blocks: list[int] | None = None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3]  # 3 stages × 3 blocks = ~20 layers

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        self.stage1 = self._make_stage(16, 16, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(16, 32, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks[2], stride=2)

        self.bn_final = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [self.BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(self.BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn_final(x), inplace=True)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

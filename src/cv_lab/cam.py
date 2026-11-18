"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Helps visualize what regions of an image the CNN is focusing on
when making predictions.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class GradCAM:
    """Grad-CAM visualizer using PyTorch forward/backward hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to catch features and gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for an input image tensor (shape: [1, 3, H, W])."""
        self.model.eval()
        self.model.zero_grad()

        # 1. Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 2. Backward pass for the target score
        target_score = output[0, target_class]
        target_score.backward()

        # 3. Global average pooling of gradients -> weights alpha_k
        # gradients shape: [1, Channels, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 5. Apply ReLU (we only care about positive influences)
        cam = F.relu(cam)

        # 6. Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # 7. Upsample to original image size
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy(), target_class


def visualize_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image_tensor: torch.Tensor,
    true_label: int,
    classes: tuple = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ),
    save_path: str | Path | None = None,
):
    """Generate and save a 3-panel visualization: Original, Heatmap, Overlay."""
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class = grad_cam.generate_cam(image_tensor.unsqueeze(0))

    # Denormalize image for display
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # 1. Original
    axes[0].imshow(img)
    axes[0].set_title(f"True: {classes[true_label]}", fontsize=11)
    axes[0].axis("off")

    # 2. CAM Heatmap
    im2 = axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)  # 50% transparency overlay
    pred_color = "green" if pred_class == true_label else "red"
    axes[2].set_title(f"Pred: {classes[pred_class]}", color=pred_color, fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f">>> Grad-CAM visualization saved to: {save_path}")
    plt.close()
    return cam, pred_class

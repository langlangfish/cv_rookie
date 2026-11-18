"""Demo script: Generate and save Grad-CAM heatmaps for test images.

Usage:
    uv run python notebooks/02_gradcam_demo.py
"""

from pathlib import Path
import torch
from torchvision import datasets, transforms

from cv_lab.models import get_model
from cv_lab.cam import visualize_gradcam
from cv_lab.device import get_device

# CIFAR-10 classes
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def main():
    device = get_device("auto")
    print(f"Using device: {device}")

    # 1. Instantiate model (ResNetSmall by default)
    model = get_model("resnet_small", num_classes=10).to(device)

    # Load trained weights if available
    checkpoint_path = Path("runs/default/best_model.pt")
    if checkpoint_path.exists():
        print(f"Loading trained weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Note: Trained checkpoint not found, using initialized weights for demonstration.")

    # 2. Pick target convolutional layer
    # For ResNetSmall, the last conv layer is stage3's last block conv2
    target_layer = model.stage3[-1].conv2

    # 3. Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # 4. Generate CAM for a few samples
    output_dir = Path("runs/gradcam_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Grad-CAM visualizations...")
    sample_indices = [0, 1, 2, 3, 4]  # 5 test images
    for idx in sample_indices:
        img_tensor, true_label = test_ds[idx]
        img_tensor = img_tensor.to(device)
        save_path = output_dir / f"sample_{idx}_{CLASSES[true_label]}.png"
        visualize_gradcam(
            model=model,
            target_layer=target_layer,
            image_tensor=img_tensor,
            true_label=true_label,
            classes=CLASSES,
            save_path=save_path,
        )

    print(f"\nAll visualizations saved to folder: {output_dir}/")


if __name__ == "__main__":
    main()

"""Device selection and info utilities."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available device.

    Args:
        preference: "auto" picks the best available; also accepts
                    "cpu", "cuda", "cuda:0", "mps".

    Returns:
        The selected torch.device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(preference)
    return dev


def device_summary(device: torch.device | None = None) -> str:
    """Return a human-readable summary of the compute device."""
    if device is None:
        device = get_device()

    lines = [f"Device : {device}"]
    if device.type == "cuda":
        lines.append(f"GPU    : {torch.cuda.get_device_name(device)}")
        mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        lines.append(f"VRAM   : {mem:.1f} GB")
    elif device.type == "mps":
        lines.append("GPU    : Apple Silicon (MPS)")
    else:
        lines.append("GPU    : None (CPU only)")
    return "\n".join(lines)

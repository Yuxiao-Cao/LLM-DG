"""
Utility functions for behavioral cloning baselines.

Includes seed setting, checkpoint saving/loading, and metrics aggregation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy, torch, and cuda.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    scaler: Any,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    filepath: str = "checkpoint.pth",
) -> None:
    """
    Save model checkpoint including weights, scaler params, and config.

    Args:
        model: PyTorch model to save
        scaler: StandardScaler with mean/std parameters
        config: Training configuration dictionary
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Dictionary of metrics to save (optional)
        filepath: Path to save checkpoint
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "scaler_params": scaler.to_dict() if hasattr(scaler, "to_dict") else None,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, filepath)

    # Also save config as separate JSON for easy inspection
    config_path = filepath.with_suffix(".config.json")
    with open(config_path, "w") as f:
        json.dump({"config": config, "scaler_params": checkpoint["scaler_params"]}, f, indent=2)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    scaler: Any = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model instance to load weights into
        scaler: StandardScaler to load parameters into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load model to

    Returns:
        Dictionary with checkpoint information (epoch, metrics, config, etc.)
    """
    # Load checkpoint with weights_only=False for compatibility with older PyTorch versions
    # and to support numpy types in scaler_params
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load scaler parameters if provided
    if scaler is not None and "scaler_params" in checkpoint:
        if checkpoint["scaler_params"] is not None:
            scaler.from_dict(checkpoint["scaler_params"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Return checkpoint info
    info = {
        "model_class": checkpoint.get("model_class"),
        "config": checkpoint.get("config", {}),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
    }

    return info


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate list of metric dictionaries into mean and std.

    Args:
        metrics_list: List of metric dictionaries, e.g.,
                      [{"safety": 80, "efficiency": 70}, {"safety": 85, ...}]

    Returns:
        Dictionary with "mean" and "std" for each metric key
    """
    if not metrics_list:
        return {}

    # Get all keys
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())

    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return result


def print_metrics_summary(metrics: Dict[str, Dict[str, float]], title: str = "Metrics") -> None:
    """
    Print aggregated metrics in a readable format.

    Args:
        metrics: Output from aggregate_metrics
        title: Title for the summary
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    for key, values in sorted(metrics.items()):
        mean_val = values["mean"]
        std_val = values["std"]
        print(f"{key:30s}: {mean_val:8.3f} +/- {std_val:7.3f}")

    print(f"{'='*60}\n")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

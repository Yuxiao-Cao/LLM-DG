#!/usr/bin/env python3
"""
Training script for behavioral cloning baselines.

Enhanced with better hyperparameters, learning rate scheduling,
and improved training loop for better convergence.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.dataset import load_and_split_data, dataframe_to_tensors
from src.baselines.bc_mlp import BCMLP, create_mlp_model
from src.baselines.bc_transformer import BCTransformer, create_transformer_model
from src.baselines.utils import (
    set_seed,
    save_checkpoint,
    get_device,
    count_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train behavioral cloning baseline for LLM-DG"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/example_data.csv",
        help="Path to CSV dataset",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "transformer"],
        default="mlp",
        help="Model architecture",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for scheduler",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.5,
        help="Delta for Huber loss (smaller = more robust to outliers)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for learning rate",
    )

    # Split arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/baselines",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--no-role-swap",
        action="store_true",
        help="Disable role-swap data augmentation",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )

    return parser.parse_args()


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)

        # Backward pass with gradient clipping for stability
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    # Step scheduler
    if scheduler is not None:
        scheduler.step()

    return {"loss": total_loss / n_samples}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            predictions = model(features)
            loss = criterion(predictions, labels)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Calculate MAE as additional metric
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    mae = np.mean(np.abs(all_predictions - all_labels))

    return {"loss": total_loss / n_samples, "mae": mae}


def create_data_loaders(
    args, train_df, val_df, scaler
) -> tuple:  # (train_loader, val_loader)
    """Create PyTorch DataLoaders from dataframes."""
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, features, labels, scenario_ids):
            self.features = torch.from_numpy(features).float()
            self.labels = torch.from_numpy(labels).float().unsqueeze(1)
            self.scenario_ids = scenario_ids

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "label": self.labels[idx],
                "scenario_id": str(self.scenario_ids[idx]),
            }

    # Extract features and labels
    train_features, train_labels, train_scenario_ids = dataframe_to_tensors(
        train_df, scaler, normalize=True
    )
    val_features, val_labels, val_scenario_ids = dataframe_to_tensors(
        val_df, scaler, normalize=True
    )

    train_dataset = SimpleDataset(train_features, train_labels, train_scenario_ids)
    val_dataset = SimpleDataset(val_features, val_labels, val_scenario_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Drop last incomplete batch for stable training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Extract dataset name from data path for file naming
    dataset_name = Path(args.data_path).stem

    # Load data
    print(f"Loading data from {args.data_path}")
    train_df, val_df, test_df, scaler = load_and_split_data(
        csv_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        apply_role_swap=not args.no_role_swap,
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(args, train_df, val_df, scaler)

    # Create model
    print(f"\nCreating {args.model.upper()} model")
    if args.model == "mlp":
        model = create_mlp_model(input_dim=4, output_bound=3.0, dropout=args.dropout).to(device)
    else:  # transformer
        model = create_transformer_model(feature_dim=2, output_bound=3.0, dropout=args.dropout).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_maes = []

    print("\nStarting training...")
    print(f"Initial learning rate: {args.lr}")
    print(f"Min learning rate: {args.min_lr}")
    print(f"Warmup epochs: {args.warmup_epochs}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        train_losses.append(train_metrics["loss"])

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_metrics["loss"])
        val_maes.append(val_metrics["mae"])

        current_lr = scheduler.get_lr()

        print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f} | Val MAE: {val_metrics['mae']:.4f} | LR: {current_lr:.2e}")

        # Early stopping with tolerance
        if val_metrics["loss"] < best_val_loss * 0.999:  # 0.1% tolerance for numerical stability
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            # Save best checkpoint
            checkpoint_path = out_dir / f"{args.model}_{dataset_name}_best.pth"
            config = {
                "model": args.model,
                "seed": args.seed,
                "epochs": epoch + 1,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "batch_size": args.batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "huber_delta": args.huber_delta,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "role_swap_augmentation": not args.no_role_swap,
                "n_parameters": n_params,
                "data_path": args.data_path,
                "dataset_name": dataset_name,
                "dropout": args.dropout,
                "warmup_epochs": args.warmup_epochs,
            }
            save_checkpoint(
                model=model,
                scaler=scaler,
                config=config,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], "val_mae": val_metrics["mae"]},
                filepath=str(checkpoint_path),
            )
            print(f"  >>> Saved best checkpoint (val_loss: {best_val_loss:.6f}, val_mae: {val_metrics['mae']:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping after {epoch + 1} epochs (patience={args.patience})")
            break

    # Save training history
    history_path = out_dir / f"{args.model}_{dataset_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_losses": [float(x) for x in train_losses],
                "val_losses": [float(x) for x in val_losses],
                "val_maes": [float(x) for x in val_maes],
                "best_val_loss": float(best_val_loss),
                "final_val_mae": float(val_maes[-1]) if val_maes else None,
                "stopped_at_epoch": epoch + 1,
            },
            f,
            indent=2,
        )
    print(f"\nTraining history saved to {history_path}")
    print(f"Best checkpoint saved to {out_dir / f'{args.model}_{dataset_name}_best.pth'}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

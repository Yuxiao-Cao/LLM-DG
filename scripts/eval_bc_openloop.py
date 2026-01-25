#!/usr/bin/env python3
"""
Evaluation script for behavioral cloning baselines (open-loop).

Loads a trained checkpoint, runs inference on test split, records per-sample
inference time, and computes evaluation metrics using the existing evaluation
functions from src/evaluation.py.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.dataset import load_and_split_data, dataframe_to_tensors
from src.baselines.bc_mlp import BCMLP, create_mlp_model
from src.baselines.bc_transformer import BCTransformer, create_transformer_model
from src.baselines.utils import (
    set_seed,
    load_checkpoint,
    get_device,
    aggregate_metrics,
    print_metrics_summary,
)
from src.data_models import InteractionScenario, VehicleState, LLMDecision
from src.evaluation import PreciseEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate behavioral cloning baseline (open-loop)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        default='checkpoints/baselines/transformer_best.pth',
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/example_data.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/baseline_bc",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if None)",
    )

    return parser.parse_args()


def create_test_dataset_and_mapping(
    args, scaler
) -> Tuple[DataLoader, pd.DataFrame, Dict[str, List[int]]]:
    """
    Create test dataset and a mapping from scenario_id to predictions.

    Returns:
        Tuple of (test_loader, test_df, scenario_id_to_indices)
    """
    # Load data with same split
    _, _, test_df, _ = load_and_split_data(
        csv_path=args.data_path,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=args.seed,
        apply_role_swap=False,  # No augmentation for test
    )

    # Extract features and labels
    test_features, test_labels, test_scenario_ids = dataframe_to_tensors(
        test_df, scaler, normalize=True
    )

    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = torch.from_numpy(features).float()
            self.labels = torch.from_numpy(labels).float()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "label": self.labels[idx],
            }

    test_dataset = SimpleDataset(test_features, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Build mapping: scenario_id -> list of indices in the test set
    # Use string keys to handle both numeric and string scenario IDs
    scenario_id_to_indices = {}
    for idx, sid in enumerate(test_scenario_ids):
        sid_str = str(sid)  # Keep as string for consistent comparison
        if sid_str not in scenario_id_to_indices:
            scenario_id_to_indices[sid_str] = []
        scenario_id_to_indices[sid_str].append(idx)

    return test_loader, test_df, scenario_id_to_indices


def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference on test set.

    Returns:
        Predictions array
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)

            # Batch prediction
            preds = model(features).cpu().numpy().flatten()
            all_predictions.extend(preds)

    return np.array(all_predictions)


def evaluate_with_metrics(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    scenario_id_to_indices: Dict[str, List[int]],
) -> List[Dict]:
    """
    Compute evaluation metrics using existing evaluation.py functions.

    For each test sample, creates an InteractionScenario and uses PreciseEvaluator.
    """
    # Create evaluator
    evaluator = PreciseEvaluator()

    results = []

    # Process each row in test_df
    for test_idx, row in test_df.iterrows():
        scenario_id = str(row["Scenario_id"])  # Keep as string to handle non-numeric IDs
        frame_id = int(row["frame_id"])

        # Get the index of this scenario in our predictions
        if scenario_id not in scenario_id_to_indices:
            print(f"Warning: Scenario {scenario_id} not in prediction indices")
            continue

        # Use the first occurrence of this scenario_id
        pred_idx = scenario_id_to_indices[scenario_id][0]
        predicted_accel = predictions[pred_idx]

        # Create InteractionScenario
        vehicle_1 = VehicleState(
            vehicle_id=str(row["track_id_1"]),
            distance=float(row["d_1"]),
            velocity=float(row["v_1"]),
            acceleration=float(row["a_1"]) if pd.notna(row["a_1"]) else None,
        )
        vehicle_2 = VehicleState(
            vehicle_id=str(row["track_id_2"]),
            distance=float(row["d_2"]),
            velocity=float(row["v_2"]),
            acceleration=float(row["a_2"]) if pd.notna(row["a_2"]) else None,
        )

        scenario = InteractionScenario(
            scenario_id=str(scenario_id),
            frame_id=frame_id,
            vehicle_1=vehicle_1,
            vehicle_2=vehicle_2,
            scenario_type=str(row["Scenario_type"]) if pd.notna(row["Scenario_type"]) else None,
            ground_truth_priority=str(row["priority"]) if pd.notna(row["priority"]) else None,
        )

        # Create LLMDecision with BC prediction
        bc_decision = LLMDecision(
            acceleration_1=float(predicted_accel),
            reasoning="BC baseline prediction",
            confidence=1.0,
        )

        # Evaluate using PreciseEvaluator
        result = evaluator.evaluate_scenario(
            scenario=scenario,
            llm_decision=bc_decision,
            baseline_decision=vehicle_1.acceleration,
        )

        results.append(
            {
                "scenario_id": str(scenario_id),
                "frame_id": frame_id,
                "predicted_acceleration": float(predicted_accel),
                "ground_truth_acceleration": float(row["a_1"]),
                "acceleration_mae": float(abs(predicted_accel - row["a_1"])),
                "safety_score": float(result.metrics.safety_score),
                "efficiency_score": float(result.metrics.efficiency_score),
                "compliance_score": float(result.metrics.compliance_score),
                "rationality_score": float(result.metrics.rationality_score),
                "overall_score": float(result.metrics.overall_score),
            }
        )

    return results


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recreate model with dropout from config if available
    # First, we need to peek at the checkpoint to get the model class and config
    import json
    config_path = Path(args.checkpoint).with_suffix('.config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            checkpoint_data = json.load(f)
        config = checkpoint_data.get('config', {})
        model_class_name = config.get('model', '')
        dropout = config.get('dropout', 0.1)
    else:
        # Fallback: try to load from checkpoint (may fail on older PyTorch)
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model_class_name = checkpoint.get("config", {}).get("model", "")
            config = checkpoint.get("config", {})
            dropout = config.get("dropout", 0.1)
        except Exception as e:
            print(f"Warning: Could not load config from checkpoint: {e}")
            print("Defaulting to MLP model")
            model_class_name = "mlp"
            config = {}
            dropout = 0.1

    # Recreate model
    if model_class_name == "mlp" or "mlp" in str(args.checkpoint).lower():
        model = create_mlp_model(input_dim=4, output_bound=3.0, dropout=dropout)
        model_name = "mlp"
    else:
        model = create_transformer_model(feature_dim=2, output_bound=3.0, dropout=dropout)
        model_name = "transformer"

    # Load scaler
    from src.baselines.dataset import StandardScaler

    scaler = StandardScaler()

    # Load checkpoint (this loads model weights and scaler params)
    load_checkpoint(args.checkpoint, model, scaler, device=device)
    model.to(device)


    print(f"Model: {model_name}")
    if config:
        print(f"Config: {json.dumps(config, indent=2)}")
    # Create test dataset and scenario mapping
    test_loader, test_df, scenario_id_to_indices = create_test_dataset_and_mapping(args, scaler)
    print(f"Test samples: {len(test_df)}")

    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, test_loader, device)

    # Also get labels for basic metrics
    test_features, test_labels, _ = dataframe_to_tensors(test_df, scaler, normalize=True)

    # Compute basic metrics
    mae = np.mean(np.abs(predictions - test_labels))
    rmse = np.sqrt(np.mean((predictions - test_labels) ** 2))
    print(f"\nAcceleration MAE:  {mae:.4f} m/s²")
    print(f"Acceleration RMSE: {rmse:.4f} m/s²")

    # Evaluate using existing metrics
    print("\nComputing evaluation metrics using src/evaluation.py...")
    eval_results = evaluate_with_metrics(
        predictions, test_df, scenario_id_to_indices
    )

    print(f"Evaluated {len(eval_results)} scenarios")

    if len(eval_results) == 0:
        print("ERROR: No evaluation results generated!")
        print(f"Scenario IDs in test_df: {test_df['Scenario_id'].unique()}")
        print(f"Scenario ID mapping keys: {list(scenario_id_to_indices.keys())}")
        return

    # Aggregate metrics
    metrics_keys = [
        "safety_score",
        "efficiency_score",
        "compliance_score",
        "rationality_score",
        "overall_score",
    ]
    metrics_list = []
    for r in eval_results:
        metrics_list.append({k: r[k] for k in metrics_keys})

    aggregated = aggregate_metrics(metrics_list)
    print_metrics_summary(aggregated, "LLM-DG Evaluation Metrics (Open-Loop)")

    # Save results to CSV
    dataset_name = Path(args.data_path).stem
    csv_path = out_dir / f"{model_name}_{dataset_name}_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "scenario_id",
            "frame_id",
            "predicted_acceleration",
            "ground_truth_acceleration",
            "acceleration_mae",
            "safety_score",
            "efficiency_score",
            "compliance_score",
            "rationality_score",
            "overall_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in eval_results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nResults saved to: {csv_path}")
    print(f"Total rows written: {len(eval_results)}")

    # Save summary JSON
    summary_path = out_dir / f"{model_name}_{dataset_name}_summary.json"
    summary = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "data_path": args.data_path,
        "dataset_name": dataset_name,
        "test_samples": len(eval_results),
        "acceleration_mae": float(mae),
        "acceleration_rmse": float(rmse),
        "eval_metrics": aggregated,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

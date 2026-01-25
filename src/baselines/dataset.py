"""
Dataset module for behavioral cloning baselines.

Loads CSV data, splits by Scenario_id, performs feature extraction and normalization.
Includes role-swap augmentation and z-score normalization fitted on training data only.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    """
    Simple z-score normalizer fitted on training data only.
    Parameters are saved to checkpoint for reproducible inference.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        """Fit scaler on data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return data * self.std + self.mean

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert scaler parameters to dict for checkpointing."""
        return {
            "mean": [float(x) for x in self.mean.tolist()] if self.mean is not None else [],
            "std": [float(x) for x in self.std.tolist()] if self.std is not None else [],
        }

    def from_dict(self, params: Dict[str, List[float]]) -> None:
        """Load scaler parameters from dict."""
        self.mean = np.array(params["mean"], dtype=np.float32) if params["mean"] else None
        self.std = np.array(params["std"], dtype=np.float32) if params["std"] else None


class BCDataset(Dataset):
    """
    PyTorch Dataset for behavioral cloning.

    Features: [d_ego, v_ego, d_opp, v_opp]
    Label: a_ego (acceleration for ego vehicle)

    Includes role-swap augmentation where each sample generates an additional
    sample with ego/opp roles swapped.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        scenario_ids: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        normalize: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            features: Raw feature array of shape (N, 4)
            labels: Label array of shape (N,)
            scenario_ids: Scenario ID for each sample
            scaler: Fitted StandardScaler for normalization
            normalize: Whether to apply z-score normalization
        """
        self.features_raw = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.scenario_ids = scenario_ids
        self.scaler = scaler
        self.normalize = normalize

        if normalize and scaler is not None:
            self.features = scaler.transform(features).astype(np.float32)
        else:
            self.features = self.features_raw

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.from_numpy(np.array([self.labels[idx]])),
            "scenario_id": str(self.scenario_ids[idx]),
        }


def load_and_split_data(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    apply_role_swap: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Load CSV data and split by Scenario_id into train/val/test sets.

    Ensures all frames from the same scenario stay in the same split.
    Fits scaler on training data only.

    Args:
        csv_path: Path to CSV file with columns:
                  Scenario_type, Scenario_id, frame_id, d_1, v_1, a_1, d_2, v_2, a_2, priority
        train_ratio: Fraction of scenarios for training (default 0.7)
        val_ratio: Fraction of scenarios for validation (default 0.1)
        test_ratio: Fraction of scenarios for testing (default 0.2)
        seed: Random seed for reproducible splits
        apply_role_swap: Whether to add role-swapped augmented samples

    Returns:
        Tuple of (train_df, val_df, test_df, fitted_scaler)
    """
    np.random.seed(seed)

    df = pd.read_csv(csv_path)

    # Get unique scenario IDs
    scenario_ids = df["Scenario_id"].unique()
    n_scenarios = len(scenario_ids)
    n_train = int(n_scenarios * train_ratio)
    n_val = int(n_scenarios * val_ratio)

    # Shuffle and split scenario IDs
    shuffled_ids = np.random.permutation(scenario_ids)
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val :]

    # Split dataframes by scenario ID
    train_df = df[df["Scenario_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["Scenario_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["Scenario_id"].isin(test_ids)].reset_index(drop=True)

    # Apply role-swap augmentation to training data
    if apply_role_swap:
        train_df = _apply_role_swap(train_df)

    # Fit scaler on training features only
    train_features = _extract_features(train_df)
    scaler = StandardScaler()
    scaler.fit(train_features)

    return train_df, val_df, test_df, scaler


def _extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract features [d_ego, v_ego, d_opp, v_opp] from dataframe."""
    # Vehicle 1 is treated as ego
    features = np.column_stack(
        [
            df["d_1"].values,
            df["v_1"].values,
            df["d_2"].values,
            df["v_2"].values,
        ]
    )
    return features


def _extract_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract labels (a_ego = a_1) from dataframe."""
    return df["a_1"].values


def _apply_role_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create augmented dataset by swapping ego/opp roles.

    For each original sample (d_1, v_1, a_1, d_2, v_2, a_2), creates a new sample
    with swapped roles: (d_2, v_2, a_2, d_1, v_1, a_1).

        The new sample has:
        - d_ego = original d_2, v_ego = original v_2, a_ego = original a_2
        - d_opp = original d_1, v_opp = original v_1

    Args:
        df: Original dataframe

    Returns:
        Dataframe with original + role-swapped samples
    """
    swapped = df.copy()

    # Swap role 1 <-> role 2
    swapped["d_1"], swapped["d_2"] = swapped["d_2"].values, swapped["d_1"].values
    swapped["v_1"], swapped["v_2"] = swapped["v_2"].values, swapped["v_1"].values
    swapped["a_1"], swapped["a_2"] = swapped["a_2"].values, swapped["a_1"].values

    # Update priority if it exists (swap vehicle_1 <-> vehicle_2)
    if "priority" in swapped.columns:
        swapped["priority"] = swapped["priority"].map(
            {"vehicle_1": "vehicle_2", "vehicle_2": "vehicle_1"}
        ).fillna(swapped["priority"])

    # Combine original and swapped
    return pd.concat([df, swapped], ignore_index=True)


def dataframe_to_tensors(
    df: pd.DataFrame, scaler: StandardScaler, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert dataframe to feature tensors and labels.

    Args:
        df: Input dataframe
        scaler: Fitted StandardScaler
        normalize: Whether to normalize features

    Returns:
        Tuple of (features, labels, scenario_ids)
    """
    features = _extract_features(df).astype(np.float32)
    labels = _extract_labels(df).astype(np.float32)
    scenario_ids = df["Scenario_id"].values

    if normalize:
        features = scaler.transform(features).astype(np.float32)

    return features, labels, scenario_ids


def get_train_val_test_loaders(
    csv_path: str,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Convenience function to create train/val/test DataLoaders.

    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for DataLoaders
        train_ratio: Fraction of scenarios for training
        val_ratio: Fraction of scenarios for validation
        test_ratio: Fraction of scenarios for testing
        seed: Random seed for reproducible splits
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    train_df, val_df, test_df, scaler = load_and_split_data(
        csv_path=csv_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Create datasets
    train_features, train_labels, train_scenario_ids = dataframe_to_tensors(
        train_df, scaler, normalize=True
    )
    val_features, val_labels, val_scenario_ids = dataframe_to_tensors(
        val_df, scaler, normalize=True
    )
    test_features, test_labels, test_scenario_ids = dataframe_to_tensors(
        test_df, scaler, normalize=True
    )

    train_dataset = BCDataset(train_features, train_labels, train_scenario_ids, scaler)
    val_dataset = BCDataset(val_features, val_labels, val_scenario_ids, scaler)
    test_dataset = BCDataset(test_features, test_labels, test_scenario_ids, scaler)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, scaler

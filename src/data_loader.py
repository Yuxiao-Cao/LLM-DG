"""
Data loading utilities for INTERACTIONS dataset
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path
from .data_models import InteractionScenario, VehicleState


class InteractionDataLoader:
    """
    Loader for vehicle interaction data from INTERACTIONS dataset
    """

    def __init__(self, csv_path: str):
        """
        Initialize data loader

        Args:
            csv_path: Path to the CSV file containing interaction data
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.scenarios = {}

    def load_data(self) -> None:
        """Load data from CSV file"""
        self.data = pd.read_csv(self.csv_path, dtype={'Scenario_id': str})
        self._group_by_scenario()

    def _group_by_scenario(self) -> None:
        """Group data by scenario_id for easy access"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        for scenario_id in self.data['Scenario_id'].unique():
            scenario_data = self.data[self.data['Scenario_id'] == scenario_id]
            self.scenarios[scenario_id] = scenario_data.reset_index(drop=True)

    def get_scenario(self, scenario_id: str, frame_id: Optional[int] = None) -> Optional[InteractionScenario]:
        """
        Get a specific scenario

        Args:
            scenario_id: ID of the scenario
            frame_id: Specific frame ID (if None, returns first frame)

        Returns:
            InteractionScenario object or None if not found
        """
        if scenario_id not in self.scenarios:
            return None

        scenario_data = self.scenarios[scenario_id]

        if frame_id is not None:
            frame_data = scenario_data[scenario_data['frame_id'] == frame_id]
            if frame_data.empty:
                return None
            row = frame_data.iloc[0]
        else:
            row = scenario_data.iloc[0]

        return self._rows_to_scenarios(pd.DataFrame([row]))[0]

    def get_all_scenarios(self) -> List[InteractionScenario]:
        """
        Get all scenarios from the dataset

        Returns:
            List of all InteractionScenario objects
        """
        scenarios = []
        for scenario_id in self.scenarios.keys():
            scenario = self.get_scenario(scenario_id)
            if scenario is not None:
                scenarios.append(scenario)
        return scenarios

    def get_sample_scenarios(self, n: int = 5, random_state: int = 42) -> List[InteractionScenario]:
        """
        Get a random sample of scenarios

        Args:
            n: Number of scenarios to sample
            random_state: Seed passed to pandas sampling for reproducibility

        Returns:
            List of sampled InteractionScenario objects
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        sample_data = self.data.sample(
            n=min(n, len(self.data)), random_state=random_state, replace=False
        )
        return self._rows_to_scenarios(sample_data)

    def _rows_to_scenarios(self, rows: pd.DataFrame) -> List[InteractionScenario]:
        """Convert data rows to scenarios while preserving row order."""
        scenarios = []
        for _, row in rows.iterrows():
            vehicle_1 = VehicleState(
                vehicle_id=str(row['track_id_1']),
                distance=float(row['d_1']),
                velocity=float(row['v_1']),
                acceleration=float(row['a_1']) if pd.notna(row['a_1']) else None
            )

            vehicle_2 = VehicleState(
                vehicle_id=str(row['track_id_2']),
                distance=float(row['d_2']),
                velocity=float(row['v_2']),
                acceleration=float(row['a_2']) if pd.notna(row['a_2']) else None
            )

            # Extract scenario type from the first column and ground truth priority from the priority column
            scenario_type = str(row['Scenario_type']) if pd.notna(row['Scenario_type']) else None
            ground_truth_priority = str(row['priority']) if pd.notna(row['priority']) else None

            scenario = InteractionScenario(
                scenario_id=str(row['Scenario_id']),
                frame_id=int(row['frame_id']),
                vehicle_1=vehicle_1,
                vehicle_2=vehicle_2,
                scenario_type=scenario_type,
                ground_truth_priority=ground_truth_priority
            )
            scenarios.append(scenario)
        return scenarios

    def get_scenarios_from_manifest(self, manifest_path: str) -> List[InteractionScenario]:
        """Load scenario frames listed in a manifest, preserving manifest order.

        The manifest must contain ``Scenario_id`` and ``frame_id`` columns. Both
        the manifest and source data must be unique on that composite key.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        manifest = pd.read_csv(manifest_path, dtype={'Scenario_id': str})
        required_columns = {'Scenario_id', 'frame_id'}
        missing_columns = required_columns.difference(manifest.columns)
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        if manifest[list(required_columns)].isna().any().any():
            raise ValueError("Manifest keys Scenario_id and frame_id cannot be empty")

        try:
            manifest = manifest.copy()
            numeric_frame_ids = pd.to_numeric(manifest['frame_id'], errors='raise')
            if not (numeric_frame_ids % 1 == 0).all():
                raise ValueError("non-integer frame_id")
            manifest['frame_id'] = numeric_frame_ids.astype('int64')
        except (TypeError, ValueError) as exc:
            raise ValueError("Manifest frame_id values must be integers") from exc

        manifest_duplicate_mask = manifest.duplicated(
            subset=['Scenario_id', 'frame_id'], keep=False
        )
        if manifest_duplicate_mask.any():
            duplicates = self._format_keys(manifest.loc[manifest_duplicate_mask])
            raise ValueError(f"Manifest contains duplicate sample keys: {duplicates}")

        source = self.data.copy()
        source['Scenario_id'] = source['Scenario_id'].astype(str)
        source_duplicate_mask = source.duplicated(
            subset=['Scenario_id', 'frame_id'], keep=False
        )
        if source_duplicate_mask.any():
            duplicates = self._format_keys(source.loc[source_duplicate_mask])
            raise ValueError(f"Source data contains duplicate sample keys: {duplicates}")

        source_by_key = source.set_index(['Scenario_id', 'frame_id'], drop=False)
        requested_keys = list(
            manifest[['Scenario_id', 'frame_id']].itertuples(index=False, name=None)
        )
        missing_keys = [key for key in requested_keys if key not in source_by_key.index]
        if missing_keys:
            formatted = ", ".join(f"{scenario_id}::{frame_id}" for scenario_id, frame_id in missing_keys)
            raise ValueError(f"Manifest records not found in source data: {formatted}")

        selected_rows = pd.DataFrame(
            [source_by_key.loc[key] for key in requested_keys]
        ).reset_index(drop=True)
        if len(selected_rows) != len(manifest):
            raise RuntimeError("Manifest selection did not preserve every requested record")
        return self._rows_to_scenarios(selected_rows)

    @staticmethod
    def _format_keys(rows: pd.DataFrame) -> str:
        keys = rows[['Scenario_id', 'frame_id']].drop_duplicates()
        return ", ".join(
            f"{scenario_id}::{int(frame_id)}"
            for scenario_id, frame_id in keys.itertuples(index=False, name=None)
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics

        Returns:
            Dictionary containing dataset statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        stats = {
            'total_scenarios': len(self.scenarios),
            'total_frames': len(self.data),
            'unique_scenarios': self.data['Scenario_id'].nunique(),
            'scenario_id_counts': self.data['Scenario_id'].value_counts().to_dict(),
            'vehicle_stats': {
                'distance_1': {
                    'min': float(self.data['d_1'].min()),
                    'max': float(self.data['d_1'].max()),
                    'mean': float(self.data['d_1'].mean()),
                    'std': float(self.data['d_1'].std())
                },
                'velocity_1': {
                    'min': float(self.data['v_1'].min()),
                    'max': float(self.data['v_1'].max()),
                    'mean': float(self.data['v_1'].mean()),
                    'std': float(self.data['v_1'].std())
                },
                'acceleration_1': {
                    'min': float(self.data['a_1'].min()),
                    'max': float(self.data['a_1'].max()),
                    'mean': float(self.data['a_1'].mean()),
                    'std': float(self.data['a_1'].std())
                },
                'distance_2': {
                    'min': float(self.data['d_2'].min()),
                    'max': float(self.data['d_2'].max()),
                    'mean': float(self.data['d_2'].mean()),
                    'std': float(self.data['d_2'].std())
                },
                'velocity_2': {
                    'min': float(self.data['v_2'].min()),
                    'max': float(self.data['v_2'].max()),
                    'mean': float(self.data['v_2'].mean()),
                    'std': float(self.data['v_2'].std())
                },
                'acceleration_2': {
                    'min': float(self.data['a_2'].min()),
                    'max': float(self.data['a_2'].max()),
                    'mean': float(self.data['a_2'].mean()),
                    'std': float(self.data['a_2'].std())
                }
            }
        }

        return stats

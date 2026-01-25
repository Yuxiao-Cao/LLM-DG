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
        self.data = pd.read_csv(self.csv_path)
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

        # Create vehicle states
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

        return InteractionScenario(
            scenario_id=str(row['Scenario_id']),
            frame_id=int(row['frame_id']),
            vehicle_1=vehicle_1,
            vehicle_2=vehicle_2,
            scenario_type=scenario_type,
            ground_truth_priority=ground_truth_priority
        )

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

    def get_sample_scenarios(self, n: int = 5) -> List[InteractionScenario]:
        """
        Get a random sample of scenarios

        Args:
            n: Number of scenarios to sample

        Returns:
            List of sampled InteractionScenario objects
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        sample_data = self.data.sample(n=min(n, len(self.data)))
        scenarios = []

        for _, row in sample_data.iterrows():
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
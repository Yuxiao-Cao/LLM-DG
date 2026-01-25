"""
Closed-loop rollout experiment for two-vehicle LLM control.

This module implements closed-loop simulation where both vehicles are controlled
by LLM decision-making over multiple time steps, evaluating compounding errors
and multi-step interaction dynamics.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from ..data_models import InteractionScenario, VehicleState, LLMDecision
from ..gamecard import GameCard
from ..llm_interface import LLMInterface
from ..evaluation import PreciseEvaluator, EvaluationConfig

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Configuration constants
MIN_ACCELERATION = -3.0  # m/s^2
MAX_ACCELERATION = 3.0   # m/s^2
PASSING_THRESHOLD = -5.0  # meters past intersection (considered passed)
COLLISION_DISTANCE = 2.0  # meters (considered collision)


@dataclass
class RolloutStep:
    """Single step data in a closed-loop rollout"""
    k: int
    state: Dict[str, float]  # {d1, v1, d2, v2}
    a1: float
    a2: float
    safety_v1: float
    safety_v2: float
    efficiency_v1: float
    efficiency_v2: float
    compliance_v1: float
    compliance_v2: float
    safety_sys: float
    efficiency_sys: float
    compliance_sys: float
    overall_sys: float
    stop_flag: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutResult:
    """Complete result of a closed-loop rollout episode"""
    dataset_name: str
    model_name: str
    seed: int
    init_index: int
    init_state: Dict[str, float]
    per_step: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_acceleration_from_llm_response(response: str) -> Optional[float]:
    """
    Robustly parse acceleration from LLM response.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed acceleration value or None if parsing fails
    """
    # First try: JSON parse with "acceleration" or "a" key
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Try various possible keys
            for key in ["acceleration_1", "acceleration", "a", "a_1"]:
                if key in data:
                    value = data[key]
                    return float(value)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Second try: regex to extract first float
    # Look for patterns like "acceleration: 2.5" or "a = -1.2"
    patterns = [
        r'acceleration[_\s]*1?:\s*([+-]?\d+\.?\d*)',
        r'acceleration[_\s]*=\s*([+-]?\d+\.?\d*)',
        r'\ba[_\s]*1?:\s*([+-]?\d+\.?\d*)',
        r'\ba[_\s]*=\s*([+-]?\d+\.?\d*)',
        r'deceleration[:\s]+([+-]?\d+\.?\d*)',
        r'm\s*/\s*s\^2[:\s]+([+-]?\d+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                return value
            except ValueError:
                continue

    # Last resort: find any floating point number in the response
    # that looks like an acceleration (between -10 and 10 typically)
    numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
    for num_str in numbers:
        try:
            val = float(num_str)
            if -10.0 <= val <= 10.0:
                return val
        except ValueError:
            continue

    return None


def clamp_acceleration(a: float) -> float:
    """Clamp acceleration to valid range [-3, 3] m/s^2"""
    return max(MIN_ACCELERATION, min(MAX_ACCELERATION, a))


def update_kinematic_state(d: float, v: float, a: float, dt: float) -> Tuple[float, float]:
    """
    Update vehicle state using simple kinematic model.

    Args:
        d: Current distance to intersection (m)
        v: Current velocity (m/s)
        a: Acceleration (m/s^2)
        dt: Time step (s)

    Returns:
        Tuple of (next_distance, next_velocity)
    """
    # Update distance: d_next = d - v*dt - 0.5*a*dt^2
    # Negative because distance decreases as we approach intersection
    d_next = d - v * dt - 0.5 * a * dt**2

    # Update velocity: v_next = v + a*dt
    v_next = v + a * dt

    # Velocity can't be negative (vehicle stops)
    v_next = max(0.0, v_next)

    return d_next, v_next


def create_swapped_scenario(original: InteractionScenario) -> InteractionScenario:
    """
    Create a scenario with vehicle roles swapped (Vehicle-2 becomes ego).

    Args:
        original: Original scenario

    Returns:
        New scenario with swapped vehicle roles
    """
    return InteractionScenario(
        scenario_id=f"{original.scenario_id}_swapped",
        frame_id=original.frame_id,
        vehicle_1=VehicleState(
            vehicle_id=original.vehicle_2.vehicle_id,
            distance=original.vehicle_2.distance,
            velocity=original.vehicle_2.velocity,
            acceleration=original.vehicle_2.acceleration
        ),
        vehicle_2=VehicleState(
            vehicle_id=original.vehicle_1.vehicle_id,
            distance=original.vehicle_1.distance,
            velocity=original.vehicle_1.velocity,
            acceleration=original.vehicle_1.acceleration
        ),
        scenario_type=original.scenario_type,
        ground_truth_priority=original.ground_truth_priority
    )


def compute_step_metrics(
    d1: float, v1: float, a1: float,
    d2: float, v2: float, a2: float,
    evaluator: PreciseEvaluator
) -> Dict[str, float]:
    """
    Compute metrics for both vehicles at a single step.

    Args:
        d1, v1, a1: Vehicle 1 state
        d2, v2, a2: Vehicle 2 state
        evaluator: PreciseEvaluator instance

    Returns:
        Dictionary with individual and system-level metrics
    """
    # Create scenarios for each vehicle (evaluated as ego)
    scenario_v1 = InteractionScenario(
        scenario_id="step_v1",
        frame_id=0,
        vehicle_1=VehicleState(vehicle_id="vehicle_1", distance=d1, velocity=v1, acceleration=a1),
        vehicle_2=VehicleState(vehicle_id="vehicle_2", distance=d2, velocity=v2, acceleration=a2),
        scenario_type="intersection"
    )

    scenario_v2 = InteractionScenario(
        scenario_id="step_v2",
        frame_id=0,
        vehicle_1=VehicleState(vehicle_id="vehicle_2", distance=d2, velocity=v2, acceleration=a2),
        vehicle_2=VehicleState(vehicle_id="vehicle_1", distance=d1, velocity=v1, acceleration=a1),
        scenario_type="intersection"
    )

    # Evaluate each vehicle's decision
    decision_v1 = LLMDecision(acceleration_1=a1, reasoning="step", confidence=1.0)
    decision_v2 = LLMDecision(acceleration_1=a2, reasoning="step", confidence=1.0)

    result_v1 = evaluator.evaluate_scenario(scenario_v1, decision_v1)
    result_v2 = evaluator.evaluate_scenario(scenario_v2, decision_v2)

    # Individual metrics
    safety_v1 = result_v1.metrics.safety_score
    efficiency_v1 = result_v1.metrics.efficiency_score
    compliance_v1 = result_v1.metrics.compliance_score

    safety_v2 = result_v2.metrics.safety_score
    efficiency_v2 = result_v2.metrics.efficiency_score
    compliance_v2 = result_v2.metrics.compliance_score

    # System-level aggregation
    safety_sys = min(safety_v1, safety_v2)
    compliance_sys = min(compliance_v1, compliance_v2)
    efficiency_sys = (efficiency_v1 + efficiency_v2) / 2.0

    # Overall system score (0.4 safety + 0.25 efficiency + 0.25 compliance) / 0.9
    # Note: No rationality score in closed-loop
    overall_sys = (0.4 * safety_sys + 0.25 * efficiency_sys + 0.25 * compliance_sys) / 0.9

    return {
        "safety_v1": safety_v1,
        "safety_v2": safety_v2,
        "efficiency_v1": efficiency_v1,
        "efficiency_v2": efficiency_v2,
        "compliance_v1": compliance_v1,
        "compliance_v2": compliance_v2,
        "safety_sys": safety_sys,
        "efficiency_sys": efficiency_sys,
        "compliance_sys": compliance_sys,
        "overall_sys": overall_sys
    }


def check_termination(d1: float, v1: float, d2: float, v2: float,
                     safety_sys: float, k: int, max_steps: int) -> Tuple[bool, str]:
    """
    Check if rollout should terminate.

    Args:
        d1, v1: Vehicle 1 state
        d2, v2: Vehicle 2 state
        safety_sys: System safety score
        k: Current step
        max_steps: Maximum steps allowed

    Returns:
        Tuple of (should_stop, reason)
    """
    # Check if either vehicle has passed intersection
    if d1 < PASSING_THRESHOLD:
        return True, "success"
    if d2 < PASSING_THRESHOLD:
        return True, "success"

    # Check collision/imminent collision (safety == 0 or very low)
    if safety_sys <= 0:
        return True, "collision"

    # Check timeout
    if k >= max_steps - 1:
        return True, "timeout"

    return False, ""


class ClosedLoopRollout:
    """
    Closed-loop rollout simulator for two-vehicle LLM control.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        gamecard: GameCard,
        evaluator: PreciseEvaluator,
        dt: float = 1.0,
        max_steps: int = 5
    ):
        """
        Initialize closed-loop rollout.

        Args:
            llm_client: LLM interface for decision generation
            gamecard: GameCard instance for prompt generation
            evaluator: Evaluator for metric computation
            dt: Time step in seconds
            max_steps: Maximum steps per episode
        """
        self.llm_client = llm_client
        self.gamecard = gamecard
        self.evaluator = evaluator
        self.dt = dt
        self.max_steps = max_steps

    def rollout_one_episode(
        self,
        init_scenario: InteractionScenario,
        init_index: int,
        dataset_name: str = "example_data",
        model_name: str = "deepseek",
        seed: int = 0,
        show_progress: bool = False,
        pbar: Optional[Any] = None
    ) -> RolloutResult:
        """
        Run a single closed-loop rollout episode.

        Args:
            init_scenario: Initial scenario (state only, actions ignored)
            init_index: Index of initial state in dataset
            dataset_name: Name of dataset
            model_name: Name of LLM model
            seed: Random seed for reproducibility
            show_progress: Whether to show progress bar for steps
            pbar: Optional external progress bar to update

        Returns:
            RolloutResult with complete trajectory data
        """
        import random
        random.seed(seed)

        # Extract initial state (ignore action labels)
        d1, v1 = init_scenario.vehicle_1.distance, init_scenario.vehicle_1.velocity
        d2, v2 = init_scenario.vehicle_2.distance, init_scenario.vehicle_2.velocity

        init_state = {"d1": d1, "v1": v1, "d2": d2, "v2": v2}

        per_step = []
        outcome = "timeout"

        # Create progress bar for steps if requested
        if show_progress and TQDM_AVAILABLE and pbar is None:
            step_pbar = tqdm(total=self.max_steps, desc=f"  Steps (init_idx={init_index})", leave=False)
        elif pbar is not None:
            step_pbar = pbar
        else:
            step_pbar = None

        for k in range(self.max_steps):
            # Update progress bar description
            if step_pbar is not None:
                step_pbar.set_description(f"  Step {k+1}/{self.max_steps} (init_idx={init_index})")
                step_pbar.update(1)

            # Build current state scenario
            # Use 0.0 as placeholder acceleration (will be ignored by evaluator)
            current_scenario = InteractionScenario(
                scenario_id=f"{init_scenario.scenario_id}_step{k}",
                frame_id=k,
                vehicle_1=VehicleState(vehicle_id="vehicle_1", distance=d1, velocity=v1, acceleration=0.0),
                vehicle_2=VehicleState(vehicle_id="vehicle_2", distance=d2, velocity=v2, acceleration=0.0),
                scenario_type=init_scenario.scenario_type or "intersection"
            )

            # a) Get LLM decision for Vehicle-1 (as ego)
            prompt_v1, _ = self.gamecard.create_precise_prompt(current_scenario)
            response_v1 = self.llm_client.generate_response(prompt_v1)
            a1_parsed = parse_acceleration_from_llm_response(response_v1)

            if a1_parsed is None:
                # Parsing failed - mark as invalid and stop
                outcome = "invalid_parse"
                break

            a1 = clamp_acceleration(a1_parsed)

            # b) Get LLM decision for Vehicle-2 (as ego - swap roles)
            swapped_scenario = create_swapped_scenario(current_scenario)
            prompt_v2, _ = self.gamecard.create_precise_prompt(swapped_scenario)
            response_v2 = self.llm_client.generate_response(prompt_v2)
            a2_parsed = parse_acceleration_from_llm_response(response_v2)

            if a2_parsed is None:
                outcome = "invalid_parse"
                break

            # Note: a2_parsed is for Vehicle-2 as ego, so use directly
            a2 = clamp_acceleration(a2_parsed)

            # e) Compute step metrics
            metrics = compute_step_metrics(d1, v1, a1, d2, v2, a2, self.evaluator)

            # f) Check termination before state update
            should_stop, reason = check_termination(
                d1, v1, d2, v2, metrics["safety_sys"], k, self.max_steps
            )

            if should_stop:
                outcome = reason

            # Record step
            step = RolloutStep(
                k=k,
                state={"d1": d1, "v1": v1, "d2": d2, "v2": v2},
                a1=a1,
                a2=a2,
                safety_v1=metrics["safety_v1"],
                safety_v2=metrics["safety_v2"],
                efficiency_v1=metrics["efficiency_v1"],
                efficiency_v2=metrics["efficiency_v2"],
                compliance_v1=metrics["compliance_v1"],
                compliance_v2=metrics["compliance_v2"],
                safety_sys=metrics["safety_sys"],
                efficiency_sys=metrics["efficiency_sys"],
                compliance_sys=metrics["compliance_sys"],
                overall_sys=metrics["overall_sys"],
                stop_flag=should_stop
            )
            per_step.append(step.to_dict())

            if should_stop:
                break

            # d) Update state
            d1, v1 = update_kinematic_state(d1, v1, a1, self.dt)
            d2, v2 = update_kinematic_state(d2, v2, a2, self.dt)

        # Close progress bar if we created it
        if step_pbar is not None and pbar is None and show_progress:
            step_pbar.close()

        # Compute summary statistics
        summary = self._compute_summary(per_step, outcome)

        return RolloutResult(
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            init_index=init_index,
            init_state=init_state,
            per_step=per_step,
            summary=summary
        )

    def _compute_summary(self, per_step: List[Dict[str, Any]], outcome: str) -> Dict[str, Any]:
        """Compute summary statistics for the episode"""
        if not per_step:
            return {
                "min_safety": 0.0,
                "mean_overall": 0.0,
                "mean_efficiency": 0.0,
                "mean_compliance": 0.0,
                "n_steps": 0,
                "outcome": outcome
            }

        safety_scores = [step["safety_sys"] for step in per_step]
        efficiency_scores = [step["efficiency_sys"] for step in per_step]
        compliance_scores = [step["compliance_sys"] for step in per_step]
        overall_scores = [step["overall_sys"] for step in per_step]

        return {
            "min_safety": min(safety_scores),
            "mean_overall": sum(overall_scores) / len(overall_scores),
            "mean_efficiency": sum(efficiency_scores) / len(efficiency_scores),
            "mean_compliance": sum(compliance_scores) / len(compliance_scores),
            "n_steps": len(per_step),
            "outcome": outcome
        }

    def rollout_batch(
        self,
        scenarios: List[InteractionScenario],
        indices: List[int],
        dataset_name: str = "example_data",
        model_name: str = "deepseek",
        seed: int = 0,
        show_progress: bool = True
    ) -> List[RolloutResult]:
        """
        Run multiple rollout episodes.

        Args:
            scenarios: List of initial scenarios
            indices: List of indices in dataset
            dataset_name: Name of dataset
            model_name: Name of LLM model
            seed: Random seed
            show_progress: Whether to show progress bars

        Returns:
            List of RolloutResult objects
        """
        results = []

        if show_progress and TQDM_AVAILABLE:
            episode_iter = tqdm(zip(scenarios, indices), total=len(scenarios),
                                desc="Episodes", unit="ep")
        else:
            episode_iter = zip(scenarios, indices)

        for scenario, idx in episode_iter:
            if show_progress and not TQDM_AVAILABLE:
                print(f"Running episode (init_index={idx})...")

            try:
                result = self.rollout_one_episode(
                    scenario, idx, dataset_name, model_name, seed + len(results),
                    show_progress=show_progress
                )
                results.append(result)

                # Update episode progress if using tqdm
                if show_progress and TQDM_AVAILABLE:
                    episode_iter.set_postfix({
                        "outcome": result.summary.get("outcome", "?"),
                        "steps": result.summary.get("n_steps", 0)
                    })

            except Exception as e:
                if show_progress and not TQDM_AVAILABLE:
                    print(f"Error in episode: {e}")
                # Create error result
                results.append(RolloutResult(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed + len(results),
                    init_index=idx,
                    init_state={"d1": 0, "v1": 0, "d2": 0, "v2": 0},
                    per_step=[],
                    summary={"error": str(e), "outcome": "error"}
                ))

        return results


def save_rollout_results(results: List[RolloutResult], output_dir: Path, run_id: str):
    """
    Save rollout results to JSONL and CSV files.

    Args:
        results: List of rollout results
        output_dir: Output directory
        run_id: Unique identifier for this run
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save individual trajectories as JSONL
    jsonl_path = output_dir / f"rollouts_{run_id}.jsonl"
    with open(jsonl_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')

    # Save summary as CSV
    csv_path = output_dir / f"summary_{run_id}.csv"
    import csv

    if results:
        # Changed "seed" to "num", added mean_efficiency and mean_compliance
        fieldnames = [
            "dataset_name", "model_name", "num", "init_index",
            "init_d1", "init_v1", "init_d2", "init_v2",
            "min_safety", "mean_efficiency", "mean_compliance", "mean_overall", "n_steps", "outcome"
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # Only include fields that are in fieldnames (filter out 'error' etc.)
                row = {
                    "dataset_name": result.dataset_name,
                    "model_name": result.model_name,
                    "num": result.seed,  # Changed from "seed" to "num"
                    "init_index": result.init_index,
                    "init_d1": result.init_state.get("d1", 0),
                    "init_v1": result.init_state.get("v1", 0),
                    "init_d2": result.init_state.get("d2", 0),
                    "init_v2": result.init_state.get("v2", 0),
                    "min_safety": result.summary.get("min_safety", 0),
                    "mean_efficiency": result.summary.get("mean_efficiency", 0),
                    "mean_compliance": result.summary.get("mean_compliance", 0),
                    "mean_overall": result.summary.get("mean_overall", 0),
                    "n_steps": result.summary.get("n_steps", 0),
                    "outcome": result.summary.get("outcome", "unknown")
                }
                writer.writerow(row)

    print(f"Saved JSONL to: {jsonl_path}")
    print(f"Saved CSV to: {csv_path}")


def compute_aggregate_statistics(results: List[RolloutResult]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all rollout results.

    Args:
        results: List of rollout results

    Returns:
        Dictionary with aggregate statistics
    """
    valid_results = [r for r in results if "error" not in r.summary]

    if not valid_results:
        return {
            "total_episodes": len(results),
            "valid_episodes": 0,
            "min_safety_mean": 0.0,
            "min_safety_std": 0.0,
            "mean_efficiency_mean": 0.0,
            "mean_efficiency_std": 0.0,
            "mean_compliance_mean": 0.0,
            "mean_compliance_std": 0.0,
            "mean_overall_mean": 0.0,
            "mean_overall_std": 0.0,
            "success_rate": 0.0
        }

    # Extract metrics
    min_safety_scores = [r.summary.get("min_safety", 0) for r in valid_results]
    mean_efficiency_scores = [r.summary.get("mean_efficiency", 0) for r in valid_results]
    mean_compliance_scores = [r.summary.get("mean_compliance", 0) for r in valid_results]
    mean_overall_scores = [r.summary.get("mean_overall", 0) for r in valid_results]

    # Count outcomes
    success_count = sum(1 for r in valid_results if r.summary.get("outcome") == "success")
    collision_count = sum(1 for r in valid_results if r.summary.get("outcome") == "collision")
    timeout_count = sum(1 for r in valid_results if r.summary.get("outcome") == "timeout")
    invalid_count = sum(1 for r in valid_results if r.summary.get("outcome") == "invalid_parse")

    # Compute statistics
    def mean_and_std(values):
        if not values:
            return 0.0, 0.0
        n = len(values)
        mean_val = sum(values) / n
        if n == 1:
            return mean_val, 0.0
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        return mean_val, variance ** 0.5

    min_safety_mean, min_safety_std = mean_and_std(min_safety_scores)
    mean_efficiency_mean, mean_efficiency_std = mean_and_std(mean_efficiency_scores)
    mean_compliance_mean, mean_compliance_std = mean_and_std(mean_compliance_scores)
    mean_overall_mean, mean_overall_std = mean_and_std(mean_overall_scores)

    success_rate = success_count / len(valid_results) if valid_results else 0.0

    return {
        "total_episodes": len(results),
        "valid_episodes": len(valid_results),
        "min_safety_mean": round(min_safety_mean, 4),
        "min_safety_std": round(min_safety_std, 4),
        "mean_efficiency_mean": round(mean_efficiency_mean, 4),
        "mean_efficiency_std": round(mean_efficiency_std, 4),
        "mean_compliance_mean": round(mean_compliance_mean, 4),
        "mean_compliance_std": round(mean_compliance_std, 4),
        "mean_overall_mean": round(mean_overall_mean, 4),
        "mean_overall_std": round(mean_overall_std, 4),
        "success_rate": round(success_rate, 4),
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "invalid_count": invalid_count
    }


def save_aggregate_statistics(results: List[RolloutResult], output_dir: Path, run_id: str):
    """
    Save aggregate statistics to a separate CSV file.

    Args:
        results: List of rollout results
        output_dir: Output directory
        run_id: Unique identifier for this run
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    stats = compute_aggregate_statistics(results)

    # Save as single-row CSV
    stats_path = output_dir / f"aggregate_stats_{run_id}.csv"
    import csv

    with open(stats_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)

    print(f"Saved aggregate statistics to: {stats_path}")

    return stats

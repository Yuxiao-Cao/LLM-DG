#!/usr/bin/env python3
"""
Closed-Loop Rollout Experiment for LLM-DG

This script runs closed-loop simulation where both vehicles are controlled
by LLM decision-making over multiple time steps.

Usage:
    python scripts/run_closed_loop_rollout.py --dataset data/example_data.csv --n_init 3

Output:
    - outputs/closed_loop_LLM/rollouts_*.jsonl: Per-trajectory detailed results
    - outputs/closed_loop_LLM/summary_*.csv: Aggregated summary statistics
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import InteractionDataLoader
from src.gamecard import GameCard
from src.llm_interface import create_llm_interface
from src.evaluation import PreciseEvaluator
from src.experiments.closed_loop import (
    ClosedLoopRollout,
    save_rollout_results,
    save_aggregate_statistics
)
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run closed-loop rollout experiment for LLM-DG"
    )

    parser.add_argument(
        "--dataset",
        default="data/example_data.csv",
        help="Path to dataset CSV file (default: data/example_data.csv)"
    )

    parser.add_argument(
        "--n_init",
        type=int,
        default=50,
        help="Number of initial states to sample (default: 50)"
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum steps per episode (default: 10)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek",
        choices=["doubao", "openai", "deepseek", "qwen", "gemini", "claude"],
        help="LLM model to use (default: deepseek)"
    )

    parser.add_argument(
        "--input_format",
        type=str,
        default="text_json",
        choices=["text_json", "json_only", "text_only"],
        help="Input format for prompts (default: text_json)"
    )

    parser.add_argument(
        "--decision_mode",
        type=str,
        default="precise",
        choices=["precise"],
        help="Decision mode (default: precise)"
    )

    parser.add_argument(
        "--cot_type",
        type=str,
        default="cot",
        choices=["cot", "nocot"],
        help="Chain-of-Thought type (default: cot)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/closed_loop_LLM/",
        help="Output directory (default: outputs/closed_loop_LLM/)"
    )

    return parser.parse_args()


def map_input_format(format_arg: str) -> str:
    """Map CLI input format to GameCard prompt format"""
    mapping = {
        "text_json": "text+json",
        "json_only": "json",
        "text_only": "text"
    }
    return mapping.get(format_arg, "text+json")


def main():
    """Main entry point"""
    args = parse_arguments()

    print("=" * 60)
    print("Closed-Loop Rollout Experiment for LLM-DG")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"n_init: {args.n_init}")
    print(f"dt: {args.dt}s")
    print(f"max_steps: {args.max_steps}")
    print(f"input_format: {args.input_format}")
    print(f"decision_mode: {args.decision_mode}")
    print(f"seed: {args.seed}")
    print(f"out_dir: {args.out_dir}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    loader = InteractionDataLoader(args.dataset)
    loader.load_data()
    stats = loader.get_statistics()
    print(f"Loaded {stats['total_scenarios']} scenarios with {stats['total_frames']} total frames")

    # Sample initial states
    import random
    random.seed(args.seed)

    available_indices = list(range(len(loader.data)))
    sampled_indices = random.sample(available_indices, min(args.n_init, len(available_indices)))
    sampled_indices.sort()

    print(f"\nSampled {len(sampled_indices)} initial states (indices: {sampled_indices})")

    # Save sampled indices for reproducibility
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    indices_file = out_dir / f"sampled_indices_seed{args.seed}.json"
    with open(indices_file, 'w') as f:
        json.dump({
            "seed": args.seed,
            "dataset": args.dataset,
            "n_init": args.n_init,
            "sampled_indices": sampled_indices
        }, f, indent=2)
    print(f"Saved sampled indices to: {indices_file}")

    # Load scenarios for sampled indices
    scenarios = []
    for idx in sampled_indices:
        row = loader.data.iloc[idx]
        from src.data_models import VehicleState, InteractionScenario

        v1 = VehicleState(
            vehicle_id=str(row['track_id_1']),
            distance=float(row['d_1']),
            velocity=float(row['v_1']),
            acceleration=0.0  # Use placeholder, ignore action labels
        )
        v2 = VehicleState(
            vehicle_id=str(row['track_id_2']),
            distance=float(row['d_2']),
            velocity=float(row['v_2']),
            acceleration=0.0  # Use placeholder, ignore action labels
        )

        scenario = InteractionScenario(
            scenario_id=str(row['Scenario_id']),
            frame_id=int(row['frame_id']),
            vehicle_1=v1,
            vehicle_2=v2,
            scenario_type=str(row['Scenario_type']) if 'Scenario_type' in row else 'intersection',
            ground_truth_priority=None
        )
        scenarios.append(scenario)

    # Initialize components
    print("\nInitializing LLM interface and evaluator...")
    llm_client = create_llm_interface(args.model)

    prompt_format = map_input_format(args.input_format)
    gamecard = GameCard(prompt_format=prompt_format, cot_type=args.cot_type)

    evaluator = PreciseEvaluator()

    # Initialize closed-loop rollout
    rollout = ClosedLoopRollout(
        llm_client=llm_client,
        gamecard=gamecard,
        evaluator=evaluator,
        dt=args.dt,
        max_steps=args.max_steps
    )

    # Extract dataset name from path
    dataset_name = Path(args.dataset).stem

    # Run rollouts
    print(f"\nRunning {len(scenarios)} closed-loop rollout episodes...")
    print("-" * 40)

    results = rollout.rollout_batch(
        scenarios=scenarios,
        indices=sampled_indices,
        dataset_name=dataset_name,
        model_name=args.model,
        seed=args.seed,
        show_progress=True
    )

    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model}_{dataset_name}_n{args.n_init}_dt{args.dt}_seed{args.seed}_{timestamp}"

    save_rollout_results(results, out_dir, run_id)

    # Save and display aggregate statistics
    print("\n" + "-" * 40)
    print("Computing aggregate statistics...")

    stats = save_aggregate_statistics(results, out_dir, run_id)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)

    print(f"\nTotal episodes: {stats['total_episodes']}")
    print(f"Valid episodes: {stats['valid_episodes']}")

    print(f"\nOutcome distribution:")
    print(f"  - Success: {stats['success_count']} ({stats['success_rate']*100:.1f}%)")
    print(f"  - Collision: {stats['collision_count']}")
    print(f"  - Timeout: {stats['timeout_count']}")
    print(f"  - Invalid parse: {stats['invalid_count']}")

    print(f"\nMin Safety: {stats['min_safety_mean']:.4f} ± {stats['min_safety_std']:.4f}")
    print(f"Mean Efficiency: {stats['mean_efficiency_mean']:.4f} ± {stats['mean_efficiency_std']:.4f}")
    print(f"Mean Compliance: {stats['mean_compliance_mean']:.4f} ± {stats['mean_compliance_std']:.4f}")
    print(f"Mean Overall: {stats['mean_overall_mean']:.4f} ± {stats['mean_overall_std']:.4f}")
    print(f"Success Rate: {stats['success_rate']:.4f}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

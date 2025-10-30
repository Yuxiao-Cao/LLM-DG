#!/usr/bin/env python3
"""
LLM-DG: Main Pipeline for Evaluating Large Language Models' Dynamic Game Decision-Making

This script implements the complete pipeline for:
1. Loading vehicle interaction data from INTERACTIONS dataset
2. Creating GameCard prompts using Chain-of-Thought methodology (precise mode)
3. Fuzzy priority determination using LLM-enhanced fuzzy logic (fuzzy mode)
4. Invoking LLM API (doubao-seed-1-6-thinking-250715) for decision generation
5. Evaluating strategy outcomes with game quality metrics
6. Calibrating decisions using opponent rationality regulator
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import time
from datetime import datetime

from numpy.f2py.crackfortran import endifs
from tqdm import tqdm

from src.data_loader import InteractionDataLoader
from src.gamecard import GameCard
from src.llm_interface import create_llm_interface
from src.evaluation import PreciseEvaluator, EvaluationConfig, OpponentRationalityRegulator, FuzzyEvaluator, FuzzyEvaluationResult
from src.data_models import EvaluationResult, FuzzyDecision


class LLMDGPipeline:
    """
    Complete pipeline for LLM-DG benchmark evaluation with optional fuzzy decision-making
    """

    def __init__(self,
                 data_path: str,
                 model_type: str,
                 prompt_format: str,
                 decision_mode: str,
                 output_dir: str,
                 cot_type: str = "cot"):
        """
        Initialize the pipeline

        Args:
            data_path: Path to interaction data CSV file
            model_type: Type of LLM interface ("doubao", "openai", "text")
            prompt_format: Format for prompts ("text", "json", "text+json")
            decision_mode: Decision-making mode ("precise" for acceleration, "fuzzy" for priority)
            output_dir: Directory to save results
            cot_type: Chain-of-Thought type ("cot" or "nocot")
        """
        self.data_path = data_path
        self.data_loader = InteractionDataLoader(data_path)
        self.llm_interface = create_llm_interface(model_type)
        self.decision_mode = decision_mode
        self.prompt_format = prompt_format
        self.model_type = model_type
        self.cot_type = cot_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize decision-making components based on mode
        if decision_mode == "precise":
            self.gamecard = GameCard(prompt_format=prompt_format, cot_type=cot_type)
            self.evaluator = PreciseEvaluator()
            self.rationality_regulator = OpponentRationalityRegulator()
        elif decision_mode == "fuzzy":
            self.gamecard = GameCard(prompt_format=prompt_format, cot_type=cot_type)
            self.fuzzy_evaluator = FuzzyEvaluator()
        else:
            raise ValueError(f"Invalid decision mode: {decision_mode}. Must be 'precise' or 'fuzzy'")

        print(f"Initialized LLM-DG Pipeline:")
        print(f"  - Data source: {data_path}")
        print(f"  - Model type: {model_type}")
        print(f"  - Decision mode: {decision_mode}")
        if decision_mode == "precise":
            print(f"  - Prompt format: {prompt_format}")
        print(f"  - CoT type: {cot_type}")
        print(f"  - Output directory: {output_dir}")

    def run_evaluation(self,
                      num_scenarios: int = 10,
                      use_rationality_calibration: bool = True,
                      opponent_type: str = "neutral") -> Union[List[EvaluationResult], List[Dict[str, Any]]]:
        """
        Run complete evaluation pipeline

        Args:
            num_scenarios: Number of scenarios to evaluate
            use_rationality_calibration: Whether to use opponent rationality calibration (precise mode only)
            opponent_type: Type of opponent model for calibration (precise mode only)

        Returns:
            List of evaluation results (EvaluationResult for precise mode, Dict for fuzzy mode)
        """
        print(f"\nStarting evaluation of {num_scenarios} scenarios in {self.decision_mode} mode...")

        # Load data
        print("Loading interaction data...")
        self.data_loader.load_data()
        stats = self.data_loader.get_statistics()
        print(f"Loaded {stats['total_scenarios']} scenarios with {stats['total_frames']} total frames")

        # Get sample scenarios
        scenarios = self.data_loader.get_sample_scenarios(num_scenarios)
        print(f"Selected {len(scenarios)} scenarios for evaluation")

        # Run evaluation
        if self.decision_mode == "precise":
            return self._run_precise_evaluation(scenarios, use_rationality_calibration, opponent_type)
        else:
            return self._run_fuzzy_evaluation(scenarios)

    def _run_precise_evaluation(self, scenarios, use_rationality_calibration, opponent_type) -> List[EvaluationResult]:
        """Run evaluation in precise mode (original functionality)"""
        results = []
        response_times = []

        for i, scenario in enumerate(tqdm(scenarios, desc="Evaluating scenarios (precise mode)")):
            try:
                result, response_time = self._evaluate_single_scenario_precise(
                    scenario,
                    use_rationality_calibration=use_rationality_calibration,
                    opponent_type=opponent_type
                )
                results.append(result)
                response_times.append(response_time)

            except Exception as e:
                print(f"Error evaluating scenario {scenario.scenario_id}: {e}")
                continue

        print(f"Successfully evaluated {len(results)}/{len(scenarios)} scenarios")
        return results

    def _run_fuzzy_evaluation(self, scenarios) -> List[FuzzyEvaluationResult]:
        """Run evaluation in fuzzy mode (priority determination)"""
        results = []

        for i, scenario in enumerate(tqdm(scenarios, desc="Evaluating scenarios (fuzzy mode)")):
            try:
                result = self._evaluate_single_scenario_fuzzy(scenario)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating scenario {scenario.scenario_id}: {e}")
                continue

        print(f"Successfully evaluated {len(results)}/{len(scenarios)} scenarios")
        return results

    def _evaluate_single_scenario_precise(self,
                                        scenario,
                                        use_rationality_calibration: bool = True,
                                        opponent_type: str = "neutral") -> tuple[EvaluationResult, float]:
        """
        Evaluate a single scenario in precise mode

        Args:
            scenario: Interaction scenario to evaluate
            use_rationality_calibration: Whether to use rationality calibration
            opponent_type: Type of opponent model

        Returns:
            Tuple of (evaluation result, response time in seconds)
        """
        # Create GameCard prompt
        precise_prompt, json_data = self.gamecard.create_precise_prompt(scenario)

        start_time = time.time()
        # Generate LLM response
        llm_response = self.llm_interface.generate_response(precise_prompt)

        response_time = time.time() - start_time

        # Parse LLM decision
        llm_decision = self.gamecard.parse_precise_response(llm_response)

        # Apply rationality calibration if requested
        if use_rationality_calibration:
            llm_decision = self.rationality_regulator.calibrate_decision(
                scenario, llm_decision, opponent_type
            )

        # Evaluate the decision
        result = self.evaluator.evaluate_scenario(
            scenario,
            llm_decision,
            baseline_decision=scenario.vehicle_1.acceleration
        )

        # Update result with response time
        result.response_time = response_time

        return result, response_time

    def _evaluate_single_scenario_fuzzy(self, scenario) -> FuzzyEvaluationResult:
        """
        Evaluate a single scenario in fuzzy mode

        Args:
            scenario: Interaction scenario to evaluate

        Returns:
            FuzzyEvaluationResult containing fuzzy evaluation results
        """
        start_time = time.time()

        # Create fuzzy prompt
        fuzzy_prompt, fuzzy_data = self.gamecard.create_fuzzy_prompt(scenario)

        # Generate LLM response
        llm_response = self.llm_interface.generate_response(fuzzy_prompt)

        response_time = time.time() - start_time

        # Parse fuzzy decision
        fuzzy_decision = self.gamecard.parse_fuzzy_response(llm_response, scenario)

        # Extract ground truth priority from scenario data
        ground_truth_priority = scenario.ground_truth_priority

        # Evaluate fuzzy decision
        evaluation_result = self.fuzzy_evaluator.evaluate_fuzzy_decision(
            scenario=scenario,
            fuzzy_decision=fuzzy_decision,
            ground_truth_priority=ground_truth_priority,
            response_time=response_time
        )

        return evaluation_result

    def _generate_filename(self, file_type: str = "results") -> str:
        """
        Generate filename with the new naming convention

        Args:
            file_type: Type of file ("results" or "report")

        Returns:
            Generated filename
        """
        # Extract data source name from file path
        # For 'DGTD_data.csv', we want 'DGTD'
        data_source_full = Path(self.data_path).stem
        # Remove '_data' suffix if present
        data_source = data_source_full.replace('_data', '')

        # Get current timestamp
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")

        # Remove special characters from prompt format
        prompt_format_clean = self.prompt_format.replace("+", "")

        # Determine file extension
        extension = "json" if file_type == "results" else "txt"

        # Construct filename using new convention
        # Results: evaluation_results_<data source>_<model type>_<fuzzy/precision>_<prompt type>_<time>.<ext>
        # Reports: evaluation_report_<data source>_<model type>_<fuzzy/precision>_<prompt type>_<time>.<ext>
        if file_type == "results":
            filename = f"evaluation_results_{data_source}_{self.model_type}_{self.decision_mode}_{prompt_format_clean}_{timestamp}.{extension}"
        else:
            filename = f"evaluation_report_{data_source}_{self.model_type}_{self.decision_mode}_{prompt_format_clean}_{timestamp}.{extension}"

        return filename

    def save_results(self, results: Union[List[EvaluationResult], List[FuzzyEvaluationResult]], filename: str = None) -> str:
        """
        Save evaluation results to file

        Args:
            results: List of evaluation results
            filename: Output filename (if None, generates timestamp-based name)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = self._generate_filename("results")

        output_path = self.output_dir / filename

        if self.decision_mode == "precise":
            results_data = self._save_precise_results(results)
        else:
            results_data = self._save_fuzzy_results(results)

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to: {output_path}")
        return str(output_path)

    def _save_precise_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Save results for precise mode"""
        results_data = {
            "metadata": {
                "total_scenarios": len(results),
                "decision_mode": "precise",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_config": {
                    "prompt_format": self.gamecard.prompt_format,
                    "model_type": type(self.llm_interface).__name__
                }
            },
            "results": [result.to_dict() for result in results]
        }

        # Calculate aggregate statistics
        if results:
            metrics = [result.metrics for result in results]
            results_data["aggregate_stats"] = {
                "safety_score": {
                    "mean": sum(m.safety_score for m in metrics) / len(metrics),
                    "min": min(m.safety_score for m in metrics),
                    "max": max(m.safety_score for m in metrics)
                },
                "efficiency_score": {
                    "mean": sum(m.efficiency_score for m in metrics) / len(metrics),
                    "min": min(m.efficiency_score for m in metrics),
                    "max": max(m.efficiency_score for m in metrics)
                },
                "compliance_score": {
                    "mean": sum(m.compliance_score for m in metrics) / len(metrics),
                    "min": min(m.compliance_score for m in metrics),
                    "max": max(m.compliance_score for m in metrics)
                },
                "rationality_score": {
                    "mean": sum(m.rationality_score for m in metrics) / len(metrics),
                    "min": min(m.rationality_score for m in metrics),
                    "max": max(m.rationality_score for m in metrics)
                },
                "overall_score": {
                    "mean": sum(m.overall_score for m in metrics) / len(metrics),
                    "min": min(m.overall_score for m in metrics),
                    "max": max(m.overall_score for m in metrics)
                }
            }

        return results_data

    def _save_fuzzy_results(self, results: List[FuzzyEvaluationResult]) -> Dict[str, Any]:
        """Save results for fuzzy mode"""
        # Convert FuzzyEvaluationResult objects to dictionaries
        results_dicts = []
        for result in results:
            result_dict = {
                "scenario_id": result.scenario_id,
                "priority_decision": {
                    "priority_vehicle": result.fuzzy_decision.priority_vehicle,
                    "confidence": result.confidence_score,
                    "risk_level": result.risk_level,
                    "scenario_type": result.scenario_type,
                    "fuzzy_reasoning": result.fuzzy_weights,
                    "textual_reasoning": result.fuzzy_decision.textual_reasoning
                },
                "ground_truth": result.ground_truth,
                "is_correct": result.is_correct,
                "response_time": result.response_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results_dicts.append(result_dict)

        # Calculate metrics using the fuzzy evaluator
        metrics = self.fuzzy_evaluator.evaluate_batch(results)

        results_data = {
            "metadata": {
                "total_scenarios": len(results),
                "decision_mode": "fuzzy",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_config": {
                    "model_type": type(self.llm_interface).__name__
                }
            },
            "results": results_dicts,
            "evaluation_metrics": {
                "accuracy": metrics.accuracy,
                "confidence_mean": metrics.confidence_mean,
                "confidence_std": metrics.confidence_std,
                "response_time_mean": metrics.response_time_mean,
                "total_scenarios": metrics.total_scenarios,
                "correct_predictions": metrics.correct_predictions,
                "risk_distribution": metrics.risk_distribution,
                "scenario_type_distribution": metrics.scenario_type_distribution,
                "fuzzy_weight_analysis": metrics.fuzzy_weight_analysis
            }
        }

        return results_data

    def generate_report(self, results: Union[List[EvaluationResult], List[FuzzyEvaluationResult]]) -> str:
        """
        Generate evaluation report

        Args:
            results: List of evaluation results

        Returns:
            Report as string
        """
        if not results:
            return "No results to report."

        if self.decision_mode == "precise":
            return self._generate_precise_report(results)
        else:
            return self._generate_fuzzy_report(results)

    def _generate_precise_report(self, results: List[EvaluationResult]) -> str:
        """Generate report for precise mode"""
        metrics = [result.metrics for result in results]
        response_times = [r.response_time for r in results if r.response_time is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        report = f"""
LLM-DG BENCHMARK EVALUATION REPORT (PRECISE MODE)
===============================================

Summary:
- Total Scenarios Evaluated: {len(results)}
- Decision Mode: Precise (acceleration-based)
- Evaluation Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
- Average LLM Response Time: {avg_response_time:.2f} seconds per scenario

Performance Metrics:
- Safety Score: {sum(m.safety_score for m in metrics) / len(metrics):.2f}/100 (±{self._std([m.safety_score for m in metrics]):.2f})
- Efficiency Score: {sum(m.efficiency_score for m in metrics) / len(metrics):.2f}/100 (±{self._std([m.efficiency_score for m in metrics]):.2f})
- Compliance Score: {sum(m.compliance_score for m in metrics) / len(metrics):.2f}/100 (±{self._std([m.compliance_score for m in metrics]):.2f})
- Rationality Score: {sum(m.rationality_score for m in metrics) / len(metrics):.2f}/100 (±{self._std([m.rationality_score for m in metrics]):.2f})
- Overall Score: {sum(m.overall_score for m in metrics) / len(metrics):.2f}/100 (±{self._std([m.overall_score for m in metrics]):.2f})

Best Performing Scenarios:
"""
        # Add top 5 scenarios
        sorted_results = sorted(results, key=lambda r: r.metrics.overall_score, reverse=True)[:5]
        for i, result in enumerate(sorted_results, 1):
            response_time_str = f" ({result.response_time:.2f}s)" if result.response_time else ""
            report += f"{i}. {result.scenario_id}: {result.metrics.overall_score:.2f}/100{response_time_str}\n"

        report += "\nWorst Performing Scenarios:\n"
        # Add bottom 5 scenarios
        sorted_results = sorted(results, key=lambda r: r.metrics.overall_score)[:5]
        for i, result in enumerate(sorted_results, 1):
            response_time_str = f" ({result.response_time:.2f}s)" if result.response_time else ""
            report += f"{i}. {result.scenario_id}: {result.metrics.overall_score:.2f}/100{response_time_str}\n"

        return report

    def _generate_fuzzy_report(self, results: List[FuzzyEvaluationResult]) -> str:
        """Generate report for fuzzy mode"""
        # Calculate comprehensive metrics using the fuzzy evaluator
        metrics = self.fuzzy_evaluator.evaluate_batch(results)

        # Generate the detailed report using the evaluator's report generation
        return self.fuzzy_evaluator.generate_evaluation_report(metrics)

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


def main():
    """Main function to run the LLM-DG pipeline"""
    parser = argparse.ArgumentParser(
        description="LLM-DG: Evaluate Large Language Models' Dynamic Game Decision-Making"
    )

    parser.add_argument(
        "--data-path",
        default="data/example_data.csv",
        help="Path to interaction data CSV file"
    )

    parser.add_argument(
        "--model-type",
        choices=["doubao", "openai", "deepseek", "qwen", "gemini", "claude"],
        default="mock",
        help="Type of LLM interface to use"
    )

    parser.add_argument(
        "--decision-mode",
        choices=["precise", "fuzzy"],
        default="precise",
        help="Decision-making mode: precise (acceleration) or fuzzy (priority)"
    )

    parser.add_argument(
        "--prompt-format",
        choices=["text", "json", "text+json"],
        default="text+json",
        help="Format for GameCard prompts (precise mode only)"
    )

    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=10,
        help="Number of scenarios to evaluate"
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="Use opponent rationality calibration (precise mode only)"
    )

    parser.add_argument(
        "--opponent-type",
        choices=["cooperative", "competitive", "neutral"],
        default="neutral",
        help="Type of opponent model for calibration (precise mode only)"
    )

    parser.add_argument(
        "--cot-type",
        choices=["cot", "nocot"],
        default="cot",
        help="Chain-of-Thought type: cot (with reasoning) or nocot (without reasoning)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.decision_mode == "fuzzy" and args.use_calibration:
        print("Warning: Rationality calibration is only available in precise mode. Ignoring --use-calibration.")

    # Initialize and run pipeline
    pipeline = LLMDGPipeline(
        data_path=args.data_path,
        model_type=args.model_type,
        prompt_format=args.prompt_format,
        decision_mode=args.decision_mode,
        output_dir=args.output_dir,
        cot_type=args.cot_type
    )

    # Run evaluation
    if args.decision_mode == "precise":
        results = pipeline.run_evaluation(
            num_scenarios=args.num_scenarios,
            use_rationality_calibration=args.use_calibration,
            opponent_type=args.opponent_type
        )

    else:
        results = pipeline.run_evaluation(
            num_scenarios=args.num_scenarios
        )

    # Save results
    results_file = pipeline.save_results(results)

    # Generate and save report
    report = pipeline.generate_report(results)
    report_file = pipeline.output_dir / pipeline._generate_filename("report")
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Evaluation report saved to: {report_file}")


if __name__ == "__main__":
    main()
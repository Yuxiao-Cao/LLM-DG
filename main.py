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
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import time
from datetime import datetime, timezone
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:  # Progress display is optional for CLI/help and minimal installs.
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_loader import InteractionDataLoader
from src.gamecard import GameCard
from src.llm_interface import create_llm_interface
from src.evaluation import PreciseEvaluator, EvaluationConfig, OpponentRationalityRegulator, FuzzyEvaluator, FuzzyEvaluationResult
from src.data_models import EvaluationResult, FuzzyDecision


@dataclass
class SampleEvaluationRecord:
    scenario_id: str
    frame_id: int
    sample_id: str
    prompt_hash: str
    raw_response: Optional[str]
    parse_status: str
    api_error: Optional[str]
    latency: float
    raw_decision: Optional[Any]
    calibrated_decision: Optional[Any]
    evaluation: Optional[Any]


def get_git_commit() -> Optional[str]:
    """Return the current commit, or None when Git metadata is unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-DG: Evaluate Large Language Models' Dynamic Game Decision-Making"
    )
    parser.add_argument("--data-path", default="data/example_data.csv")
    parser.add_argument("--model-type", choices=["doubao", "openai", "deepseek", "qwen", "gemini", "claude"], default="deepseek")
    parser.add_argument("--decision-mode", choices=["precise", "fuzzy"], default="precise")
    parser.add_argument("--prompt-format", choices=["text", "json", "text+json"], default="text+json")
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/open_loop_LLM/")
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--opponent-type", choices=["cooperative", "competitive", "neutral"], default="neutral")
    parser.add_argument("--cot-type", choices=["cot", "nocot"], default="cot")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260712)
    parser.add_argument("--manifest-path")
    parser.add_argument(
        "--strict-parse", action=argparse.BooleanOptionalAction, default=True,
        help="Require structured responses (default: enabled; use --no-strict-parse for legacy fallback)"
    )
    return parser


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
                 cot_type: str = "cot",
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_tokens: int = 100000,
                 generation_seed: Optional[int] = None,
                 sample_seed: int = 20260712,
                 manifest_path: Optional[str] = None,
                 strict_parse: bool = True,
                 llm_interface=None):
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
        self.llm_interface = llm_interface or create_llm_interface(model_type)
        self.decision_mode = decision_mode
        self.prompt_format = prompt_format
        self.model_type = model_type
        self.cot_type = cot_type
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.generation_seed = generation_seed
        self.sample_seed = sample_seed
        self.manifest_path = manifest_path
        self.strict_parse = strict_parse
        self.calibration_enabled = False
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.calibration_enabled = bool(
            use_rationality_calibration and self.decision_mode == "precise"
        )
        print(f"\nStarting evaluation in {self.decision_mode} mode...")

        # Load data
        print("Loading interaction data...")
        self.data_loader.load_data()
        stats = self.data_loader.get_statistics()
        print(f"Loaded {stats['total_scenarios']} scenarios with {stats['total_frames']} total frames")

        # Get sample scenarios
        if self.manifest_path:
            scenarios = self.data_loader.get_scenarios_from_manifest(self.manifest_path)
        else:
            scenarios = self.data_loader.get_sample_scenarios(
                num_scenarios, random_state=self.sample_seed
            )
        print(f"Selected {len(scenarios)} scenarios for evaluation")

        # Run evaluation
        if self.decision_mode == "precise":
            return self._run_precise_evaluation(scenarios, use_rationality_calibration, opponent_type)
        else:
            return self._run_fuzzy_evaluation(scenarios)

    def _run_precise_evaluation(self, scenarios, use_rationality_calibration, opponent_type) -> List[SampleEvaluationRecord]:
        """Run evaluation in precise mode (original functionality)"""
        results = []
        for scenario in tqdm(scenarios, desc="Evaluating scenarios (precise mode)"):
            results.append(self._evaluate_single_scenario_precise(
                scenario, use_rationality_calibration, opponent_type
            ))

        print(f"Successfully evaluated {len(results)}/{len(scenarios)} scenarios")
        return results

    def _run_fuzzy_evaluation(self, scenarios) -> List[SampleEvaluationRecord]:
        """Run evaluation in fuzzy mode (priority determination)"""
        results = []

        for scenario in tqdm(scenarios, desc="Evaluating scenarios (fuzzy mode)"):
            results.append(self._evaluate_single_scenario_fuzzy(scenario))

        print(f"Successfully evaluated {len(results)}/{len(scenarios)} scenarios")
        return results

    def _evaluate_single_scenario_precise(self,
                                        scenario,
                                        use_rationality_calibration: bool = True,
                                        opponent_type: str = "neutral") -> SampleEvaluationRecord:
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

        return self._generate_parse_evaluate(
            scenario, precise_prompt, use_rationality_calibration, opponent_type
        )

    def _evaluate_single_scenario_fuzzy(self, scenario) -> SampleEvaluationRecord:
        """
        Evaluate a single scenario in fuzzy mode

        Args:
            scenario: Interaction scenario to evaluate

        Returns:
            FuzzyEvaluationResult containing fuzzy evaluation results
        """
        fuzzy_prompt, fuzzy_data = self.gamecard.create_fuzzy_prompt(scenario)
        return self._generate_parse_evaluate(scenario, fuzzy_prompt, False, "neutral")

    def _generate_parse_evaluate(self, scenario, prompt, use_calibration, opponent_type):
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        started = time.perf_counter()
        raw_response = None
        try:
            generation = self.llm_interface.generate_response(
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                generation_seed=self.generation_seed
            )
            latency = time.perf_counter() - started
            raw_response = self._response_text(generation)
        except Exception as exc:
            return SampleEvaluationRecord(
                scenario.scenario_id, scenario.frame_id, scenario.sample_id,
                prompt_hash, raw_response, "api_error", str(exc),
                time.perf_counter() - started, None, None, None
            )

        try:
            if self.decision_mode == "precise":
                raw_decision = self.gamecard.parse_precise_response(
                    raw_response, strict=self.strict_parse
                )
                should_score = (
                    raw_decision.is_valid and raw_decision.parse_status == "strict_json"
                ) if self.strict_parse else True
                if not should_score:
                    return SampleEvaluationRecord(
                        scenario.scenario_id, scenario.frame_id, scenario.sample_id,
                        prompt_hash, raw_response, raw_decision.parse_status, None,
                        latency, raw_decision, None, None
                    )
                calibrated = (
                    self.rationality_regulator.calibrate_decision(
                        scenario, raw_decision, opponent_type
                    ) if use_calibration else None
                )
                evaluated_decision = calibrated or raw_decision
                evaluation = self.evaluator.evaluate_scenario(
                    scenario, evaluated_decision,
                    baseline_decision=scenario.vehicle_1.acceleration
                )
                evaluation.response_time = latency
            else:
                raw_decision = self.gamecard.parse_fuzzy_response(
                    raw_response, scenario, strict=self.strict_parse
                )
                should_score = (
                    raw_decision.is_valid and raw_decision.parse_status == "strict_json"
                ) if self.strict_parse else True
                if not should_score:
                    return SampleEvaluationRecord(
                        scenario.scenario_id, scenario.frame_id, scenario.sample_id,
                        prompt_hash, raw_response, raw_decision.parse_status, None,
                        latency, raw_decision, None, None
                    )
                calibrated = None
                evaluation = self.fuzzy_evaluator.evaluate_fuzzy_decision(
                    scenario=scenario, fuzzy_decision=raw_decision,
                    ground_truth_priority=scenario.ground_truth_priority,
                    response_time=latency
                )
            parse_status = raw_decision.parse_status
        except Exception as exc:
            return SampleEvaluationRecord(
                scenario.scenario_id, scenario.frame_id, scenario.sample_id,
                prompt_hash, raw_response, "parse_error", None, latency,
                None, None, None
            )

        return SampleEvaluationRecord(
            scenario.scenario_id, scenario.frame_id, scenario.sample_id,
            prompt_hash, raw_response, parse_status, None, latency,
            raw_decision, calibrated, evaluation
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        """Accept legacy strings and structured GenerationResult-like objects."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("text", "content", "raw_response"):
                if response.get(key) is not None:
                    return str(response[key])
        for attribute in ("text", "content", "raw_response"):
            value = getattr(response, attribute, None)
            if value is not None:
                return str(value)
        raise TypeError("Generation result does not expose text/content/raw_response")

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
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        temperature = str(self.temperature).replace(".", "p")

        # Determine file extension
        extension = "json" if file_type == "results" else "txt"

        # Construct filename using new convention
        # Results: evaluation_results_<data source>_<model type>_<fuzzy/precision>_<prompt type>_<time>.<ext>
        # Reports: evaluation_report_<data source>_<model type>_<fuzzy/precision>_<prompt type>_<time>.<ext>
        prefix = "evaluation_results" if file_type == "results" else "evaluation_report"
        filename = f"{prefix}_{self.model_type}_{self.decision_mode}_temp{temperature}_{timestamp}.{extension}"

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
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _metadata(self, total_scenarios: int) -> Dict[str, Any]:
        requested = {
            "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens, "generation_seed": self.generation_seed
        }
        return {
            "total_scenarios": total_scenarios,
            "model_type": self.model_type,
            "provider": self.model_type,
            "model_name": getattr(self.llm_interface, "model_name", None),
            "decision_mode": self.decision_mode,
            "prompt_format": self.prompt_format,
            "cot_type": self.cot_type,
            "requested_parameters": requested,
            "effective_parameters": dict(requested),
            "parameter_capabilities": self.llm_interface.get_parameter_capabilities()
            if hasattr(self.llm_interface, "get_parameter_capabilities") else None,
            "generation_seed": self.generation_seed,
            "sample_seed": self.sample_seed,
            "manifest_path": self.manifest_path,
            "data_path": str(self.data_path),
            "git_commit": get_git_commit(),
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "calibration_enabled": self.calibration_enabled,
            "strict_parse": self.strict_parse
        }

    @staticmethod
    def _model_dump(value):
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value

    def _record_to_dict(self, record: SampleEvaluationRecord) -> Dict[str, Any]:
        evaluation = record.evaluation
        if isinstance(evaluation, EvaluationResult):
            metrics = evaluation.metrics.to_dict()
        elif evaluation is not None:
            metrics = {
                "ground_truth": evaluation.ground_truth,
                "is_correct": evaluation.is_correct,
                "confidence_score": evaluation.confidence_score,
                "risk_level": evaluation.risk_level,
                "scenario_type": evaluation.scenario_type,
                "fuzzy_weights": evaluation.fuzzy_weights
            }
        else:
            metrics = None
        generation_parameters = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "generation_seed": self.generation_seed
        }
        return {
            "scenario_id": record.scenario_id,
            "frame_id": record.frame_id,
            "sample_id": record.sample_id,
            "model_name": getattr(self.llm_interface, "model_name", None),
            "requested_parameters": generation_parameters,
            "effective_parameters": dict(generation_parameters),
            "prompt_hash": record.prompt_hash,
            "raw_response": record.raw_response,
            "parse_status": record.parse_status,
            "api_error": record.api_error,
            "latency": record.latency,
            "raw_decision": self._model_dump(record.raw_decision),
            "calibrated_decision": self._model_dump(record.calibrated_decision),
            "evaluation_metrics": metrics
        }

    @staticmethod
    def _parse_statistics(results: List[SampleEvaluationRecord]) -> Dict[str, Any]:
        total = len(results)
        api_success = sum(r.parse_status != "api_error" for r in results)
        strict_valid = sum(
            r.parse_status == "strict_json"
            and r.raw_decision is not None
            and r.raw_decision.is_valid
            for r in results
        )
        fallback = sum(
            r.parse_status in {"regex_fallback", "default_fallback"}
            for r in results
        )
        invalid = sum(
            r.raw_decision is not None and not r.raw_decision.is_valid
            for r in results
        )
        parse_errors = sum(r.parse_status == "parse_error" for r in results)

        def rate(count):
            return count / total if total else 0.0

        return {
            "total_samples": total,
            "api_success_count": api_success,
            "api_success_rate": rate(api_success),
            "strict_json_valid_count": strict_valid,
            "strict_json_valid_rate": rate(strict_valid),
            "fallback_count": fallback,
            "fallback_rate": rate(fallback),
            "invalid_count": invalid,
            "invalid_rate": rate(invalid),
            "parse_error_count": parse_errors,
            "parse_error_rate": rate(parse_errors)
        }

    def _save_precise_results(self, results: List[SampleEvaluationRecord]) -> Dict[str, Any]:
        """Save results for precise mode"""
        results_data = {
            "metadata": self._metadata(len(results)),
            "results": [self._record_to_dict(result) for result in results],
            "parse_statistics": self._parse_statistics(results)
        }

        # Calculate aggregate statistics
        if results:
            metrics = [r.evaluation.metrics for r in results if r.evaluation is not None]
        else:
            metrics = []
        if metrics:
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

    def _save_fuzzy_results(self, results: List[SampleEvaluationRecord]) -> Dict[str, Any]:
        """Save results for fuzzy mode"""
        # Convert FuzzyEvaluationResult objects to dictionaries
        results_dicts = [self._record_to_dict(result) for result in results]

        # Calculate metrics using the fuzzy evaluator
        valid_results = [r.evaluation for r in results if r.evaluation is not None]
        metrics = self.fuzzy_evaluator.evaluate_batch(valid_results)

        results_data = {
            "metadata": self._metadata(len(results)),
            "results": results_dicts,
            "parse_statistics": self._parse_statistics(results),
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
        evaluations = [
            item.evaluation if isinstance(item, SampleEvaluationRecord) else item
            for item in results
            if not isinstance(item, SampleEvaluationRecord) or item.evaluation is not None
        ]
        if not evaluations:
            return "No results to report."

        if self.decision_mode == "precise":
            return self._generate_precise_report(evaluations)
        else:
            return self._generate_fuzzy_report(evaluations)

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
    args = build_parser().parse_args()

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
        cot_type=args.cot_type,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        generation_seed=args.generation_seed,
        sample_seed=args.sample_seed,
        manifest_path=args.manifest_path,
        strict_parse=args.strict_parse
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

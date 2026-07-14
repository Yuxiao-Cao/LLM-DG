#!/usr/bin/env python3
"""Run serial temperature-stability experiments on a fixed manifest."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import InteractionDataLoader
from src.data_models import EvaluationResult
from src.evaluation import FuzzyEvaluator, PreciseEvaluator
from src.gamecard import GameCard
from src.llm_interface import create_llm_interface


@dataclass
class StabilityTask:
    sequence_id: int
    temperature: float
    repeat_id: int
    scenario: Any
    prompt: str
    prompt_sha256: str
    job_id: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a serial temperature-stability experiment on a fixed manifest"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--model-type", required=True,
                        choices=["doubao", "openai", "deepseek", "qwen", "gemini", "claude"])
    parser.add_argument("--decision-mode", choices=["precise", "fuzzy"], required=True)
    parser.add_argument("--prompt-format", choices=["text", "json", "text+json"], default="text+json")
    parser.add_argument("--cot-type", choices=["cot", "nocot"], default="cot")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--schedule-seed", type=int, default=20260712)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _temperature_label(value: float) -> str:
    return format(value, ".12g")


def build_tasks(
    scenarios: Iterable[Any], gamecard: GameCard, model_type: str,
    decision_mode: str, temperatures: List[float], repeats: int,
    schedule_seed: int
) -> List[StabilityTask]:
    if repeats <= 0:
        raise ValueError("--repeats must be greater than zero")
    if not temperatures:
        raise ValueError("At least one temperature is required")

    prompt_by_sample = {}
    scenario_by_sample = {}
    for scenario in scenarios:
        if decision_mode == "precise":
            prompt, _ = gamecard.create_precise_prompt(scenario)
        else:
            prompt, _ = gamecard.create_fuzzy_prompt(scenario)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_by_sample[scenario.sample_id] = (prompt, prompt_hash)
        scenario_by_sample[scenario.sample_id] = scenario

    task_specs = []
    for temperature in temperatures:
        for repeat_id in range(repeats):
            for sample_id, scenario in scenario_by_sample.items():
                prompt, prompt_hash = prompt_by_sample[sample_id]
                job_id = "|".join([
                    model_type, decision_mode, f"temperature={_temperature_label(temperature)}",
                    f"repeat={repeat_id}", f"sample={sample_id}", f"prompt={prompt_hash}"
                ])
                task_specs.append((temperature, repeat_id, scenario, prompt, prompt_hash, job_id))

    random.Random(schedule_seed).shuffle(task_specs)
    return [
        StabilityTask(index, temperature, repeat_id, scenario, prompt, prompt_hash, job_id)
        for index, (temperature, repeat_id, scenario, prompt, prompt_hash, job_id)
        in enumerate(task_specs)
    ]


def load_fixed_scenarios(data_path: str, manifest_path: str) -> List[Any]:
    loader = InteractionDataLoader(data_path)
    loader.load_data()
    return loader.get_scenarios_from_manifest(manifest_path)


def load_successful_job_ids(output_path: Path) -> Set[str]:
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume: invalid JSONL at line {line_number}: {exc}"
                ) from exc
            if record.get("job_id") and record.get("api_error") is None:
                completed.add(record["job_id"])
    return completed


def unpack_generation_result(result: Any) -> Tuple[str, Any, Any, Optional[Dict[str, Any]]]:
    if isinstance(result, str):
        return result, None, None, None
    if isinstance(result, dict):
        text = next((result.get(key) for key in ("text", "content", "raw_response")
                     if result.get(key) is not None), None)
        usage = result.get("usage")
        fingerprint = result.get("system_fingerprint")
        effective = result.get("effective_parameters")
    else:
        text = next((getattr(result, key, None) for key in ("text", "content", "raw_response")
                     if getattr(result, key, None) is not None), None)
        usage = getattr(result, "usage", None)
        fingerprint = getattr(result, "system_fingerprint", None)
        effective = getattr(result, "effective_parameters", None)
    if text is None:
        raise TypeError("Generation result does not expose text/content/raw_response")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    return str(text), usage, fingerprint, effective


def model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def execute_task(
    task: StabilityTask, interface: Any, gamecard: GameCard,
    decision_mode: str, top_p: float, max_tokens: int,
    generation_seed: Optional[int], max_retries: int, retry_backoff: float,
    precise_evaluator: Optional[PreciseEvaluator] = None,
    fuzzy_evaluator: Optional[FuzzyEvaluator] = None,
) -> Dict[str, Any]:
    requested = {
        "temperature": task.temperature, "top_p": top_p,
        "max_tokens": max_tokens, "generation_seed": generation_seed
    }
    attempts = []
    raw_response = None
    usage = None
    fingerprint = None
    effective = None
    api_error = None
    total_started = time.perf_counter()

    for attempt_id in range(max_retries + 1):
        attempt_started = time.perf_counter()
        try:
            generated = interface.generate_response(task.prompt, **requested)
            raw_response, usage, fingerprint, effective = unpack_generation_result(generated)
            attempts.append({
                "attempt_id": attempt_id,
                "latency": time.perf_counter() - attempt_started,
                "api_error": None
            })
            api_error = None
            break
        except Exception as exc:
            api_error = str(exc)
            attempts.append({
                "attempt_id": attempt_id,
                "latency": time.perf_counter() - attempt_started,
                "api_error": api_error
            })
            if attempt_id < max_retries and retry_backoff > 0:
                time.sleep(retry_backoff * (2 ** attempt_id))

    raw_decision = None
    metrics = None
    parse_status = "api_error"
    is_valid = False
    parse_error = None
    if api_error is None:
        try:
            if decision_mode == "precise":
                raw_decision = gamecard.parse_precise_response(raw_response, strict=True)
                strict_valid = (
                    raw_decision.is_valid and raw_decision.parse_status == "strict_json"
                )
                if strict_valid:
                    evaluation = (precise_evaluator or PreciseEvaluator()).evaluate_scenario(
                        task.scenario, raw_decision,
                        baseline_decision=task.scenario.vehicle_1.acceleration
                    )
                    metrics = evaluation.metrics.to_dict()
            else:
                raw_decision = gamecard.parse_fuzzy_response(
                    raw_response, task.scenario, strict=True
                )
                strict_valid = (
                    raw_decision.is_valid and raw_decision.parse_status == "strict_json"
                )
                if strict_valid:
                    evaluation = (fuzzy_evaluator or FuzzyEvaluator()).evaluate_fuzzy_decision(
                        scenario=task.scenario, fuzzy_decision=raw_decision,
                        ground_truth_priority=task.scenario.ground_truth_priority,
                        response_time=time.perf_counter() - total_started
                    )
                    metrics = {
                        "ground_truth": evaluation.ground_truth,
                        "is_correct": evaluation.is_correct,
                        "confidence_score": evaluation.confidence_score,
                        "risk_level": evaluation.risk_level,
                        "scenario_type": evaluation.scenario_type,
                        "fuzzy_weights": evaluation.fuzzy_weights
                    }
            parse_status = raw_decision.parse_status
            is_valid = strict_valid
            parse_error = raw_decision.parse_error
        except Exception as exc:
            parse_status = "parse_error"
            is_valid = False
            parse_error = str(exc)
            raw_decision = None
            metrics = None

    return {
        "job_id": task.job_id,
        "sequence_id": task.sequence_id,
        "scenario_id": task.scenario.scenario_id,
        "frame_id": task.scenario.frame_id,
        "sample_id": task.scenario.sample_id,
        "model_type": task.job_id.split("|", 1)[0],
        "provider": getattr(interface, "provider", task.job_id.split("|", 1)[0]),
        "model_name": getattr(interface, "model_name", None),
        "decision_mode": decision_mode,
        "temperature": task.temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "requested_parameters": requested,
        "effective_parameters": effective or dict(requested),
        "parameter_capabilities": interface.get_parameter_capabilities()
        if hasattr(interface, "get_parameter_capabilities") else None,
        "repeat_id": task.repeat_id,
        "prompt_sha256": task.prompt_sha256,
        "raw_response": raw_response,
        "api_usage": usage,
        "system_fingerprint": fingerprint,
        "latency": time.perf_counter() - total_started,
        "attempts": attempts,
        "api_error": api_error,
        "parse_status": parse_status,
        "is_valid": is_valid,
        "parse_error": parse_error,
        "raw_decision": model_dump(raw_decision),
        "metrics": metrics
    }


def run_tasks(
    tasks: List[StabilityTask], output_path: Path, interface: Any,
    gamecard: GameCard, decision_mode: str, top_p: float, max_tokens: int,
    generation_seed: Optional[int], max_retries: int, retry_backoff: float,
    resume: bool = False
) -> int:
    completed = load_successful_job_ids(output_path) if resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            if task.job_id in completed:
                continue
            record = execute_task(
                task, interface, gamecard, decision_mode, top_p, max_tokens,
                generation_seed, max_retries, retry_backoff
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            written += 1
    return written


def main() -> None:
    args = build_parser().parse_args()
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff cannot be negative")

    scenarios = load_fixed_scenarios(args.data_path, args.manifest_path)
    gamecard = GameCard(args.prompt_format, args.cot_type)
    tasks = build_tasks(
        scenarios, gamecard, args.model_type, args.decision_mode,
        args.temperatures, args.repeats, args.schedule_seed
    )
    output_path = Path(args.output).resolve()

    if args.dry_run:
        print("Temperature stability dry run")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Model: {args.model_type}")
        print(f"  Temperatures: {args.temperatures}")
        print(f"  Repeats: {args.repeats}")
        print(f"  Samples: {len(scenarios)}")
        print(f"  Output: {output_path}")
        return

    interface = create_llm_interface(args.model_type)
    if not hasattr(interface, "provider"):
        interface.provider = args.model_type
    written = run_tasks(
        tasks, output_path, interface, gamecard, args.decision_mode,
        args.top_p, args.max_tokens, args.generation_seed,
        args.max_retries, args.retry_backoff, args.resume
    )
    print(f"Completed: wrote {written} task records to {output_path}")


if __name__ == "__main__":
    main()

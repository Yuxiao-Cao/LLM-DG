#!/usr/bin/env python3
"""Analyze repeated temperature-stability JSONL outputs without API access."""

import argparse
import glob
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_FIELDS = {
    "job_id", "sample_id", "model_type", "decision_mode", "temperature",
    "repeat_id", "api_error", "parse_status", "is_valid", "raw_decision",
    "metrics", "latency"
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready statistics from temperature stability JSONL files"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more JSONL paths or glob patterns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--include-invalid", action="store_true")
    return parser


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(matches)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No input JSONL files matched --inputs")
    return unique


def read_jsonl(paths: Sequence[Path]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    records = []
    malformed = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed.append({
                        "file": str(path), "line": line_number, "error": str(exc)
                    })
                    continue
                record["_present_fields"] = list(record.keys())
                record["_source_file"] = str(path)
                record["_source_line"] = line_number
                records.append(record)
    return pd.DataFrame(records), malformed


def _model_column(data: pd.DataFrame) -> str:
    if "model_name" in data and data["model_name"].notna().any():
        return "model_name"
    return "model_type"


def completeness_report(data: pd.DataFrame, malformed: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_fields = []
    for index, record in data.iterrows():
        present = set(record.get("_present_fields", record.index))
        missing = sorted(REQUIRED_FIELDS.difference(present))
        if missing:
            missing_fields.append({
                "row": int(index), "job_id": record.get("job_id"), "fields": missing
            })

    duplicate_ids = []
    if "job_id" in data:
        duplicate_ids = sorted(
            str(value) for value in data.loc[data["job_id"].duplicated(False), "job_id"].dropna().unique()
        )

    incomplete = []
    globally_expected_ids = []
    if not data.empty and {"model_type", "temperature", "sample_id", "repeat_id"}.issubset(data.columns):
        globally_expected_ids = sorted(int(value) for value in data["repeat_id"].dropna().unique())
        for (model, temperature), model_group in data.groupby(["model_type", "temperature"], dropna=False):
            expected_ids = globally_expected_ids
            for sample_id, sample_group in model_group.groupby("sample_id", dropna=False):
                actual_ids = sorted(int(value) for value in sample_group["repeat_id"].dropna().unique())
                missing_ids = sorted(set(expected_ids).difference(actual_ids))
                duplicate_repeats = sorted(
                    int(value) for value in sample_group.loc[
                        sample_group["repeat_id"].duplicated(False), "repeat_id"
                    ].dropna().unique()
                )
                if missing_ids or duplicate_repeats:
                    incomplete.append({
                        "model_type": model, "temperature": float(temperature),
                        "sample_id": sample_id, "expected_repeat_ids": expected_ids,
                        "actual_repeat_ids": actual_ids, "missing_repeat_ids": missing_ids,
                        "duplicate_repeat_ids": duplicate_repeats
                    })

    api_errors = []
    if "api_error" in data:
        for _, row in data[data["api_error"].notna()].iterrows():
            api_errors.append({
                "job_id": row.get("job_id"), "api_error": row.get("api_error"),
                "source_file": row.get("_source_file"), "source_line": row.get("_source_line")
            })
    return {
        "input_record_count": len(data),
        "malformed_jsonl_lines": malformed,
        "duplicate_job_ids": duplicate_ids,
        "missing_required_fields": missing_fields,
        "expected_repeat_ids_inferred_from_all_inputs": globally_expected_ids,
        "repeat_expectation_rule": "union of repeat_id values observed across all input records",
        "incomplete_repeat_groups": incomplete,
        "api_error_count": len(api_errors),
        "api_errors": api_errors,
        "is_complete": not (malformed or duplicate_ids or missing_fields or incomplete or api_errors)
    }


def mean_absolute_pairwise_difference(values: Sequence[float]) -> float:
    values = [float(value) for value in values if pd.notna(value)]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else math.nan
    return float(np.mean([abs(left - right) for left, right in combinations(values, 2)]))


def majority_agreement(values: Sequence[Any]) -> float:
    values = [value for value in values if pd.notna(value)]
    if not values:
        return math.nan
    counts = pd.Series(values).value_counts()
    return float(counts.iloc[0] / len(values))


def pairwise_agreement(values: Sequence[Any]) -> float:
    values = [value for value in values if pd.notna(value)]
    if len(values) < 2:
        return 1.0 if len(values) == 1 else math.nan
    pairs = list(combinations(values, 2))
    return float(sum(left == right for left, right in pairs) / len(pairs))


def normalized_decision_entropy(values: Sequence[Any]) -> float:
    values = [value for value in values if pd.notna(value)]
    if not values:
        return math.nan
    probabilities = pd.Series(values).value_counts(normalize=True).to_numpy(dtype=float)
    if len(probabilities) == 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(probabilities))


def _nested(record: pd.Series, column: str, key: str) -> Any:
    value = record.get(column)
    return value.get(key) if isinstance(value, dict) else None


def _eligible(group: pd.DataFrame, include_invalid: bool) -> pd.DataFrame:
    api_ok = group["api_error"].isna()
    return group[api_ok] if include_invalid else group[api_ok & group["is_valid"].fillna(False)]


def precise_per_scenario(data: pd.DataFrame, include_invalid: bool = False) -> pd.DataFrame:
    precise = data[data["decision_mode"] == "precise"].copy()
    if precise.empty:
        return pd.DataFrame()
    model_col = _model_column(precise)
    rows = []
    for (model, temperature, sample_id), group in precise.groupby([model_col, "temperature", "sample_id"]):
        selected = _eligible(group, include_invalid)
        accelerations = [
            _nested(row, "raw_decision", "acceleration_1") for _, row in selected.iterrows()
        ]
        accelerations = [float(value) for value in accelerations if value is not None]
        scores = [_nested(row, "metrics", "overall_score") for _, row in selected.iterrows()]
        scores = [float(value) for value in scores if value is not None]
        rows.append({
            "model": model, "temperature": float(temperature), "sample_id": sample_id,
            "expected_repeats": int(group["repeat_id"].nunique()),
            "observed_calls": len(group), "included_decisions": len(accelerations),
            "acceleration_mean": np.mean(accelerations) if accelerations else math.nan,
            "acceleration_within_scenario_sd": np.std(accelerations, ddof=0) if accelerations else math.nan,
            "acceleration_mapd": mean_absolute_pairwise_difference(accelerations),
            "overall_score_mean": np.mean(scores) if scores else math.nan,
            "overall_score_within_scenario_sd": np.std(scores, ddof=0) if scores else math.nan,
            "valid_response_rate": float(group["is_valid"].fillna(False).mean()),
            "strict_json_valid_rate": float(((group["parse_status"] == "strict_json") & group["is_valid"].fillna(False)).mean()),
            "api_success_rate": float(group["api_error"].isna().mean())
        })
    return pd.DataFrame(rows)


def fuzzy_per_scenario(data: pd.DataFrame, include_invalid: bool = False) -> pd.DataFrame:
    fuzzy = data[data["decision_mode"] == "fuzzy"].copy()
    if fuzzy.empty:
        return pd.DataFrame()
    model_col = _model_column(fuzzy)
    rows = []
    for (model, temperature, sample_id), group in fuzzy.groupby([model_col, "temperature", "sample_id"]):
        selected = _eligible(group, include_invalid)
        decisions = [_nested(row, "raw_decision", "priority_vehicle") for _, row in selected.iterrows()]
        decisions = [value for value in decisions if value is not None]
        accuracy = [_nested(row, "metrics", "is_correct") for _, row in selected.iterrows()]
        accuracy = [float(value) for value in accuracy if value is not None]
        rows.append({
            "model": model, "temperature": float(temperature), "sample_id": sample_id,
            "expected_repeats": int(group["repeat_id"].nunique()),
            "observed_calls": len(group), "included_decisions": len(decisions),
            "majority_agreement": majority_agreement(decisions),
            "pairwise_agreement": pairwise_agreement(decisions),
            "normalized_decision_entropy": normalized_decision_entropy(decisions),
            "accuracy_mean": np.mean(accuracy) if accuracy else math.nan,
            "accuracy_within_scenario_sd": np.std(accuracy, ddof=0) if accuracy else math.nan,
            "strict_json_valid_rate": float(((group["parse_status"] == "strict_json") & group["is_valid"].fillna(False)).mean()),
            "api_success_rate": float(group["api_error"].isna().mean())
        })
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: Sequence[float], samples: int, rng: np.random.Generator) -> Tuple[float, float]:
    array = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if array.size == 0 or samples <= 0:
        return math.nan, math.nan
    means = np.mean(rng.choice(array, size=(samples, array.size), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_per_scenario(
    per_scenario: pd.DataFrame, raw_data: pd.DataFrame,
    bootstrap_samples: int, bootstrap_seed: int
) -> pd.DataFrame:
    if per_scenario.empty:
        return pd.DataFrame()
    metric_columns = [
        column for column in per_scenario.columns
        if column not in {"model", "temperature", "sample_id", "expected_repeats", "observed_calls", "included_decisions"}
    ]
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    model_col = _model_column(raw_data)
    for (model, temperature), group in per_scenario.groupby(["model", "temperature"]):
        row = {"model": model, "temperature": temperature, "scenario_count": len(group)}
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, bootstrap_samples, rng)
            row[f"{metric}_scenario_mean"] = np.mean(values) if len(values) else math.nan
            row[f"{metric}_scenario_median"] = np.median(values) if len(values) else math.nan
            row[f"{metric}_cross_scenario_sd"] = np.std(values, ddof=0) if len(values) else math.nan
            row[f"{metric}_bootstrap_ci95_low"] = low
            row[f"{metric}_bootstrap_ci95_high"] = high
        latency = raw_data[
            (raw_data[model_col] == model) &
            (raw_data["temperature"].astype(float) == float(temperature)) &
            raw_data["api_error"].isna()
        ]["latency"].dropna().astype(float)
        row["latency_mean"] = latency.mean() if len(latency) else math.nan
        row["latency_median"] = latency.median() if len(latency) else math.nan
        row["latency_p95"] = latency.quantile(0.95) if len(latency) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_from_scores(left: pd.Series, right: pd.Series) -> float:
    common = left.dropna().index.intersection(right.dropna().index)
    if len(common) < 2:
        return math.nan
    left_rank = left.loc[common].rank(method="average")
    right_rank = right.loc[common].rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def rank_stability(data: pd.DataFrame, include_invalid: bool = False) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    model_col = _model_column(data)
    working = data[data["api_error"].isna()].copy()
    if not include_invalid:
        working = working[working["is_valid"].fillna(False)]
    working["score"] = working.apply(
        lambda row: _nested(row, "metrics", "overall_score")
        if row["decision_mode"] == "precise"
        else _nested(row, "metrics", "is_correct"), axis=1
    )
    working = working[working["score"].notna()]
    ranking_rows = []
    score_tables = {}
    for (temperature, repeat_id), group in working.groupby(["temperature", "repeat_id"]):
        sample_sets = [set(model_group["sample_id"]) for _, model_group in group.groupby(model_col)]
        common_samples = set.intersection(*sample_sets) if sample_sets else set()
        filtered = group[group["sample_id"].isin(common_samples)]
        scores = filtered.groupby(model_col)["score"].mean()
        ranks = scores.rank(ascending=False, method="average")
        score_tables[(float(temperature), int(repeat_id))] = scores
        for model in scores.index:
            ranking_rows.append({
                "analysis_type": "repeat_ranking", "model": model,
                "temperature": float(temperature), "repeat_id": int(repeat_id),
                "score": float(scores[model]), "rank": float(ranks[model]),
                "common_sample_count": len(common_samples), "correlation": math.nan,
                "filter_rule": "intersection of valid scored sample_ids across models"
            })

    for repeat_id in sorted({key[1] for key in score_tables}):
        left = score_tables.get((0.0, repeat_id), pd.Series(dtype=float))
        right = score_tables.get((0.7, repeat_id), pd.Series(dtype=float))
        correlation = spearman_from_scores(left, right)
        ranking_rows.append({
            "analysis_type": "temperature_0_vs_0.7_spearman", "model": None,
            "temperature": None, "repeat_id": repeat_id, "score": math.nan,
            "rank": math.nan, "common_sample_count": len(left.index.intersection(right.index)),
            "correlation": correlation,
            "filter_rule": "models present with scores at both temperatures for the same repeat"
        })

    for temperature in sorted({key[0] for key in score_tables}):
        repeat_ids = sorted(key[1] for key in score_tables if key[0] == temperature)
        for left_id, right_id in combinations(repeat_ids, 2):
            left, right = score_tables[(temperature, left_id)], score_tables[(temperature, right_id)]
            ranking_rows.append({
                "analysis_type": "within_temperature_repeat_spearman", "model": None,
                "temperature": temperature, "repeat_id": f"{left_id}::{right_id}",
                "score": math.nan, "rank": math.nan,
                "common_sample_count": len(left.index.intersection(right.index)),
                "correlation": spearman_from_scores(left, right),
                "filter_rule": "models present with scores in both repeats"
            })
    return pd.DataFrame(ranking_rows)


def write_markdown_report(
    output_path: Path, report: Dict[str, Any], precise_summary: pd.DataFrame,
    fuzzy_summary: pd.DataFrame, rank_table: pd.DataFrame
) -> None:
    lines = [
        "# Temperature Stability Analysis", "",
        f"- Input records: {report['input_record_count']}",
        f"- Duplicate job IDs: {len(report['duplicate_job_ids'])}",
        f"- Incomplete repeat groups: {len(report['incomplete_repeat_groups'])}",
        f"- API errors: {report['api_error_count']}", "",
        "## Interpretation", "",
        "`*_within_scenario_sd` measures variation across repeated generations of the same scene. ",
        "`*_cross_scenario_sd` measures variation of scenario-level statistics across different scenes. ",
        "Bootstrap intervals are 95% confidence intervals for the mean across scenarios.", ""
    ]
    preferred = {
        "Precise summary": [
            "model", "temperature", "scenario_count",
            "acceleration_within_scenario_sd_scenario_mean",
            "acceleration_mapd_scenario_mean", "overall_score_mean_scenario_mean",
            "strict_json_valid_rate_scenario_mean", "api_success_rate_scenario_mean",
            "latency_mean", "latency_p95"
        ],
        "Fuzzy summary": [
            "model", "temperature", "scenario_count",
            "majority_agreement_scenario_mean", "pairwise_agreement_scenario_mean",
            "normalized_decision_entropy_scenario_mean", "accuracy_mean_scenario_mean",
            "strict_json_valid_rate_scenario_mean", "api_success_rate_scenario_mean"
        ],
        "Rank stability": [
            "analysis_type", "model", "temperature", "repeat_id",
            "score", "rank", "correlation", "common_sample_count"
        ]
    }
    for title, table in (("Precise summary", precise_summary), ("Fuzzy summary", fuzzy_summary), ("Rank stability", rank_table)):
        lines.extend([f"## {title}", ""])
        if table.empty:
            lines.extend(["No applicable records.", ""])
        else:
            columns = [column for column in preferred[title] if column in table.columns]
            lines.extend([dataframe_to_markdown(table[columns]), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate dependency."""
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator] + rows)


def analyze(
    input_paths: Sequence[Path], output_dir: Path, bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42, include_invalid: bool = False
) -> Dict[str, Any]:
    data, malformed = read_jsonl(input_paths)
    report = completeness_report(data, malformed)
    output_dir.mkdir(parents=True, exist_ok=True)

    precise_scenario = precise_per_scenario(data, include_invalid) if not data.empty else pd.DataFrame()
    fuzzy_scenario = fuzzy_per_scenario(data, include_invalid) if not data.empty else pd.DataFrame()
    precise_summary = summarize_per_scenario(
        precise_scenario, data, bootstrap_samples, bootstrap_seed
    ) if not precise_scenario.empty else pd.DataFrame()
    fuzzy_summary = summarize_per_scenario(
        fuzzy_scenario, data, bootstrap_samples, bootstrap_seed
    ) if not fuzzy_scenario.empty else pd.DataFrame()
    ranks = rank_stability(data, include_invalid) if not data.empty else pd.DataFrame()

    outputs = {
        "precise_per_scenario.csv": precise_scenario,
        "precise_summary.csv": precise_summary,
        "fuzzy_per_scenario.csv": fuzzy_scenario,
        "fuzzy_summary.csv": fuzzy_summary,
        "rank_stability.csv": ranks,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False, lineterminator="\n")
    (output_dir / "completeness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stability_summary = {
        "inputs": [str(path) for path in input_paths],
        "include_invalid": include_invalid,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "precise_model_temperature_groups": len(precise_summary),
        "fuzzy_model_temperature_groups": len(fuzzy_summary),
        "rank_analysis_rows": len(ranks),
        "precise_summary": json.loads(precise_summary.to_json(orient="records")),
        "fuzzy_summary": json.loads(fuzzy_summary.to_json(orient="records")),
        "rank_correlations": json.loads(
            ranks[ranks["analysis_type"] != "repeat_ranking"].to_json(orient="records")
        ) if not ranks.empty else [],
        "completeness": {
            "is_complete": report["is_complete"],
            "api_error_count": report["api_error_count"],
            "duplicate_job_id_count": len(report["duplicate_job_ids"]),
            "incomplete_repeat_group_count": len(report["incomplete_repeat_groups"])
        }
    }
    (output_dir / "stability_summary.json").write_text(
        json.dumps(stability_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_report(
        output_dir / "stability_report.md", report, precise_summary, fuzzy_summary, ranks
    )
    return stability_summary


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be greater than zero")
    paths = expand_inputs(args.inputs)
    summary = analyze(
        paths, Path(args.output_dir), args.bootstrap_samples,
        args.bootstrap_seed, args.include_invalid
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a deterministic, versionable evaluation manifest from a dataset."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "Scenario_id", "frame_id", "Scenario_type", "track_id_1", "track_id_2"
}
OUTPUT_COLUMNS = [
    "manifest_order", "Scenario_id", "frame_id", "Scenario_type",
    "track_id_1", "track_id_2", "d_1", "v_1", "d_2", "v_2"
]
UNIQUE_KEY = ["Scenario_id", "frame_id"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic open-loop evaluation manifest"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-scenarios", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario-type")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.metadata.json")


def create_manifest(
    data_path: str,
    num_scenarios: int,
    output: str,
    seed: int = 42,
    scenario_type: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    source_path = Path(data_path).resolve()
    output_path = Path(output).resolve()
    metadata_path = metadata_path_for(output_path)

    if num_scenarios <= 0:
        raise ValueError("--num-scenarios must be greater than zero")
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source_path}")

    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing file(s): "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace them."
        )

    data = pd.read_csv(source_path, dtype={"Scenario_id": str})
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if data[UNIQUE_KEY].isna().any().any():
        raise ValueError("Unique key fields Scenario_id and frame_id cannot be empty")

    duplicates = data.duplicated(subset=UNIQUE_KEY, keep=False)
    if duplicates.any():
        duplicate_keys = data.loc[duplicates, UNIQUE_KEY].drop_duplicates()
        formatted = ", ".join(
            f"{scenario_id}::{frame_id}"
            for scenario_id, frame_id in duplicate_keys.itertuples(index=False, name=None)
        )
        raise ValueError(f"Dataset contains duplicate unique keys: {formatted}")

    candidates = data
    if scenario_type is not None:
        candidates = candidates[candidates["Scenario_type"] == scenario_type]
    if num_scenarios > len(candidates):
        filter_note = f" after filtering Scenario_type={scenario_type!r}" if scenario_type else ""
        raise ValueError(
            f"Requested {num_scenarios} samples, but only {len(candidates)} are available{filter_note}"
        )

    sampled = candidates.sample(
        n=num_scenarios, random_state=seed, replace=False
    ).copy()
    sampled.insert(0, "manifest_order", range(len(sampled)))
    for optional_column in OUTPUT_COLUMNS:
        if optional_column not in sampled.columns:
            sampled[optional_column] = None
    manifest = sampled[OUTPUT_COLUMNS]

    metadata = {
        "source_data_path": str(source_path),
        "source_data_sha256": sha256_file(source_path),
        "seed": seed,
        "requested_sample_count": num_scenarios,
        "actual_sample_count": len(manifest),
        "scenario_type_filter": scenario_type,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(source_path.parent),
        "unique_key": "Scenario_id + frame_id",
        "unique_key_columns": UNIQUE_KEY,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, lineterminator="\n")
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output_path, metadata_path


def main() -> None:
    args = build_parser().parse_args()
    manifest_path, metadata_path = create_manifest(
        data_path=args.data_path,
        num_scenarios=args.num_scenarios,
        output=args.output,
        seed=args.seed,
        scenario_type=args.scenario_type,
        overwrite=args.overwrite,
    )
    print(f"Manifest written to: {manifest_path}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()

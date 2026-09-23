from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..data_loader import read_dataset_csv, write_dataset_csv

EXPECTED_RESULT_FILES = 4
EXPECTED_ROWS = 2720
EXPECTED_MODELS = 4
EXPECTED_DILEMMAS = 680
KEY_COLUMNS = ("dilemma_id", "model")


@dataclass(frozen=True)
class MasterDatasetSummary:
    """Short summary of the combined analysis master dataset."""

    loaded_files: int
    rows: int
    unique_dilemmas: int
    unique_models: int
    duplicate_dilemma_model_pairs: int
    missing_final_answers: int
    output_path: Path
    warnings: tuple[str, ...]


def build_analysis_master(
    *,
    results_dir: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> MasterDatasetSummary:
    """Build the analysis master CSV from the final model result CSV files."""
    result_files = sorted(Path(results_dir).glob("*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No result CSV files found in: {Path(results_dir)}")

    dataframes = [read_dataset_csv(file) for file in result_files]
    master_dataset = pd.concat(dataframes, ignore_index=True)
    _validate_required_columns(master_dataset)

    master_dataset = _sort_master_dataset(master_dataset)
    duplicate_pairs = int(master_dataset.duplicated(subset=list(KEY_COLUMNS)).sum())
    missing_final_answers = int(_missing_final_answer(master_dataset).sum())
    warnings = _collect_warnings(
        loaded_files=len(result_files),
        rows=len(master_dataset),
        unique_models=master_dataset["model"].nunique(),
        unique_dilemmas=master_dataset["dilemma_id"].nunique(),
        duplicate_pairs=duplicate_pairs,
        missing_final_answers=missing_final_answers,
    )

    saved_path = write_dataset_csv(master_dataset, output_path, overwrite=overwrite)

    return MasterDatasetSummary(
        loaded_files=len(result_files),
        rows=len(master_dataset),
        unique_dilemmas=master_dataset["dilemma_id"].nunique(),
        unique_models=master_dataset["model"].nunique(),
        duplicate_dilemma_model_pairs=duplicate_pairs,
        missing_final_answers=missing_final_answers,
        output_path=saved_path,
        warnings=warnings,
    )


def format_master_dataset_summary(summary: MasterDatasetSummary) -> str:
    """Format the build summary for console output."""
    lines = [
        f"Loaded files: {summary.loaded_files}",
        f"Rows: {summary.rows}",
        f"Unique dilemmas: {summary.unique_dilemmas}",
        f"Models: {summary.unique_models}",
        f"Duplicate dilemma-model pairs: {summary.duplicate_dilemma_model_pairs}",
        f"Missing final answers: {summary.missing_final_answers}",
    ]

    if summary.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in summary.warnings)

    lines.extend(["", "Saved:", str(summary.output_path)])
    return "\n".join(lines)


def _validate_required_columns(dataset: pd.DataFrame) -> None:
    required_columns = {*KEY_COLUMNS, "final_answer", "justification", "raw_response"}
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Result dataset is missing required columns: " + ", ".join(missing_columns)
        )


def _sort_master_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def _missing_final_answer(dataset: pd.DataFrame) -> pd.Series:
    final_answers = dataset["final_answer"]
    return final_answers.isna() | final_answers.astype("string").str.strip().eq("")


def _collect_warnings(
    *,
    loaded_files: int,
    rows: int,
    unique_models: int,
    unique_dilemmas: int,
    duplicate_pairs: int,
    missing_final_answers: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if loaded_files != EXPECTED_RESULT_FILES:
        warnings.append(f"Expected {EXPECTED_RESULT_FILES} result files, found {loaded_files}.")
    if rows != EXPECTED_ROWS:
        warnings.append(f"Expected {EXPECTED_ROWS} rows, found {rows}.")
    if unique_models != EXPECTED_MODELS:
        warnings.append(f"Expected {EXPECTED_MODELS} models, found {unique_models}.")
    if unique_dilemmas != EXPECTED_DILEMMAS:
        warnings.append(f"Expected {EXPECTED_DILEMMAS} dilemmas, found {unique_dilemmas}.")
    if duplicate_pairs:
        warnings.append(f"Found {duplicate_pairs} duplicate dilemma-model pairs.")
    if missing_final_answers:
        warnings.append(f"Found {missing_final_answers} rows without a final answer.")
    return tuple(warnings)

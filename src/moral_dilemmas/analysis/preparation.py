import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_loader import read_dataset_csv, write_dataset_csv

EXPECTED_RESULT_FILES = 4
EXPECTED_ROWS = 2720
EXPECTED_MODELS = 4
EXPECTED_DILEMMAS = 680
GLMM_CONTRASTIVE_DILEMMA_THRESHOLD = 50
MASTER_KEY_COLUMNS = ("dilemma_id", "model")
VALUE_OUTCOME_COLUMNS = (
    "dilemma_id",
    "topic_group",
    "model",
    "value",
    "value_in_a",
    "value_in_b",
    "value_chosen",
    "is_contrastive",
)


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


@dataclass(frozen=True)
class ValueOutcomesSummary:
    """Short summary of the long-format value outcomes dataset."""

    input_observations: int
    long_format_observations: int
    unique_raw_values: int
    eligible_values: tuple[str, ...]
    top_contrastive_counts: tuple[tuple[str, int], ...]
    missing_value_chosen: int
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
    _validate_master_required_columns(master_dataset)

    master_dataset = _sort_master_dataset(master_dataset)
    duplicate_pairs = int(master_dataset.duplicated(subset=list(MASTER_KEY_COLUMNS)).sum())
    missing_final_answers = int(_missing_final_answer(master_dataset).sum())
    warnings = _collect_master_warnings(
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


def build_value_outcomes_long(
    *,
    master_dataset_path: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> ValueOutcomesSummary:
    """Build one long-format value-outcome row per dilemma, model, and relevant value."""
    master_dataset = read_dataset_csv(master_dataset_path)
    _validate_value_outcome_required_columns(master_dataset)

    value_outcomes = _create_value_outcome_dataframe(master_dataset)
    contrastive_counts = _contrastive_dilemma_counts(value_outcomes)
    eligible_values = tuple(
        contrastive_counts[contrastive_counts >= GLMM_CONTRASTIVE_DILEMMA_THRESHOLD].index.tolist()
    )
    top_contrastive_counts = tuple(
        (str(value), int(count)) for value, count in contrastive_counts.head(10).items()
    )
    missing_value_chosen = int(value_outcomes["value_chosen"].isna().sum())
    warnings = _collect_value_outcome_warnings(missing_value_chosen=missing_value_chosen)

    saved_path = write_dataset_csv(value_outcomes, output_path, overwrite=overwrite)

    return ValueOutcomesSummary(
        input_observations=len(master_dataset),
        long_format_observations=len(value_outcomes),
        unique_raw_values=value_outcomes["value"].nunique(),
        eligible_values=eligible_values,
        top_contrastive_counts=top_contrastive_counts,
        missing_value_chosen=missing_value_chosen,
        output_path=saved_path,
        warnings=warnings,
    )


def format_master_dataset_summary(summary: MasterDatasetSummary) -> str:
    """Format the master dataset build summary for console output."""
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


def format_value_outcomes_summary(summary: ValueOutcomesSummary) -> str:
    """Format the value outcomes build summary for console output."""
    lines = [
        f"Input observations: {summary.input_observations}",
        f"Long-format observations: {summary.long_format_observations}",
        "",
        f"Unique raw values: {summary.unique_raw_values}",
        "",
        (
            f"Values with >= {GLMM_CONTRASTIVE_DILEMMA_THRESHOLD} "
            f"contrastive dilemmas: {len(summary.eligible_values)}"
        ),
    ]

    if summary.top_contrastive_counts:
        lines.append("")
        lines.append("Top contrastive counts:")
        lines.extend(f"{value}: {count}" for value, count in summary.top_contrastive_counts)

    lines.append("")
    lines.append("Eligible values:")
    if summary.eligible_values:
        lines.extend(f"- {value}" for value in summary.eligible_values)
    else:
        lines.append("- none")

    if summary.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in summary.warnings)

    lines.extend(["", "Saved:", str(summary.output_path)])
    return "\n".join(lines)


def _create_value_outcome_dataframe(master_dataset: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for row in master_dataset.itertuples(index=False):
        row_data = row._asdict()
        values_a = parse_value_labels(row_data["action_a_values"])
        values_b = parse_value_labels(row_data["action_b_values"])

        for value in sorted(values_a | values_b):
            value_in_a = int(value in values_a)
            value_in_b = int(value in values_b)
            records.append(
                {
                    "dilemma_id": row_data["dilemma_id"],
                    "topic_group": row_data["topic_group"],
                    "model": row_data["model"],
                    "value": value,
                    "value_in_a": value_in_a,
                    "value_in_b": value_in_b,
                    "value_chosen": _value_chosen(
                        final_answer=row_data["final_answer"],
                        value_in_a=value_in_a,
                        value_in_b=value_in_b,
                    ),
                    "is_contrastive": int(value_in_a != value_in_b),
                }
            )

    return (
        pd.DataFrame.from_records(records, columns=VALUE_OUTCOME_COLUMNS)
        .sort_values(["dilemma_id", "model", "value"])
        .reset_index(drop=True)
    )


def parse_value_labels(raw_values: Any) -> set[str]:
    """Parse an option's raw value labels without harmonizing label names."""
    if isinstance(raw_values, list | tuple | set):
        return {str(value) for value in raw_values}

    if pd.isna(raw_values):
        return set()

    raw_text = str(raw_values).strip()
    if not raw_text:
        return set()

    parsed = ast.literal_eval(raw_text)
    if isinstance(parsed, str):
        return {parsed}
    if isinstance(parsed, list | tuple | set):
        return {str(value) for value in parsed}

    raise ValueError(f"Expected value labels to be a list-like object, got: {raw_text}")


def _value_chosen(
    *,
    final_answer: object,
    value_in_a: int,
    value_in_b: int,
) -> int | None:
    normalized_answer = str(final_answer).strip().upper()
    if normalized_answer == "A":
        return value_in_a
    if normalized_answer == "B":
        return value_in_b
    return None


def _validate_master_required_columns(dataset: pd.DataFrame) -> None:
    required_columns = {*MASTER_KEY_COLUMNS, "final_answer", "justification", "raw_response"}
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Result dataset is missing required columns: " + ", ".join(missing_columns)
        )


def _validate_value_outcome_required_columns(dataset: pd.DataFrame) -> None:
    required_columns = {
        "dilemma_id",
        "topic_group",
        "model",
        "final_answer",
        "action_a_values",
        "action_b_values",
    }
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Analysis master dataset is missing required columns: " + ", ".join(missing_columns)
        )


def _sort_master_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sort_values(list(MASTER_KEY_COLUMNS)).reset_index(drop=True)


def _missing_final_answer(dataset: pd.DataFrame) -> pd.Series:
    final_answers = dataset["final_answer"]
    return final_answers.isna() | final_answers.astype("string").str.strip().eq("")


def _contrastive_dilemma_counts(value_outcomes: pd.DataFrame) -> pd.Series:
    return (
        value_outcomes.loc[value_outcomes["is_contrastive"] == 1]
        .groupby("value")["dilemma_id"]
        .nunique()
        .sort_values(ascending=False)
    )


def _collect_master_warnings(
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


def _collect_value_outcome_warnings(*, missing_value_chosen: int) -> tuple[str, ...]:
    if not missing_value_chosen:
        return ()
    return (f"Found {missing_value_chosen} value rows without a selected final answer.",)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_loader import read_dataset_csv, write_dataset_csv

EXPECTED_MODELS = 4
GLMM_INPUT_COLUMNS = (
    "dilemma_id",
    "topic_group",
    "model",
    "value",
    "value_chosen",
)


@dataclass(frozen=True)
class GlmmInputSummary:
    """Short summary of the exported GLMM input dataset."""

    eligible_values: int
    rows_written: int
    unique_dilemmas: int
    models: int
    output_path: Path


def build_glmm_input_dataset(
    *,
    value_outcomes_path: str | Path,
    contrastive_value_path: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> GlmmInputSummary:
    """Build the long-format contrastive input dataset"""
    value_outcomes = read_dataset_csv(value_outcomes_path)
    contrastive_value = read_dataset_csv(contrastive_value_path)
    _validate_required_columns(value_outcomes, contrastive_value)

    eligible_values = _eligible_values(contrastive_value)
    glmm_input = _create_glmm_input(value_outcomes, eligible_values)
    _validate_glmm_input(glmm_input)
    saved_path = write_dataset_csv(glmm_input, output_path, overwrite=overwrite)

    return GlmmInputSummary(
        eligible_values=len(eligible_values),
        rows_written=len(glmm_input),
        unique_dilemmas=glmm_input["dilemma_id"].nunique(),
        models=glmm_input["model"].nunique(),
        output_path=saved_path,
    )


def format_glmm_input_summary(summary: GlmmInputSummary) -> str:
    """Format the GLMM input export summary for console output."""
    return "\n".join(
        [
            f"Eligible values: {summary.eligible_values}",
            f"Rows written: {summary.rows_written}",
            f"Unique dilemmas: {summary.unique_dilemmas}",
            f"Models: {summary.models}",
            "",
            "Saved:",
            str(summary.output_path),
        ]
    )


def _eligible_values(contrastive_value: pd.DataFrame) -> list[str]:
    eligible_mask = contrastive_value["eligible_for_glmm"].map(_parse_boolean)
    return contrastive_value.loc[eligible_mask, "value"].drop_duplicates().sort_values().tolist()


def _create_glmm_input(value_outcomes: pd.DataFrame, eligible_values: list[str]) -> pd.DataFrame:
    glmm_input = value_outcomes.loc[
        value_outcomes["value"].isin(eligible_values) & value_outcomes["is_contrastive"].eq(1),
        GLMM_INPUT_COLUMNS,
    ].copy()
    glmm_input["value_chosen"] = glmm_input["value_chosen"].astype(
        int
    )  # export the GLMM outcome as binary integers.

    return glmm_input.sort_values(["value", "dilemma_id", "model"]).reset_index(drop=True)


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true" or normalized == "True":
        return True
    if normalized == "false" or normalized == "False":
        return False
    raise ValueError(f"Expected boolean value, got: {value}")


def _validate_required_columns(
    value_outcomes: pd.DataFrame,
    contrastive_value: pd.DataFrame,
) -> None:
    _validate_columns(
        value_outcomes,
        {
            "dilemma_id",
            "topic_group",
            "model",
            "value",
            "value_chosen",
            "is_contrastive",
        },
        "Value outcomes dataset",
    )
    _validate_columns(
        contrastive_value,
        {
            "value",
            "eligible_for_glmm",
        },
        "Contrastive value summary",
    )


def _validate_columns(dataset: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing_columns))


def _validate_glmm_input(glmm_input: pd.DataFrame) -> None:
    if glmm_input["model"].nunique() != EXPECTED_MODELS:
        raise ValueError(f"Expected {EXPECTED_MODELS} models in GLMM input.")

    if not glmm_input["value_chosen"].isin([0, 1]).all():
        raise ValueError("GLMM input value_chosen must contain only 0 and 1.")

    duplicate_rows = glmm_input.duplicated(subset=["value", "dilemma_id", "model"])
    if duplicate_rows.any():
        raise ValueError("GLMM input contains duplicate value/dilemma_id/model rows.")

    rows_per_value_dilemma = glmm_input.groupby(["value", "dilemma_id"])["model"].nunique()
    incomplete_pairs = rows_per_value_dilemma[rows_per_value_dilemma != EXPECTED_MODELS]
    if not incomplete_pairs.empty:
        raise ValueError(
            "Expected four model observations for every value/dilemma_id pair in GLMM input."
        )

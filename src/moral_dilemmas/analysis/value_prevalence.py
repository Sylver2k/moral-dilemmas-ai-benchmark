from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..data_loader import read_dataset_csv, write_dataset_csv

EXPERIMENT_DILEMMA_COUNT = 680
GLMM_CONTRASTIVE_DILEMMA_THRESHOLD = 50
VALUE_PREVALENCE_COLUMNS = (
    "model",
    "value",
    "available_count",
    "selected_count",
    "prevalence",
    "rank",
)
CONTRASTIVE_VALUE_COLUMNS = (
    "model",
    "value",
    "contrastive_count",
    "selected_contrastive_count",
    "contrastive_selection_rate",
    "eligible_for_glmm",
    "contrastive_rank",
)


@dataclass(frozen=True)
class ValuePrevalenceSummary:
    """Short summary of descriptive value prevalence results."""

    models: int
    unique_raw_values: int
    rows_written: int
    top_values_by_model: tuple[tuple[str, tuple[tuple[int, str, float], ...]], ...]
    output_path: Path


@dataclass(frozen=True)
class ContrastiveValueSummary:
    """Short summary of contrastive value selection rates."""

    contrastive_values: int
    glmm_eligible_values: int
    rows_written: int
    top_values_by_model: tuple[tuple[str, tuple[tuple[int, str, float, int], ...]], ...]
    output_path: Path


def build_value_prevalence_summary(
    *,
    value_outcomes_path: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> ValuePrevalenceSummary:
    """Summarize how often each model selected options containing each value."""
    value_outcomes = read_dataset_csv(value_outcomes_path)
    _validate_required_columns(value_outcomes)

    prevalence_summary = _create_value_prevalence_summary(value_outcomes)
    saved_path = write_dataset_csv(prevalence_summary, output_path, overwrite=overwrite)

    return ValuePrevalenceSummary(
        models=prevalence_summary["model"].nunique(),
        unique_raw_values=prevalence_summary["value"].nunique(),
        rows_written=len(prevalence_summary),
        top_values_by_model=_top_values_by_model(prevalence_summary),
        output_path=saved_path,
    )


def build_contrastive_value_summary(
    *,
    value_outcomes_path: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> ContrastiveValueSummary:
    """Summarize value selection rates in contrastive dilemmas only."""
    value_outcomes = read_dataset_csv(value_outcomes_path)
    _validate_required_columns(value_outcomes, extra_columns={"is_contrastive"})

    contrastive_summary = _create_contrastive_value_summary(value_outcomes)
    saved_path = write_dataset_csv(contrastive_summary, output_path, overwrite=overwrite)

    return ContrastiveValueSummary(
        contrastive_values=contrastive_summary["value"].nunique(),
        glmm_eligible_values=contrastive_summary.loc[
            contrastive_summary["eligible_for_glmm"], "value"
        ].nunique(),
        rows_written=len(contrastive_summary),
        top_values_by_model=_top_contrastive_values_by_model(contrastive_summary),
        output_path=saved_path,
    )


def format_value_prevalence_summary(summary: ValuePrevalenceSummary) -> str:
    """Format the value prevalence summary for console output."""
    lines = [
        f"Models: {summary.models}",
        f"Unique raw values: {summary.unique_raw_values}",
        f"Rows written: {summary.rows_written}",
    ]

    for model, top_values in summary.top_values_by_model:
        lines.extend(["", f"Model: {model}"])
        lines.extend(f"{rank}. {value} {prevalence:.4f}" for rank, value, prevalence in top_values)

    lines.extend(["", "Saved:", str(summary.output_path)])
    return "\n".join(lines)


def format_contrastive_value_summary(summary: ContrastiveValueSummary) -> str:
    """Format the contrastive value summary for console output."""
    lines = [
        f"Values with at least 1 contrastive dilemma: {summary.contrastive_values}",
        (
            f"Values with >= {GLMM_CONTRASTIVE_DILEMMA_THRESHOLD} "
            f"contrastive dilemmas: {summary.glmm_eligible_values}"
        ),
        f"Rows written: {summary.rows_written}",
    ]

    for model, top_values in summary.top_values_by_model:
        lines.extend(["", f"Model: {model}"])
        if top_values:
            lines.extend(
                f"{rank}. {value} rate={rate:.4f} n={contrastive_count}"
                for rank, value, rate, contrastive_count in top_values
            )
        else:
            lines.append("No eligible values")

    lines.extend(["", "Saved:", str(summary.output_path)])
    return "\n".join(lines)


def _create_value_prevalence_summary(value_outcomes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        value_outcomes.groupby(["model", "value"], as_index=False)
        .agg(
            available_count=("dilemma_id", "nunique"),
            selected_count=("value_chosen", "sum"),
        )
        .astype({"selected_count": int})
    )
    summary["prevalence"] = summary["selected_count"] / EXPERIMENT_DILEMMA_COUNT
    summary = summary.sort_values(
        ["model", "selected_count", "value"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    summary["rank"] = summary.groupby("model").cumcount() + 1

    return summary.loc[:, VALUE_PREVALENCE_COLUMNS]


def _create_contrastive_value_summary(value_outcomes: pd.DataFrame) -> pd.DataFrame:
    contrastive_outcomes = value_outcomes.loc[value_outcomes["is_contrastive"] == 1].copy()
    summary = (
        contrastive_outcomes.groupby(["model", "value"], as_index=False)
        .agg(
            contrastive_count=("dilemma_id", "nunique"),
            selected_contrastive_count=("value_chosen", "sum"),
        )
        .astype({"selected_contrastive_count": int})
    )
    summary["contrastive_selection_rate"] = (
        summary["selected_contrastive_count"] / summary["contrastive_count"]
    )
    summary["eligible_for_glmm"] = (
        summary["contrastive_count"] >= GLMM_CONTRASTIVE_DILEMMA_THRESHOLD
    )
    summary = summary.sort_values(
        ["model", "contrastive_selection_rate", "contrastive_count", "value"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    summary["contrastive_rank"] = summary.groupby("model").cumcount() + 1

    return summary.loc[:, CONTRASTIVE_VALUE_COLUMNS]


def _top_values_by_model(
    prevalence_summary: pd.DataFrame,
) -> tuple[tuple[str, tuple[tuple[int, str, float], ...]], ...]:
    top_values: list[tuple[str, tuple[tuple[int, str, float], ...]]] = []

    for model, model_summary in prevalence_summary.groupby("model", sort=True):
        top_rows = model_summary.head(3)
        top_values.append(
            (
                str(model),
                tuple(
                    (int(row.rank), str(row.value), float(row.prevalence))
                    for row in top_rows.itertuples(index=False)
                ),
            )
        )

    return tuple(top_values)


def _top_contrastive_values_by_model(
    contrastive_summary: pd.DataFrame,
) -> tuple[tuple[str, tuple[tuple[int, str, float, int], ...]], ...]:
    top_values: list[tuple[str, tuple[tuple[int, str, float, int], ...]]] = []

    for model, model_summary in contrastive_summary.groupby("model", sort=True):
        top_rows = model_summary.loc[model_summary["eligible_for_glmm"]].head(3)
        top_values.append(
            (
                str(model),
                tuple(
                    (
                        rank,
                        str(row.value),
                        float(row.contrastive_selection_rate),
                        int(row.contrastive_count),
                    )
                    for rank, row in enumerate(top_rows.itertuples(index=False), start=1)
                ),
            )
        )

    return tuple(top_values)


def _validate_required_columns(
    value_outcomes: pd.DataFrame,
    *,
    extra_columns: set[str] | None = None,
) -> None:
    required_columns = {
        "dilemma_id",
        "model",
        "value",
        "value_chosen",
    }
    if extra_columns:
        required_columns.update(extra_columns)

    missing_columns = sorted(required_columns.difference(value_outcomes.columns))
    if missing_columns:
        raise ValueError(
            "Value outcomes dataset is missing required columns: " + ", ".join(missing_columns)
        )

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_loader import read_dataset_csv, write_dataset_csv

N_BOOTSTRAP = 10_000
BOOTSTRAP_RANDOM_SEED = 2187
CONFIDENCE_LEVEL = 0.95
CI_LOWER_QUANTILE = (1 - CONFIDENCE_LEVEL) / 2
CI_UPPER_QUANTILE = 1 - CI_LOWER_QUANTILE
EXPECTED_DILEMMAS = 680
BOOTSTRAP_BATCH_SIZE = 250

PREVALENCE_CI_COLUMNS = ("model", "value", "prevalence", "ci_lower", "ci_upper")
CONTRASTIVE_CI_COLUMNS = (
    "model",
    "value",
    "contrastive_selection_rate",
    "ci_lower",
    "ci_upper",
)
MODEL_DIFFERENCE_COLUMNS = ("model_1", "model_2", "value", "difference", "ci_lower", "ci_upper")


@dataclass(frozen=True)
class BootstrapRunSummary:
    """Short summary of the bootstrap artifacts written to disk."""

    n_bootstrap: int
    bootstrap_unit: str
    dilemmas_per_replication: int
    confidence_level: float
    random_seed: int
    prevalence_output_path: Path
    contrastive_output_path: Path
    model_differences_output_path: Path


def build_bootstrap_datasets(
    *,
    value_outcomes_path: str | Path,
    value_prevalence_path: str | Path,
    contrastive_value_path: str | Path,
    prevalence_output_path: str | Path,
    contrastive_output_path: str | Path,
    model_differences_output_path: str | Path,
    overwrite: bool = True,
) -> BootstrapRunSummary:
    """Build percentile bootstrap intervals using shared dilemma-level resamples."""
    value_outcomes = read_dataset_csv(value_outcomes_path)
    value_prevalence = read_dataset_csv(value_prevalence_path)
    contrastive_value = read_dataset_csv(contrastive_value_path)
    _validate_required_columns(value_outcomes, value_prevalence, contrastive_value)

    dilemma_ids = np.array(sorted(value_outcomes["dilemma_id"].unique()))
    _validate_dilemma_ids(dilemma_ids)
    bootstrap_dilemma_positions = _create_bootstrap_dilemma_positions(len(dilemma_ids))

    prevalence_ci = _create_prevalence_ci(
        value_outcomes=value_outcomes,
        value_prevalence=value_prevalence,
        dilemma_ids=dilemma_ids,
        bootstrap_dilemma_positions=bootstrap_dilemma_positions,
    )
    contrastive_ci, contrastive_rates = _create_contrastive_ci(
        value_outcomes=value_outcomes,
        contrastive_value=contrastive_value,
        dilemma_ids=dilemma_ids,
        bootstrap_dilemma_positions=bootstrap_dilemma_positions,
    )
    model_differences = _create_model_difference_ci(
        contrastive_value=contrastive_value,
        contrastive_rates=contrastive_rates,
    )

    prevalence_path = write_dataset_csv(prevalence_ci, prevalence_output_path, overwrite=overwrite)
    contrastive_path = write_dataset_csv(
        contrastive_ci,
        contrastive_output_path,
        overwrite=overwrite,
    )
    model_differences_path = write_dataset_csv(
        model_differences,
        model_differences_output_path,
        overwrite=overwrite,
    )

    return BootstrapRunSummary(
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_unit="dilemma",
        dilemmas_per_replication=len(dilemma_ids),
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=BOOTSTRAP_RANDOM_SEED,
        prevalence_output_path=prevalence_path,
        contrastive_output_path=contrastive_path,
        model_differences_output_path=model_differences_path,
    )


def format_bootstrap_run_summary(summary: BootstrapRunSummary) -> str:
    """Format the bootstrap run summary for console output."""
    return "\n".join(
        [
            f"Bootstrap replications: {summary.n_bootstrap}",
            f"Bootstrap unit: {summary.bootstrap_unit}",
            f"Dilemmas per replication: {summary.dilemmas_per_replication}",
            f"Confidence interval: {summary.confidence_level:.0%} percentile",
            f"Seed: {summary.random_seed}",
            "",
            f"{summary.prevalence_output_path.name} written",
            f"{summary.contrastive_output_path.name} written",
            f"{summary.model_differences_output_path.name} written",
        ]
    )


def _create_bootstrap_dilemma_positions(n_dilemmas: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    return rng.integers(
        0,
        n_dilemmas,
        size=(N_BOOTSTRAP, n_dilemmas),
        dtype=np.int16,
    )


def _create_prevalence_ci(
    *,
    value_outcomes: pd.DataFrame,
    value_prevalence: pd.DataFrame,
    dilemma_ids: np.ndarray,
    bootstrap_dilemma_positions: np.ndarray,
) -> pd.DataFrame:
    models = sorted(value_prevalence["model"].unique())
    values = sorted(value_prevalence["value"].unique())
    selected = _create_selected_value_cube(
        value_outcomes=value_outcomes,
        dilemma_ids=dilemma_ids,
        models=models,
        values=values,
    )
    bootstrap_prevalence = np.empty(
        (N_BOOTSTRAP, len(models), len(values)),
        dtype=np.float32,
    )

    for batch_start in range(0, N_BOOTSTRAP, BOOTSTRAP_BATCH_SIZE):
        batch_end = min(batch_start + BOOTSTRAP_BATCH_SIZE, N_BOOTSTRAP)
        batch_positions = bootstrap_dilemma_positions[batch_start:batch_end]
        bootstrap_prevalence[batch_start:batch_end] = selected[batch_positions].sum(axis=1) / len(
            dilemma_ids
        )

    lower = np.quantile(bootstrap_prevalence, CI_LOWER_QUANTILE, axis=0)
    upper = np.quantile(bootstrap_prevalence, CI_UPPER_QUANTILE, axis=0)
    interval_rows = _interval_rows(models=models, values=values, lower=lower, upper=upper)
    result = value_prevalence.merge(interval_rows, on=["model", "value"], how="left")

    return (
        result.loc[:, PREVALENCE_CI_COLUMNS]
        .sort_values(["model", "prevalence"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _create_contrastive_ci(
    *,
    value_outcomes: pd.DataFrame,
    contrastive_value: pd.DataFrame,
    dilemma_ids: np.ndarray,
    bootstrap_dilemma_positions: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    eligible_summary = contrastive_value.loc[contrastive_value["eligible_for_glmm"]].copy()
    models = sorted(eligible_summary["model"].unique())
    values = sorted(eligible_summary["value"].unique())
    selected_contrastive = _create_selected_value_cube(
        value_outcomes=value_outcomes.loc[value_outcomes["is_contrastive"] == 1],
        dilemma_ids=dilemma_ids,
        models=models,
        values=values,
    )
    contrastive = _create_contrastive_value_matrix(
        value_outcomes=value_outcomes,
        dilemma_ids=dilemma_ids,
        values=values,
    )
    bootstrap_rates = np.empty(
        (N_BOOTSTRAP, len(models), len(values)),
        dtype=np.float32,
    )

    for batch_start in range(0, N_BOOTSTRAP, BOOTSTRAP_BATCH_SIZE):
        batch_end = min(batch_start + BOOTSTRAP_BATCH_SIZE, N_BOOTSTRAP)
        batch_positions = bootstrap_dilemma_positions[batch_start:batch_end]
        selected_sums = selected_contrastive[batch_positions].sum(axis=1)
        contrastive_sums = contrastive[batch_positions].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bootstrap_rates[batch_start:batch_end] = selected_sums / contrastive_sums[:, None, :]

    lower = np.nanquantile(bootstrap_rates, CI_LOWER_QUANTILE, axis=0)
    upper = np.nanquantile(bootstrap_rates, CI_UPPER_QUANTILE, axis=0)
    interval_rows = _interval_rows(models=models, values=values, lower=lower, upper=upper)
    result = eligible_summary.merge(interval_rows, on=["model", "value"], how="left")
    
    return (
        result.loc[:, CONTRASTIVE_CI_COLUMNS]
        .sort_values(["model", "contrastive_selection_rate"], ascending=[True, False])
        .reset_index(drop=True),
        bootstrap_rates,
    )


def _create_model_difference_ci(
    *,
    contrastive_value: pd.DataFrame,
    contrastive_rates: np.ndarray,
) -> pd.DataFrame:
    eligible_summary = contrastive_value.loc[contrastive_value["eligible_for_glmm"]].copy()
    models = sorted(eligible_summary["model"].unique())
    values = sorted(eligible_summary["value"].unique())
    observed_rates = eligible_summary.pivot(
        index="value",
        columns="model",
        values="contrastive_selection_rate",
    )
    records: list[dict[str, object]] = []

    for model_1, model_2 in combinations(models, 2):
        model_1_index = models.index(model_1)
        model_2_index = models.index(model_2)
        differences = (
            contrastive_rates[:, model_1_index, :] - contrastive_rates[:, model_2_index, :]
        )
        lower = np.nanquantile(differences, CI_LOWER_QUANTILE, axis=0)
        upper = np.nanquantile(differences, CI_UPPER_QUANTILE, axis=0)

        for value_index, value in enumerate(values):
            records.append(
                {
                    "model_1": model_1,
                    "model_2": model_2,
                    "value": value,
                    "difference": (
                        observed_rates.at[value, model_1] - observed_rates.at[value, model_2]
                    ),
                    "ci_lower": lower[value_index],
                    "ci_upper": upper[value_index],
                }
            )

    return (
        pd.DataFrame.from_records(records, columns=MODEL_DIFFERENCE_COLUMNS)
        .sort_values(["model_1", "model_2", "value"])
        .reset_index(drop=True)
    )


def _create_selected_value_cube(
    *,
    value_outcomes: pd.DataFrame,
    dilemma_ids: np.ndarray,
    models: list[str],
    values: list[str],
) -> np.ndarray:
    dilemma_index = {dilemma_id: index for index, dilemma_id in enumerate(dilemma_ids)}
    model_index = {model: index for index, model in enumerate(models)}
    value_index = {value: index for index, value in enumerate(values)}
    selected = np.zeros((len(dilemma_ids), len(models), len(values)), dtype=np.uint8)

    relevant_rows = value_outcomes.loc[
        value_outcomes["model"].isin(models)
        & value_outcomes["value"].isin(values)
        & value_outcomes["value_chosen"].eq(1)
    ]
    for row in relevant_rows.itertuples(index=False):
        selected[
            dilemma_index[row.dilemma_id],
            model_index[row.model],
            value_index[row.value],
        ] = 1

    return selected


def _create_contrastive_value_matrix(
    *,
    value_outcomes: pd.DataFrame,
    dilemma_ids: np.ndarray,
    values: list[str],
) -> np.ndarray:
    dilemma_index = {dilemma_id: index for index, dilemma_id in enumerate(dilemma_ids)}
    value_index = {value: index for index, value in enumerate(values)}
    contrastive = np.zeros((len(dilemma_ids), len(values)), dtype=np.uint8)
    contrastive_rows = value_outcomes.loc[
        value_outcomes["value"].isin(values) & value_outcomes["is_contrastive"].eq(1)
    ]

    for row in contrastive_rows.drop_duplicates(["dilemma_id", "value"]).itertuples(index=False):
        contrastive[dilemma_index[row.dilemma_id], value_index[row.value]] = 1

    return contrastive


def _interval_rows(
    *,
    models: list[str],
    values: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "model": model,
                "value": value,
                "ci_lower": lower[model_index, value_index],
                "ci_upper": upper[model_index, value_index],
            }
            for model_index, model in enumerate(models)
            for value_index, value in enumerate(values)
        ]
    )


def _validate_required_columns(
    value_outcomes: pd.DataFrame,
    value_prevalence: pd.DataFrame,
    contrastive_value: pd.DataFrame,
) -> None:
    _validate_columns(
        value_outcomes,
        {
            "dilemma_id",
            "model",
            "value",
            "value_chosen",
            "is_contrastive",
        },
        "Value outcomes dataset",
    )
    _validate_columns(
        value_prevalence,
        {
            "model",
            "value",
            "prevalence",
        },
        "Value prevalence summary",
    )
    _validate_columns(
        contrastive_value,
        {
            "model",
            "value",
            "contrastive_selection_rate",
            "eligible_for_glmm",
        },
        "Contrastive value summary",
    )


def _validate_columns(dataset: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing_columns))


def _validate_dilemma_ids(dilemma_ids: np.ndarray) -> None:
    if len(dilemma_ids) != EXPECTED_DILEMMAS:
        raise ValueError(f"Expected {EXPECTED_DILEMMAS} dilemmas, found {len(dilemma_ids)}.")

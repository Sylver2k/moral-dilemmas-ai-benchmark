from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from ..data_loader import read_dataset_csv

SHEET_NAMES = (
    "value_model_summary",
    "global_glmm_results",
    "pairwise_results",
    "bootstrap_model_differences",
)


@dataclass(frozen=True)
class FinalWorkbookSummary:
    """Short summary of the consolidated final analysis workbook."""

    output_path: Path
    row_counts: dict[str, int]


def build_final_analysis_workbook(
    *,
    value_prevalence_path: str | Path,
    bootstrap_prevalence_path: str | Path,
    contrastive_value_path: str | Path,
    bootstrap_contrastive_path: str | Path,
    global_glmm_path: str | Path,
    pairwise_path: str | Path,
    bootstrap_model_differences_path: str | Path,
    output_path: str | Path,
) -> FinalWorkbookSummary:
    """Consolidate final quantitative results into one formatted Excel workbook."""
    # Load existing result artifacts
    value_prevalence = read_dataset_csv(value_prevalence_path)
    bootstrap_prevalence = read_dataset_csv(bootstrap_prevalence_path)
    contrastive_value = read_dataset_csv(contrastive_value_path)
    bootstrap_contrastive = read_dataset_csv(bootstrap_contrastive_path)
    global_glmm = read_dataset_csv(global_glmm_path)
    pairwise = read_dataset_csv(pairwise_path)
    bootstrap_model_differences = read_dataset_csv(bootstrap_model_differences_path)

    value_model_summary = _create_value_model_summary(
        value_prevalence=value_prevalence,
        bootstrap_prevalence=bootstrap_prevalence,
        contrastive_value=contrastive_value,
        bootstrap_contrastive=bootstrap_contrastive,
    )
    global_glmm_results = _create_global_glmm_results(global_glmm)
    pairwise_results = _create_pairwise_results(pairwise, global_glmm_results)
    bootstrap_model_differences_sheet = _create_bootstrap_model_differences(
        bootstrap_model_differences
    )

    sheets = {
        "value_model_summary": value_model_summary,
        "global_glmm_results": global_glmm_results,
        "pairwise_results": pairwise_results,
        "bootstrap_model_differences": bootstrap_model_differences_sheet,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_workbook(destination, sheets)

    return FinalWorkbookSummary(
        output_path=destination,
        row_counts={sheet_name: len(sheet_data) for sheet_name, sheet_data in sheets.items()},
    )


def format_final_workbook_summary(summary: FinalWorkbookSummary) -> str:
    """Format the final workbook build summary for console output."""
    lines = ["Creating final analysis workbook..."]
    lines.extend(
        f"{sheet_name}: {summary.row_counts[sheet_name]} rows" for sheet_name in SHEET_NAMES
    )
    lines.extend(["", f"Written: {summary.output_path}"])
    return "\n".join(lines)


def _create_value_model_summary(
    *,
    value_prevalence: pd.DataFrame,
    bootstrap_prevalence: pd.DataFrame,
    contrastive_value: pd.DataFrame,
    bootstrap_contrastive: pd.DataFrame,
) -> pd.DataFrame:
    _validate_unique_keys(value_prevalence, ["model", "value"], "value_prevalence_summary")
    _validate_unique_keys(bootstrap_prevalence, ["model", "value"], "bootstrap_prevalence_ci")
    _validate_unique_keys(contrastive_value, ["model", "value"], "contrastive_value_summary")
    _validate_unique_keys(bootstrap_contrastive, ["model", "value"], "bootstrap_contrastive_ci")

    # Merge prevalence estimates with bootstrap confidence intervals
    summary = value_prevalence.rename(columns={"rank": "prevalence_rank"}).merge(
        bootstrap_prevalence.loc[:, ["model", "value", "ci_lower", "ci_upper"]].rename(
            columns={
                "ci_lower": "prevalence_ci_lower",
                "ci_upper": "prevalence_ci_upper",
            }
        ),
        on=["model", "value"],
        how="left",
        validate="one_to_one",
    )

    # Add contrastive estimates and eligible-value intervals
    summary = summary.merge(
        contrastive_value,
        on=["model", "value"],
        how="left",
        validate="one_to_one",
    ).merge(
        bootstrap_contrastive.loc[:, ["model", "value", "ci_lower", "ci_upper"]].rename(
            columns={
                "ci_lower": "contrastive_ci_lower",
                "ci_upper": "contrastive_ci_upper",
            }
        ),
        on=["model", "value"],
        how="left",
        validate="one_to_one",
    )

    columns = [
        "model",
        "value",
        "available_count",
        "selected_count",
        "prevalence",
        "prevalence_rank",
        "prevalence_ci_lower",
        "prevalence_ci_upper",
        "contrastive_count",
        "selected_contrastive_count",
        "contrastive_selection_rate",
        "contrastive_rank",
        "contrastive_ci_lower",
        "contrastive_ci_upper",
        "eligible_for_glmm",
    ]
    return summary.loc[:, columns].sort_values(["model", "prevalence_rank"]).reset_index(drop=True)


def _create_global_glmm_results(global_glmm: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "value",
        "n_dilemmas",
        "n_observations",
        "lr_statistic",
        "df",
        "p_value",
        "p_adjusted",
        "significant",
        "convergence_ok",
        "singular_fit",
    ]
    _validate_columns(global_glmm, columns, "glmm_global_results_adjusted")
    if global_glmm["value"].duplicated().any():
        raise ValueError("global_glmm_results contains duplicate values.")

    return (
        global_glmm.loc[:, columns]
        .sort_values(["p_adjusted", "p_value", "value"], na_position="last")
        .reset_index(drop=True)
    )


def _create_pairwise_results(
    pairwise: pd.DataFrame,
    global_glmm_results: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["value", "model_1", "model_2", "p_value", "p_adjusted", "significant"]
    _validate_columns(pairwise, columns, "glmm_pairwise_results")

    significant_values = set(global_glmm_results.loc[global_glmm_results["significant"], "value"])
    pairwise_values = set(pairwise["value"])
    unexpected_values = sorted(pairwise_values.difference(significant_values))
    if unexpected_values:
        raise ValueError(
            "Pairwise results contain values without significant global results: "
            + ", ".join(unexpected_values)
        )

    return (
        pairwise.loc[:, columns]
        .sort_values(["value", "p_adjusted", "model_1", "model_2"], na_position="last")
        .reset_index(drop=True)
    )


def _create_bootstrap_model_differences(
    bootstrap_model_differences: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["model_1", "model_2", "value", "difference", "ci_lower", "ci_upper"]
    _validate_columns(
        bootstrap_model_differences,
        columns,
        "bootstrap_model_differences",
    )
    return (
        bootstrap_model_differences.loc[:, columns]
        .sort_values(["value", "model_1", "model_2"])
        .reset_index(drop=True)
    )


def _write_workbook(destination: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        for sheet_name, sheet_data in sheets.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_sheet(writer.book[sheet_name])


def _format_sheet(worksheet) -> None:
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.font = Font(bold=True)

    for column_cells in worksheet.columns:
        header = column_cells[0].value
        column_letter = get_column_letter(column_cells[0].column)
        max_length = max(
            len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells
        )
        worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 36)

        number_format = _number_format_for_column(str(header))
        if number_format:
            for cell in column_cells[1:]:
                cell.number_format = number_format


def _number_format_for_column(column_name: str) -> str | None:
    if column_name in {"p_value", "p_adjusted"}:
        return "0.000000"
    if column_name in {
        "prevalence",
        "prevalence_ci_lower",
        "prevalence_ci_upper",
        "contrastive_selection_rate",
        "contrastive_ci_lower",
        "contrastive_ci_upper",
        "difference",
        "ci_lower",
        "ci_upper",
        "lr_statistic",
    }:
        return "0.0000"
    return None


def _validate_unique_keys(dataset: pd.DataFrame, keys: list[str], label: str) -> None:
    _validate_columns(dataset, keys, label)
    if dataset.duplicated(subset=keys).any():
        raise ValueError(f"{label} contains duplicate keys: {', '.join(keys)}.")


def _validate_columns(dataset: pd.DataFrame, columns: list[str], label: str) -> None:
    missing_columns = sorted(set(columns).difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"{label} is missing columns: " + ", ".join(missing_columns))

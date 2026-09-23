from pathlib import Path

import pandas as pd
import pytest

from moral_dilemmas.analysis import (
    build_contrastive_value_summary,
    build_value_prevalence_summary,
)
from moral_dilemmas.data_loader import read_dataset_csv


def value_outcome_row(
    *,
    dilemma_id: int,
    model: str,
    value: str,
    value_chosen: int,
    is_contrastive: int = 1,
) -> dict[str, object]:
    return {
        "dilemma_id": dilemma_id,
        "topic_group": "Group",
        "model": model,
        "value": value,
        "value_in_a": 1,
        "value_in_b": 0,
        "value_chosen": value_chosen,
        "is_contrastive": is_contrastive,
    }


def test_build_value_prevalence_summary_uses_fixed_experiment_denominator(
    tmp_path: Path,
) -> None:
    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    output_path = tmp_path / "analysis" / "value_prevalence_summary.csv"
    pd.DataFrame(
        [
            value_outcome_row(dilemma_id=1, model="model-a", value="care", value_chosen=1),
            value_outcome_row(dilemma_id=2, model="model-a", value="care", value_chosen=0),
            value_outcome_row(dilemma_id=1, model="model-a", value="honesty", value_chosen=1),
            value_outcome_row(dilemma_id=1, model="model-b", value="trust", value_chosen=1),
        ]
    ).to_csv(value_outcomes_path, index=False)

    summary = build_value_prevalence_summary(
        value_outcomes_path=value_outcomes_path,
        output_path=output_path,
    )

    prevalence_summary = read_dataset_csv(output_path)
    assert summary.models == 2
    assert summary.unique_raw_values == 3
    assert summary.rows_written == 3
    assert prevalence_summary.columns.tolist() == [
        "model",
        "value",
        "available_count",
        "selected_count",
        "prevalence",
        "rank",
    ]

    model_a = prevalence_summary.loc[prevalence_summary["model"] == "model-a"]
    assert model_a[["value", "rank"]].to_records(index=False).tolist() == [
        ("care", 1),
        ("honesty", 2),
    ]
    care = model_a.loc[model_a["value"] == "care"].iloc[0]
    assert care["available_count"] == 2
    assert care["selected_count"] == 1
    assert care["prevalence"] == pytest.approx(1 / 680)


def test_build_value_prevalence_summary_rejects_missing_required_columns(
    tmp_path: Path,
) -> None:
    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    pd.DataFrame([{"dilemma_id": 1}]).to_csv(value_outcomes_path, index=False)

    try:
        build_value_prevalence_summary(
            value_outcomes_path=value_outcomes_path,
            output_path=tmp_path / "value_prevalence_summary.csv",
        )
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("Expected missing required columns to raise ValueError.")


def test_build_contrastive_value_summary_uses_only_contrastive_rows(
    tmp_path: Path,
) -> None:
    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    output_path = tmp_path / "analysis" / "contrastive_value_summary.csv"
    pd.DataFrame(
        [
            value_outcome_row(dilemma_id=1, model="model-a", value="care", value_chosen=1),
            value_outcome_row(dilemma_id=2, model="model-a", value="care", value_chosen=0),
            value_outcome_row(dilemma_id=3, model="model-a", value="care", value_chosen=1),
            value_outcome_row(
                dilemma_id=4,
                model="model-a",
                value="care",
                value_chosen=1,
                is_contrastive=0,
            ),
            value_outcome_row(dilemma_id=1, model="model-a", value="honesty", value_chosen=1),
            value_outcome_row(dilemma_id=2, model="model-a", value="honesty", value_chosen=1),
            value_outcome_row(dilemma_id=1, model="model-b", value="care", value_chosen=0),
        ]
    ).to_csv(value_outcomes_path, index=False)

    summary = build_contrastive_value_summary(
        value_outcomes_path=value_outcomes_path,
        output_path=output_path,
    )

    contrastive_summary = read_dataset_csv(output_path)
    assert summary.contrastive_values == 2
    assert summary.glmm_eligible_values == 0
    assert summary.rows_written == 3
    assert contrastive_summary.columns.tolist() == [
        "model",
        "value",
        "contrastive_count",
        "selected_contrastive_count",
        "contrastive_selection_rate",
        "eligible_for_glmm",
        "contrastive_rank",
    ]

    model_a = contrastive_summary.loc[contrastive_summary["model"] == "model-a"]
    assert model_a[["value", "contrastive_rank"]].to_records(index=False).tolist() == [
        ("honesty", 1),
        ("care", 2),
    ]
    care = model_a.loc[model_a["value"] == "care"].iloc[0]
    assert care["contrastive_count"] == 3
    assert care["selected_contrastive_count"] == 2
    assert care["contrastive_selection_rate"] == pytest.approx(2 / 3)
    assert not care["eligible_for_glmm"]


def test_build_contrastive_value_summary_marks_glmm_eligible_values(
    tmp_path: Path,
) -> None:
    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    output_path = tmp_path / "analysis" / "contrastive_value_summary.csv"
    rows = [
        value_outcome_row(
            dilemma_id=dilemma_id,
            model="model-a",
            value="care",
            value_chosen=int(dilemma_id <= 30),
        )
        for dilemma_id in range(1, 51)
    ]
    pd.DataFrame(rows).to_csv(value_outcomes_path, index=False)

    summary = build_contrastive_value_summary(
        value_outcomes_path=value_outcomes_path,
        output_path=output_path,
    )

    contrastive_summary = read_dataset_csv(output_path)
    care = contrastive_summary.iloc[0]
    assert summary.glmm_eligible_values == 1
    assert care["contrastive_count"] == 50
    assert care["selected_contrastive_count"] == 30
    assert care["eligible_for_glmm"]

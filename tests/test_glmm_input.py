from pathlib import Path

import pandas as pd
import pytest

from moral_dilemmas.analysis import build_glmm_input_dataset
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


def test_build_glmm_input_dataset_filters_to_eligible_contrastive_rows(
    tmp_path: Path,
) -> None:
    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    contrastive_value_path = tmp_path / "contrastive_value_summary.csv"
    output_path = tmp_path / "analysis" / "glmm_input.csv"
    models = ["model-a", "model-b", "model-c", "model-d"]
    rows = [
        value_outcome_row(
            dilemma_id=dilemma_id,
            model=model,
            value="care",
            value_chosen=int(model in {"model-a", "model-c"}),
        )
        for dilemma_id in [1, 2]
        for model in models
    ]
    rows.extend(
        value_outcome_row(
            dilemma_id=1,
            model=model,
            value="trust",
            value_chosen=1,
        )
        for model in models
    )
    rows.extend(
        value_outcome_row(
            dilemma_id=3,
            model=model,
            value="care",
            value_chosen=1,
            is_contrastive=0,
        )
        for model in models
    )
    pd.DataFrame(rows).to_csv(value_outcomes_path, index=False)
    pd.DataFrame(
        [{"model": model, "value": "care", "eligible_for_glmm": "True"} for model in models]
        + [{"model": model, "value": "trust", "eligible_for_glmm": "False"} for model in models]
    ).to_csv(contrastive_value_path, index=False)

    summary = build_glmm_input_dataset(
        value_outcomes_path=value_outcomes_path,
        contrastive_value_path=contrastive_value_path,
        output_path=output_path,
    )

    glmm_input = read_dataset_csv(output_path)
    assert summary.eligible_values == 1
    assert summary.rows_written == 8
    assert summary.unique_dilemmas == 2
    assert summary.models == 4
    assert glmm_input.columns.tolist() == [
        "dilemma_id",
        "topic_group",
        "model",
        "value",
        "value_chosen",
    ]
    assert glmm_input["value"].unique().tolist() == ["care"]
    assert glmm_input[["value", "dilemma_id", "model"]].to_records(index=False).tolist()[0] == (
        "care",
        1,
        "model-a",
    )


def test_build_glmm_input_dataset_rejects_incomplete_model_sets(
    tmp_path: Path,
) -> None:
    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    contrastive_value_path = tmp_path / "contrastive_value_summary.csv"
    pd.DataFrame(
        [
            value_outcome_row(dilemma_id=1, model="model-a", value="care", value_chosen=1),
            value_outcome_row(dilemma_id=1, model="model-b", value="care", value_chosen=0),
            value_outcome_row(dilemma_id=1, model="model-c", value="care", value_chosen=1),
        ]
    ).to_csv(value_outcomes_path, index=False)
    pd.DataFrame([{"value": "care", "eligible_for_glmm": True}]).to_csv(
        contrastive_value_path,
        index=False,
    )

    with pytest.raises(ValueError, match="Expected 4 models"):
        build_glmm_input_dataset(
            value_outcomes_path=value_outcomes_path,
            contrastive_value_path=contrastive_value_path,
            output_path=tmp_path / "glmm_input.csv",
        )

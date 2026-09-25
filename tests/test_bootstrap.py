from pathlib import Path

import pandas as pd
import pytest

import moral_dilemmas.analysis.bootstrap as bootstrap
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


def test_build_bootstrap_datasets_writes_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap, "N_BOOTSTRAP", 200)
    monkeypatch.setattr(bootstrap, "EXPECTED_DILEMMAS", 3)
    monkeypatch.setattr(bootstrap, "BOOTSTRAP_BATCH_SIZE", 50)

    value_outcomes_path = tmp_path / "value_outcomes_long.csv"
    value_prevalence_path = tmp_path / "value_prevalence_summary.csv"
    contrastive_value_path = tmp_path / "contrastive_value_summary.csv"
    prevalence_output_path = tmp_path / "bootstrap_prevalence_ci.csv"
    contrastive_output_path = tmp_path / "bootstrap_contrastive_ci.csv"
    differences_output_path = tmp_path / "bootstrap_model_differences.csv"

    pd.DataFrame(
        [
            value_outcome_row(dilemma_id=1, model="model-a", value="care", value_chosen=1),
            value_outcome_row(dilemma_id=2, model="model-a", value="care", value_chosen=1),
            value_outcome_row(dilemma_id=3, model="model-a", value="trust", value_chosen=1),
            value_outcome_row(dilemma_id=1, model="model-b", value="care", value_chosen=1),
            value_outcome_row(dilemma_id=2, model="model-b", value="trust", value_chosen=1),
            value_outcome_row(dilemma_id=3, model="model-b", value="trust", value_chosen=1),
        ]
    ).to_csv(value_outcomes_path, index=False)
    pd.DataFrame(
        [
            {"model": "model-a", "value": "care", "prevalence": 2 / 3},
            {"model": "model-a", "value": "trust", "prevalence": 1 / 3},
            {"model": "model-b", "value": "care", "prevalence": 1 / 3},
            {"model": "model-b", "value": "trust", "prevalence": 2 / 3},
        ]
    ).to_csv(value_prevalence_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "model-a",
                "value": "care",
                "contrastive_selection_rate": 1.0,
                "eligible_for_glmm": True,
            },
            {
                "model": "model-a",
                "value": "trust",
                "contrastive_selection_rate": 1.0,
                "eligible_for_glmm": False,
            },
            {
                "model": "model-b",
                "value": "care",
                "contrastive_selection_rate": 1.0,
                "eligible_for_glmm": True,
            },
            {
                "model": "model-b",
                "value": "trust",
                "contrastive_selection_rate": 1.0,
                "eligible_for_glmm": False,
            },
        ]
    ).to_csv(contrastive_value_path, index=False)

    summary = bootstrap.build_bootstrap_datasets(
        value_outcomes_path=value_outcomes_path,
        value_prevalence_path=value_prevalence_path,
        contrastive_value_path=contrastive_value_path,
        prevalence_output_path=prevalence_output_path,
        contrastive_output_path=contrastive_output_path,
        model_differences_output_path=differences_output_path,
    )

    prevalence_ci = read_dataset_csv(prevalence_output_path)
    contrastive_ci = read_dataset_csv(contrastive_output_path)
    differences = read_dataset_csv(differences_output_path)
    assert summary.n_bootstrap == 200
    assert summary.dilemmas_per_replication == 3
    assert prevalence_ci.columns.tolist() == [
        "model",
        "value",
        "prevalence",
        "ci_lower",
        "ci_upper",
    ]
    assert contrastive_ci.columns.tolist() == [
        "model",
        "value",
        "contrastive_selection_rate",
        "ci_lower",
        "ci_upper",
    ]
    assert differences.columns.tolist() == [
        "model_1",
        "model_2",
        "value",
        "difference",
        "ci_lower",
        "ci_upper",
    ]
    assert len(prevalence_ci) == 4
    assert len(contrastive_ci) == 2
    assert len(differences) == 1
    model_a_care = prevalence_ci.loc[
        (prevalence_ci["model"] == "model-a") & (prevalence_ci["value"] == "care")
    ].iloc[0]
    assert model_a_care["prevalence"] == pytest.approx(2 / 3)
    assert differences.iloc[0]["difference"] == pytest.approx(0)

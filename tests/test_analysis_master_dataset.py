from pathlib import Path

import pandas as pd

from moral_dilemmas.analysis import (
    build_analysis_master,
    build_value_outcomes_long,
)
from moral_dilemmas.analysis.preparation import parse_value_labels
from moral_dilemmas.data_loader import read_dataset_csv


def analysis_row(
    *,
    dilemma_id: int,
    model: str,
    final_answer: str | None = "A",
) -> dict[str, object]:
    return {
        "run_id": f"{model}_run",
        "model": model,
        "dilemma_id": dilemma_id,
        "basic_situation": f"Basic {dilemma_id}",
        "dilemma_situation": f"Dilemma {dilemma_id}",
        "topic": 1,
        "topic_group": "Group",
        "action_a_type": "to_do",
        "action_a": f"Do {dilemma_id}",
        "action_a_negative_consequence": "Consequence A",
        "action_a_values": "care, responsibility",
        "action_b_type": "not_to_do",
        "action_b": f"Do not {dilemma_id}",
        "action_b_negative_consequence": "Consequence B",
        "action_b_values": "fairness",
        "prompt": "Prompt",
        "final_answer": final_answer,
        "selected_action_type": "to_do" if final_answer == "A" else None,
        "justification": "Because",
        "raw_response": "FINAL ANSWER: A\nBecause" if final_answer == "A" else "",
        "error": None,
    }


def test_build_analysis_master_combines_results_and_sorts_output(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    output_path = tmp_path / "analysis" / "analysis_master.csv"
    results_dir.mkdir()
    pd.DataFrame(
        [
            analysis_row(dilemma_id=2, model="model-b"),
            analysis_row(dilemma_id=1, model="model-b", final_answer=None),
        ]
    ).to_csv(results_dir / "model_b.csv", index=False)
    pd.DataFrame(
        [
            analysis_row(dilemma_id=2, model="model-a"),
            analysis_row(dilemma_id=1, model="model-a"),
        ]
    ).to_csv(results_dir / "model_a.csv", index=False)

    summary = build_analysis_master(results_dir=results_dir, output_path=output_path)

    master_dataset = read_dataset_csv(output_path)
    assert summary.output_path == output_path
    assert summary.loaded_files == 2
    assert summary.rows == 4
    assert summary.unique_dilemmas == 2
    assert summary.unique_models == 2
    assert summary.duplicate_dilemma_model_pairs == 0
    assert summary.missing_final_answers == 1
    assert master_dataset[["dilemma_id", "model"]].to_records(index=False).tolist() == [
        (1, "model-a"),
        (1, "model-b"),
        (2, "model-a"),
        (2, "model-b"),
    ]
    assert "Expected 4 result files, found 2." in summary.warnings
    assert "Found 1 rows without a final answer." in summary.warnings


def test_build_analysis_master_rejects_missing_required_columns(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame([{"dilemma_id": 1}]).to_csv(results_dir / "broken.csv", index=False)

    try:
        build_analysis_master(
            results_dir=results_dir,
            output_path=tmp_path / "analysis" / "analysis_master.csv",
        )
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("Expected missing required columns to raise ValueError.")


def test_parse_value_labels_preserves_raw_distinct_labels() -> None:
    assert parse_value_labels("['judgement', 'judgment', 'trust', 'trust']") == {
        "judgement",
        "judgment",
        "trust",
    }


def test_build_value_outcomes_long_creates_value_level_rows(tmp_path: Path) -> None:
    master_path = tmp_path / "analysis_master.csv"
    output_path = tmp_path / "analysis" / "value_outcomes_long.csv"
    pd.DataFrame(
        [
            {
                **analysis_row(dilemma_id=2, model="model-b", final_answer=None),
                "action_a_values": "['care', 'trust', 'trust']",
                "action_b_values": "['fairness', 'trust']",
            },
            {
                **analysis_row(dilemma_id=1, model="model-a", final_answer="B"),
                "action_a_values": "['care', 'honesty']",
                "action_b_values": "['fairness', 'honesty']",
            },
        ]
    ).to_csv(master_path, index=False)

    summary = build_value_outcomes_long(
        master_dataset_path=master_path,
        output_path=output_path,
    )

    value_outcomes = read_dataset_csv(output_path)
    assert summary.input_observations == 2
    assert summary.long_format_observations == 6
    assert summary.unique_raw_values == 4
    assert summary.missing_value_chosen == 3
    assert value_outcomes[["dilemma_id", "model", "value"]].to_records(index=False).tolist() == [
        (1, "model-a", "care"),
        (1, "model-a", "fairness"),
        (1, "model-a", "honesty"),
        (2, "model-b", "care"),
        (2, "model-b", "fairness"),
        (2, "model-b", "trust"),
    ]

    model_a_rows = value_outcomes.loc[value_outcomes["model"] == "model-a"].set_index("value")
    assert model_a_rows.loc["care", "value_chosen"] == 0
    assert model_a_rows.loc["fairness", "value_chosen"] == 1
    assert model_a_rows.loc["honesty", "is_contrastive"] == 0

    model_b_rows = value_outcomes.loc[value_outcomes["model"] == "model-b"]
    assert model_b_rows["value_chosen"].isna().all()
    assert "Found 3 value rows without a selected final answer." in summary.warnings

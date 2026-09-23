from pathlib import Path

import pandas as pd

from moral_dilemmas.analysis import build_analysis_master
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

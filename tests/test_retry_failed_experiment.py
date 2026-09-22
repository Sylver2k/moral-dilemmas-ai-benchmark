from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from moral_dilemmas.data_loader import DailyDilemmasLoader, read_dataset_csv
from retry_failed_experiment import (
    create_error_dataset,
    export_error_dataset,
    run_retry_experiment,
)


def result_row(
    dilemma_id: int,
    *,
    model: str = "gemma4:31b",
    final_answer: str | None = "A",
    error: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": "run",
        "model": model,
        "dilemma_id": dilemma_id,
        "basic_situation": f"Basic {dilemma_id}",
        "dilemma_situation": f"Dilemma {dilemma_id}",
        "topic": 1,
        "topic_group": "Group",
        "action_a_type": "to_do",
        "action_a": f"Do {dilemma_id}",
        "action_a_negative_consequence": "Consequence A",
        "action_a_values": "Value A",
        "action_b_type": "not_to_do",
        "action_b": f"Do not {dilemma_id}",
        "action_b_negative_consequence": "Consequence B",
        "action_b_values": "Value B",
        "prompt": "Prompt",
        "final_answer": final_answer,
        "selected_action_type": "to_do" if final_answer == "A" else None,
        "justification": "Because",
        "raw_response": "FINAL ANSWER: A\nBecause" if final_answer == "A" else "",
        "error": error,
    }


def test_create_error_dataset_returns_rows_without_final_answer(tmp_path: Path) -> None:
    results = pd.DataFrame(
        [
            result_row(1, final_answer="A"),
            result_row(2, final_answer=None, error="TimeoutError: timeout"),
            result_row(3, final_answer="", error="Expected exactly one FINAL ANSWER line."),
            result_row(4, model="qwen3.6:35b", final_answer=None, error="Other model"),
        ]
    )
    source = tmp_path / "results.csv"
    results.to_csv(source, index=False)

    error_dataset = create_error_dataset(source, model_name="gemma4:31b")

    expected = results.loc[[1, 2], DailyDilemmasLoader.EXPERIMENT_COLUMNS].reset_index(drop=True)
    assert_frame_equal(error_dataset, expected)


def test_create_error_dataset_can_include_all_models(tmp_path: Path) -> None:
    results = pd.DataFrame(
        [
            result_row(1, final_answer="B"),
            result_row(2, final_answer=None, error="TimeoutError: timeout"),
            result_row(3, model="qwen3.6:35b", final_answer=None, error="Other model"),
        ]
    )
    source = tmp_path / "results.csv"
    results.to_csv(source, index=False)

    error_dataset = create_error_dataset(source, model_name=None)

    assert error_dataset["dilemma_id"].tolist() == [2, 3]


def test_create_error_dataset_validates_required_columns(tmp_path: Path) -> None:
    source = tmp_path / "broken.csv"
    pd.DataFrame([{"dilemma_id": 1}]).to_csv(source, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        create_error_dataset(source)


def test_export_error_dataset_roundtrip(tmp_path: Path) -> None:
    error_dataset = pd.DataFrame(
        [result_row(2, final_answer=None, error="TimeoutError")],
        columns=[
            *DailyDilemmasLoader.EXPERIMENT_COLUMNS,
            "run_id",
            "model",
            "prompt",
            "final_answer",
            "selected_action_type",
            "justification",
            "raw_response",
            "error",
        ],
    ).loc[:, DailyDilemmasLoader.EXPERIMENT_COLUMNS]
    output = tmp_path / "retried" / "error_dataset.csv"

    exported = export_error_dataset(error_dataset, output)

    assert exported == output
    assert_frame_equal(read_dataset_csv(output), error_dataset)


def test_run_retry_experiment_returns_none_when_no_failed_targets(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    pd.DataFrame(
        [
            result_row(1, final_answer="A"),
            result_row(2, final_answer="B"),
        ]
    ).to_csv(source, index=False)

    result = run_retry_experiment(
        results_dataset_path=source,
        model_name="gemma4:31b",
        error_dataset_path=tmp_path / "error_dataset.csv",
        output_path=tmp_path / "retry_results.csv",
        run_id="retry",
    )

    assert result is None
    assert not (tmp_path / "error_dataset.csv").exists()
    assert not (tmp_path / "retry_results.csv").exists()

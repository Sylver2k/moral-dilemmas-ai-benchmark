from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from moral_dilemmas.data_loader import DailyDilemmasLoader, read_dataset_csv
from retry_failed_experiment import (
    create_error_dataset,
    export_error_dataset,
    merge_retry_results,
    run_retry_experiment,
)


def result_row(
    dilemma_id: int,
    *,
    run_id: str = "run",
    model: str = "gemma4:31b",
    final_answer: str | None = "A",
    error: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
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


def test_merge_retry_results_replaces_only_failed_rows_and_preserves_order(
    tmp_path: Path,
) -> None:
    original_results = pd.DataFrame(
        [
            result_row(1, final_answer="A"),
            result_row(2, final_answer=None, error="TimeoutError: timeout"),
            result_row(3, final_answer="B"),
        ]
    )
    retry_results = pd.DataFrame(
        [
            result_row(2, run_id="retry", final_answer="B", error=None),
            result_row(3, run_id="retry", final_answer="A", error=None),
        ]
    )
    original_path = tmp_path / "original.csv"
    retry_path = tmp_path / "retry.csv"
    output_path = tmp_path / "retried" / "merged" / "merged.csv"
    original_results.to_csv(original_path, index=False)
    retry_results.to_csv(retry_path, index=False)

    merged_path = merge_retry_results(
        original_results_path=original_path,
        retry_results_path=retry_path,
        output_path=output_path,
        model_name="gemma4:31b",
    )

    merged_results = read_dataset_csv(merged_path)
    expected = read_dataset_csv(original_path).astype("object")
    retry_results_from_csv = read_dataset_csv(retry_path)
    expected.loc[1, :] = retry_results_from_csv.loc[0, original_results.columns].to_numpy()

    assert merged_path == output_path
    assert merged_results["dilemma_id"].tolist() == [1, 2, 3]
    assert_frame_equal(merged_results, expected, check_dtype=False)


def test_merge_retry_results_does_not_replace_failures_for_other_models(
    tmp_path: Path,
) -> None:
    original_results = pd.DataFrame(
        [
            result_row(1, model="gemma4:31b", final_answer=None, error="TimeoutError"),
            result_row(2, model="qwen3.6:35b", final_answer=None, error="TimeoutError"),
        ]
    )
    retry_results = pd.DataFrame(
        [
            result_row(1, run_id="retry", model="gemma4:31b", final_answer="A"),
            result_row(2, run_id="retry", model="qwen3.6:35b", final_answer="B"),
        ]
    )
    original_path = tmp_path / "original.csv"
    retry_path = tmp_path / "retry.csv"
    output_path = tmp_path / "merged.csv"
    original_results.to_csv(original_path, index=False)
    retry_results.to_csv(retry_path, index=False)

    merge_retry_results(
        original_results_path=original_path,
        retry_results_path=retry_path,
        output_path=output_path,
        model_name="gemma4:31b",
    )

    merged_results = read_dataset_csv(output_path)
    expected = read_dataset_csv(original_path).astype("object")
    retry_results_from_csv = read_dataset_csv(retry_path)
    expected.loc[0, :] = retry_results_from_csv.loc[0, original_results.columns].to_numpy()
    assert_frame_equal(merged_results, expected, check_dtype=False)


def test_merge_retry_results_rejects_duplicate_retry_rows(tmp_path: Path) -> None:
    original_results = pd.DataFrame(
        [result_row(1, final_answer=None, error="TimeoutError")]
    )
    retry_results = pd.DataFrame(
        [
            result_row(1, run_id="retry-1", final_answer="A"),
            result_row(1, run_id="retry-2", final_answer="B"),
        ]
    )
    original_path = tmp_path / "original.csv"
    retry_path = tmp_path / "retry.csv"
    original_results.to_csv(original_path, index=False)
    retry_results.to_csv(retry_path, index=False)

    with pytest.raises(ValueError, match="duplicate model/dilemma_id"):
        merge_retry_results(
            original_results_path=original_path,
            retry_results_path=retry_path,
            output_path=tmp_path / "merged.csv",
        )

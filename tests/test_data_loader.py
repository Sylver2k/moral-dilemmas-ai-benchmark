from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from moral_dilemmas.data_loader import (
    DailyDilemmasLoader,
    read_dataset_csv,
    write_dataset_csv,
)


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dilemma_idx": dilemma_id,
                "basic_situation": "Situation",
                "dilemma_situation": "Dilemma",
                "topic": 1,
                "topic_group": "Group",
                "idx": dilemma_id * 2 + offset,
                "action_type": action_type,
                "action": f"{action_type} {dilemma_id}",
                "negative_consequence": f"Consequence {action_type}",
                "values_aggregated": f"Value {action_type}",
            }
            for dilemma_id in (3, 1, 2)
            for offset, action_type in enumerate(("to_do", "not_to_do"))
        ]
    )


def test_assignments_preserve_action_fields_and_ignore_input_order(
    raw: pd.DataFrame, tmp_path: Path
) -> None:
    before = raw.copy(deep=True)
    source = tmp_path / "raw.csv"
    raw.to_csv(source, index=False)
    loader = DailyDilemmasLoader(source)
    unified = loader.create_unified_dataset()
    assert unified["dilemma_id"].tolist() == [3, 1, 2]
    experiment = loader.create_experiment_dataset(action_order_seed=2187)
    assert experiment["dilemma_id"].tolist() == [1, 2, 3]
    assert experiment["action_a_type"].tolist() == ["not_to_do", "to_do", "not_to_do"]
    raw.iloc[::-1].to_csv(source, index=False)
    assert_frame_equal(experiment, loader.create_experiment_dataset(action_order_seed=2187))
    for _, row in experiment.iterrows():
        for label in ("a", "b"):
            action_type = row[f"action_{label}_type"]
            assert row[f"action_{label}"] == f"{action_type} {row['dilemma_id']}"
            assert row[f"action_{label}_negative_consequence"] == f"Consequence {action_type}"
            assert row[f"action_{label}_values"] == f"Value {action_type}"
    sampled = loader.create_sampled_experiment_dataset(
        action_order_seed=2187, samples_per_topic_group=2, sampling_seed=12
    )
    expected = experiment[experiment["dilemma_id"].isin(sampled["dilemma_id"])].reset_index(
        drop=True
    )
    assert_frame_equal(sampled, expected)
    assert_frame_equal(raw, before)


def test_csv_roundtrip_and_overwrite(raw: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dataset.csv"
    assert write_dataset_csv(raw, path) == path
    assert_frame_equal(read_dataset_csv(path), raw)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_dataset_csv(raw.iloc[:2], path)
    assert_frame_equal(read_dataset_csv(path), raw)
    write_dataset_csv(raw.iloc[:2], path, overwrite=True)
    assert_frame_equal(read_dataset_csv(path), raw.iloc[:2])


def test_create_and_export_sampled_dataset(raw: pd.DataFrame, tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    raw.to_csv(source, index=False)
    loader = DailyDilemmasLoader(source)
    output = tmp_path / "processed" / "sample.csv"
    result = loader.export_sampled_experiment_dataset(
        output,
        action_order_seed=2187,
        sampling_seed=12,
        samples_per_topic_group=2,
    )
    assert result == output
    expected = loader.create_sampled_experiment_dataset(
        action_order_seed=2187,
        sampling_seed=12,
        samples_per_topic_group=2,
    )
    assert len(expected) == 2
    assert_frame_equal(read_dataset_csv(output), expected)
    with pytest.raises(FileExistsError):
        loader.export_sampled_experiment_dataset(
            output,
            action_order_seed=2187,
            sampling_seed=12,
            samples_per_topic_group=1,
        )
    assert_frame_equal(read_dataset_csv(output), expected)
    loader.export_sampled_experiment_dataset(
        output,
        action_order_seed=2187,
        sampling_seed=12,
        samples_per_topic_group=1,
        overwrite=True,
    )
    assert len(read_dataset_csv(output)) == 1


def test_export_all_dataset_stages(raw: pd.DataFrame, tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    raw.to_csv(source, index=False)
    loader = DailyDilemmasLoader(source)
    output_dir = tmp_path / "stages"
    paths = loader.export_all_datasets(
        output_dir, action_order_seed=2187, sampling_seed=12, samples_per_topic_group=2
    )
    expected = {
        "unified": loader.create_unified_dataset(),
        "experiment": loader.create_experiment_dataset(action_order_seed=2187),
        "sampled_experiment": loader.create_sampled_experiment_dataset(
            action_order_seed=2187, sampling_seed=12, samples_per_topic_group=2
        ),
    }
    assert paths.keys() == expected.keys()
    for stage, dataset in expected.items():
        assert paths[stage] == output_dir / f"daily_dilemmas_{stage}.csv"
        assert_frame_equal(read_dataset_csv(paths[stage]), dataset)
    with pytest.raises(FileExistsError):
        loader.export_all_datasets(
            output_dir, action_order_seed=2187, sampling_seed=12, samples_per_topic_group=2
        )
    loader.export_all_datasets(
        output_dir,
        action_order_seed=2187,
        sampling_seed=12,
        samples_per_topic_group=1,
        overwrite=True,
    )
    assert len(read_dataset_csv(paths["sampled_experiment"])) == 1

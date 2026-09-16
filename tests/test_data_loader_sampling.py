from pathlib import Path

import pandas as pd
import pytest

from moral_dilemmas.data_loader import DailyDilemmasLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Dilemmas_with_values_aggregated.csv"
EXPERIMENT_SEED = 2187
SAMPLING_SEED = 2187
EXPECTED_TOPIC_GROUPS = 17
EXPECTED_DILEMMAS_PER_TOPIC_GROUP = 80
SAMPLES_PER_TOPIC_GROUP = 40


@pytest.fixture
def loader() -> DailyDilemmasLoader:
    return DailyDilemmasLoader(RAW_DATASET_PATH)


@pytest.fixture
def experiment_dataset(loader: DailyDilemmasLoader) -> pd.DataFrame:
    return loader.create_experiment_dataset(action_order_seed=EXPERIMENT_SEED)


def test_full_experiment_dataset_matches_expected_sampling_input(
    experiment_dataset: pd.DataFrame,
    loader: DailyDilemmasLoader,
) -> None:

    assert len(experiment_dataset) == EXPECTED_TOPIC_GROUPS * EXPECTED_DILEMMAS_PER_TOPIC_GROUP
    assert experiment_dataset["dilemma_id"].nunique() == len(experiment_dataset)

    topic_group_counts = experiment_dataset["topic_group"].value_counts()
    assert len(topic_group_counts) == EXPECTED_TOPIC_GROUPS
    assert (topic_group_counts == EXPECTED_DILEMMAS_PER_TOPIC_GROUP).all()


def test_sampled_experiment_dataset_has_40_dilemmas_per_topic_group(
    experiment_dataset: pd.DataFrame,
    loader: DailyDilemmasLoader,
) -> None:
    sampled_dataset = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_SEED,
        sampling_seed=SAMPLING_SEED,
        samples_per_topic_group=SAMPLES_PER_TOPIC_GROUP,
    )

    assert len(sampled_dataset) == EXPECTED_TOPIC_GROUPS * SAMPLES_PER_TOPIC_GROUP
    assert sampled_dataset["dilemma_id"].nunique() == len(sampled_dataset)

    topic_group_counts = sampled_dataset["topic_group"].value_counts()
    assert len(topic_group_counts) == EXPECTED_TOPIC_GROUPS
    assert (topic_group_counts == SAMPLES_PER_TOPIC_GROUP).all()


def test_sampling_is_reproducible_with_same_seed(loader: DailyDilemmasLoader) -> None:
    first_sample = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_SEED,
        sampling_seed=SAMPLING_SEED,
        samples_per_topic_group=SAMPLES_PER_TOPIC_GROUP,
    )
    second_sample = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_SEED,
        sampling_seed=SAMPLING_SEED,
        samples_per_topic_group=SAMPLES_PER_TOPIC_GROUP,
    )

    assert first_sample["dilemma_id"].tolist() == second_sample["dilemma_id"].tolist()


def test_different_sampling_seed_changes_selected_dilemmas(
    experiment_dataset: pd.DataFrame,
    loader: DailyDilemmasLoader,
) -> None:
    first_sample = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_SEED,
        sampling_seed=SAMPLING_SEED,
        samples_per_topic_group=SAMPLES_PER_TOPIC_GROUP,
    )
    second_sample = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_SEED,
        sampling_seed=SAMPLING_SEED + 1,
        samples_per_topic_group=SAMPLES_PER_TOPIC_GROUP,
    )

    assert set(first_sample["dilemma_id"]) != set(second_sample["dilemma_id"])


def test_sampled_experiment_dataset_keeps_experiment_schema(
    experiment_dataset: pd.DataFrame,
    loader: DailyDilemmasLoader,
) -> None:
    sampled_dataset = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_SEED,
        sampling_seed=SAMPLING_SEED,
        samples_per_topic_group=SAMPLES_PER_TOPIC_GROUP,
    )

    assert list(sampled_dataset.columns) == list(DailyDilemmasLoader.EXPERIMENT_COLUMNS)

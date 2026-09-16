from pathlib import Path
from random import Random

import pandas as pd


def read_dataset_csv(dataset_path: str | Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame."""
    return pd.read_csv(dataset_path)


def write_dataset_csv(
    dataset: pd.DataFrame, output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write an already prepared DataFrame as CSV"""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Set overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False, encoding="utf-8", mode="w" if overwrite else "x")
    return destination


class DailyDilemmasLoader:
    """Prepare datasets from one raw DailyDilemmas CSV."""

    SHARED_COLUMNS = (
        "dilemma_idx",
        "basic_situation",
        "dilemma_situation",
        "topic",
        "topic_group",
    )
    ACTION_TYPES = ("to_do", "not_to_do")
    UNIFIED_COLUMNS = (
        *SHARED_COLUMNS,
        "to_do_idx",
        "to_do_action",
        "to_do_negative_consequence",
        "to_do_values",
        "not_to_do_idx",
        "not_to_do_action",
        "not_to_do_negative_consequence",
        "not_to_do_values",
    )
    EXPERIMENT_COLUMNS = (
        "dilemma_id",
        "basic_situation",
        "dilemma_situation",
        "topic",
        "topic_group",
        "action_a_type",
        "action_a",
        "action_a_negative_consequence",
        "action_a_values",
        "action_b_type",
        "action_b",
        "action_b_negative_consequence",
        "action_b_values",
    )

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)

    def load_raw_dataset(self) -> pd.DataFrame:
        """Read the source CSV."""
        return read_dataset_csv(self.dataset_path)

    def create_unified_dataset(self) -> pd.DataFrame:
        """Pair the to_do/not_to_do rows."""
        raw_dataset = self.load_raw_dataset()
        records: list[dict[str, object]] = []

        for _, group in raw_dataset.groupby("dilemma_idx", sort=False):
            first = group.iloc[0]
            record: dict[str, object] = {column: first[column] for column in self.SHARED_COLUMNS}

            for action_type in self.ACTION_TYPES:
                action = group.loc[group["action_type"] == action_type].iloc[0]
                record.update(
                    {
                        f"{action_type}_idx": action["idx"],
                        f"{action_type}_action": action["action"],
                        f"{action_type}_negative_consequence": action["negative_consequence"],
                        f"{action_type}_values": action["values_aggregated"],
                    }
                )

            records.append(record)

        return pd.DataFrame.from_records(records, columns=self.UNIFIED_COLUMNS)

    def create_experiment_dataset(self, *, action_order_seed: int) -> pd.DataFrame:
        """Assign A/B positions reproducibly in sorted dilemma ID order."""
        randomizer = Random(action_order_seed)
        unified_dataset = self.create_unified_dataset().sort_values("dilemma_idx")
        records: list[dict[str, object]] = []

        for _, row in unified_dataset.iterrows():
            action_a_type, action_b_type = self.ACTION_TYPES
            if randomizer.choice([True, False]):
                action_a_type, action_b_type = action_b_type, action_a_type

            records.append(
                {
                    "dilemma_id": row["dilemma_idx"],
                    "basic_situation": row["basic_situation"],
                    "dilemma_situation": row["dilemma_situation"],
                    "topic": row["topic"],
                    "topic_group": row["topic_group"],
                    **self._experiment_action_values(row, "a", action_a_type),
                    **self._experiment_action_values(row, "b", action_b_type),
                }
            )

        experiment_dataset = pd.DataFrame.from_records(records, columns=self.EXPERIMENT_COLUMNS)
        return experiment_dataset

    def create_sampled_experiment_dataset(
        self,
        *,
        action_order_seed: int,
        samples_per_topic_group: int = 40,
        sampling_seed: int,
    ) -> pd.DataFrame:
        """Assign A/B labels on the full dataset, then sample equally per topic group."""
        experiment_dataset = self.create_experiment_dataset(action_order_seed=action_order_seed)

        return (
            experiment_dataset.groupby("topic_group", group_keys=False)
            .sample(n=samples_per_topic_group, random_state=sampling_seed)
            .sort_values("dilemma_id")
            .reset_index(drop=True)
        )

    def export_sampled_experiment_dataset(
        self,
        output_path: str | Path,
        *,
        action_order_seed: int,
        sampling_seed: int,
        samples_per_topic_group: int = 40,
        overwrite: bool = False,
    ) -> Path:
        """Create the sampled experiment dataset and export it, returning its CSV path."""
        dataset = self.create_sampled_experiment_dataset(
            action_order_seed=action_order_seed,
            sampling_seed=sampling_seed,
            samples_per_topic_group=samples_per_topic_group,
        )

        return write_dataset_csv(dataset, output_path, overwrite=overwrite)

    def export_all_datasets(
        self,
        output_dir: str | Path,
        *,
        action_order_seed: int,
        sampling_seed: int,
        samples_per_topic_group: int = 40,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Export every stage as csv and return their paths."""

        datasets = {
            "unified": self.create_unified_dataset(),
            "experiment": self.create_experiment_dataset(action_order_seed=action_order_seed),
            "sampled_experiment": self.create_sampled_experiment_dataset(
                action_order_seed=action_order_seed,
                sampling_seed=sampling_seed,
                samples_per_topic_group=samples_per_topic_group,
            ),
        }
        return {
            stage: write_dataset_csv(
                dataset,
                Path(output_dir) / f"daily_dilemmas_{stage}.csv",
                overwrite=overwrite,
            )
            for stage, dataset in datasets.items()
        }

    @staticmethod
    def _experiment_action_values(
        row: pd.Series,
        label: str,
        action_type: str,
    ) -> dict[str, object]:
        return {
            f"action_{label}_type": action_type,
            f"action_{label}": row[f"{action_type}_action"],
            f"action_{label}_negative_consequence": row[f"{action_type}_negative_consequence"],
            f"action_{label}_values": row[f"{action_type}_values"],
        }

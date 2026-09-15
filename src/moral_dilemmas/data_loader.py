from pathlib import Path
from random import Random

import pandas as pd


class DailyDilemmasLoader:
    """Load, transform, and export the DailyDilemmas dataset."""

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
        self._raw_dataset: pd.DataFrame | None = None
        self._unified_dataset: pd.DataFrame | None = None

    @staticmethod
    def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
        """Load a CSV dataset from a path."""
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        return pd.read_csv(path)

    def load_raw_dataset(self) -> pd.DataFrame:
        """Load the raw DailyDilemmas CSV, returning a defensive copy."""
        if self._raw_dataset is None:
            self._raw_dataset = self.load_dataset(self.dataset_path)
            self._unified_dataset = None

        return self._raw_dataset.copy(deep=True)

    def create_unified_dataset(self) -> pd.DataFrame:
        """Return one row per dilemma with both actions in separate columns."""
        if self._unified_dataset is not None:
            return self._unified_dataset.copy(deep=True)

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

        unified_dataset = pd.DataFrame.from_records(records, columns=self.UNIFIED_COLUMNS)
        self._unified_dataset = unified_dataset
        return unified_dataset.copy(deep=True)

    def create_experiment_dataset(self, seed: int) -> pd.DataFrame:
        """Return one row per dilemma with reproducibly assigned A/B actions."""
        randomizer = Random(seed)
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

        return pd.DataFrame.from_records(records, columns=self.EXPERIMENT_COLUMNS)

    def export_unified_dataset(
        self,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Create and write the unified dataset as a UTF-8 CSV file."""
        destination = Path(output_path)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Set overwrite=True to replace it."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.create_unified_dataset().to_csv(
            destination,
            index=False,
            encoding="utf-8",
        )
        return destination

    def export_experiment_dataset(
        self,
        output_path: str | Path,
        *,
        seed: int,
        overwrite: bool = False,
    ) -> Path:
        """Create and write the experiment dataset as a CSV file."""
        destination = Path(output_path)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Set overwrite=True to replace it."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.create_experiment_dataset(seed=seed).to_csv(
            destination,
            index=False,
            encoding="utf-8",
        )
        return destination

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

from pathlib import Path

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

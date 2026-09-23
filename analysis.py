from pathlib import Path

from src.moral_dilemmas.analysis import (
    build_analysis_master,
    build_value_outcomes_long,
    format_master_dataset_summary,
    format_value_outcomes_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
ANALYSIS_DATA_DIR = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_MASTER_PATH = ANALYSIS_DATA_DIR / "analysis_master.csv"
VALUE_OUTCOMES_LONG_PATH = ANALYSIS_DATA_DIR / "value_outcomes_long.csv"


def build_master_dataset() -> Path:
    """Build the master analysis dataset from final result CSV files."""
    summary = build_analysis_master(
        results_dir=RESULTS_DIR,
        output_path=ANALYSIS_MASTER_PATH,
        overwrite=True,
    )

    print(format_master_dataset_summary(summary))

    return summary.output_path


def build_value_outcomes_dataset() -> Path:
    """Build the long-format value outcome dataset from the analysis master."""
    summary = build_value_outcomes_long(
        master_dataset_path=ANALYSIS_MASTER_PATH,
        output_path=VALUE_OUTCOMES_LONG_PATH,
        overwrite=True,
    )

    print()
    print(format_value_outcomes_summary(summary))

    return summary.output_path


def prepare_analysis_data() -> None:
    build_master_dataset()
    build_value_outcomes_dataset()


if __name__ == "__main__":
    prepare_analysis_data()

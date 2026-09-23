from pathlib import Path

from src.moral_dilemmas.analysis import (
    build_analysis_master,
    format_master_dataset_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
ANALYSIS_DATA_DIR = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_MASTER_PATH = ANALYSIS_DATA_DIR / "analysis_master.csv"


def build_master_dataset() -> Path:
    """Build the master analysis dataset from final result CSV files."""
    summary = build_analysis_master(
        results_dir=RESULTS_DIR,
        output_path=ANALYSIS_MASTER_PATH,
        overwrite=True,
    )

    print(format_master_dataset_summary(summary))
    
    return summary.output_path


def construct_master_dataset() -> None:
    build_master_dataset()


if __name__ == "__main__":
    construct_master_dataset()

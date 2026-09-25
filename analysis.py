from pathlib import Path

from src.moral_dilemmas.analysis import (
    build_analysis_master,
    build_bootstrap_datasets,
    build_contrastive_value_summary,
    build_value_outcomes_long,
    build_value_prevalence_summary,
    format_bootstrap_run_summary,
    format_contrastive_value_summary,
    format_master_dataset_summary,
    format_value_outcomes_summary,
    format_value_prevalence_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
ANALYSIS_DATA_DIR = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_MASTER_PATH = ANALYSIS_DATA_DIR / "analysis_master.csv"
VALUE_OUTCOMES_LONG_PATH = ANALYSIS_DATA_DIR / "value_outcomes_long.csv"
VALUE_PREVALENCE_SUMMARY_PATH = ANALYSIS_DATA_DIR / "value_prevalence_summary.csv"
CONTRASTIVE_VALUE_SUMMARY_PATH = ANALYSIS_DATA_DIR / "contrastive_value_summary.csv"
BOOTSTRAP_PREVALENCE_CI_PATH = ANALYSIS_DATA_DIR / "bootstrap_prevalence_ci.csv"
BOOTSTRAP_CONTRASTIVE_CI_PATH = ANALYSIS_DATA_DIR / "bootstrap_contrastive_ci.csv"
BOOTSTRAP_MODEL_DIFFERENCES_PATH = ANALYSIS_DATA_DIR / "bootstrap_model_differences.csv"


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

    print("---------")
    print(format_value_outcomes_summary(summary))

    return summary.output_path


def build_value_prevalence_dataset() -> Path:
    """Build the descriptive value prevalence summary."""
    summary = build_value_prevalence_summary(
        value_outcomes_path=VALUE_OUTCOMES_LONG_PATH,
        output_path=VALUE_PREVALENCE_SUMMARY_PATH,
        overwrite=True,
    )

    print("---------")
    print(format_value_prevalence_summary(summary))

    return summary.output_path


def build_contrastive_value_dataset() -> Path:
    """Build the contrastive value selection-rate summary."""
    summary = build_contrastive_value_summary(
        value_outcomes_path=VALUE_OUTCOMES_LONG_PATH,
        output_path=CONTRASTIVE_VALUE_SUMMARY_PATH,
        overwrite=True,
    )

    print("---------")
    print(format_contrastive_value_summary(summary))

    return summary.output_path


def build_bootstrap_analysis_datasets() -> tuple[Path, Path, Path]:
    """Build bootstrap confidence intervals and model-difference artifacts."""
    summary = build_bootstrap_datasets(
        value_outcomes_path=VALUE_OUTCOMES_LONG_PATH,
        value_prevalence_path=VALUE_PREVALENCE_SUMMARY_PATH,
        contrastive_value_path=CONTRASTIVE_VALUE_SUMMARY_PATH,
        prevalence_output_path=BOOTSTRAP_PREVALENCE_CI_PATH,
        contrastive_output_path=BOOTSTRAP_CONTRASTIVE_CI_PATH,
        model_differences_output_path=BOOTSTRAP_MODEL_DIFFERENCES_PATH,
        overwrite=True,
    )

    print("---------")
    print(format_bootstrap_run_summary(summary))

    return (
        summary.prevalence_output_path,
        summary.contrastive_output_path,
        summary.model_differences_output_path,
    )


def prepare_analysis_data() -> None:
    build_master_dataset()
    build_value_outcomes_dataset()
    build_value_prevalence_dataset()
    build_contrastive_value_dataset()


if __name__ == "__main__":
    prepare_analysis_data()
    # build_bootstrap_analysis_datasets()

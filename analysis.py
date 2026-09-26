from pathlib import Path

from src.moral_dilemmas.analysis import (
    build_analysis_master,
    build_bootstrap_datasets,
    build_contrastive_value_summary,
    build_final_analysis_workbook,
    build_glmm_input_dataset,
    build_value_outcomes_long,
    build_value_prevalence_summary,
    format_bootstrap_run_summary,
    format_contrastive_value_summary,
    format_final_workbook_summary,
    format_glmm_input_summary,
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
GLMM_INPUT_PATH = ANALYSIS_DATA_DIR / "glmm_input.csv"
BOOTSTRAP_PREVALENCE_CI_PATH = ANALYSIS_DATA_DIR / "bootstrap_prevalence_ci.csv"
BOOTSTRAP_CONTRASTIVE_CI_PATH = ANALYSIS_DATA_DIR / "bootstrap_contrastive_ci.csv"
BOOTSTRAP_MODEL_DIFFERENCES_PATH = ANALYSIS_DATA_DIR / "bootstrap_model_differences.csv"
GLMM_GLOBAL_RESULTS_ADJUSTED_PATH = ANALYSIS_DATA_DIR / "glmm_global_results_adjusted.csv"
GLMM_PAIRWISE_RESULTS_PATH = ANALYSIS_DATA_DIR / "glmm_pairwise_results.csv"
FINAL_ANALYSIS_WORKBOOK_PATH = ANALYSIS_DATA_DIR / "final_analysis_results.xlsx"


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


def build_glmm_input_dataset_for_r() -> Path:
    """Build the R input dataset for later GLMM analysis."""
    summary = build_glmm_input_dataset(
        value_outcomes_path=VALUE_OUTCOMES_LONG_PATH,
        contrastive_value_path=CONTRASTIVE_VALUE_SUMMARY_PATH,
        output_path=GLMM_INPUT_PATH,
        overwrite=True,
    )

    print("---------")
    print(format_glmm_input_summary(summary))

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


def build_final_analysis_workbook_file() -> Path:
    """Build the consolidated final analysis Excel workbook."""
    summary = build_final_analysis_workbook(
        value_prevalence_path=VALUE_PREVALENCE_SUMMARY_PATH,
        bootstrap_prevalence_path=BOOTSTRAP_PREVALENCE_CI_PATH,
        contrastive_value_path=CONTRASTIVE_VALUE_SUMMARY_PATH,
        bootstrap_contrastive_path=BOOTSTRAP_CONTRASTIVE_CI_PATH,
        global_glmm_path=GLMM_GLOBAL_RESULTS_ADJUSTED_PATH,
        pairwise_path=GLMM_PAIRWISE_RESULTS_PATH,
        bootstrap_model_differences_path=BOOTSTRAP_MODEL_DIFFERENCES_PATH,
        output_path=FINAL_ANALYSIS_WORKBOOK_PATH,
    )

    print("---------")
    print(format_final_workbook_summary(summary))

    return summary.output_path


def prepare_analysis_data() -> None:
    build_master_dataset()
    build_value_outcomes_dataset()
    build_value_prevalence_dataset()
    build_contrastive_value_dataset()
    build_glmm_input_dataset_for_r()


if __name__ == "__main__":
    prepare_analysis_data()
    # build_bootstrap_analysis_datasets()
    # build_final_analysis_workbook_file()

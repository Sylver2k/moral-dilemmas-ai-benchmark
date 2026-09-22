from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.moral_dilemmas import (
    DEFAULT_SYSTEM_PROMPT,
    LLM,
    DailyDilemmasLoader,
    ExperimentRunner,
    read_dataset_csv,
    write_dataset_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RETRIED_DATA_DIR = PROJECT_ROOT / "data" / "retried"
ERROR_DATASET_DIR = RETRIED_DATA_DIR / "error_datasets"
MERGED_RESULTS_DIR = RETRIED_DATA_DIR / "merged"

MODEL_NAME = "gemma4:31b"  # qwen3.6:35b | gemma4:31b | mistral-small3.2:24b | gpt-oss:20b
RESULTS_DATASET = PROJECT_ROOT / "data" / "results" / "gemma4-31b_seed-2187_run.csv"
EXPERIMENT_TIMEOUT = 240


def get_retry_run_id(model_name: str = MODEL_NAME) -> str:
    normalized_model_name = model_name.replace(":", "-")
    source_stem = Path(RESULTS_DATASET).stem
    return f"{normalized_model_name}_retry_{source_stem}"


RUN_ID = get_retry_run_id()
ERROR_DATASET_PATH = ERROR_DATASET_DIR / f"{RUN_ID}_error_dataset.csv"
RETRY_RESULTS_PATH = RETRIED_DATA_DIR / f"{RUN_ID}.csv"
MERGED_RESULTS_PATH = MERGED_RESULTS_DIR / f"{RUN_ID}_merged.csv"


def create_error_dataset(
    results_dataset_path: str | Path = RESULTS_DATASET,
    *,
    model_name: str | None = MODEL_NAME,
) -> pd.DataFrame:
    """Creates a new dataset containing only failed dilemmas from a prior result file."""
    results_dataset = read_dataset_csv(results_dataset_path)
    _validate_results_dataset(results_dataset)

    retry_candidates = results_dataset.copy()
    if model_name is not None:
        retry_candidates = retry_candidates.loc[retry_candidates["model"] == model_name]

    retry_candidates = retry_candidates.loc[_missing_final_answer(retry_candidates)]

    return (
        retry_candidates.loc[:, DailyDilemmasLoader.EXPERIMENT_COLUMNS]
        .drop_duplicates(subset=["dilemma_id"], keep="first")
        .reset_index(drop=True)
    )


def export_error_dataset(
    error_dataset: pd.DataFrame,
    output_path: str | Path = ERROR_DATASET_PATH,
    *,
    overwrite: bool = True,
) -> Path:
    """Export the failed-dilemma retry dataset."""
    return write_dataset_csv(error_dataset, output_path, overwrite=overwrite)


def run_retry_experiment(
    *,
    results_dataset_path: str | Path = RESULTS_DATASET,
    model_name: str = MODEL_NAME,
    error_dataset_path: str | Path = ERROR_DATASET_PATH,
    output_path: str | Path = RETRY_RESULTS_PATH,
    run_id: str = RUN_ID,
    overwrite: bool = True,
) -> Path | None:
    """Extract failed dilemmas from an earlier result CSV and retry them."""
    error_dataset = create_error_dataset(results_dataset_path, model_name=model_name)

    if error_dataset.empty:
        print("No missing final answers found. Nothing to retry.")
        return None

    exported_error_dataset_path = export_error_dataset(
        error_dataset,
        error_dataset_path,
        overwrite=overwrite,
    )

    llm = LLM(
        model=model_name,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        timeout=EXPERIMENT_TIMEOUT,
    )
    runner = ExperimentRunner(
        experiment_dataset_path=exported_error_dataset_path,
        output_path=output_path,
        run_id=run_id,
        llm=llm,
    )
    return runner.run(overwrite=overwrite)


def merge_retry_results(
    *,
    original_results_path: str | Path = RESULTS_DATASET,
    retry_results_path: str | Path = RETRY_RESULTS_PATH,
    output_path: str | Path = MERGED_RESULTS_PATH,
    model_name: str | None = MODEL_NAME,
    overwrite: bool = True,
) -> Path:
    """Replace failed original result rows with matching retried rows without changing order."""
    original_results = read_dataset_csv(original_results_path)
    retry_results = read_dataset_csv(retry_results_path)
    _validate_retry_merge_inputs(original_results, retry_results)

    merged_results = original_results.copy().astype("object")
    failed_original_mask = _missing_final_answer(merged_results)
    if model_name is not None:
        failed_original_mask &= merged_results["model"] == model_name

    retry_lookup = _create_retry_lookup(retry_results, model_name=model_name)

    for row_index in merged_results.index[failed_original_mask]:
        key = (
            str(merged_results.at[row_index, "model"]),
            int(merged_results.at[row_index, "dilemma_id"]),
        )
        if key in retry_lookup.index:
            merged_results.loc[row_index, original_results.columns] = retry_lookup.loc[
                key,
                original_results.columns,
            ].to_numpy()

    return write_dataset_csv(merged_results, output_path, overwrite=overwrite)


def _missing_final_answer(results_dataset: pd.DataFrame) -> pd.Series:
    final_answers = results_dataset["final_answer"]
    return final_answers.isna() | final_answers.astype("string").str.strip().eq("")


def _validate_results_dataset(results_dataset: pd.DataFrame) -> None:
    required_columns = {
        "model",
        "final_answer",
        *DailyDilemmasLoader.EXPERIMENT_COLUMNS,
    }
    missing_columns = sorted(required_columns.difference(results_dataset.columns))
    if missing_columns:
        raise ValueError(
            "Results dataset is missing required columns: " + ", ".join(missing_columns)
        )


def _validate_retry_merge_inputs(
    original_results: pd.DataFrame,
    retry_results: pd.DataFrame,
) -> None:
    _validate_results_dataset(original_results)
    _validate_results_dataset(retry_results)

    missing_retry_columns = sorted(set(original_results.columns).difference(retry_results.columns))
    if missing_retry_columns:
        raise ValueError(
            "Retry results dataset is missing original result columns: "
            + ", ".join(missing_retry_columns)
        )


def _create_retry_lookup(
    retry_results: pd.DataFrame,
    *,
    model_name: str | None,
) -> pd.DataFrame:
    retry_candidates = retry_results.copy()
    if model_name is not None:
        retry_candidates = retry_candidates.loc[retry_candidates["model"] == model_name]

    duplicate_keys = retry_candidates.duplicated(subset=["model", "dilemma_id"], keep=False)
    if duplicate_keys.any():
        duplicates = retry_candidates.loc[duplicate_keys, ["model", "dilemma_id"]]
        duplicate_labels = [
            f"{row.model}/{row.dilemma_id}" for row in duplicates.itertuples(index=False)
        ]
        raise ValueError(
            "Retry results contain duplicate model/dilemma_id rows: "
            + ", ".join(sorted(set(duplicate_labels)))
        )

    return retry_candidates.assign(
        model=retry_candidates["model"].astype(str),
        dilemma_id=retry_candidates["dilemma_id"].astype(int),
    ).set_index(["model", "dilemma_id"], drop=False)


def execute_retry_pipeline() -> Path | None:
    """Run the retry pipeline for the configured result file and model."""
    load_dotenv()
    return run_retry_experiment()


def merge_results() -> Path:
    """Merge the configured original and retry result files."""
    return merge_retry_results(
        original_results_path=RESULTS_DATASET,
        retry_results_path=RETRY_RESULTS_PATH,
        output_path=MERGED_RESULTS_PATH,
        model_name=MODEL_NAME,
        overwrite=True,
    )


if __name__ == "__main__":
    execute_retry_pipeline()
    # merge_results()

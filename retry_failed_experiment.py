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


def main() -> Path | None:
    load_dotenv()
    return run_retry_experiment()


if __name__ == "__main__":
    main()

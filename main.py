from pathlib import Path

from dotenv import load_dotenv

from src.moral_dilemmas import (
    DEFAULT_SYSTEM_PROMPT,
    LLM,
    DailyDilemmasLoader,
    DilemmaPromptBuilder,
    DilemmaPromptData,
    ExperimentRunner,
    read_dataset_csv,
    write_dataset_csv,
)


def get_run_id() -> str:
    normalized_model_name = MODEL_NAME.replace(":", "-")
    run_id = f"{normalized_model_name}_seed-{EXPERIMENT_RANDOM_SEED}_run"

    return run_id


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATASETS_PATH = PROJECT_ROOT / "data" / "processed"
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Dilemmas_with_values_aggregated.csv"
UNIFIED_DATASET_PATH = PROCESSED_DATASETS_PATH / "daily_dilemmas_unified.csv"
EXPERIMENT_DATASET_PATH = PROCESSED_DATASETS_PATH / "daily_dilemmas_experiment.csv"
SAMPLED_EXPERIMENT_DATASET_PATH = PROCESSED_DATASETS_PATH / "daily_dilemmas_sampled_experiment.csv"
EXPERIMENT_RANDOM_SEED = 2187
MODEL_NAME = "gpt-oss:20b"  # qwen3.6:35b | gemma4:31b | mistral-small3.2:24b | gpt-oss:20b
RUN_ID = get_run_id()
RESULTS_PATH = PROJECT_ROOT / "data" / "results" / f"{RUN_ID}.csv"


def dataset_loader_debug() -> None:
    loader = DailyDilemmasLoader(RAW_DATASET_PATH)

    raw_dataset = loader.load_raw_dataset()
    unified_dataset = loader.create_unified_dataset()
    exported_path = write_dataset_csv(unified_dataset, UNIFIED_DATASET_PATH, overwrite=True)
    experiment_dataset = loader.create_experiment_dataset(action_order_seed=EXPERIMENT_RANDOM_SEED)
    experiment_path = write_dataset_csv(
        experiment_dataset,
        EXPERIMENT_DATASET_PATH,
        overwrite=True,
    )
    sampled_dataset = loader.create_sampled_experiment_dataset(
        action_order_seed=EXPERIMENT_RANDOM_SEED,
        sampling_seed=EXPERIMENT_RANDOM_SEED,
        samples_per_topic_group=40,
    )
    sampled_path = write_dataset_csv(
        sampled_dataset,
        SAMPLED_EXPERIMENT_DATASET_PATH,
        overwrite=True,
    )

    print("DailyDilemmas loader check")
    print(f"Raw dataset: {raw_dataset.shape[0]} rows, {raw_dataset.shape[1]} columns")
    print(f"Unified dataset: {unified_dataset.shape[0]} rows, {unified_dataset.shape[1]} columns")
    print(
        f"Experiment dataset: {experiment_dataset.shape[0]} rows, "
        f"{experiment_dataset.shape[1]} columns"
    )
    print(
            f"Sampled experiment dataset: {sampled_dataset.shape[0]} rows, "
            f"{sampled_dataset.shape[1]} columns"
        )
    print(f"Unified CSV exported to: {exported_path}")
    print(f"Experiment CSV exported to: {experiment_path}")
    print(f"Sampled experiment CSV exported to: {sampled_path}")

    first_dilemma = sampled_dataset.iloc[0]
    print()
    print("First unified dilemma")
    print(f"Dilemma ID: {first_dilemma['dilemma_id']}")
    print(f"Situation: {first_dilemma['dilemma_situation']}")
    print(f"Option to do: {first_dilemma['action_a']}")
    print(f"Option not to do: {first_dilemma['action_b']}")


def llm_debug() -> None:

    llm = LLM(
        model=MODEL_NAME,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    response = llm.generate("Should I report a colleague misusing company resources?")
    print(response)


def prompt_builder_debug() -> None:
    experiment_dataset = read_dataset_csv(EXPERIMENT_DATASET_PATH)
    first_dilemma = experiment_dataset.iloc[0]

    dilemma = DilemmaPromptData(
        dilemma_id=int(first_dilemma["dilemma_id"]),
        dilemma_situation=str(first_dilemma["dilemma_situation"]),
        action_a=str(first_dilemma["action_a"]),
        action_b=str(first_dilemma["action_b"]),
        action_a_consequence=str(first_dilemma["action_a_negative_consequence"]),
        action_b_consequence=str(first_dilemma["action_b_negative_consequence"]),
        topic=int(first_dilemma["topic"]),
        topic_group=str(first_dilemma["topic_group"]),
    )

    prompt = DilemmaPromptBuilder().build(dilemma)
    print(prompt)


def run_experiment() -> Path:
    llm = LLM(
        model=MODEL_NAME,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    runner = ExperimentRunner(
        experiment_dataset_path=SAMPLED_EXPERIMENT_DATASET_PATH,
        output_path=RESULTS_PATH,
        run_id=RUN_ID,
        llm=llm,
    )
    return runner.run(overwrite=True)


if __name__ == "__main__":
    load_dotenv()

    run_experiment()

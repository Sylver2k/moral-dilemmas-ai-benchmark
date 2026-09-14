from pathlib import Path

from dotenv import load_dotenv

from src.moral_dilemmas import (
    DEFAULT_SYSTEM_PROMPT,
    LLM,
    DailyDilemmasLoader,
    DilemmaPromptBuilder,
    DilemmaPromptData,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Dilemmas_with_values_aggregated.csv"
UNIFIED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "daily_dilemmas_unified.csv"


def dataset_loader_debug() -> None:
    loader = DailyDilemmasLoader(RAW_DATASET_PATH)

    raw_dataset = loader.load_raw_dataset()
    unified_dataset = loader.create_unified_dataset()
    exported_path = loader.export_unified_dataset(UNIFIED_DATASET_PATH, overwrite=True)

    print("DailyDilemmas loader check")
    print(f"Raw dataset: {raw_dataset.shape[0]} rows, {raw_dataset.shape[1]} columns")
    print(f"Unified dataset: {unified_dataset.shape[0]} rows, {unified_dataset.shape[1]} columns")
    print(f"Unified CSV exported to: {exported_path}")

    first_dilemma = unified_dataset.iloc[0]
    print()
    print("First unified dilemma")
    print(f"Dilemma ID: {first_dilemma['dilemma_idx']}")
    print(f"Situation: {first_dilemma['dilemma_situation']}")
    print(f"Option to do: {first_dilemma['to_do_action']}")
    print(f"Option not to do: {first_dilemma['not_to_do_action']}")


def llm_debug() -> None:

    llm = LLM(
        model="gpt-oss:20b",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    response = llm.generate("Should I report a colleague misusing company resources?")
    print(response)


def prompt_builder_debug() -> None:
    loader = DailyDilemmasLoader(RAW_DATASET_PATH)
    first_dilemma = loader.create_unified_dataset().iloc[0]

    dilemma = DilemmaPromptData(
        dilemma_id=int(first_dilemma["dilemma_idx"]),
        dilemma_situation=str(first_dilemma["dilemma_situation"]),
        action_a=str(first_dilemma["to_do_action"]),
        action_b=str(first_dilemma["not_to_do_action"]),
        action_a_consequence=str(first_dilemma["to_do_negative_consequence"]),
        action_b_consequence=str(first_dilemma["not_to_do_negative_consequence"]),
        topic=int(first_dilemma["topic"]),
        topic_group=str(first_dilemma["topic_group"]),
    )

    prompt = DilemmaPromptBuilder().build(dilemma)
    print(prompt)


if __name__ == "__main__":
    load_dotenv()

    prompt_builder_debug()

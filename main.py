from pathlib import Path

from dotenv import load_dotenv

from src.moral_dilemmas import (
    DEFAULT_SYSTEM_PROMPT,
    LLM,
    DailyDilemmasLoader,
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


if __name__ == "__main__":
    load_dotenv()

    llm_debug()

from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from .data_loader import read_dataset_csv
from .llm import LLM
from .prompt_builder import DilemmaPromptBuilder, DilemmaPromptData
from .results import ResultDilemmaData, ResultStore, parse_llm_response


class ExperimentRunner:
    """Run one LLM over one prepared experiment CSV."""

    def __init__(
        self,
        experiment_dataset_path: str | Path,
        output_path: str | Path,
        run_id: str,
        llm: LLM,
        prompt_builder: DilemmaPromptBuilder | None = None,
    ) -> None:
        self.experiment_dataset_path = Path(experiment_dataset_path)
        self.output_path = Path(output_path)
        self.run_id = run_id
        self.llm = llm
        self.prompt_builder = prompt_builder or DilemmaPromptBuilder()

    def run(self, *, overwrite: bool = False) -> Path:
        """Run the experiment and export one result CSV."""
        started_at = perf_counter()
        experiment_dataset = read_dataset_csv(self.experiment_dataset_path)
        result_store = ResultStore(run_id=self.run_id, model=self.llm.model)
        successful_prompts = 0

        for _, row in tqdm(
            experiment_dataset.iterrows(),
            total=len(experiment_dataset),
            desc=f"Running {self.llm.model}",
        ):
            dilemma = self._result_dilemma_from_row(row)
            prompt = self.prompt_builder.build(self._prompt_data_from_row(row))

            try:
                raw_response = self.llm.generate(prompt)
            except Exception as error:
                result_store.add_error(
                    dilemma=dilemma,
                    prompt=prompt,
                    error=f"{type(error).__name__}: {error}",
                )
                continue

            result_store.add_result(
                dilemma=dilemma,
                prompt=prompt,
                parsed_response=parse_llm_response(raw_response),
            )
            successful_prompts += 1

        output_path = result_store.export_csv(self.output_path, overwrite=overwrite)

        elapsed_seconds = int(perf_counter() - started_at)
        minutes, seconds = divmod(elapsed_seconds, 60)
        hours, minutes = divmod(minutes, 60)

        print(
            f"Run finished with {successful_prompts}/{len(experiment_dataset)} successful prompts."
        )
        print(f"Run took {hours}h {minutes}min {seconds}s.")

        return output_path

    @staticmethod
    def _prompt_data_from_row(row: pd.Series) -> DilemmaPromptData:
        return DilemmaPromptData(
            dilemma_id=int(row["dilemma_id"]),
            dilemma_situation=str(row["dilemma_situation"]),
            action_a=str(row["action_a"]),
            action_b=str(row["action_b"]),
            action_a_consequence=str(row["action_a_negative_consequence"]),
            action_b_consequence=str(row["action_b_negative_consequence"]),
            topic=int(row["topic"]),
            topic_group=str(row["topic_group"]),
        )

    @staticmethod
    def _result_dilemma_from_row(row: pd.Series) -> ResultDilemmaData:
        return ResultDilemmaData(
            dilemma_id=int(row["dilemma_id"]),
            basic_situation=str(row["basic_situation"]),
            dilemma_situation=str(row["dilemma_situation"]),
            topic=int(row["topic"]),
            topic_group=str(row["topic_group"]),
            action_a_type=str(row["action_a_type"]),
            action_a=str(row["action_a"]),
            action_a_negative_consequence=str(row["action_a_negative_consequence"]),
            action_a_values=str(row["action_a_values"]),
            action_b_type=str(row["action_b_type"]),
            action_b=str(row["action_b"]),
            action_b_negative_consequence=str(row["action_b_negative_consequence"]),
            action_b_values=str(row["action_b_values"]),
        )

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = (
    "run_id",
    "model",
    "dilemma_id",
    "basic_situation",
    "dilemma_situation",
    "topic",
    "topic_group",
    "action_a_type",
    "action_a",
    "action_a_negative_consequence",
    "action_a_values",
    "action_b_type",
    "action_b",
    "action_b_negative_consequence",
    "action_b_values",
    "prompt",
    "final_answer",
    "selected_action_type",
    "justification",
    "raw_response",
    "error",
)

FINAL_ANSWER_PATTERN = re.compile(
    r"^\s*FINAL ANSWER\s*:\s*([AB])\s*$", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class ParsedResponse:
    """Structured fields parsed from one raw LLM response."""

    raw_response: str
    final_answer: str | None
    justification: str
    parse_error: str | None = None


@dataclass(frozen=True)
class ResultDilemmaData:
    """Dilemma metadata stored alongside one model response."""

    dilemma_id: int
    basic_situation: str
    dilemma_situation: str
    topic: int | None
    topic_group: str | None
    action_a_type: str
    action_a: str
    action_a_negative_consequence: str | None
    action_a_values: str | None
    action_b_type: str
    action_b: str
    action_b_negative_consequence: str | None
    action_b_values: str | None


@dataclass(frozen=True)
class ResultRecord:
    """One row in a model result CSV."""

    run_id: str
    model: str
    dilemma_id: int
    basic_situation: str
    dilemma_situation: str
    topic: int | None
    topic_group: str | None
    action_a_type: str
    action_a: str
    action_a_negative_consequence: str | None
    action_a_values: str | None
    action_b_type: str
    action_b: str
    action_b_negative_consequence: str | None
    action_b_values: str | None
    prompt: str
    final_answer: str | None
    selected_action_type: str | None
    justification: str
    raw_response: str
    error: str | None


def parse_llm_response(raw_response: str) -> ParsedResponse:
    """Parse the expected FINAL ANSWER line and following justification."""
    matches = list(FINAL_ANSWER_PATTERN.finditer(raw_response))
    if len(matches) != 1:
        return ParsedResponse(
            raw_response=raw_response,
            final_answer=None,
            justification="",
            parse_error=f"Expected exactly one FINAL ANSWER line, found {len(matches)}.",
        )

    match = matches[0]
    final_answer = match.group(1).upper()
    justification = raw_response[match.end() :].strip()
    return ParsedResponse(
        raw_response=raw_response,
        final_answer=final_answer,
        justification=justification,
    )


class ResultStore:
    """Collect and export results for one model run."""

    def __init__(self, run_id: str, model: str) -> None:
        self.run_id = run_id
        self.model = model
        self._records: list[ResultRecord] = []

    def get_records(self) -> list[ResultRecord]:
        """Return a copy of the stored result records."""
        return self._records.copy()

    def add_result(
        self,
        dilemma: ResultDilemmaData,
        prompt: str,
        parsed_response: ParsedResponse,
    ) -> None:
        """Add one parsed model response to the result store."""
        self._records.append(
            ResultRecord(
                run_id=self.run_id,
                model=self.model,
                **asdict(dilemma),
                prompt=prompt,
                final_answer=parsed_response.final_answer,
                selected_action_type=self._selected_action_type(dilemma, parsed_response),
                justification=parsed_response.justification,
                raw_response=parsed_response.raw_response,
                error=parsed_response.parse_error,
            )
        )

    def add_error(
        self,
        dilemma: ResultDilemmaData,
        prompt: str,
        error: str,
        *,
        raw_response: str = "",
    ) -> None:
        """Add one failed model response to the result store."""
        self._records.append(
            ResultRecord(
                run_id=self.run_id,
                model=self.model,
                **asdict(dilemma),
                prompt=prompt,
                final_answer=None,
                selected_action_type=None,
                justification="",
                raw_response=raw_response,
                error=error,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the collected results as a DataFrame with stable column order."""
        rows = [asdict(record) for record in self._records]
        return pd.DataFrame.from_records(rows, columns=RESULT_COLUMNS)

    def export_csv(self, output_path: str | Path, *, overwrite: bool = False) -> Path:
        """Export all stored results as a CSV file."""
        destination = Path(output_path)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Set overwrite=True to replace it."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(destination, index=False, encoding="utf-8")

        return destination

    @staticmethod
    def _selected_action_type(
        dilemma: ResultDilemmaData,
        parsed_response: ParsedResponse,
    ) -> str | None:
        if parsed_response.final_answer == "A":
            return dilemma.action_a_type
        if parsed_response.final_answer == "B":
            return dilemma.action_b_type
        return None

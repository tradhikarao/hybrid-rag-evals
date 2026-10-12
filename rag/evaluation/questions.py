import csv
from dataclasses import dataclass
from pathlib import Path

from rag.config import EVAL_QUESTIONS_PATH


@dataclass(frozen=True)
class Question:
    text: str
    expected_sources: list[str]  # files that contain the answer
    answerable: bool  # False when the docs don't cover it and the system should say "I don't know"
    reference: str  # ground-truth answer, used by ragas


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[Question]:
    with open(path, newline="") as f:
        return [
            Question(
                text=row["question"],
                expected_sources=row["expected_sources"].split(";"),
                answerable=row["answerable"].strip().lower() == "yes",
                reference=row["reference"],
            )
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

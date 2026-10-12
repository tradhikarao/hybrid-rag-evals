"""Choosing when the system should answer "I don't know" instead of answering."""

from dataclasses import dataclass

from rag.config import ABSTAIN_THRESHOLD

SWEEP_THRESHOLDS = [-10, -8, -6, -4, -2, 0, 2]


@dataclass(frozen=True)
class AbstentionRow:
    question: str
    answerable: bool
    top_score: float  # best cross-encoder score for the question


def count_mistakes(rows: list[AbstentionRow], threshold: float) -> tuple[int, int]:
    """(false refusals, false answers) when answering only if top_score >= threshold."""
    false_refusals = sum(row.answerable and row.top_score < threshold for row in rows)
    false_answers = sum(not row.answerable and row.top_score >= threshold for row in rows)
    return false_refusals, false_answers


def report(rows: list[AbstentionRow], threshold: float = ABSTAIN_THRESHOLD) -> None:
    print("\n--- Top cross-encoder score per question ---")
    for row in sorted(rows, key=lambda row: row.top_score, reverse=True):
        label = "yes" if row.answerable else "no"
        decision = "answer" if row.top_score >= threshold else "I don't know"
        print(f"[{row.top_score:7.2f}] answerable={label:<3} -> {decision:<12} | {row.question}")

    answerable = sum(row.answerable for row in rows)
    false_refusals, false_answers = count_mistakes(rows, threshold)
    print(f"\nThreshold {threshold}: false refusals {false_refusals}/{answerable} answerable, "
          f"false answers {false_answers}/{len(rows) - answerable} unanswerable")

    # Raising the threshold means fewer false answers but more false refusals; lowering does the opposite.
    print("\n--- Threshold sweep ---")
    for sweep in SWEEP_THRESHOLDS:
        false_refusals, false_answers = count_mistakes(rows, sweep)
        print(f"threshold {sweep:>4}: false refusals {false_refusals}, false answers {false_answers}")

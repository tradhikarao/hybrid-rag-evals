from dataclasses import dataclass

from rag.evaluation.questions import Question
from rag.generation import format_chunk, generate
from rag.retrieval import RetrieverSuite


@dataclass(frozen=True)
class Generation:
    retriever: str
    question: str
    reference: str
    contexts: list[str]  # one entry per retrieved chunk
    answer: str

    @property
    def context(self) -> str:
        return "\n\n".join(self.contexts)


def collect_generations(suite: RetrieverSuite, questions: list[Question], k: int) -> list[Generation]:
    """Answer every question once per retriever, so the retrievers can be compared on the same inputs."""
    generations = []
    for question in questions:
        for retriever, hits in suite.search(question.text, k).items():
            answer = generate(question.text, hits)
            print(f"[{retriever}] {question.text}\n  -> {answer}")
            generations.append(Generation(
                retriever=retriever,
                question=question.text,
                reference=question.reference,
                contexts=[format_chunk(hit) for hit in hits],
                answer=answer,
            ))
    return generations

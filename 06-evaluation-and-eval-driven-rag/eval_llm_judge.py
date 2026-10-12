"""Generate an answer per retriever, then grade each with an LLM judge (faithfulness and relevance).

Run:
    python 06-evaluation-and-eval-driven-rag/eval_llm_judge.py
"""

from collections import defaultdict

from rag.config import EVAL_TOP_K
from rag.documents import load_documents
from rag.evaluation.generations import collect_generations
from rag.evaluation.judge import RUBRICS, judge
from rag.evaluation.questions import load_questions
from rag.retrieval import RetrieverSuite


def main():
    generations = collect_generations(RetrieverSuite(load_documents()), load_questions(), EVAL_TOP_K)

    scores = defaultdict(list)  # (retriever, criterion) -> scores
    for generation in generations:
        for criterion in RUBRICS:
            verdict = judge(criterion, generation.question, generation.context, generation.answer)
            print(f"[{generation.retriever}] {criterion} score={verdict['score']}  {verdict['reasoning']}")
            scores[generation.retriever, criterion].append(verdict["score"])

    print(f"\n{'retriever':<15}" + "".join(f"{criterion:>14}" for criterion in RUBRICS))
    for retriever in dict.fromkeys(g.retriever for g in generations):
        averages = "".join(f"{sum(s := scores[retriever, c]) / len(s):14.2f}" for c in RUBRICS)
        print(f"{retriever:<15}{averages}")


if __name__ == "__main__":
    main()

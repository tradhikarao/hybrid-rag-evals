"""Score each retriever with hit-rate@k and MRR, then explore an "I don't know" threshold.

Run:
    python 06-evaluation-and-eval-driven-rag/eval_retrieval.py
"""

from rag.config import EVAL_TOP_K
from rag.documents import load_documents
from rag.evaluation import abstention
from rag.evaluation.metrics import first_hit_rank, hit_rate_at_k, mean_reciprocal_rank
from rag.evaluation.questions import load_questions
from rag.retrieval import RetrieverSuite

HIT_RATE_KS = (1, 2, 3)


def main():
    suite = RetrieverSuite(load_documents())
    questions = load_questions()

    ranks: dict[str, list] = {}  # retriever name -> first correct rank per question
    abstention_rows = []
    for question in questions:
        print(f"Question: {question.text} (expected: {question.expected_sources})")
        results = suite.search(question.text, EVAL_TOP_K)
        for retriever, hits in results.items():
            ranks.setdefault(retriever, []).append(first_hit_rank(hits, question.expected_sources))
        abstention_rows.append(abstention.AbstentionRow(
            question=question.text,
            answerable=question.answerable,
            top_score=results["cross-encoder"][0].score,
        ))

    header = "".join(f"{f'hit@{k}':>8}" for k in HIT_RATE_KS)
    print(f"\n{'retriever':<15}{header}{'MRR':>8}")
    for retriever, retriever_ranks in ranks.items():
        rates = "".join(f"{hit_rate_at_k(retriever_ranks, k):8.2f}" for k in HIT_RATE_KS)
        print(f"{retriever:<15}{rates}{mean_reciprocal_rank(retriever_ranks):8.2f}")

    abstention.report(abstention_rows)


if __name__ == "__main__":
    main()

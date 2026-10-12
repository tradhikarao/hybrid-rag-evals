import pytest

from rag.documents import Document, Hit
from rag.evaluation.abstention import AbstentionRow, count_mistakes
from rag.evaluation.metrics import first_hit_rank, hit_rate_at_k, mean_reciprocal_rank
from rag.retrieval import reciprocal_rank_fusion


def hit(index: int, score: float = 0.0) -> Hit:
    return Hit(index, Document(f"doc{index}.txt", f"text {index}"), score)


def test_metrics_match_the_worked_example_in_05b_evals():
    ranks = [3, 1, 5, 2, 1]
    assert hit_rate_at_k(ranks, 1) == pytest.approx(0.40)
    assert mean_reciprocal_rank(ranks) == pytest.approx(0.607, abs=1e-3)


def test_first_hit_rank_is_one_based_and_none_when_missing():
    hits = [hit(0), hit(1), hit(2)]
    assert first_hit_rank(hits, ["doc1.txt"]) == 2
    assert first_hit_rank(hits, ["other.txt"]) is None
    assert mean_reciprocal_rank([None, 1]) == 0.5


def test_rrf_rewards_documents_both_retrievers_found():
    keyword = [hit(0), hit(1)]
    semantic = [hit(2), hit(1)]
    fused = reciprocal_rank_fusion(keyword, semantic, k=5)
    assert [h.index for h in fused] == [1, 0, 2]  # doc 1 appears in both lists
    assert fused[0].score == pytest.approx(1 / 7 + 1 / 7)


def test_rrf_breaks_ties_with_raw_scores():
    fused = reciprocal_rank_fusion([hit(0, 1.0)], [hit(1, 2.0)])
    assert [h.index for h in fused] == [1, 0]


def test_count_mistakes():
    rows = [
        AbstentionRow("answerable, low score", True, -5.0),
        AbstentionRow("answerable, high score", True, 3.0),
        AbstentionRow("unanswerable, high score", False, 1.0),
        AbstentionRow("unanswerable, low score", False, -8.0),
    ]
    assert count_mistakes(rows, threshold=0.0) == (1, 1)

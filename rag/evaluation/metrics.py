"""Retrieval metrics that don't care which retriever produced the ranking."""

from typing import Optional

from rag.documents import Hit


def first_hit_rank(hits: list[Hit], expected_sources: list[str]) -> Optional[int]:
    """1-based rank of the first hit from an expected source, or None if there isn't one."""
    for rank, hit in enumerate(hits, start=1):
        if hit.source in expected_sources:
            return rank
    return None


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1 / rank if rank else 0.0


def hit_rate_at_k(ranks: list[Optional[int]], k: int) -> float:
    """Share of questions whose first correct hit landed in the top k."""
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[Optional[int]]) -> float:
    return sum(reciprocal_rank(rank) for rank in ranks) / len(ranks)

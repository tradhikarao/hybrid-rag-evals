"""The retrieval building blocks: semantic, keyword, fusion and reranking.

Every function returns `Hit`s sorted best-first, so the stages compose freely.
"""

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.config import CROSS_ENCODER_MODEL, EMBEDDING_MODEL, RERANK_TOP_N, RRF_K
from rag.documents import Document, Hit


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # cosine similarity is the dot product of unit-length vectors, so normalize both sides first
    query_unit = query_vec / np.linalg.norm(query_vec)
    doc_units = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return doc_units @ query_unit


class SemanticRetriever:
    """Dense retrieval: embed every document once, then rank by cosine similarity."""

    def __init__(self, docs: list[Document], model_name: str = EMBEDDING_MODEL):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.doc_vecs = self.model.encode([doc.text for doc in docs])

    def search(self, query: str, k: int) -> list[Hit]:
        scores = cosine_similarity(self.model.encode(query), self.doc_vecs)
        top_indices = np.argsort(scores)[::-1][:k]
        return [Hit(int(i), self.docs[i], float(scores[i])) for i in top_indices]


class KeywordRetriever:
    """Sparse retrieval: BM25 over the document texts."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize([doc.text for doc in docs]))

    def search(self, query: str, k: int) -> list[Hit]:
        indices, scores = self.bm25.retrieve(bm25s.tokenize(query), k=min(k, len(self.docs)))
        return [Hit(int(i), self.docs[int(i)], float(s)) for i, s in zip(indices[0], scores[0])]


def reciprocal_rank_fusion(*rankings: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Merge ranked lists: each document scores sum(1 / (k + rank)) over the lists it appears in.

    RRF only looks at ranks, so ties are broken by the sum of the raw retriever scores
    (cosine and BM25 live on different scales, so that sum only means something as a tiebreak).
    """
    rrf_scores: dict[int, float] = {}
    raw_scores: dict[int, float] = {}
    documents: dict[int, Document] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            rrf_scores[hit.index] = rrf_scores.get(hit.index, 0.0) + 1.0 / (k + rank)
            raw_scores[hit.index] = raw_scores.get(hit.index, 0.0) + hit.score
            documents[hit.index] = hit.document

    fused = sorted(rrf_scores, key=lambda i: (rrf_scores[i], raw_scores[i]), reverse=True)
    return [Hit(i, documents[i], rrf_scores[i]) for i in fused]


class CrossEncoderReranker:
    """Rescore (query, passage) pairs jointly; slower than retrieval but more accurate."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int = RERANK_TOP_N) -> list[Hit]:
        candidates = hits[:top_n]  # hits arrive best-first, so these are the strongest candidates
        scores = self.model.predict([[query, hit.text] for hit in candidates])
        rescored = [Hit(hit.index, hit.document, float(score)) for hit, score in zip(candidates, scores)]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)


class RetrieverSuite:
    """All four retrieval strategies over one corpus, built once and reused per query."""

    def __init__(self, docs: list[Document]):
        self.semantic = SemanticRetriever(docs)
        self.keyword = KeywordRetriever(docs)
        self.reranker = CrossEncoderReranker()

    def search(self, query: str, k: int) -> dict[str, list[Hit]]:
        semantic = self.semantic.search(query, k)
        keyword = self.keyword.search(query, k)
        fused = reciprocal_rank_fusion(keyword, semantic)
        return {
            "keyword": keyword,
            "semantic": semantic,
            "RRF": fused,
            "cross-encoder": self.reranker.rerank(query, fused),
        }

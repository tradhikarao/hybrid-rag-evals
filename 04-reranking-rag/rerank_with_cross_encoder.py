"""Retrieve-then-rerank: hybrid retrieval first, then a cross-encoder rescoring of the candidates.

Run:
    python 04-reranking-rag/rerank_with_cross_encoder.py "Is there a discount?"
"""

from rag.cli import query_from_argv
from rag.display import print_hits
from rag.documents import load_documents
from rag.retrieval import CrossEncoderReranker, KeywordRetriever, SemanticRetriever, reciprocal_rank_fusion

TOP_K = 5


def main():
    query = query_from_argv()
    print(f"Query is {query}")

    docs = load_documents()
    semantic = SemanticRetriever(docs).search(query, TOP_K)
    keyword = KeywordRetriever(docs).search(query, TOP_K)

    fused = reciprocal_rank_fusion(keyword, semantic)
    print_hits("Fused (RRF) results", fused)
    print_hits("Cross-encoder reranked results", CrossEncoderReranker().rerank(query, fused))


if __name__ == "__main__":
    main()

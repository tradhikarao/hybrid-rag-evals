"""Hybrid retrieval: fuse semantic and keyword rankings with reciprocal rank fusion (RRF).

Run:
    python 03-hybrid-retriever/hybrid_retriever.py "Is there a discount?"
"""

from rag.cli import query_from_argv
from rag.display import print_hits
from rag.documents import load_documents
from rag.retrieval import KeywordRetriever, SemanticRetriever, reciprocal_rank_fusion

TOP_K = 5
RRF_K = 5  # small constant so rank differences matter on a corpus of only five documents


def main():
    query = query_from_argv()
    print(f"Query is {query}")

    docs = load_documents()
    semantic = SemanticRetriever(docs).search(query, TOP_K)
    keyword = KeywordRetriever(docs).search(query, TOP_K)

    print_hits("Semantic results (cosine similarity)", semantic)
    print_hits("Keyword (BM25) results", keyword)
    print_hits("Fused (RRF) results", reciprocal_rank_fusion(keyword, semantic, k=RRF_K))


if __name__ == "__main__":
    main()

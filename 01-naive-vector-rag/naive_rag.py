"""The simplest RAG pipeline, built with no framework.

1. Load a handful of small text docs (one chunk per doc, since they're already short).
2. Embed every doc, and embed the user's question the same way.
3. Rank docs by cosine similarity to the question.
4. Stuff the top-k docs into a prompt and ask Claude to answer.

Run:
    python 01-naive-vector-rag/naive_rag.py "How much does the Pro plan cost?"
"""

from rag.cli import query_from_argv
from rag.display import print_hits
from rag.documents import load_documents
from rag.generation import generate
from rag.retrieval import SemanticRetriever

TOP_K = 5


def main():
    query = query_from_argv()
    print(f"Query is {query}")

    hits = SemanticRetriever(load_documents()).search(query, TOP_K)
    print_hits("Semantic results (cosine similarity)", hits)

    print(f"\n--- Answer ---\n{generate(query, hits)}")


if __name__ == "__main__":
    main()

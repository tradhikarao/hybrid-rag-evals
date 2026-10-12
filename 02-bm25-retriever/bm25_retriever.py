"""Classic keyword (lexical) retrieval with BM25.

Run:
    python 02-bm25-retriever/bm25_retriever.py "Is there a discount?"
"""

from rag.cli import query_from_argv
from rag.display import print_hits
from rag.documents import load_documents
from rag.retrieval import KeywordRetriever

TOP_K = 5


def main():
    query = query_from_argv()
    print(f"Query is {query}")

    print_hits("Keyword (BM25) results", KeywordRetriever(load_documents()).search(query, TOP_K))


if __name__ == "__main__":
    main()

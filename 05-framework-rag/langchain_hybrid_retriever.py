"""Hybrid retrieval rebuilt with LangChain, mapping each piece onto what 03 did by hand.

Run:
    python 05-framework-rag/langchain_hybrid_retriever.py "Is there a discount?"
"""

import re  # regular expressions, used to split text into word tokens for BM25

from bm25s.stopwords import STOPWORDS_EN  # same stopword list 02/03 used, so BM25 ignores the same words
from langchain.retrievers import EnsembleRetriever  # LangChain's version of our reciprocalRankFusion
from langchain_community.retrievers import BM25Retriever  # LangChain's version of our keywordSearch
from langchain_core.documents import Document  # LangChain's container for one chunk: page_content + metadata
from langchain_core.vectorstores import InMemoryVectorStore  # LangChain's version of our embed + cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings  # wraps a sentence-transformers model for LangChain

from rag.cli import query_from_argv
from rag.config import EMBEDDING_MODEL
from rag.documents import load_documents

TOP_K = 5  # how many of the most relevant chunks each retriever returns
RRF_K = 5  # the constant in 1 / (RRF_K + rank); 03 uses k=5 too. LangChain's default is 60


def to_langchain(docs) -> list[Document]:
    # page_content is the text that gets embedded/searched; metadata rides along so we can print the source later
    return [Document(page_content=doc.text, metadata={"source": doc.source}) for doc in docs]


def bm25_tokenize(text: str) -> list[str]:
    # BM25Retriever's default tokenizer is just text.split(): no lowercasing, punctuation stays glued to words
    # ("models?" != "models"), and stopwords count. This mimics bm25s.tokenize from 02/03 instead:
    # lowercase, keep words of 2+ characters, drop stopwords.
    return [token for token in re.findall(r"(?u)\b\w\w+\b", text.lower()) if token not in STOPWORDS_EN]


def print_results(title: str, docs: list[Document]) -> None:
    print(f"\n--- {title} ---")
    for rank, doc in enumerate(docs, start=1):
        # [:80] shows just a preview of the chunk text; replace newlines so each result stays on one line
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"Rank {rank}: {doc.metadata['source']}: {preview}...")


def main():
    query = query_from_argv()
    print(f"Query is {query}")

    docs = to_langchain(load_documents())

    # Semantic side: embed every doc with sentence-transformers and store the vectors in memory.
    # normalize_embeddings=True scales vectors to length 1, like our cosine_similarity did by hand.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # Keyword side: BM25 over the same docs, using our bm25s-style tokenizer
    keyword_retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenize, k=TOP_K)

    # Fusion: weighted reciprocal rank fusion. Equal weights = plain RRF, same as 03.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, semantic_retriever],
        weights=[0.5, 0.5],
        c=RRF_K,
    )

    # Show each retriever on its own first, so you can see how the fused ranking was built.
    # similarity_search_with_score also returns the cosine similarity, which as_retriever() hides.
    print("\n--- Semantic results (with cosine similarity) ---")
    for rank, (doc, score) in enumerate(vector_store.similarity_search_with_score(query, k=TOP_K), start=1):
        print(f"Rank {rank}: [{score:.3f}] {doc.metadata['source']}")

    print_results("Keyword (BM25) results", keyword_retriever.invoke(query))
    print_results("Fused (RRF) results", hybrid_retriever.invoke(query))



if __name__ == "__main__":
    main()

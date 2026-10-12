import os  # lets us read environment variables, like the OPENAI_API_KEY
import sys  # lets us read command-line arguments (sys.argv)
from pathlib import Path  # object-oriented filesystem paths, nicer than raw strings

import numpy as np  # numerical arrays + fast math (dot products, norms, sorting)
from anthropic import Anthropic
from dotenv import load_dotenv  # loads key=value pairs from a .env file into the environment
from openai import OpenAI  # the OpenAI client class used to call embeddings + chat endpoints
from sentence_transformers import SentenceTransformer
import anthropic
import bm25s


# __file__ is this script's own path; .resolve() makes it absolute;
# .parent goes up to this file's folder, .parent again goes up to the repo root,
# then we point into the shared "data" folder that sits next to all the numbered projects.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#EMBEDDING_MODEL = "text-embedding-3-small"  # which OpenAI model turns text into vectors
TOP_K = 5  # how many of the most relevant chunks to retrieve per question
# Load the model
embeddingModel = SentenceTransformer("BAAI/bge-base-en-v1.5")

def load_documents() -> list[dict]:
    docs = []  # will hold one dict per document: {"source": filename, "text": contents}
    for path in sorted(DATA_DIR.glob("*.txt")):  # find every .txt file in data/, sorted for stable order
        docs.append({"source": path.name, "text": path.read_text().strip()})  # read file text, trim whitespace, store it
    return docs  # hand back the full list of loaded documents


def embed(docs: list[dict]) -> np.ndarray:
    vectorList = []
    # for each doc, generate embedding and store in a list of vectors
    for doc in docs:
        vector = embeddingModel.encode(doc['text'])
        vectorList.append(vector)
    print(len(vectorList))
    return vectorList

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # cosine similarity = dot product of two UNIT-length vectors, so we normalize both sides to length 1 first
    query_norm = query_vec / np.linalg.norm(query_vec)  # scale the query vector down to unit length
    # np.linalg.norm(..., axis=1) computes each row's length; keepdims=True keeps it a column so division broadcasts correctly
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)  # scale every doc vector to unit length
    return doc_norms @ query_norm  # matrix-vector multiply: dot product of each doc row with the query = similarity score


def retrieve(docs: list[dict], doc_vecs: np.ndarray, query: str, k: int) -> list[dict]:
    queryVec = embeddingModel.encode(query)
    valArray = cosine_similarity(queryVec, doc_vecs)
    print(f"Cosine similarity array shape is {valArray.shape}")
    sortedIndicesList = np.argsort(valArray)[::-1]
   # print(valArray)
    print(sortedIndicesList)
    top_matches = []
    for index in sortedIndicesList:
        match = {}

        print(f"Index in Retreive method is {index}")
        print(f"Source in Retreive method is {docs[index]['source']}")
        print(f"Text in Retreive method is {docs[index]['text']}")
        print(f"Score in Retreive method is {valArray[index]}")

        match['source'] = docs[index]['source']
        match['text'] = docs[index]['text']
        match['score'] = valArray[index]
        match['index'] = index
        top_matches.append(match)
    return top_matches[:k]


def reciprocalRankFusion(keywordResults: tuple[np.ndarray, np.ndarray], semanticResults: list[dict], docs: list[dict], k: int = 60) -> list[dict]:

    for row in semanticResults:
        # :.3f formats the score to 3 decimal places; [:80] shows just a preview of the chunk text
        print(f"[{row['score']:.3f}], {row['index']}, {row['source']}: {row['text'][:80]}...")

    keywordIndices, keywordScores = keywordResults
        # Let's see what we got!
    for index, score in zip(keywordIndices[0], keywordScores[0]):
        print(f"Keyword results: (score: {score:.2f}): {index}")

    # rank 1 = best match. Both retrievers already return their results best-first,
    # so the rank is just each doc's position in that list (enumerate starting at 1).
    semantic_ranks = {}
    semantic_scores = {}
    for rank, row in enumerate(semanticResults, start=1):
        doc_index = row["index"]
        semantic_ranks[doc_index] = rank
        semantic_scores[doc_index] = row["score"]

    keyword_ranks = {}
    keyword_scores = {}
    for rank, (doc_index, score) in enumerate(zip(keywordIndices[0], keywordScores[0]), start=1):
        keyword_ranks[doc_index] = rank
        keyword_scores[doc_index] = score

    # a doc that only one retriever found still gets fused in, using just that one term
    all_indices = set(semantic_ranks) | set(keyword_ranks)

    fused = []
    for doc_index in all_indices:
        rrf_score = 0.0
        if doc_index in semantic_ranks:
            print(f"semantic ranking of {doc_index} is {semantic_ranks[doc_index]}")
            rrf_score += 1.0 / (k + semantic_ranks[doc_index])
        if doc_index in keyword_ranks:
            rrf_score += 1.0 / (k + keyword_ranks[doc_index])
            print(f"keyword ranking of {doc_index} is {keyword_ranks[doc_index]}")

        # RRF only uses rank, so it can't separate ties on its own. As a tiebreak,
        # fall back to the raw scores that produced those ranks (0 for a retriever
        # that didn't return this doc at all). Semantic scores are cosine similarity
        # (roughly 0-1) and BM25 scores are unbounded, so this sum is only meaningful
        # as a same-rrf_score tiebreak, not as a score in its own right.
        raw_score = semantic_scores.get(doc_index, 0.0) + float(keyword_scores.get(doc_index, 0.0))

        fused.append({"index": doc_index, "rrf_score": rrf_score, "raw_score": raw_score, **docs[doc_index]})

    fused.sort(key=lambda row: (row["rrf_score"], row["raw_score"]), reverse=True)

    print("\n--- Fused (RRF) results ---")
    for row in fused:
        print(f"[{row['rrf_score']:.4f}] (raw: {row['raw_score']:.3f}), {row['index']}, {row['source']}: {row['text'][:80]}...")

    return fused

def keywordSearch(docs: list[dict], query: str, k: int):
    corpus = []
    for doc in docs:
        corpus.append(doc['text'])

    docs_tokens = bm25s.tokenize(corpus)

    retreiver = bm25s.BM25()  # no corpus, so retrieve() returns doc indices instead of doc texts
    retreiver.index(docs_tokens)

    query_tokens = bm25s.tokenize(query)
    results, scores = retreiver.retrieve(query_tokens, k=k)

    for rank in range(k):
        doc_index = int(results[0][rank])  # true position in docs / corpus
        score = scores[0][rank]
        print(f"Rank {rank + 1}: doc index {doc_index}, source {docs[doc_index]['source']}, "
              f"score {score:.3f}, text {docs[doc_index]['text'][:80]}...")

    return results, scores



def main():
    load_dotenv()  # read .env (if present) and copy its variables into the process environment
    if not os.getenv(
            "OPENAI_API_KEY"):  # fail fast with a clear message instead of a confusing error from the API call later
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env to .env at the repo root "
            "and fill in your key."
        )

    docs = load_documents()  # load every .txt file from data/
    if not docs:  # guard against running this before any sample docs exist
        raise RuntimeError(f"No .txt files found in {DATA_DIR}")

    # sys.argv[0] is the script name; sys.argv[1:] is every word typed after it on the command line,
    # e.g. running `python main.py How much is Pro?` gives sys.argv[1:] == ["How", "much", "is", "Pro?"]
    default_query = "Is there a discount offered for any of these pricing models?"
    cli_args = sys.argv[1:]
    if cli_args:
        # glue the separate words back into one string, e.g. ["How", "much", "is", "Pro?"] -> "How much is Pro?"
        query = " ".join(cli_args)
    else:
        # no arguments were passed on the command line, so fall back to the default question
        query = default_query
    print(f"Query is {query}")

    #get documents and store in a list of dictionaries
    docs = load_documents()

    #vectorize the documents and store in a local variable
    vectorizedDocs = embed(docs)
    #check cosine similarity of query against the embeddings
    semanticScores = retrieve(docs,vectorizedDocs,query,5)

    keywordResults = keywordSearch(docs, query, 5)

    reciprocalRankFusion(keywordResults,semanticScores, docs,k=5)

if __name__ == "__main__":  # only run main() when this file is executed directly (not when imported elsewhere)
    main()
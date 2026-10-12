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
from sentence_transformers import CrossEncoder

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


def embed(texts: list[str]) -> np.ndarray:
    # one API call embeds ALL texts at once (cheaper/faster than one call per text)
    response = embeddingModel.encode(texts)
    vectors = response.tolist()
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # cosine similarity = dot product of two UNIT-length vectors, so we normalize both sides to length 1 first
    query_norm = query_vec / np.linalg.norm(query_vec)  # scale the query vector down to unit length
    # np.linalg.norm(..., axis=1) computes each row's length; keepdims=True keeps it a column so division broadcasts correctly
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)  # scale every doc vector to unit length
    return doc_norms @ query_norm  # matrix-vector multiply: dot product of each doc row with the query = similarity score


def SemanticallyRetrieve(docs: list[dict], doc_vecs: np.ndarray, query: str, k: int) -> list[dict]:
    # embed() expects a list of texts and returns a list of vectors, so wrap the
    # single query in a list, then take the first (only) vector back out
    embedded_queries = embed([query])
    query_vec = embedded_queries[0]

    # one similarity score per document, in the same order as docs/doc_vecs
    scores = cosine_similarity(query_vec, doc_vecs)

    # sort scores from highest to lowest, then keep only the top k
    sorted_indices_ascending = np.argsort(scores)  # lowest score first
    sorted_indices_descending = sorted_indices_ascending[::-1]  # highest score first
    top_indices = sorted_indices_descending[:k]  # keep only the top k

    # build the final list: each top document plus its similarity score attached
    top_matches = []
    for i in top_indices:
        match = dict(docs[i])  # copy the document dict so we don't mutate the original
        match["score"] = float(scores[i])
        match["index"]=i
        top_matches.append(match)
    return top_matches

def getSemanticSearchResults(query: str, docs: list[dict]) -> list[dict]:

    client = anthropic.Anthropic()  # reads Claude API key from the environment automatically

    # each doc is a dict like {"source": ..., "text": ...}; the embedding model only needs the text
    document_texts = [document["text"] for document in docs]
    # embed every document in one batch call -> matrix of shape (num_documents, embedding_dim),
    # where row i is the embedding vector for docs[i]
    document_embeddings = embed(document_texts)
    print(document_embeddings.shape)
    print(f"Question: {query}\n")  # echo back what's being asked

    retrieved = SemanticallyRetrieve(docs, document_embeddings, query, TOP_K)  # embed the query and find the closest docs

    return retrieved
    #answer = generate(client, query, retrieved)  # ask the chat model to answer using only the retrieved chunks
    #print("\n--- Answer ---")
    #print(answer)

def getKeywordSearchResults(query: str, docs: list[dict])-> tuple[np.ndarray, np.ndarray]:
    corpus = [doc['text'] for doc in docs]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))

    results, scores = retriever.retrieve(bm25s.tokenize(query), k=5)
    return results, scores

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

def rerankWithCrossEncoder(query: str, hybridRetreiverResults: list[dict[str, float]], top_n: int = 20) -> list[dict]:
    # hybridRetreiverResults is already sorted best-first by RRF, so the first
    # top_n rows are its best candidates; only those get the (expensive) cross-encoder pass
    candidates = hybridRetreiverResults[:top_n]

    crossEncoderModel = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

    pairs: list[list[str]] = []
    for candidate in candidates:
        pairs.append([query, candidate['text']])

    # Predict relevance scores
    scores = crossEncoderModel.predict(pairs)
    print(scores)

    # predict() takes one argument: a list of [query, passage] pairs, one per candidate doc
    #pairs = [[query, row["text"]] for row in candidates]
    #scores = crossEncoderModel.predict(pairs)

    reranked = []
    for row, score in zip(candidates, scores):
        reranked.append({**row, "cross_encoder_score": float(score)})
    reranked.sort(key=lambda row: row["cross_encoder_score"], reverse=True)

    print("\n--- Cross-encoder reranked results ---")
    for row in reranked:
        print(f"[{row['cross_encoder_score']:.4f}], {row['index']}, {row['source']}: {row['text'][:80]}...")

    return reranked


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

    semanticRetrieverResults = getSemanticSearchResults(query, docs)

    keywordResults = getKeywordSearchResults(query, docs)

    fusedResults = reciprocalRankFusion(keywordResults, semanticRetrieverResults, docs)

    rerankWithCrossEncoder(query, fusedResults)

if __name__ == "__main__":  # only run main() when this file is executed directly (not when imported elsewhere)
    main()
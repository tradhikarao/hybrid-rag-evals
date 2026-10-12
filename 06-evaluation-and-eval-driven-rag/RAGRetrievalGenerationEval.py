#Step 2: Compute hit-rate@k and MRR
#Write one scoring function that doesn't care which retriever produced the list:

#Input: the ranked list of source file names for one question (best first) + that question's expected_sources.
#Find the rank: the 1-based position of the first file that's in expected_sources. None if not found.
#Output: hit@1, hit@2, hit@3 (1 or 0 each) and the reciprocal rank (1 / rank, or 0 if not found).
#Then loop over all questions and average each column. That average is hit-rate@k and MRR.

#Use k = 1, 2, 3, not 5. There are only 5 files, so every retriever has hit-rate@5 = 100%.

#Sanity-check the scorer by hand first: feed it the exercise from 05b-evals/1-retrieval-metrics.md (ranks 3, 1, 5, 2, 1) and confirm you get hit@1 = 0.40 and MRR ≈ 0.607.

import os  # lets us read environment variables, like the OPENAI_API_KEY
import sys  # lets us read command-line arguments (sys.argv)
from pathlib import Path  # object-oriented filesystem paths, nicer than raw strings

import numpy as np  # numerical arrays + fast math (dot products, norms, sorting)
from anthropic import Anthropic
from dotenv import load_dotenv  # loads key=value pairs from a .env file into the environment
from numpy.ma.extras import average
from openai import OpenAI  # the OpenAI client class used to call embeddings + chat endpoints
from sentence_transformers import SentenceTransformer
import anthropic
import bm25s
from sentence_transformers import CrossEncoder
import csv
# __file__ is this script's own path; .resolve() makes it absolute;
# .parent goes up to this file's folder, .parent again goes up to the repo root,
# then we point into the shared "data" folder that sits next to all the numbered projects.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#EMBEDDING_MODEL = "text-embedding-3-small"  # which OpenAI model turns text into vectors
TOP_K = 5  # how many of the most relevant chunks to retrieve per question
# if the best cross-encoder score is below this, the system should say "I don't know" instead of answering.
# A starting guess only; use the threshold sweep printed by reportAbstention() to choose it.
ABSTAIN_THRESHOLD = -2.0
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

def getSemanticSearchResults(query: str, docs: list[dict], top_K: int) -> list[dict]:

    client = anthropic.Anthropic()  # reads Claude API key from the environment automatically

    # each doc is a dict like {"source": ..., "text": ...}; the embedding model only needs the text
    document_texts = [document["text"] for document in docs]
    # embed every document in one batch call -> matrix of shape (num_documents, embedding_dim),
    # where row i is the embedding vector for docs[i]
    document_embeddings = embed(document_texts)
    #print(document_embeddings.shape)
    #print(f"Question: {query}\n")  # echo back what's being asked

    retrieved = SemanticallyRetrieve(docs, document_embeddings, query, top_K)  # embed the query and find the closest docs

    return retrieved
    #answer = generate(client, query, retrieved)  # ask the chat model to answer using only the retrieved chunks
    #print("\n--- Answer ---")
    #print(answer)

def getKeywordSearchResults(query: str, docs: list[dict], top_K=5)-> tuple[np.ndarray, np.ndarray]:
    corpus = [doc['text'] for doc in docs]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))

    results, scores = retriever.retrieve(bm25s.tokenize(query), k=top_K)
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
            #print(f"semantic ranking of {doc_index} is {semantic_ranks[doc_index]}")
            rrf_score += 1.0 / (k + semantic_ranks[doc_index])
        if doc_index in keyword_ranks:
            rrf_score += 1.0 / (k + keyword_ranks[doc_index])
            #print(f"keyword ranking of {doc_index} is {keyword_ranks[doc_index]}")

        # RRF only uses rank, so it can't separate ties on its own. As a tiebreak,
        # fall back to the raw scores that produced those ranks (0 for a retriever
        # that didn't return this doc at all). Semantic scores are cosine similarity
        # (roughly 0-1) and BM25 scores are unbounded, so this sum is only meaningful
        # as a same-rrf_score tiebreak, not as a score in its own right.
        raw_score = semantic_scores.get(doc_index, 0.0) + float(keyword_scores.get(doc_index, 0.0))

        fused.append({"index": doc_index, "rrf_score": rrf_score, "raw_score": raw_score, **docs[doc_index]})

    fused.sort(key=lambda row: (row["rrf_score"], row["raw_score"]), reverse=True)

    #print("\n--- Fused (RRF) results ---")
    #for row in fused:
     #   print(f"[{row['rrf_score']:.4f}] (raw: {row['raw_score']:.3f}), {row['index']}, {row['source']}: {row['text'][:80]}...")

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

    #print("\n--- Cross-encoder reranked results ---")
    #for row in reranked:
    #    print(f"[{row['cross_encoder_score']:.4f}], {row['index']}, {row['source']}: {row['text'][:80]}...")

    return reranked

def ScoreRetrievers(query:str, docs: list[dict], top_K=3):
    semanticMRRList = []
    keywordMRRList = []
    rrfMRRList = []
    hitRateSumKeyword3 = 0
    hitRateSumSemantic3 = 0
    hitRateSumKeyword1 = 0
    hitRateSumSemantic1 = 0
    hitRateSumRRF1 = 0
    hitRateSumRRF3 = 0
    abstentionRows = []  # one entry per question: its answerable label + the best cross-encoder score

    with open("EvalQuestions.csv", newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            question = row["question"]
            expected_sources = row["expected_sources"].split(";")
            print(f"Question is {question} and expected source is {expected_sources}")
            keywordIndices, keywordScores = getKeywordSearchResults(question, docs, top_K)
            keywordMRR = 0
            semanticMRR = 0
            keywordRank = 0
            semanticRank = 0
            rrfMRR = 0
            rrfRank = 0
            for rank, (doc_index, score) in enumerate(zip(keywordIndices[0], keywordScores[0]), start=1):
                if docs[int(doc_index)]['source'] in expected_sources:
                    keywordRank = rank
                    break

            if keywordRank != 0:
                keywordMRR = 1/keywordRank
            keywordMRRList.append(keywordMRR)
            #print(f"Keyword Hit Rate @1 is {keywordHitRate}, MRR is {keywordMRR}")
            results = getSemanticSearchResults(question,docs, top_K)
            for rank, result in enumerate(results, start=1):
               # print(f"In Semantic results: score is {result['score']}, Rank is {rank}")
               # print(f"doc name is {result['source']}")
                if result['source'] in expected_sources:
                    semanticRank = rank
                    break
            rrfResults = reciprocalRankFusion([keywordIndices, keywordScores], results, docs)
            for rank, result in enumerate(rrfResults, start=1):
                if result['source'] in expected_sources:
                    rrfRank = rank
                    break

            crossEncoderResults = rerankWithCrossEncoder(question,rrfResults)
            # crossEncoderResults is sorted best-first, so [0] holds the highest (top) score.
            # Record it with the answerable label so reportAbstention() can test an "I don't know" threshold.
            abstentionRows.append({
                "question": question,
                "answerable": row["answerable"].strip().lower() == "yes",
                "topScore": crossEncoderResults[0]["cross_encoder_score"],
            })
            for rank, result in enumerate(crossEncoderResults, start=1):
                print(f"Cross encoder results: {result['cross_encoder_score']}, {result['source']}")
                if result['source'] in expected_sources:
                    crossEncoderRank = rank
                    break

            if rrfRank != 0:
                rrfMRR = 1/rrfRank
            rrfMRRList.append(rrfMRR)

            if semanticRank != 0:
                semanticMRR = 1/semanticRank
            semanticMRRList.append(semanticMRR)

            if semanticRank >= 1 and semanticRank <= 3:
                hitRateSumSemantic3 = hitRateSumSemantic3 + 1

            if keywordRank >= 1 and keywordRank <= 3:
                hitRateSumKeyword3 = hitRateSumKeyword3 + 1

            if semanticRank == 1:
                hitRateSumSemantic1 = hitRateSumSemantic1 + 1
            if keywordRank == 1:
                hitRateSumKeyword1 = hitRateSumKeyword1 + 1

            if rrfRank >= 1 and rrfRank <= 3:
                hitRateSumRRF3 = hitRateSumRRF3 + 1
            if rrfRank == 1:
                hitRateSumRRF1 = hitRateSumRRF1 + 1

            #print(f"Semantic Hit Rate @1 is {SemanticHitRate}, MRR is {semanticMRR}")
            #avgMRR = (keywordMRR + semanticMRR)/2
            #print(f"Avg MRR is {avgMRR}")
        avgKeywordMRR = sum(keywordMRRList)/len(keywordMRRList)
        avgSemanticMRR = sum(semanticMRRList)/len(semanticMRRList)
        avgRRFMRR = sum(rrfMRRList)/len(rrfMRRList)

        avgHitRateKeyword3 = hitRateSumKeyword3/len(keywordMRRList)
        avgHitRateSemantic3 = hitRateSumSemantic3 /len(semanticMRRList)
        avgHitRateRRF3 = hitRateSumRRF3/len(rrfMRRList)

        avgHitRateKeyword1 = hitRateSumKeyword1 / len(keywordMRRList)
        avgHitRateSemantic1 = hitRateSumSemantic1 / len(semanticMRRList)
        avgHitRateRRF1 = hitRateSumRRF1 / len(rrfMRRList)

        print(f"Keyword MRR1 is {avgKeywordMRR:.2f} and the avg Hit Rate os {avgHitRateKeyword1}")
        print(f"Semantic MRR1 is {avgSemanticMRR:.2f} and the avg Hit Rate is {avgHitRateSemantic1}")
        print(f"RRF MRR1 is {avgRRFMRR:.2f} and the avg Hit Rate is {avgHitRateRRF1}")

        print(f"Keyword MRR3 is {avgKeywordMRR:.2f} and the avg Hit Rate os {avgHitRateKeyword3}")
        print(f"Semantic MRR3 is {avgSemanticMRR:.2f} and the avg Hit Rate is {avgHitRateSemantic3}")
        print(f"RRF MRR3 is {avgRRFMRR:.2f} and the avg Hit Rate is {avgHitRateRRF3}")

    return abstentionRows


def countAbstentionMistakes(abstentionRows: list[dict], threshold: float) -> tuple[int, int]:
    # The system answers when the top cross-encoder score is at or above the threshold,
    # and says "I don't know" when it's below.
    falseRefusals = 0  # said "I don't know", but the docs DID have the answer
    falseAnswers = 0   # answered, but the docs did NOT cover it (where hallucination happens)
    for row in abstentionRows:
        systemAnswers = row["topScore"] >= threshold
        if row["answerable"] and not systemAnswers:
            falseRefusals = falseRefusals + 1
        if not row["answerable"] and systemAnswers:
            falseAnswers = falseAnswers + 1
    return falseRefusals, falseAnswers


def reportAbstention(abstentionRows: list[dict], threshold: float = ABSTAIN_THRESHOLD) -> None:
    # Step 1: every question's top score next to its label, highest score first,
    # so you can see where the answerable and unanswerable groups sit
    print("\n--- Top cross-encoder score per question ---")
    for row in sorted(abstentionRows, key=lambda row: row["topScore"], reverse=True):
        label = "yes" if row["answerable"] else "no"
        decision = "answer" if row["topScore"] >= threshold else "I don't know"
        print(f"[{row['topScore']:7.2f}] answerable={label:<3} -> {decision:<12} | {row['question']}")

    # Step 2: count both kinds of mistake at the chosen threshold
    falseRefusals, falseAnswers = countAbstentionMistakes(abstentionRows, threshold)
    answerableCount = sum(1 for row in abstentionRows if row["answerable"])
    unanswerableCount = len(abstentionRows) - answerableCount
    print(f"\nThreshold {threshold}: false refusals {falseRefusals}/{answerableCount} answerable, "
          f"false answers {falseAnswers}/{unanswerableCount} unanswerable")

    # Step 3: try a range of thresholds to see the trade-off. Raising the threshold
    # means fewer false answers but more false refusals, and lowering it does the opposite.
    print("\n--- Threshold sweep ---")
    for sweepThreshold in [-10, -8, -6, -4, -2, 0, 2]:
        falseRefusals, falseAnswers = countAbstentionMistakes(abstentionRows, sweepThreshold)
        print(f"threshold {sweepThreshold:>4}: false refusals {falseRefusals}, false answers {falseAnswers}")


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

    semanticRetrieverResults = getSemanticSearchResults(query, docs, top_K=5)

    keywordResults = getKeywordSearchResults(query, docs, 5)

    fusedResults = reciprocalRankFusion(keywordResults, semanticRetrieverResults, docs)

    rerankWithCrossEncoder(query, fusedResults)

    abstentionRows = ScoreRetrievers(query, docs)

    reportAbstention(abstentionRows)


if __name__ == "__main__":  # only run main() when this file is executed directly (not when imported elsewhere)
    main()

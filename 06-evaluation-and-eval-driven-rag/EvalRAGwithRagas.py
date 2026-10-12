from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


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
import json
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

   # for row in semanticResults:
        # :.3f formats the score to 3 decimal places; [:80] shows just a preview of the chunk text
    #    print(f"[{row['score']:.3f}], {row['index']}, {row['source']}: {row['text'][:80]}...")

    keywordIndices, keywordScores = keywordResults
        # Let's see what we got!
    #for index, score in zip(keywordIndices[0], keywordScores[0]):
     #   print(f"Keyword results: (score: {score:.2f}): {index}")

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

def SaveOutput(docs: list[dict], top_K=3) -> list[dict]:
   client = Anthropic()
   results = []

   def record(retriever: str, question: str, reference: str, contexts: list[str]):
       # ragas expects retrieved_contexts as a list[str] (one entry per chunk), so keep it a list
       # and only join it into a single string for the prompt sent to generate().
       generatedResult = generate(client, question, "\n".join(contexts), docs)
       print(f"{retriever} Generated Text is {generatedResult}")
       results.append({"type": retriever, "user_input": question, "retrieved_contexts": contexts,
                       "response": generatedResult, "reference": reference})

   with open("EvalQuestions.csv", newline="") as f:
       for row in csv.DictReader(f, skipinitialspace=True):
           question = row["question"]
           reference = row["reference"]

           keywordIndices, keywordScores = getKeywordSearchResults(question, docs, top_K)
           record("keyword", question, reference,
                  [docs[int(i)]['source'] + ':' + docs[int(i)]['text'] for i in keywordIndices[0]])

           semanticResults = getSemanticSearchResults(question, docs, top_K)
           record("semantic", question, reference,
                  [r['source'] + ':' + r['text'] for r in semanticResults])

           rrfResults = reciprocalRankFusion([keywordIndices, keywordScores], semanticResults, docs)
           record("RRF", question, reference,
                  [r['source'] + ':' + r['text'] for r in rrfResults])

           crossEncoderResults = rerankWithCrossEncoder(question, rrfResults)
           record("cross-encoder", question, reference,
                  [r['source'] + ':' + r['text'] for r in crossEncoderResults])

   return results

def generate(client: Anthropic, query: str, contextText: str, docs: list[dict]) -> str:
   # contextText = ""
    #for val in sortedMatchesList:
    #    print(f"In Generate method, Source is {val['source']}")
    #    contextText = contextText + val['source'] + "\n" + val['text'] + "\n\n"

    prompt = (
        "Answer the question using ONLY the context below. "  # instruction: don't let the model use outside knowledge
        "If the context doesn't contain the answer, say you don't know.\n\n"  # instruction: avoid making things up
        f"Context:\n{contextText}\n\nQuestion: {query}"  # the actual retrieved text + the user's question
    )

    #print(f"The Prompt is {prompt}")
    response = client.messages.create(
        system="You are an advaned Hybrid Retrievers - RAG assistant. Use the provided prompt that contains the context and user query",
        model="claude-sonnet-5",  # Or the latest available model
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])

    for block in response.content:
        if block.type == "text":
            return block.text


def EvaluateFaithfullnessCriteria(question: str, retreivedResults: str, generatedAnswer: str) -> dict:
    prompt = f'''You are grading an answer produced by a Hybrid Retrievers - RAG system. Your task is to evaluate the Faithfullness criteria between the context tag and 
                    the answer tag. Evalate if every claim in answer tag is supported by the context tag.
                    Only judge against the provided context and do not use any outside knowledge. Length is not quality and dont go with positional bias too.
                    Use the below scoring criteria as a guide. First provide short reasoning for the score and then the score in a JSON format. Reason before you decide the score
                    <question>{question}</question>
                    <context>{retreivedResults}</context>
                    <answer>{generatedAnswer}</answer>
                    Faithfulness Scoring (1–5)
                    5: every claim is directly supported by the context.
                    4: all key claims supported; a minor detail is slightly reworded or inferred.
                    3: mostly supported, but at least one claim isn't in the context.
                    2: the main claim is unsupported, or significant parts are invented.
                    1: contradicts the context or is mostly invented.
                    If the answer only says it doesn't know and makes no other claims, score it as a 5.
                    Respond with ONLY a JSON object of this form:
                    {{"reasoning": "<claim-by-claim check>", "score": <integer 1-5>}}
                    '''

    openAIClient = OpenAI()
    response = openAIClient.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an intelligent Hybrid Retrievers - RAG grading assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return json.loads(response.choices[0].message.content)

def EvaluateRelevanceCriteria(question: str, retreivedResults: str, generatedAnswer: str) -> dict:
    prompt = f'''You are grading an answer produced by a Hybrid Retrievers - RAG system. Your task is to evaluate the Relevance criteria between the question  and 
                    the generated answer tag. Evaluate if every answer is truly relevant to the question.
                    Only judge against the provided context and do not use any outside knowledge. Length is not quality and dont go with positional bias too.
                    Use the below scoring criteria as a guide. First provide short reasoning for the score and then the score in a JSON format. Reason before you decide the score
                    <question>{question}</question>
                    <answer>{generatedAnswer}</answer>
                    Relevance Scoring (1–5)
                    5: directly and completely answers the question.
                    4: answers it, with minor gaps or unneeded extra information.   
                    3: partially answers it, or answers a closely related question.
                    2: on-topic but doesn't answer what was asked.
                    1: unrelated to the question.
                    If the answer only says it doesn't know and makes no other claims, score it as a 5.
                    Respond with ONLY a JSON object of this form:
                    {{"reasoning": "<claim-by-claim check>", "score": <integer 1-5>}}
                    '''

    openAIClient = OpenAI()
    response = openAIClient.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an intelligent Hybrid Retrievers - RAG grading assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return json.loads(response.choices[0].message.content)

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
    #print(f"Query is {query}")

    #semanticRetrieverResults = getSemanticSearchResults(query, docs, top_K=5)

    #keywordResults = getKeywordSearchResults(query, docs, 5)

    #fusedResults = reciprocalRankFusion(keywordResults, semanticRetrieverResults, docs)

    #rerankWithCrossEncoder(query, fusedResults)

    results = SaveOutput(docs, 3)

    # Judge LLM + embeddings for the ragas metrics (answer_relevancy needs the embeddings)
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # Evaluate each retriever separately so the scores can be compared side by side.
    # "type" is not a ragas column, so strip it before building the dataset.
    for retriever in ["keyword", "semantic", "RRF", "cross-encoder"]:
        # Step 1: keep only the results that were produced by this retriever
        results_for_this_retriever = [
            result for result in results
            if result["type"] == retriever
        ]

        # Step 2: drop the "type" key from each result, leaving only the columns ragas understands
        # (user_input, retrieved_contexts, response, reference)
        ragas_samples = []
        for result in results_for_this_retriever:
            sample = {}
            for column_name, value in result.items():
                if column_name != "type":
                    sample[column_name] = value
            ragas_samples.append(sample)

        ragas_result = evaluate(
            dataset=EvaluationDataset.from_list(ragas_samples),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        print(f"\n=== {retriever} ===")
        print(ragas_result.to_pandas())

if __name__ == "__main__":  # only run main() when this file is executed directly (not when imported elsewhere)
    main()

"""
Naive Vector Hybrid Retrievers - RAG — the simplest possible Hybrid Retrievers - RAG pipeline, built with no framework.

Steps:
1. Load a handful of small text docs.
2. Chunk them (here: one chunk per doc, since they're already short).
3. Embed every chunk with OpenAI's embedding model.
4. Embed the user's question the same way.
5. Rank chunks by cosine similarity to the question.
6. Stuff the top-k chunks into a prompt and ask an OpenAI chat model to answer.

Run:
    python main.py "How much does the Pro plan cost?"
"""

import os  # lets us read environment variables, like the OPENAI_API_KEY
import sys  # lets us read command-line arguments (sys.argv)
from pathlib import Path  # object-oriented filesystem paths, nicer than raw strings

import numpy as np  # numerical arrays + fast math (dot products, norms, sorting)
from anthropic import Anthropic
from dotenv import load_dotenv  # loads key=value pairs from a .env file into the environment
from openai import OpenAI  # the OpenAI client class used to call embeddings + chat endpoints
from sentence_transformers import SentenceTransformer


# __file__ is this script's own path; .resolve() makes it absolute;
# .parent goes up to this file's folder, .parent again goes up to the repo root,
# then we point into the shared "data" folder that sits next to all the numbered projects.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#EMBEDDING_MODEL = "text-embedding-3-small"  # which OpenAI model turns text into vectors
TOP_K = 2  # how many of the most relevant chunks to retrieve per question
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

    # one API call embeds ALL texts at once (cheaper/faster than one call per text)
   # response = embeddingModel.encode(texts)
   # vectors = response.tolist()
   # return np.array(vectors, dtype=np.float32)


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
        top_matches.append(match)
    return top_matches[:k]



def generate(client: Anthropic, query: str, sortedMatchesList: list[dict], docs: list[dict]) -> str:
    contextText = ""
    for val in sortedMatchesList:
        print(f"In Generate method, Source is {val['source']}")
        contextText = contextText + val['source'] + "\n" + val['text'] + "\n\n"

    prompt = (
        "Answer the question using ONLY the context below. "  # instruction: don't let the model use outside knowledge
        "If the context doesn't contain the answer, say you don't know.\n\n"  # instruction: avoid making things up
        f"Context:\n{contextText}\n\nQuestion: {query}"  # the actual retrieved text + the user's question
    )

    print(f"The Prompt is {prompt}")
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


def main():
   load_dotenv()
   try:
       anthropic_key = os.environ['ANTHROPIC_API_KEY']
   except KeyError:
       print("Anthropic key was expected but not found")
   else:
       print("Anthropic key was found")

   anthropicModel = Anthropic()

   #load docs and get text from each, store in a list of dictionaries
   docs = load_documents()
   vectorList = embed(docs)
   default_query= "what is the lowest tier in terms of pricing?"
   sortedIndicesList = retrieve(docs, vectorList, default_query, 5)
   result = generate(anthropicModel,default_query,sortedIndicesList, docs)

   print(result)
if __name__ == "__main__":  # only run main() when this file is executed directly (not when imported elsewhere)
    main()

import os  # lets us read environment variables, like the OPENAI_API_KEY
import sys  # lets us read command-line arguments (sys.argv)
from pathlib import Path  # object-oriented filesystem paths, nicer than raw strings
import bm25s
import numpy as np  # numerical arrays + fast math (dot products, norms, sorting)
from transformers.models.xlm.tokenization_xlm import lowercase_and_remove_accent

# __file__ is this script's own path; .resolve() makes it absolute;
# .parent goes up to this file's folder, .parent again goes up to the repo root,
# then we point into the shared "data" folder that sits next to all the numbered projects.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#EMBEDDING_MODEL = "text-embedding-3-small"  # which OpenAI model turns text into vectors
TOP_K = 2  # how many of the most relevant chunks to retrieve per question

def load_documents() -> list[dict]:
    docs = []  # will hold one dict per document: {"source": filename, "text": contents}
    for path in sorted(DATA_DIR.glob("*.txt")):  # find every .txt file in data/, sorted for stable order
        docs.append({"source": path.name, "text": path.read_text().strip()})  # read file text, trim whitespace, store it
    return docs  # hand back the full list of loaded documents


def main():
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

    docs = load_documents()
    corpus = []
    for doc in docs:
        corpus.append(doc['text'])

    docs_tokens = bm25s.tokenize(corpus)

    retreiver = bm25s.BM25(corpus=corpus)
    retreiver.index(docs_tokens)

    query_tokens = bm25s.tokenize(query)
    results, scores = retreiver.retrieve(query_tokens,k=5)

    #for index, (result, score) in enumerate(zip(results, scores)):
    #    print(f"Index is {index}, Result is {result}, Score is {score}")
    #print(type(retreiver.retrieve(query_tokens)))

    for index in range(5):
        print(f"Result is {results[0][index]}, Score is {scores[0][index]}")

if __name__ == "__main__":  # only run main() when this file is executed directly (not when imported elsewhere)
    main()


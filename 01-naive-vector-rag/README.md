# 01 — Naive Vector RAG

The simplest RAG pipeline: embed documents, embed the question, rank by
cosine similarity, stuff the top matches into a prompt, ask the model to
answer. No framework, no vector database server — just numpy.

## Run

```bash
git clone https://github.com/tradhikarao/hybrid-rag-evals.git
cd hybrid-rag-evals
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# put ANTHROPIC_API_KEY in .env at the repo root, if not already done
python 01-naive-vector-rag/naive_rag.py "How much does the Pro plan cost?"
```

Try a few other questions against `data/`:
- "Is my data encrypted?"
- "What happens if the API rate limit is exceeded?"
- "Do free users get live chat support?"

## What to notice

- Print the retrieved chunks *before* the answer (the script already does
  this) — this is the single most useful debugging habit in RAG. If the
  answer is wrong, check retrieval first, not the prompt.
- Try a question phrased very differently from the source text's wording
  (e.g. "Is my data protected from hackers?" instead of "encrypted"). Vector
  similarity should still find `02_security.txt` — that's the point of
  semantic search over keyword matching.
- Try a question with an exact term but low semantic "meaning" density, like
  "What does a 429 status code mean?" This is where the next project (BM25)
  starts to matter — exact-token matches aren't always what embeddings rank
  highest.

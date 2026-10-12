# RAG Learning Path

A set of small, self-contained projects that build up RAG (Retrieval-Augmented
Generation) systems from simple to complex. Each numbered folder is runnable on
its own and teaches one concept.

| # | Project | Teaches |
|---|---------|---------|
| 01 | [naive-vector-rag](01-naive-vector-rag) | Embeddings + cosine similarity + generation, no framework |
| 02 | [bm25-retriever](02-bm25-retriever) | Classic keyword (lexical) retrieval |
| 03 | [hybrid-retriever](03-hybrid-retriever) | Combining dense + sparse retrieval via score fusion |
| 04 | [reranking-rag](04-reranking-rag) | Retrieve-then-rerank with a cross-encoder |
| 05 | [framework-rag](05-framework-rag) | Rebuilding hybrid RAG with LangChain/LlamaIndex |
| 05b | [eval-basics](05b-evals) | No-code warm-up: hit rate, MRR, LLM-as-a-judge |
| 06 | [evaluation-and-eval-driven-rag](06-evaluation-and-eval-driven-rag) | Measuring whether changes actually help |

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env .env   # then fill in OPENAI_API_KEY
```

## Order

Go in order 01 → 06. Projects 01–04 are built from scratch in plain Python so
the mechanics are visible before any framework abstracts them away. 05
revisits 03's problem using a framework so the abstractions map onto code
you've already written by hand. 06 adds a way to measure whether any of this
is actually working better than the last version.

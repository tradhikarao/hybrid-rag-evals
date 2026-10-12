# 04 — Reranking RAG

Take the hybrid retriever's top-N candidates and rerank them with a
cross-encoder before generation.

## Concept

Bi-encoders (what 01/03 use) embed the query and each document
*independently*, then compare vectors — fast, but the model never sees the
query and document together. A cross-encoder takes (query, document) pairs
*jointly* as input and scores relevance directly — slower (can't
precompute), but more accurate. The standard production pattern is: cheap
retriever gets top-N candidates, cross-encoder reranks to top-k.

## TODO

- [ ] `pip install sentence-transformers` and load a small cross-encoder,
      e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- [ ] Get top-N (e.g. N=5) candidates from the hybrid retriever in 03.
- [ ] Score each (query, candidate) pair with the cross-encoder and re-sort.
- [ ] Compare the final top-k before vs. after reranking on a few queries.

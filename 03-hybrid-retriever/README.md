# 03 — Hybrid Retriever

Combine the vector retriever (01) and BM25 retriever (02) into one hybrid
retriever, then compare all three on the same queries.

## Concept

Run both retrievers over the full corpus, then merge their rankings with a
fusion method — Reciprocal Rank Fusion (RRF) is the simplest to implement:

```
score(doc) = sum(1 / (k + rank_in_list_i))  for each retriever i, k ~= 60
```

RRF works purely on rank position, so it sidesteps the problem of BM25 and
cosine-similarity scores being on incompatible scales.

## TODO

- [ ] Reuse the embedding + BM25 logic from 01/02 (import or copy as
      functions — a small amount of duplication is fine at this stage).
- [ ] Get top-N ranked lists from both retrievers for a query.
- [ ] Implement RRF to merge the two ranked lists into one.
- [ ] Print a three-way comparison (vector-only / BM25-only / hybrid) for a
      handful of test queries, especially ones picked in 02 where BM25 and
      vector search disagreed.

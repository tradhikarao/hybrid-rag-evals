# 05 — Framework Rebuild

Reimplement the hybrid RAG pipeline from 03 using a framework (LangChain or
LlamaIndex — pick one) to see how its abstractions map onto code you've
already written by hand.

## Concept

By this point the mechanics (chunk, embed, index, retrieve, fuse, prompt,
generate) should feel familiar. Frameworks package these steps into
reusable classes (`Retriever`, `VectorStore`, `EnsembleRetriever` /
`QueryFusionRetriever`, etc.). Rebuilding the same pipeline in a framework
should feel like recognizing the pieces, not learning new concepts.

## TODO

- [ ] Pick LangChain or LlamaIndex.
- [ ] Reimplement 03's corpus + hybrid retrieval using the framework's
      built-in vector store + BM25 retriever + ensemble/fusion retriever.
- [ ] Run the same test queries as 03 and confirm comparable results.
- [ ] Note anywhere the framework's default behavior differs from the
      from-scratch version (chunking defaults, fusion weights, etc.) — those
      differences are usually where framework "magic" causes confusion later.

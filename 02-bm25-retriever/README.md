# 02 — BM25 Retriever

Same corpus as 01, but retrieve with classic keyword (lexical) search instead
of embeddings, using `rank_bm25`.

## Concept

BM25 scores documents by term frequency and inverse document frequency — it
rewards exact token overlap between the query and the document, with no
notion of "meaning." This makes it strong on exact names, codes, and jargon
(e.g. "429 status code", "AES-256") and weak on paraphrase (e.g. "Is my data
protected?" when the doc says "encrypted").

## TODO

- [ ] Load `data/*.txt`, tokenize (simple `.lower().split()` is fine to start).
- [ ] Build a `BM25Okapi` index over the tokenized corpus.
- [ ] Given a query, tokenize it and call `bm25.get_scores(...)`.
- [ ] Print top-k chunks with scores, same format as 01, so results are
      directly comparable.
- [ ] Run the *same* questions used in 01 and note where BM25 and vector
      search disagree — that disagreement is the lesson.

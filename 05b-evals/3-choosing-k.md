# Lesson 3 — What is "k", and how should it be set?

Your code uses "k" for three different things. Here's each one and how to set it.

## 1. `TOP_K`: how many chunks to retrieve

This is the main one: how many documents or chunks you retrieve and paste
into the LLM prompt. It appears as `TOP_K` in your files, as `k=` in
`retreiver.retrieve(query_tokens, k=k)`, and as `search_kwargs={"k": TOP_K}`
in LangChain.

It's a trade-off:

- **Too small (e.g. 1):** if the right doc is ranked 2nd, the LLM never sees
  it and can't answer.
- **Too large:** more cost, and more irrelevant text in the prompt. That
  distracts the LLM and makes hallucinated or mixed-up answers more likely,
  which lowers faithfulness.

**How to set it: measure it with your 06 eval.**

1. Compute hit@1, hit@2, hit@3. Find the **smallest k where hit rate stops
   improving much**. For example, if hit@1 = 0.6, hit@2 = 0.9 and hit@3 = 0.9,
   choose k = 2.
2. Check generation quality with the LLM judge at that k and one above it.
   Sometimes an extra chunk helps answers that need two docs. Sometimes it
   lowers faithfulness.
3. Choose the k with the best answer scores.

Typical real-world values are 3–10 chunks. With 5 small files in this repo,
expect **2 or 3** to be right. `TOP_K = 5` sends everything, which is really
no retrieval at all.

## 2. The k in hit@k and MRR@k: where the metric stops counting

This k isn't a setting of your retriever. It's the **cutoff you evaluate at**.
It should **match `TOP_K`**, since that's what the LLM actually sees. It's
also why you report several values (hit@1, 2, 3): they're what you use to
*choose* `TOP_K` in step 1 above.

## 3. The RRF constant (`k=60` / `c=5`): how fusion weights ranks

In `reciprocalRankFusion(..., k=...)` and LangChain's
`EnsembleRetriever(c=...)`, each doc gets `1 / (k + rank)` from each
retriever. This constant has **nothing to do with how many docs you get
back**. It controls **how much being ranked first matters** compared to being
ranked lower:

| Constant | Rank 1 | Rank 2 | Rank 5 | Effect |
|---|---|---|---|---|
| 5 | 1/6 = 0.167 | 1/7 = 0.143 | 1/10 = 0.100 | Top ranks count much more: a retriever's #1 pick carries a lot of weight |
| 60 | 1/61 = 0.0164 | 1/62 = 0.0161 | 1/65 = 0.0154 | Ranks count almost equally: docs that **both** retrievers returned win |

**How to set it:** start with **60**. It's the value from the original RRF
paper and works well without tuning. With only 5 docs, a small value like 5
makes fusion more sensitive to each retriever's top pick. Module 03 used 5.
This is a good "Step 6" experiment in 06: run the eval with 5 and with 60, and
keep whichever gives better hit@k.

## Related: `top_n` in reranking

In `rerankWithCrossEncoder(..., top_n=20)`, this is how many fused candidates
the cross-encoder rescores before you keep the best `TOP_K`. Set it **larger
than `TOP_K`** so the reranker can promote a doc from lower down, but not so
large that it slows down (cross-encoders are slow). Around 20–50 is typical.
With 5 docs, just rerank all 5.

## Summary

| Name | What it controls | How to set it |
|---|---|---|
| `TOP_K` | chunks sent to the LLM | tune with hit@k + judge scores |
| hit@k / MRR@k cutoff | where the metric stops counting | match `TOP_K` |
| RRF constant | how much top ranks dominate fusion | start at 60, test 5 vs 60 |
| `top_n` (rerank) | candidates the cross-encoder rescores | larger than `TOP_K`; all 5 here |

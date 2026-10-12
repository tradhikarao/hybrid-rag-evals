# 06 — Evaluation-Driven RAG

Introduce a small evaluation set and a way to measure whether a change
(chunk size, k, reranking, fusion weights) actually helps, instead of
eyeballing outputs.

> **New to evals?** Do [05b-eval-basics](../05b-evals) first. It
> explains hit rate, MRR and LLM-as-a-judge with worked examples and paper
> exercises. The steps below assume you've done it.

## Concept

- **Retrieval metrics**: hit-rate@k (did the correct source doc appear in
  the top-k?) and MRR (mean reciprocal rank). Both are computable from a
  hand-labeled (question -> expected source file) set.
- **Generation metrics**: harder to measure automatically; a lightweight
  approach is "LLM-as-judge": ask a model to score the generated answer's
  faithfulness to the retrieved context and relevance to the question.

## Step-by-step

### Step 1: Write the eval set (~10 questions)

Create a data file in this folder (JSON or CSV) where each entry has:

- `question`: what a user would type
- `expected_sources`: a **list** of file names that correctly answer it
  (usually one; two when the answer genuinely lives in two files)

Example entries (write the rest yourself):

| question | expected_sources |
|---|---|
| Can I get a cheaper price if I pay for a whole year? | `01_pricing.txt` |
| What happens if I send too many API requests? | `04_api_limits.txt` |

Guidelines for a useful set:

- **Cover every file**, about 2 questions per file.
- **Mix question styles** so the set can tell retrievers apart:
  - *keyword-style* questions that reuse words from the doc ("What is the
    HTTP 429 rate limit?"). BM25 should do well here.
  - *paraphrased* questions that use **none** of the doc's words ("Do I get
    money back if the service goes down?" for `03_uptime_sla.txt`). Semantic
    search should do well here.
  - one or two **tricky** questions: "maintenance" appears in both
    `03_uptime_sla.txt` and the stray `email maintenance` line in
    `04_api_limits.txt`, and "email support" appears in both `01_pricing.txt`
    and `05_support.txt`.
- Write questions **the way a customer would**, not by copying sentences out
  of the doc. Copying makes every retriever look perfect.
- Freeze it. Once you start comparing retrievers, don't edit the set, or
  earlier numbers are no longer comparable.

### Step 2: Compute hit-rate@k and MRR

Write one scoring function that doesn't care which retriever produced the list:

- **Input:** the ranked list of source file names for one question (best
  first) + that question's `expected_sources`.
- **Find the rank:** the 1-based position of the first file that's in
  `expected_sources`. None if not found.
- **Output:** hit@1, hit@2, hit@3 (1 or 0 each) and the reciprocal rank
  (1 / rank, or 0 if not found).

Then loop over all questions and **average** each column. That average is
hit-rate@k and MRR.

**Use k = 1, 2, 3, not 5.** There are only 5 files, so every retriever has
hit-rate@5 = 100%.

**Sanity-check the scorer by hand first:** feed it the exercise from
`05b-eval-basics/1-retrieval-metrics.md` (ranks 3, 1, 5, 2, 1) and confirm you
get hit@1 = 0.40 and MRR ≈ 0.607.

### Step 3: Plug in each retriever

Each retriever needs to produce **a ranked list of source file names** for a
question. What you have to convert:

| Retriever | Where | What it returns | Turn it into file names by… |
|---|---|---|---|
| Semantic | `rag/retrieval.py` → `SemanticRetriever.search()` | list of `Hit`s, best first | reading `.source` from each |
| BM25 | `rag/retrieval.py` → `KeywordRetriever.search()` | list of `Hit`s, best first | reading `.source` from each |
| Hybrid (RRF) | `rag/retrieval.py` → `reciprocal_rank_fusion()` | fused list of `Hit`s | reading `.source` from each |
| Reranked | `rag/retrieval.py` → `CrossEncoderReranker.rerank()` | reranked list of `Hit`s | reading `.source` from each |
| LangChain hybrid | `05-framework-rag` → `hybrid_retriever.invoke()` | list of `Document`s | `doc.metadata['source']` |

Tips:

- **Make every retriever return the full ranking** (all 5 files) during
  evaluation, so MRR can see ranks beyond `TOP_K`.
- **Load models and build indexes once**, outside the question loop.
  Re-loading the embedding model for every question is very slow.
- Remove or silence the debug `print`s in those functions first, or the
  output will be unreadable.
- The folder names start with digits, so you can't `import` them directly. The
  simplest beginner approach is to copy the functions you need into a file in
  this folder.

### Step 4: Summary table (the payoff)

Print one row per retriever:

| Retriever | hit@1 | hit@2 | hit@3 | MRR |
|---|---|---|---|---|
| Naive vector | | | | |
| BM25 | | | | |
| Hybrid (RRF) | | | | |
| Reranked | | | | |
| LangChain hybrid | | | | |

Then also print **per-question ranks** for each retriever. The averages tell you
*which* is best. The per-question view tells you *why* (for example "BM25
misses every paraphrased question").

With 10 questions, one question changes a score by 0.1. Treat differences
smaller than that as a tie.

### Step 5: LLM-as-judge for generated answers

Follow the build plan at the end of `05b-eval-basics/2-llm-as-judge.md`:

1. For each eval question: retrieve with your best retriever, generate an
   answer (you can reuse `generate()` from `rag/generation.py`,
   which already calls Claude), and save question + context + answer.
2. Write the judge prompt yourself: one criterion per call, anchored 1–5
   rubric, inputs in tags, reasoning before score, fixed output format.
3. **Calibrate before trusting it:** hand-score ~10 answers (including a few
   deliberately bad ones), run the judge on them, and count how often it lands
   within ±1 of you.
4. Report average faithfulness and relevance, and read the reasoning for
   every score ≤ 3.

### Step 6: Use it (the "eval-driven" part)

Pick **one** change, re-run the whole eval, and compare against the table
from Step 4. Change only one thing at a time, or you won't know which change
caused the difference. Ideas:

- `TOP_K` 1 vs 2 vs 3 (generation scores)
- RRF constant `k = 5` vs `60`
- Reranking top 3 vs all 5
- Adding a stemmer to BM25

Keep a short log of each change and its before/after numbers in this README.

## Checklist

- [ ] Finish the exercises in `05b-eval-basics`.
- [ ] Write ~10 question/expected-source pairs covering the `data/` corpus.
- [ ] Hand-check the scoring function against the 05b exercise.
- [ ] Compute hit-rate@1/2/3 and MRR for each retriever built so far (naive
      vector, BM25, hybrid, reranked, LangChain).
- [ ] Produce one summary table + per-question ranks.
- [ ] Build and calibrate an LLM-as-judge scorer (1–5 on faithfulness +
      relevance).
- [ ] Run at least one eval-driven experiment and record before/after.

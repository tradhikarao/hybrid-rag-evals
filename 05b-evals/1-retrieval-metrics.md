# Lesson 1 — Hit rate@k and MRR

Both metrics need just two ingredients:

1. **An eval set:** questions, each labeled with the file that *should* be
   retrieved (the "expected source").
2. **A ranked list** from your retriever for each question, best match first.
   Every retriever you've built already produces one.

## Step 0: find the rank

For each question, walk down the ranked list and find where the expected file
first appears. That position, counting from **1**, is its **rank**.

```
Question:  "Is there a discount for paying yearly?"
Expected:  01_pricing.txt
Ranked:    1. 04_api_limits.txt
           2. 01_pricing.txt   <- here, so rank = 2
           3. 05_support.txt
```

If the expected file isn't in the list at all, it has **no rank** (a miss).

Everything below is arithmetic on these ranks.

## Hit rate@k: "Was the right doc in the top k?"

For one question:

- **hit = 1** if rank ≤ k
- **hit = 0** otherwise (including misses)

For the whole eval set, **hit rate@k = the average of those 1s and 0s**,
which is the fraction of questions where the retriever "found it" within the top k.

Why "@k"? Because in RAG only the top k chunks are pasted into the LLM prompt.
If `TOP_K = 2`, a doc at rank 3 might as well not exist. So choose k to match
how many chunks you actually send to the LLM.

## MRR: "How high up was it, on average?"

Hit rate treats rank 1 and rank k the same. MRR rewards putting the right doc
**at the top**.

For one question, the **reciprocal rank (RR)** is:

- **RR = 1 / rank**, so rank 1 gives 1.0, rank 2 gives 0.5, rank 3 gives 0.333, rank 4 gives 0.25
- **RR = 0** for a miss. With **MRR@k**, anything ranked below k also counts as 0.

**MRR = the average of the RRs** over all questions (Mean Reciprocal Rank).

Why 1/rank? It drops fast: moving from rank 2 to rank 1 doubles the score,
but moving from rank 5 to rank 4 barely matters. That matches reality: LLMs pay
the most attention to what's at the top of the context.

## Worked example

Four questions, one retriever:

| Question | Expected | Rank | hit@1 | hit@3 | RR | RR@3 |
|---|---|---|---|---|---|---|
| Q1 | 01_pricing.txt | 1 | 1 | 1 | 1.0 | 1.0 |
| Q2 | 03_uptime_sla.txt | 2 | 0 | 1 | 0.5 | 0.5 |
| Q3 | 02_security.txt | 4 | 0 | 0 | 0.25 | **0** (rank 4 > 3) |
| Q4 | 04_api_limits.txt | 1 | 1 | 1 | 1.0 | 1.0 |
| **Average** | | | **0.50** | **0.75** | **0.6875** | **0.625** |

Read it as: "half the time the right doc is #1, three quarters of the time
it's in the top 3, and on average it sits at roughly rank 1.5 (1 / 0.6875 ≈ 1.45)."

## Why keep hit@k if MRR has more detail?

MRR does have more detail, but it answers a different question from the one
your RAG system actually faces.

**Your LLM only sees the top k chunks.** With `TOP_K = 2`, the right doc at
rank 2 is just as useful as at rank 1, and at rank 3 it's useless. Hit@k
measures exactly that cutoff: "what fraction of questions give the LLM the
information it needs?"

MRR averages that cutoff away, so two very different retrievers can get the
same MRR:

| | Q1 | Q2 | Q3 | Q4 | MRR@3 | hit@2 |
|---|---|---|---|---|---|---|
| Retriever A | rank 1 | rank 1 | not in top 3 | not in top 3 | (1 + 1 + 0 + 0) / 4 = **0.50** | **0.50** |
| Retriever B | rank 2 | rank 2 | rank 2 | rank 2 | (0.5 × 4) / 4 = **0.50** | **1.00** |

MRR calls them a tie. For RAG with `TOP_K = 2`, **B is clearly better**:
every question gets the right document into the prompt. With A, half the
questions can't be answered at all. Only hit@k shows that.

The two metrics answer different questions:

- **Hit@k: "Can the LLM answer?"** This is the main metric for RAG. It's also
  easy to explain: "80% of questions retrieve the right doc."
- **MRR: "How close to the top is it?"** This matters because LLMs pay the most
  attention to the start of the context. It's also useful as a tiebreaker, and
  when choosing what k to use.

Look at both. If they disagree, like A and B above, hit@k at your real
`TOP_K` usually decides. (How to choose `TOP_K` itself is in
[3-choosing-k.md](3-choosing-k.md).)

## Two traps specific to this repo

1. **Your corpus has only 5 files.** If k = 5, every retriever returns every
   file, so **hit rate@5 is always 100%** and tells you nothing. Use
   **k = 1, 2 and 3**. MRR over the full list is still meaningful.
2. **Some questions have two right answers.** "Do Pro users get email support?"
   is answered by both `01_pricing.txt` and `05_support.txt`. Either (a) avoid
   ambiguous questions, or (b) allow a *list* of expected sources and use the
   rank of whichever one appears first. Option (b) is more realistic.

## Your turn (pencil and paper)

Five questions. Here is the rank of the expected file for each:

| Question | Rank |hit rate@1 | hit rate@3 | MRR |  MRR@3
|---|---|---|---|---|---|
| Q1 | 3 | 0 | 1 | 1/3 | 1/3
| Q2 | 1 | 1 | 1 | 1   | 1
| Q3 | 5 | 0 | 0 | 1/5 | 0
| Q4 | 2 | 0 | 1 | 1/2 | 1/2
| Q5 | 1 | 1 | 1 | 1   | 1


Compute: **hit rate@1, hit rate@2, hit rate@3, MRR (full list) and MRR@3.**

Then answer: if a change moved Q1 from rank 3 to rank 1, which metrics change,
and by how much?

---

<details>
<summary>Answers (try first!)</summary>

- hit@1 = Q2, Q5, so 2/5 = **0.40**
- hit@2 = Q2, Q4, Q5, so 3/5 = **0.60**
- hit@3 = Q1, Q2, Q4, Q5, so 4/5 = **0.80**
- MRR = (1/3 + 1 + 1/5 + 1/2 + 1) / 5 = 3.033 / 5 = **0.607**
- MRR@3 = (1/3 + 1 + 0 + 1/2 + 1) / 5 = 2.833 / 5 = **0.567**

Moving Q1 from rank 3 to 1: hit@1 goes 0.40 → 0.60, hit@2 goes 0.60 → 0.80,
hit@3 is unchanged (it was already a hit), and MRR rises by (1 − 1/3) / 5 = 0.133.
Notice that **one question moves the metrics by 0.13–0.20**. With a small eval
set, don't over-read small differences between retrievers.

</details>

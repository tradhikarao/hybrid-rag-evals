# 05b — Eval Basics (do this before 06)

A no-code warm-up for evaluation. You'll learn the three ideas that 06 is
built on, practice them with pencil and paper, and only then write code in 06.

## Why evals?

Up to now you've judged each retriever by running one query and reading the
output. That's "eyeballing", and it has two problems:

1. **One query isn't enough.** A change that helps the discount question might
   break three other questions you never tried.
2. **You can't compare fairly.** "The hybrid looked better" isn't a number you
   can put next to "the reranker looked better".

An **eval** fixes both: a fixed list of questions with known right answers,
plus a scoring rule that turns each retriever's output into a number.

## The two things you can measure in RAG

| Stage | Question it answers | How you measure it | Lesson |
|---|---|---|---|
| **Retrieval** | Did we fetch the right document? | Hit rate@k and MRR: pure arithmetic, no AI needed | [1-retrieval-metrics.md](1-retrieval-metrics.md) |
| **Generation** | Is the final answer correct and grounded? | LLM-as-a-judge: a second model grades the answer | [2-llm-as-judge.md](2-llm-as-judge.md) |

Measure retrieval first. If the right document never reaches the LLM, no
prompt can save the answer.

## How to work through this folder

1. Read `1-retrieval-metrics.md` and do the exercise on paper. Check your
   answers at the bottom.
2. Read `2-llm-as-judge.md`, and score the example answers yourself *before*
   you look at the suggested scores.
3. Read `3-choosing-k.md`: the three different "k"s in your code and how to
   set each one.
4. Go to `06-evaluation-and-eval-driven-rag/README.md` and follow the
   step-by-step build.

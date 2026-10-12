# Lesson 2 — LLM-as-a-judge

## The problem it solves

Retrieval metrics only ask "did the right *document* come back?" They can't
tell you whether the LLM then wrote a good *answer*. The LLM can:

- get the right document and still **make things up** ("20% student discount"),
- give a true statement that **doesn't answer the question**, or
- be right but so vague that it's useless.

Answers are free text, so you can't check them with `==`. You could grade them
by hand, but that doesn't scale and you won't redo it after every change.
**LLM-as-a-judge** means: write down how *you* would grade an answer, then
have a second LLM call apply those rules for you.

## The two scores in your 06 TODO

Grade each on its own. Never combine them into one "quality" score, because
then you can't tell *what* went wrong.

| Metric | Question the judge answers | What the judge needs to see |
|---|---|---|
| **Faithfulness** (groundedness) | Is every claim in the answer supported by the retrieved context? | the **context** + the **answer** |
| **Relevance** | Does the answer actually address the question that was asked? | the **question** + the **answer** |

Key idea: faithfulness is judged **only against the context, not against the
judge's own knowledge.** If the context says 15% and the answer says 15%, it's
faithful, even if the real-world company offers something else. You're testing
"did the model stick to its sources?"

The two scores fail independently:

| | Faithful | Unfaithful |
|---|---|---|
| **Relevant** | the goal | hallucination on-topic (dangerous, looks correct) |
| **Irrelevant** | true but off-topic | worst case |

## Step 1: write the rubric (the most important part)

A judge is only as good as its rubric. "Rate faithfulness 1–5" on its own gives
random-ish numbers, because the model has to guess what a 3 means.
**Anchor every score with a description:**

**Faithfulness (1–5)**
- **5**: every claim is directly supported by the context.
- **4**: all key claims supported; a minor detail is slightly reworded or inferred.
- **3**: mostly supported, but at least one claim isn't in the context.
- **2**: the main claim is unsupported, or significant parts are invented.
- **1**: contradicts the context or is mostly invented.

**Relevance (1–5)**
- **5**: directly and completely answers the question.
- **4**: answers it, with minor gaps or unneeded extra information.
- **3**: partially answers it, or answers a closely related question.
- **2**: on-topic but doesn't answer what was asked.
- **1**: unrelated to the question.

> Beginner tip: many teams start with **pass/fail** instead of 1–5. It's
> easier to agree on "is every claim supported: yes/no?" than on "is this a 3
> or a 4?". Your TODO says 1–5, which is fine; the anchors above are what make it work.

## How faithfulness is decided: claim by claim

Don't judge the answer as a whole. **Break it into individual claims** and
check each one against the retrieved context on its own. This works the same
whether you grade by hand or with an LLM judge.

**1. Split the answer into claims**, meaning the smallest statements that can
be true or false. Take this answer to the discount question: *"Yes, annual
billing on Pro or Team saves 15%, and Enterprise customers can negotiate
custom pricing."*

| # | Claim | Label |
|---|---|---|
| 1 | A discount exists (the "Yes") | ✅ supported |
| 2 | It's for annual billing | ✅ supported |
| 3 | It applies to Pro or Team | ✅ supported |
| 4 | It's 15% | ✅ supported |
| 5 | Enterprise customers can negotiate custom pricing | ❌ not in context |

**2. Label each claim** with one of three labels:

- **Supported:** the context states it, or it follows directly. Rewording is
  fine ("saves 15%" = "15% discount").
- **Not in context:** the context doesn't say it. This counts against the
  answer **even if it's true in real life**. You're testing "did the model
  stick to its sources?", not "is this a fact?"
- **Contradicted:** the context says something different (e.g. "20%" when it
  says 15%). This is the worst label, because it's the kind of confident wrong
  answer a user would believe.

**3. Turn the labels into a score.** Two common ways:

- **Fraction supported:** 4/5 = 80% here. Automated eval tools such as RAGAS
  do this.
- **Map to your rubric:** "at least one claim not in context" is a **3** in
  the rubric above. Any contradicted claim should cap the score at 1–2.

**Rules to decide in advance** (write them into your judge prompt):

- **Ignore filler.** "Great question!" and "I hope this helps" aren't claims.
- **"The context doesn't mention X" is faithful.** Saying "I don't know" when
  the context doesn't know is the behavior you want.
- **Simple calculations are borderline.** "Pro costs about $101.90 a year"
  (9.99 × 12 × 0.85) isn't written anywhere, but it follows from the numbers.
  Decide once whether calculations like this count as supported, and state
  that rule.
- **Judge only against the context that was actually retrieved**, not the
  whole `data/` folder. If retrieval brought back the wrong doc and the LLM
  answered from its own knowledge, the answer is unfaithful even if it's
  correct. That low score points you to a retrieval problem.

## Step 2: the anatomy of a judge prompt

You'll write this prompt yourself. It should have these parts, in order:

1. **Role and task:** "You are grading an answer produced by a RAG system…"
2. **The one criterion being judged**, with its full rubric. Use **one
   criterion per judge call.** Asking for faithfulness and relevance in the
   same call makes each score less reliable.
3. **Ground rules**, for example "Judge only against the provided context. Do
   not use outside knowledge. Length is not quality."
4. **The inputs, clearly separated**, for example wrapped in tags like
   `<question>…</question>`, `<context>…</context>`, `<answer>…</answer>`, so the
   judge never confuses the answer with the instructions.
5. **Output format:** ask for a short **reasoning first, then the score**, in
   a fixed format such as JSON with `reasoning` and `score` fields. Reasoning
   before the score makes the judge think before committing, and gives *you* an
   explanation to read when you disagree. A fixed format means your code can
   read the score reliably. For faithfulness, the reasoning should be the
   claim-by-claim list from the section above: ask the judge to list the
   claims, label each one, and only then give the score.

Also set **temperature to 0** (or as low as the API allows) so the same input
gives (nearly) the same grade each run.

## Step 3: practise being the judge yourself

Use the rubric above. The question is `"Is there a discount offered for any of
these pricing models?"` and the context is `01_pricing.txt`. **Score each
answer for faithfulness and relevance before opening the answers.**

- **A:** "Yes. Annual billing for the Pro or Team plan gives a 15% discount compared to monthly billing." 
- **B:** "Yes, students get 20% off any plan."
- **C:** "The Pro plan costs $9.99 per month and includes 2TB of storage."
- **D:** "Yes, annual billing on Pro or Team saves 15%, and Enterprise customers can negotiate custom pricing."
- **E:** "Nimbus offers several plans with different features and prices, and depending on your needs and billing preferences there may be options that could reduce your overall cost."

Question | Relevance | Faithfulness
A | 5 | 5
B | 5 | 0
C | 1 | 5
D | 5 | 2
E | 3 | 0

<details>
<summary>Suggested scores</summary>

| | Faithfulness | Relevance | Why |
|---|---|---|---|
| A | 5 | 5 | Every claim is in the context; directly answers the question. |
| B | 1 | 4 | On-topic, but the discount is invented. Shows why faithfulness matters. |
| C | 5 | 2 | True per the context, but says nothing about discounts. |
| D | 3 | 5 | The 15% part is supported; the Enterprise pricing claim isn't in this context (Enterprise appears only in `05_support.txt`, and says nothing about pricing). |
| E | 4–5 | 2 | Vague enough that nothing is false, but it dodges the question. Watch for this: judges often reward long, hedged answers too generously. |

If your scores differed by a point, that's normal. That disagreement is
exactly what the next step is about.

</details>

## Step 4: check the judge against yourself (calibration)

A judge is itself a system that can be wrong, so **evaluate your evaluator**:

1. Collect ~10 real answers from your RAG pipeline, and **deliberately add a
   few bad ones** (like B, C and E above) so you know the judge sees failures.
2. Score them yourself first and write your scores down.
3. Run the judge on the same answers.
4. Compare. Count how often the judge is within ±1 point of you (or exactly
   matches, for pass/fail).
5. Where you disagree, read the judge's reasoning. Usually the fix is a
   clearer anchor or a new ground rule in the rubric, not a different model.

Only trust the judge's averages once it mostly agrees with you.

## Common judge mistakes to watch for

- **Leniency:** everything scores 4–5. Fix with stricter anchors and bad examples.
- **Verbosity bias:** longer answers score higher. Add "length is not quality" (answer E).
- **Using outside knowledge:** the judge "knows" a fact and calls an unsupported claim faithful. Restate "only the context counts".
- **Self-preference:** a model tends to like text written by itself. If you can, use a different (ideally stronger) model as judge than the one generating answers.
- **Noise:** re-running can change a score by a point. Low temperature helps; with a tiny eval set, don't treat a 0.2 difference in the average as real.
- **Parsing failures:** the judge sometimes breaks the output format. Your code should notice, and not silently count it as 0.

## Which model to use as the judge

You need an LLM API for this part (sentence-transformers can't do it). You
already have the `anthropic` SDK in your venv, so a Claude model is an easy
choice. A small, cheap model is fine to start with. If calibration shows it
disagrees with you a lot, try a stronger model before rewriting everything.

## Your own judge, or a framework like RAGAS?

It's mostly not a choice between the two: **RAGAS faithfulness is itself an
LLM judge.** It runs the claim-by-claim method above, using an LLM for each
step (split into claims, check each against the context, score = supported ÷
total). So the real choice is **writing your own judge prompt vs. using
RAGAS's pre-written one.** Either way, the quality mostly depends on which LLM
does the judging.

| | Your own judge | RAGAS |
|---|---|---|
| **Learning value** | High: you see and control every step | Low: the prompt is hidden inside the library |
| **Setup effort** | Write a prompt and parse the score | Install and connect your data; fast once working |
| **Control** | Your rubric and your rules (e.g. whether calculations count as supported) | Its rules; changing them means digging into the library |
| **Output** | Whatever you design (your 1–5 rubric) | A 0–1 fraction |
| **Other metrics** | Only what you write | Answer relevancy, context precision/recall and more, ready-made |
| **Beginner problems** | Prompt design, parsing failures | Its API has changed a lot between versions, and it defaults to OpenAI models, so you'd need to configure Claude and your sentence-transformers embeddings instead. Also check that your installed version supports your Python version. |

**Recommended order:**

1. **Build your own judge for 06.** You'll understand exactly what a
   faithfulness score means, and you'll recognize what RAGAS does later.
2. **Then run RAGAS on the same saved answers** as a cross-check. If your judge
   says 5/5 and RAGAS says 0.4 on the same answer, read both explanations. That
   disagreement usually reveals a loose rule in one of them.
3. **Check both against your own hand scores.** Neither is automatically
   right. RAGAS isn't more trustworthy just because it's a framework, and both
   depend on the LLM they use.

**A third option:** small **non-LLM models** called NLI (natural language
inference) or hallucination-detection models. They run locally through
sentence-transformers or Hugging Face, like your cross-encoder. They take a
(context, claim) pair and output how likely it is that the context supports
the claim. They're free and fast, but weaker at splitting an answer into claims
and at reasoning over a few sentences together. They're a useful cheap first
filter, but not a replacement for your main judge.

## Plain-language build plan (you write the code)

1. For each eval question, run your RAG pipeline and save: the question, the
   retrieved context text, and the generated answer.
2. Write a function that takes (criterion, rubric, question, context, answer),
   builds the judge prompt from Step 2, calls the model, and returns the
   reasoning and the score.
3. Call it twice per answer: once for faithfulness, once for relevance.
4. Store the results in a table with columns: question, answer, faithfulness,
   relevance, and both reasonings.
5. Print the average of each score, and **read the reasoning for every low
   score**. That's where you learn what to fix in your pipeline.

When you've written it, ask for a review. It's worth checking the prompt
and the score parsing before you trust the numbers.

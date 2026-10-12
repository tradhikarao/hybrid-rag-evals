"""LLM-as-a-judge: score a generated answer 1-5 on one criterion."""

import json
from functools import lru_cache

from openai import OpenAI

from rag.config import JUDGE_MODEL

RUBRICS = {
    "faithfulness": (
        "Evaluate whether every claim in the answer is supported by the context.",
        """5: every claim is directly supported by the context.
4: all key claims supported; a minor detail is slightly reworded or inferred.
3: mostly supported, but at least one claim isn't in the context.
2: the main claim is unsupported, or significant parts are invented.
1: contradicts the context or is mostly invented.""",
    ),
    "relevance": (
        "Evaluate whether the answer is truly relevant to the question.",
        """5: directly and completely answers the question.
4: answers it, with minor gaps or unneeded extra information.
3: partially answers it, or answers a closely related question.
2: on-topic but doesn't answer what was asked.
1: unrelated to the question.""",
    ),
}

PROMPT = """You are grading an answer produced by a RAG system on one criterion: {criterion}.
{task} Judge only against the material provided and use no outside knowledge.
Length is not quality, and don't let position bias sway you.
First give short reasoning, then the score.

<question>{question}</question>
<context>{context}</context>
<answer>{answer}</answer>

{criterion} scoring (1-5):
{rubric}
If the answer only says it doesn't know and makes no other claims, score it 5.
Respond with ONLY a JSON object of this form:
{{"reasoning": "<claim-by-claim check>", "score": <integer 1-5>}}"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def judge(criterion: str, question: str, context: str, answer: str) -> dict:
    task, rubric = RUBRICS[criterion]
    prompt = PROMPT.format(criterion=criterion.capitalize(), task=task, rubric=rubric,
                           question=question, context=context, answer=answer)
    response = _client().chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an intelligent RAG grading assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)

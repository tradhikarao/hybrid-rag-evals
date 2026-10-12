from functools import lru_cache

from anthropic import Anthropic

from rag.config import GENERATION_MODEL
from rag.documents import Hit

SYSTEM_PROMPT = "You are a helpful RAG assistant. Answer from the context provided in the user's message."


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    return Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def format_chunk(hit: Hit) -> str:
    return f"{hit.source}: {hit.text}"


def generate(query: str, hits: list[Hit]) -> str:
    """Answer `query` using only the retrieved `hits`."""
    context = "\n\n".join(format_chunk(hit) for hit in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    response = _client().messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")

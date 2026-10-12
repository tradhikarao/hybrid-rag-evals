"""Shared settings for every project in the learning path."""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_QUESTIONS_PATH = ROOT / "06-evaluation-and-eval-driven-rag" / "EvalQuestions.csv"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # local sentence-transformers model, no API key needed
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
GENERATION_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 5  # chunks each retriever returns per question
RERANK_TOP_N = 20  # fused candidates that get the (expensive) cross-encoder pass
RRF_K = 60  # the constant in 1 / (RRF_K + rank)
EVAL_TOP_K = 3  # retrieval depth when scoring retrievers (hit@1..3 and MRR)
# If the best cross-encoder score is below this, the system should say "I don't know".
# A starting guess only; use the threshold sweep from the retrieval eval to choose it.
ABSTAIN_THRESHOLD = -2.0

DEFAULT_QUERY = "Is there a discount offered for any of these pricing models?"

load_dotenv(ROOT / ".env")  # makes ANTHROPIC_API_KEY / OPENAI_API_KEY visible to the SDKs

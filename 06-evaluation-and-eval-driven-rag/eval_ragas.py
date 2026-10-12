"""Score each retriever's generated answers with ragas (faithfulness, relevancy, context precision/recall).

Run:
    python 06-evaluation-and-eval-driven-rag/eval_ragas.py
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from rag.config import EVAL_TOP_K, JUDGE_EMBEDDING_MODEL, JUDGE_MODEL
from rag.documents import load_documents
from rag.evaluation.generations import collect_generations
from rag.evaluation.questions import load_questions
from rag.retrieval import RetrieverSuite

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def main():
    generations = collect_generations(RetrieverSuite(load_documents()), load_questions(), EVAL_TOP_K)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBEDDING_MODEL))  # for answer_relevancy

    # Evaluate each retriever separately so the scores can be compared side by side.
    for retriever in dict.fromkeys(g.retriever for g in generations):
        samples = [
            {
                "user_input": g.question,
                "retrieved_contexts": g.contexts,
                "response": g.answer,
                "reference": g.reference,
            }
            for g in generations
            if g.retriever == retriever
        ]
        result = evaluate(
            dataset=EvaluationDataset.from_list(samples),
            metrics=METRICS,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        print(f"\n=== {retriever} ===")
        print(result.to_pandas())


if __name__ == "__main__":
    main()

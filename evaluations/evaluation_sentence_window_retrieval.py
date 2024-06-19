import json
from typing import Tuple, List

from architectures.sentence_window_retrieval import indexing, rag_sentence_window_retrieval
from architectures.basic_rag import basic_rag


def read_question_answers(base_path: str) -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers


def main():

    base_path = "../datasets/ARAGOG/"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 128
    top_k = 3

    questions, answers = read_question_answers(base_path)
    doc_store, doc_chunk_store = indexing(embedding_model, chunk_size, base_path)

    q = questions[0]

    rag_window_retrieval = rag_sentence_window_retrieval(doc_store, doc_chunk_store, embedding_model, top_k)
    rag_window_retrieval.run(
        data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
    )

    rag = basic_rag(doc_store, embedding_model, top_k)
    rag.run(
        data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
    )


if __name__ == '__main__':
    main()


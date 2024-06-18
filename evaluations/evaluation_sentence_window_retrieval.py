import json
from typing import Tuple, List

from architectures.sentence_window_retrieval import indexing


def read_question_answers(base_path: str) -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers


def main():

    base_path = "../datasets/ARAGOG/"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 256
    top_k = 1

    questions, answers = read_question_answers(base_path)
    doc_store, doc_chunk_store = indexing("sentence-transformers/all-MiniLM-L6-v2", chunk_size, base_path)





if __name__ == '__main__':
    main()


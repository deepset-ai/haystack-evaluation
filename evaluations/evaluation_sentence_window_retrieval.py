import json
import os
from pathlib import Path
from typing import Tuple, List

from openai import BadRequestError
from tqdm import tqdm

from architectures.sentence_window_retrieval import rag_sentence_window_retrieval
from architectures.basic_rag import basic_rag
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator, SASEvaluator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult


def read_question_answers(base_path: str) -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers


def indexing(embedding_model: str, chunk_size: int, base_path: str) -> InMemoryDocumentStore:
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size, split_overlap=5))  # splitting by word
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pdf_files = [full_path / "papers_for_questions" / f_name for f_name in os.listdir(files_path)]
    pipeline.run({"converter": {"sources": pdf_files}})

    return document_store


def run_evaluation(sample_questions, sample_answers, retrieved_contexts, predicted_answers, embedding_model):
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator(raise_on_failure=False))
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator(raise_on_failure=False))
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

    eval_pipeline_results = eval_pipeline.run(
        {"context_relevance": {"questions": sample_questions, "contexts": retrieved_contexts},
         "faithfulness": {"questions": sample_questions, "contexts": retrieved_contexts, "predicted_answers": predicted_answers},
         "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": sample_answers}
         }
    )

    results = {
        "context_relevance": eval_pipeline_results['context_relevance'],
        "faithfulness": eval_pipeline_results['faithfulness'],
        "sas": eval_pipeline_results['sas']
    }

    inputs = {'questions': sample_questions,
              'contexts': retrieved_contexts,
              'true_answers': sample_answers,
              'predicted_answers': predicted_answers}

    return results, inputs


def run_rag(rag, questions):
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = rag.run(
                data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}})
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response['answer_builder']['answers'][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)

    return retrieved_contexts, predicted_answers


def main():

    base_path = "../datasets/ARAGOG/"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 128
    top_k = 3

    questions, answers = read_question_answers(base_path)
    doc_store = indexing(embedding_model, chunk_size, base_path)

    # RAG with sentence window retrieval
    rag_window_retrieval = rag_sentence_window_retrieval(doc_store, embedding_model, top_k)
    retrieved_contexts, predicted_answers = run_rag(rag_window_retrieval, questions)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_rag_window = EvaluationRunResult(run_name="window-retrieval", inputs=inputs, results=results)

    # Baseline RAG
    rag = basic_rag(doc_store, embedding_model, top_k)
    retrieved_contexts, predicted_answers = run_rag(rag, questions)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_base_rag = EvaluationRunResult(run_name="base-rag", inputs=inputs, results=results)

    eval_results_base_rag.comparative_individual_scores_report(eval_results_rag_window)


if __name__ == '__main__':
    main()

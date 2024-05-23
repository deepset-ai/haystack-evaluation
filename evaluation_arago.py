import json
import os
from pathlib import Path
from typing import Tuple, List

from openai import BadRequestError

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator, SASEvaluator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult
from tqdm import tqdm

from architectures.basic_rag import basic_rag
from architectures.hyde_rag import rag_with_hyde
from utils import timeit

files_path = "datasets/ARAGOG/papers_for_questions"


@timeit
def indexing(embedding_model: str, chunk_size: int):
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=chunk_size))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pdf_files = [files_path+"/"+f_name for f_name in os.listdir(files_path)]
    pipeline.run({"converter": {"sources": pdf_files}})

    return document_store


def read_question_answers() -> Tuple[List[str], List[str]]:
    with open("datasets/ARAGOG/eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers


@timeit
def run_basic_rag(doc_store, sample_questions, embedding_model, top_k):
    """
    A function to run the basic rag model on a set of sample questions and answers
    """

    rag = basic_rag(document_store=doc_store, embedding_model=embedding_model, top_k=top_k)

    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(sample_questions):
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


def run_hyde_rag(doc_store, sample_questions, sample_answers, embedding_model):

    hyde_rag = rag_with_hyde(document_store=doc_store, embedding_model=embedding_model, top_k=3)

    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(sample_questions):
        response = hyde_rag.run(
            data={"hyde": {"query": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}})
        predicted_answers.append(response["answer_builder"]["answers"][0].data)
        retrieved_contexts.append([d.content for d in response['answer_builder']['answers'][0].documents])

    context_relevance = ContextRelevanceEvaluator()
    faithfulness = FaithfulnessEvaluator()
    sas = SASEvaluator(model=embedding_model)
    sas.warm_up()
    results = {
        "context_relevance": context_relevance.run(sample_questions, retrieved_contexts),
        "faithfulness": faithfulness.run(sample_questions, retrieved_contexts, predicted_answers),
        "sas": sas.run(predicted_answers, sample_answers)
    }
    inputs = {'questions': sample_questions, "true_answers": sample_answers, "predicted_answers": predicted_answers}

    return EvaluationRunResult(run_name="hyde_rag", inputs=inputs, results=results)


@timeit
def run_evaluation(sample_questions, sample_answers, retrieved_contexts, predicted_answers, embedding_model):
    context_relevance = ContextRelevanceEvaluator(raise_on_failure=False)
    faithfulness = FaithfulnessEvaluator(raise_on_failure=False)
    sas = SASEvaluator(model=embedding_model)
    sas.warm_up()

    results = {
        "context_relevance": context_relevance.run(sample_questions, retrieved_contexts),
        "faithfulness": faithfulness.run(sample_questions, retrieved_contexts, predicted_answers),
        "sas": sas.run(predicted_answers, sample_answers),
    }

    inputs = {'questions': sample_questions, "true_answers": sample_answers, "predicted_answers": predicted_answers}

    return results, inputs


def parameter_tuning(questions, answers):
    """
    Run the basic RAG model with different parameters, and evaluate the results.

    The parameters to be tuned are: embedding model, top_k, and chunk_size.
    """
    embedding_models = {
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/msmarco-distilroberta-base-v2",
        "sentence-transformers/all-mpnet-base-v2"
    }
    top_k_values = [1, 2, 3]
    chunk_sizes = [64, 128, 256]

    # create results directory if it does not exist using Pathlib
    out_path = Path("aragog_results")
    out_path.mkdir(exist_ok=True)

    for embedding_model in embedding_models:
        for top_k in top_k_values:
            for chunk_size in chunk_sizes:
                name_params = f"{embedding_model.split('/')[-1]}__top_k:{top_k}__chunk_size:{chunk_size}"
                print(name_params)
                print("Indexing documents")
                doc_store = indexing(embedding_model, chunk_size)
                print("Running RAG pipeline")
                retrieved_contexts, predicted_answers = run_basic_rag(doc_store, questions, embedding_model, top_k)
                print(f"Running evaluation")
                results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
                eval_results = EvaluationRunResult(run_name=name_params, inputs=inputs, results=results)
                eval_results.score_report().to_csv(f"{out_path}/score_report_{name_params}.csv")
                eval_results.to_pandas().to_csv(f"{out_path}/detailed_{name_params}.csv")


def main():
    questions, answers = read_question_answers()
    parameter_tuning(questions, answers)


if __name__ == '__main__':
    main()

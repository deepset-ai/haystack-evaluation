import argparse
import json
import os
import random
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
from utils.utils import timeit

base_path = "../datasets/ARAGOG/"


@timeit
def indexing(embedding_model: str, chunk_size: int):
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size))  # splitting by word
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pdf_files = [full_path / "papers_for_questions" / f_name for f_name in os.listdir(files_path)]
    pipeline.run({"converter": {"sources": pdf_files}})

    return document_store


def read_question_answers() -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
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


@timeit
def run_evaluation(sample_questions, sample_answers, retrieved_contexts, predicted_answers, embedding_model):
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator(raise_on_failure=False))
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator(raise_on_failure=False))
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

    eval_pipeline_results = eval_pipeline.run(
        {
            "context_relevance": {"questions": sample_questions, "contexts": retrieved_contexts},
            "faithfulness": {
                "questions": sample_questions, "contexts": retrieved_contexts, "predicted_answers": predicted_answers
            },
            "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": sample_answers},
        }
    )

    results = {
        "context_relevance": eval_pipeline_results['context_relevance'],
        "faithfulness": eval_pipeline_results['faithfulness'],
        "sas": eval_pipeline_results['sas']
    }

    inputs = {'questions': sample_questions, 'true_answers': sample_answers, 'predicted_answers': predicted_answers}

    return results, inputs


def parameter_tuning(questions, answers, out_path: str):
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
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    for embedding_model in embedding_models:
        for chunk_size in chunk_sizes:
            print(f"Indexing documents with {embedding_model} model with a chunk_size={chunk_size}")
            doc_store = indexing(embedding_model, chunk_size)
            for top_k in top_k_values:
                name_params = f"{embedding_model.split('/')[-1]}__top_k:{top_k}__chunk_size:{chunk_size}"
                print(name_params)
                print("Running RAG pipeline")
                retrieved_contexts, predicted_answers = run_basic_rag(doc_store, questions, embedding_model, top_k)
                print(f"Running evaluation")
                results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
                eval_results = EvaluationRunResult(run_name=name_params, inputs=inputs, results=results)
                eval_results.score_report().to_csv(f"{out_path}/score_report_{name_params}.csv", index=False)
                eval_results.to_pandas().to_csv(f"{out_path}/detailed_{name_params}.csv", index=False)


def create_args():
    parser = argparse.ArgumentParser(description='Run the ARAGOG dataset evaluation on a RAG pipeline')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='The output directory for the results',
        required=True
    )
    parser.add_argument('--sample', type=int, help='The number of questions to sample')
    return parser.parse_args()


def main():
    args = create_args()
    questions, answers = read_question_answers()

    if args.sample:
        random.seed(42)
        questions = random.sample(questions, args.sample)
        answers = random.sample(answers, args.sample)

    parameter_tuning(questions, answers, args.output_dir)


if __name__ == '__main__':
    main()

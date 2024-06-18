import json
import os
from pathlib import Path
from typing import Tuple, List

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_experimental.evaluation.harness.rag import (
    RAGEvaluationHarness,
    RAGEvaluationMetric,
    RAGEvaluationInput,
    RAGExpectedComponent, RAGExpectedComponentMetadata,
)

from architectures.basic_rag import basic_rag
from architectures.hyde_rag import rag_with_hyde
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
    pipeline.run({"converter": {"sources": pdf_files[0:3]}})

    return document_store


def read_question_answers() -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers


@timeit
def eval_pipeline(questions, answers, pipeline, components, run_name):

    pipeline_eval_harness = RAGEvaluationHarness(
        pipeline,
        metrics={
            RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY,  # how to specify the embedding model to use?
            RAGEvaluationMetric.ANSWER_FAITHFULNESS,
            # RAGEvaluationMetric.CONTEXT_RELEVANCE
        },
        rag_components=components
    )

    hyde_eval_harness_input = RAGEvaluationInput(
        queries=questions,
        ground_truth_answers=answers,
        additional_rag_inputs={
            "prompt_builder": {"question": [q for q in questions]},
            "answer_builder": {"query": [q for q in questions]},
        },
    )
    return pipeline_eval_harness.run(inputs=hyde_eval_harness_input, run_name=run_name)


def main():
    questions, answers = read_question_answers()
    embedding_model = "sentence-transformers/msmarco-distilroberta-base-v2"
    chunk_size = 128
    top_k = 3
    doc_store = indexing(embedding_model, chunk_size)

    rag = basic_rag(document_store=doc_store, embedding_model=embedding_model, top_k=top_k)
    rag_components = {
        RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
            name="query_embedder", input_mapping={"query": "text"}),
        RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
            name="retriever", output_mapping={"retrieved_documents": "documents"}),
        RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
            name="llm", output_mapping={"replies": "replies"})
    }
    baseline_rag_eval_output = eval_pipeline(questions[:25], answers[:25], rag, rag_components, "baseline_rag")

    """
    hyde_rag = rag_with_hyde(document_store=doc_store, embedding_model=embedding_model, top_k=top_k)
    hyde_components = {
        RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
            name="hyde", input_mapping={"query": "query"}),
        RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
            name="retriever", output_mapping={"retrieved_documents": "documents"}),
        RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
            name="llm", output_mapping={"replies": "replies"})
    }
    
    hyde_rag_eval_output = eval_pipeline(questions[:25], answers[:25], hyde_rag, hyde_components, "hyde_rag")

    comparative_df = baseline_rag_eval_output.results.comparative_individual_scores_report(
        hyde_rag_eval_output.results, keep_columns=["response"]
    )
    comparative_df.to_csv("comparative_scores.csv")
    """


if __name__ == '__main__':
    main()
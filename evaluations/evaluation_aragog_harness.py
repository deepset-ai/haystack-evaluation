# pip install haystack-experimental git+https://github.com/deepset-ai/haystack-experimental.git

import json
import os
from pathlib import Path
from typing import Tuple, List

from architectures.basic_rag import basic_rag
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


def main():

    questions, answers = read_question_answers()
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 32
    top_k = 1
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

    emb_eval_harness = RAGEvaluationHarness(
        rag,
        metrics={RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY},
        rag_components=rag_components
    )

    input_questions = questions[:3]
    gold_answers = answers[:3]

    eval_harness_input = RAGEvaluationInput(
        queries=input_questions,
        ground_truth_answers=gold_answers,
        additional_rag_inputs={
            "prompt_builder": {"question": [q for q in input_questions]},
            "answer_builder": {"query": [q for q in input_questions]},
        },
    )

    emb_eval_run = emb_eval_harness.run(inputs=eval_harness_input, run_name="emb_eval_run")

    print(emb_eval_run)


if __name__ == '__main__':
    main()
import json
import os
from typing import List

from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator, SASEvaluator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult
from tqdm import tqdm

from architectures.basic_rag import basic_rag
from architectures.hyde_rag import rag_with_hyde

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
base_path = "datasets/SQuAD-2.0/transformed_squad/"


def load_transformed_squad():
    with open(base_path+"questions.jsonl", "r") as f:
        questions = [json.loads(x) for x in f.readlines()]
    for idx, question in enumerate(questions):
        question["query_id"] = f"query_{idx}"

    def create_document(text: str, name: str):
        return Document(content=text, meta={"name": name})

    # walk through the files in the directory and transform each line of each text file into a Document
    documents = []
    for root, dirs, files in os.walk(base_path):
        for article in files:
            with open(f"{root}/{article}", "r") as f:
                raw_texts = f.read().split("\n")
                for text in raw_texts:
                    documents.append(create_document(text, article.replace(".txt", "")))

    return questions, documents


def indexing(documents: List[Document]):
    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


def run_basic_rag(doc_store, sample_questions, sample_answers):
    """
    A function to run the basic rag model on a set of sample questions and answers
    """

    rag = basic_rag(document_store=doc_store, embedding_model=embedding_model, top_k=3)

    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(sample_questions):
        response = rag.run(
            data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}})
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
    inputs = {'questions': sample_questions, 'answers': predicted_answers}

    return EvaluationRunResult(run_name="basic_rag", inputs=inputs, results=results)


def run_hyde_rag(doc_store, sample_questions, sample_answers):

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
    inputs = {'questions': sample_questions, 'predicted_answers': predicted_answers}

    return EvaluationRunResult(run_name="hyde_rag", inputs=inputs, results=results)


def main():

    questions, documents = load_transformed_squad()
    doc_store = indexing(documents)

    limit = 5
    questions = questions[0:limit]
    sample_ground_truth_answers = ground_truth_answers[0:limit]

    basic_rag_results = run_basic_rag(doc_store, sample_questions, sample_ground_truth_answers)
    hyde_rag_results = run_hyde_rag(doc_store, sample_questions, sample_ground_truth_answers)

    comparative_df = basic_rag_results.comparative_individual_scores_report(hyde_rag_results)


import json
import os
import random
from typing import List

from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import (
    DocumentMRREvaluator,
    DocumentMAPEvaluator,
    DocumentRecallEvaluator,
    FaithfulnessEvaluator,
    SASEvaluator
)
from haystack.components.evaluators.document_recall import RecallMode
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
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


def run_basic_rag(doc_store, samples):

    rag = basic_rag(document_store=doc_store, embedding_model=embedding_model, top_k=3)

    # ground truth data
    questions = []
    ground_truth_docs = []
    ground_truth_answers = []

    # predicted data
    retrieved_docs = []
    predicted_contexts = []
    predicted_answers = []

    for sample in tqdm(samples):
        q = sample["question"]
        answer = sample["answers"]["text"]
        ground_truth_documents = [doc for doc in doc_store.storage.values() if doc.meta["name"] == sample["document"]]
        response = rag.run(
            data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
        )

        # gather ground truth data
        ground_truth_docs.append(ground_truth_documents)
        ground_truth_answers.append(answer[0])
        questions.append(q)

        # gather response data
        retrieved_docs.append(response["answer_builder"]["answers"][0].documents)
        predicted_contexts.append([doc.content for doc in response["answer_builder"]["answers"][0].documents])
        predicted_answers.append(response["answer_builder"]["answers"][0].data)

    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
    eval_pipeline.add_component("doc_map", DocumentMAPEvaluator())
    eval_pipeline.add_component("doc_recall_single_hit", DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT))
    eval_pipeline.add_component("doc_recall_multi_hit", DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT))
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

    eval_pipeline_results = eval_pipeline.run(
        {
            "doc_mrr": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "faithfulness": {"questions": questions, "contexts": predicted_contexts, "predicted_answers": predicted_answers},
            "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": ground_truth_answers},
            "doc_map": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_single_hit": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_multi_hit": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs}
        }
    )

    results = {        
        "doc_mrr": eval_pipeline_results['doc_mrr'],
        "faithfulness": eval_pipeline_results['faithfulness'],
        "sas": eval_pipeline_results['sas'],
        "doc_map": eval_pipeline_results['doc_map'],
        "doc_recall_single_hit": eval_pipeline_results['doc_recall_single_hit'],
        "doc_recall_multi_hit": eval_pipeline_results['doc_recall_multi_hit']
    }

    inputs = {'questions': questions, 'true_answers': ground_truth_answers, 'predicted_answers': predicted_answers}

    return EvaluationRunResult(run_name="basic_rag", inputs=inputs, results=results)


def run_hyde_rag(doc_store, samples):

    hyde_rag = rag_with_hyde(document_store=doc_store, embedding_model=embedding_model, top_k=3)

    # ground truth data
    questions = []
    ground_truth_docs = []
    ground_truth_answers = []

    # predicted data
    retrieved_docs = []
    predicted_contexts = []
    predicted_answers = []

    for sample in tqdm(samples):
        q = sample["question"]
        answer = sample["answers"]["text"]
        ground_truth_documents = [doc for doc in doc_store.storage.values() if doc.meta["name"] == sample["document"]]
        response = hyde_rag.run(
            data={"hyde": {"query": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
        )

        # gather ground truth data
        ground_truth_docs.append(ground_truth_documents)
        ground_truth_answers.append(answer[0])
        questions.append(q)

        # gather response data
        retrieved_docs.append(response["answer_builder"]["answers"][0].documents)
        predicted_contexts.append([doc.content for doc in response["answer_builder"]["answers"][0].documents])
        predicted_answers.append(response["answer_builder"]["answers"][0].data)

    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
    eval_pipeline.add_component("doc_map", DocumentMAPEvaluator())
    eval_pipeline.add_component("doc_recall_single_hit", DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT))
    eval_pipeline.add_component("doc_recall_multi_hit", DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT))
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

    eval_pipeline_results = eval_pipeline.run(
        {
            "doc_mrr": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "faithfulness": {"questions": questions, "contexts": predicted_contexts, "predicted_answers": predicted_answers},
            "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": ground_truth_answers},
            "doc_map": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_single_hit": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_multi_hit": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs}
        }
    )

    results = {
        "doc_mrr": eval_pipeline_results['doc_mrr'],
        "faithfulness": eval_pipeline_results['faithfulness'],
        "sas": eval_pipeline_results['sas'],
        "doc_map": eval_pipeline_results['doc_map'],
        "doc_recall_single_hit": eval_pipeline_results['doc_recall_single_hit'],
        "doc_recall_multi_hit": eval_pipeline_results['doc_recall_multi_hit']
    }

    inputs = {'questions': questions, 'true_answers': ground_truth_answers, 'predicted_answers': predicted_answers}

    return EvaluationRunResult(run_name="hyde_rag", inputs=inputs, results=results)


def main():

    all_questions, documents = load_transformed_squad()
    doc_store = indexing(documents)

    limit = 10
    samples = random.sample(all_questions, limit)

    basic_rag_results = run_basic_rag(doc_store, samples)
    hyde_rag_results = run_hyde_rag(doc_store, samples)

    comparative_df = basic_rag_results.comparative_individual_scores_report(hyde_rag_results)

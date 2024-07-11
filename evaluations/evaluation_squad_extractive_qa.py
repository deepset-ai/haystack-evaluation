import argparse
import json
import os
import random
from pathlib import Path

from architectures.extractive_qa import get_extractive_qa_pipeline
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    SASEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult
from tqdm import tqdm
from utils.utils import timeit

base_path = "../datasets/SQuAD-2.0/transformed_squad/"


def load_transformed_squad():
    with open(base_path + "questions.jsonl", "r") as f:
        questions = [json.loads(x) for x in f.readlines()]
    for idx, question in enumerate(questions):
        question["query_id"] = f"query_{idx}"

    def create_document(text: str, name: str):
        return Document(content=text, meta={"name": name})

    documents = []
    for root, _, files in os.walk(base_path + "articles"):
        for article in files:
            with open(f"{root}/{article}", "r") as f:
                article_text = f.read()
                documents.append(create_document(article_text, article.replace(".txt", "")))

    return questions, documents


@timeit
def indexing(documents, embedding_model, chunk_size):
    document_store = InMemoryDocumentStore()
    doc_splitter = DocumentSplitter(split_by="sentence", split_length=chunk_size)
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_splitter, name="doc_splitter")
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_splitter.documents", "doc_embedder.documents")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_splitter": {"documents": documents}})

    return document_store


@timeit
def run_extractive_qa(doc_store, questions, embedding_model, top_k_retriever):
    extractive_qa = get_extractive_qa_pipeline(
        document_store=doc_store, embedding_model=embedding_model, top_k_retriever=top_k_retriever
    )

    # predicted data
    retrieved_docs = []
    predicted_answers = []

    for q in tqdm(questions):
        response = extractive_qa.run(
            data={"embedder": {"text": q}, "retriever": {"top_k": top_k_retriever}, "reader": {"query": q, "top_k": 1}}
        )
        retrieved_docs.append([answer.document for answer in response["reader"]["answers"]])
        predicted_answers.append(response["reader"]["answers"][0].data)

    return retrieved_docs, predicted_answers


@timeit
def run_evaluation(
    embedding_model, ground_truth_docs, retrieved_docs, questions, predicted_answers, ground_truth_answers
):
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
    eval_pipeline.add_component("doc_map", DocumentMAPEvaluator())
    eval_pipeline.add_component("doc_recall_single_hit", DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT))
    eval_pipeline.add_component("doc_recall_multi_hit", DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT))
    eval_pipeline.add_component("answer_exact", AnswerExactMatchEvaluator())
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

    # get the original documents from the retrieved documents which were split
    original_retrieved_docs = []
    for doc in retrieved_docs:
        original_docs = []
        for split_doc in doc:
            for original_doc in ground_truth_docs:
                if split_doc.meta["name"] == original_doc[0].meta["name"]:
                    original_docs.append(original_doc[0])
        original_retrieved_docs.append(original_docs)

    eval_pipeline_results = eval_pipeline.run(
        {
            "doc_mrr": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": ground_truth_answers},
            "answer_exact": {"predicted_answers": predicted_answers, "ground_truth_answers": ground_truth_answers},
            "doc_map": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_single_hit": {
                "ground_truth_documents": ground_truth_docs,
                "retrieved_documents": retrieved_docs,
            },
            "doc_recall_multi_hit": {
                "ground_truth_documents": ground_truth_docs,
                "retrieved_documents": retrieved_docs,
            },
        }
    )

    results = {
        "doc_mrr": eval_pipeline_results["doc_mrr"],
        "sas": eval_pipeline_results["sas"],
        "doc_map": eval_pipeline_results["doc_map"],
        "doc_recall_single_hit": eval_pipeline_results["doc_recall_single_hit"],
        "doc_recall_multi_hit": eval_pipeline_results["doc_recall_multi_hit"],
    }

    inputs = {
        "questions": questions,
        "true_answers": ground_truth_answers,
        "predicted_answers": predicted_answers,
        "retrieved_docs": retrieved_docs,
    }

    return results, inputs


def parameter_tuning(queries, documents, output_path: str):
    """
    Run the basic Extractive QA model with different parameters, and evaluate the results.

    The parameters to be tuned are: embedding model, top_k, and chunk_size.
    """
    embedding_models = {
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/msmarco-distilroberta-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
    }
    top_k_values = [1, 2, 3]
    chunk_sizes = [5, 10, 15]

    out_path = Path(output_path)
    out_path.mkdir(exist_ok=True)

    questions = []
    ground_truth_answers = []
    ground_truth_docs = []
    for sample in queries:
        questions.append(sample["question"])
        ground_truth_answers.append(sample["answers"]["text"][0])
        ground_truth_docs.append([doc for doc in documents if doc.meta["name"] == sample["document"]])

    for embedding_model in embedding_models:
        for top_k in top_k_values:
            for chunk_size in chunk_sizes:
                name_params = f"{embedding_model.split('/')[-1]}__top_k:{top_k}__chunk_size:{chunk_size}"
                print(name_params)
                print("Indexing documents")
                doc_store = indexing(documents, embedding_model, chunk_size)
                print("Running Extractive QA pipeline")
                retrieved_docs, predicted_answers = run_extractive_qa(doc_store, questions, embedding_model, top_k)
                print("Running evaluation")
                results, inputs = run_evaluation(
                    embedding_model,
                    ground_truth_docs,
                    retrieved_docs,
                    questions,
                    predicted_answers,
                    ground_truth_answers,
                )
                eval_results = EvaluationRunResult(run_name=name_params, inputs=inputs, results=results)
                eval_results.score_report().to_csv(f"{out_path}/score_report_{name_params}.csv", index=False)
                eval_results.to_pandas().to_csv(f"{out_path}/detailed_{name_params}.csv", index=False)


def create_args():
    parser = argparse.ArgumentParser(description="Run the ARAGOG dataset evaluation on a RAG pipeline")
    parser.add_argument("--output-dir", type=str, help="The output directory for the results", required=True)
    parser.add_argument("--sample", type=int, help="The number of questions to sample", default=100)
    return parser.parse_args()


@timeit
def main():
    random.seed(42)
    args = create_args()
    all_queries, documents = load_transformed_squad()

    # the total number of questions is 98k, so we take a sample of 100 or whatever the user specifies
    queries = random.sample(all_queries, args.sample)
    if args.sample:
        queries = random.sample(all_queries, args.sample)

    print(f"Running evaluation on {args.sample} questions")
    parameter_tuning(queries, documents, args.output_dir)


if __name__ == "__main__":
    main()

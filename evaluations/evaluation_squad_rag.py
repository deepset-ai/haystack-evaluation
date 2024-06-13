import json
import os
import random
from pathlib import Path

from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    DocumentMRREvaluator,
    DocumentMAPEvaluator,
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

from architectures.basic_rag import basic_rag

base_path = "datasets/SQuAD-2.0/transformed_squad/"


def load_transformed_squad():
    with open(base_path+"questions.jsonl", "r") as f:
        questions = [json.loads(x) for x in f.readlines()]
    for idx, question in enumerate(questions):
        question["query_id"] = f"query_{idx}"

    def create_document(text: str, name: str):
        return Document(content=text, meta={"name": name})

    documents = []
    for root, dirs, files in os.walk(base_path+"articles"):
        for article in files:
            with open(f"{root}/{article}", "r") as f:
                article_text = f.read()
                documents.append(create_document(article_text, article.replace(".txt", "")))

    return questions, documents


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


def run_basic_rag(doc_store, questions, embedding_model, top_k):

    rag = basic_rag(document_store=doc_store, embedding_model=embedding_model, top_k=top_k)

    # predicted data
    retrieved_docs = []
    retrieved_contexts = []
    predicted_answers = []

    for q in tqdm(questions):
        response = rag.run(
            data={"query_embedder": {"text": q},
                  "prompt_builder": {"question": q},
                  "answer_builder": {"query": q}}
        )

        # gather response data
        retrieved_docs.append(response["answer_builder"]["answers"][0].documents)
        retrieved_contexts.append([doc.content for doc in response["answer_builder"]["answers"][0].documents])
        predicted_answers.append(response["answer_builder"]["answers"][0].data)

    return retrieved_docs, predicted_answers, retrieved_contexts


def run_evaluation(embedding_model, ground_truth_docs, retrieved_docs, questions, predicted_answers, ground_truth_answers):
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
            "doc_mrr": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": original_retrieved_docs},
            "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": ground_truth_answers},
            "answer_exact": {"predicted_answers": predicted_answers, "ground_truth_answers": ground_truth_answers},
            "doc_map": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": original_retrieved_docs},
            "doc_recall_single_hit": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": original_retrieved_docs},
            "doc_recall_multi_hit": {"ground_truth_documents": ground_truth_docs, "retrieved_documents": original_retrieved_docs}
        }
    )

    results = {
        "doc_mrr": eval_pipeline_results['doc_mrr'],
        "sas": eval_pipeline_results['sas'],
        "doc_map": eval_pipeline_results['doc_map'],
        "doc_recall_single_hit": eval_pipeline_results['doc_recall_single_hit'],
        "doc_recall_multi_hit": eval_pipeline_results['doc_recall_multi_hit']
    }

    inputs = {'questions': questions,
              'true_answers': ground_truth_answers,
              'predicted_answers': predicted_answers,
              'contexts': retrieved_docs
              }

    return results, inputs


def parameter_tuning(queries, documents):
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
    chunk_sizes = [5, 10, 15]

    # create results directory if it does not exist using Pathlib
    out_path = Path("squad_results")
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
                print("Running RAG pipeline")
                retrieved_docs, predicted_answers, retrieved_contexts = run_basic_rag(
                    doc_store, questions, embedding_model, top_k
                )
                print(f"Running evaluation")
                results, inputs = run_evaluation(
                    embedding_model, ground_truth_docs, retrieved_docs, questions, predicted_answers,
                    ground_truth_answers
                )
                eval_results = EvaluationRunResult(run_name=name_params, inputs=inputs, results=results)
                eval_results.score_report().to_csv(f"{out_path}/score_report_{name_params}.csv")
                eval_results.to_pandas().to_csv(f"{out_path}/detailed_{name_params}.csv")


def main():
    random.seed(42)
    all_queries, documents = load_transformed_squad()
    queries = random.sample(all_queries, 100)  # take a sample of 100 questions
    parameter_tuning(queries, documents)


if __name__ == "__main__":
    main()

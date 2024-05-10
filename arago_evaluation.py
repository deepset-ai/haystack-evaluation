import json
import os

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator, SASEvaluator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult
from tqdm import tqdm

from architectures.basic_rag import basic_rag
from architectures.hyde_rag import rag_with_hyde

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
files_path = "datasets/ARAGOG/papers_for_questions"


def indexing():
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=256))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pdf_files = [files_path+"/"+f_name for f_name in os.listdir(files_path)]
    pipeline.run({"converter": {"sources": pdf_files}})

    return document_store


def read_question_answers():
    with open("datasets/ARAGOG/eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers


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
        "sas": sas.run(predicted_answers, sample_answers),
        'predicted_answers': predicted_answers,
    }
    inputs = {'questions': sample_questions}

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
        'predicted_answers': predicted_answers,
        "context_relevance": context_relevance.run(sample_questions, retrieved_contexts),
        "faithfulness": faithfulness.run(sample_questions, retrieved_contexts, predicted_answers),
        "sas": sas.run(predicted_answers, sample_answers)
    }
    inputs = {'questions': sample_questions}

    return EvaluationRunResult(run_name="hyde_rag", inputs=inputs, results=results)


def main():
    doc_store = indexing()
    questions, ground_truth_answers = read_question_answers()
    limit = 5
    sample_questions = questions[0:limit]
    sample_ground_truth_answers = ground_truth_answers[0:limit]

    basic_rag_results = run_basic_rag(doc_store, sample_questions, sample_ground_truth_answers)
    hyde_rag_results = run_hyde_rag(doc_store, sample_questions, sample_ground_truth_answers)

    comparative_df = basic_rag_results.comparative_individual_scores_report(hyde_rag_results)

import json
import os
import random

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

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
files_path = "datasets/MiniESGBench/"


def indexing():
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=128))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pdf_files = [files_path+"source_files/"+f_name for f_name in os.listdir(files_path+"source_files/")]
    pipeline.run({"converter": {"sources": pdf_files}})

    return document_store


def read_question_answers():
    with open(files_path+"/rag_dataset.json", "r") as f:
        data = json.load(f)
        questions = []
        contexts = []
        answers = []
        for entry in data['examples']:
            questions.append(entry['query'])
            contexts.append(entry['reference_contexts'])
            answers.append(entry['reference_answer'])

    return questions, contexts, answers


def run_basic_rag(doc_store, questions_sample, answers_sample, contexts_sample):
    """
    A function to run the basic rag model on a set of sample questions and answers
    """

    rag = basic_rag(document_store=doc_store, embedding_model=embedding_model, top_k=2)

    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions_sample):
        response = rag.run(
            data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}})
        predicted_answers.append(response["answer_builder"]["answers"][0].data)
        retrieved_contexts.append([d.content for d in response['answer_builder']['answers'][0].documents])

    context_relevance = ContextRelevanceEvaluator()
    faithfulness = FaithfulnessEvaluator()
    sas = SASEvaluator(model=embedding_model)
    sas.warm_up()
    results = {
        "context_relevance": context_relevance.run(questions_sample, retrieved_contexts),
        "faithfulness": faithfulness.run(questions_sample, retrieved_contexts, predicted_answers),
        "sas": sas.run(predicted_answers, answers_sample),
    }
    inputs = {'questions': questions_sample, "true_answers": answers_sample, "predicted_answers": predicted_answers}

    return EvaluationRunResult(run_name="basic_rag", inputs=inputs, results=results)


def main():
    doc_store = indexing()
    questions, contexts, answers = read_question_answers()

    limit = 5
    questions_sample = random.sample(questions, limit)
    contexts_sample = random.sample(contexts, limit)
    answers_sample = random.sample(answers, limit)

    basic_rag_results = run_basic_rag(doc_store, questions_sample, answers_sample, contexts_sample)




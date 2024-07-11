import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from random import choice
from typing import Dict, List, Set

from haystack import Document, Pipeline, component
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


@dataclass
class QueryRelevance:
    topic_id: int
    iteration: int
    document_id: str
    relevance_judgement: int


def read_query_relevance(query_relevance_file: str) -> Dict[int, List[QueryRelevance]]:
    """
    Read the qrel file and return a dictionary with the query_id as key and a list of relevant document ids as value.

    Topic ID: The ID of the topic or query.
    Iteration: This is usually 0 and can be ignored.
    Document ID: The ID of the document being judged.
    Relevance Judgment: A number indicating the relevance of the document to the topic (usually binary, 0 for not
                        relevant and 1 for relevant, but sometimes more granular scales are used).
    """

    query_relevance: Dict[int, List[QueryRelevance]] = defaultdict(lambda: [])
    with open(query_relevance_file, "r") as f:
        for line in f:
            query_id, iteration, doc_id, relevance = line.strip().split()
            query = QueryRelevance(
                topic_id=int(query_id), iteration=iteration, document_id=doc_id, relevance_judgement=int(relevance)
            )
            query_relevance[int(query_id)].append(query)

    return query_relevance


def read_topics(topic_file: str):
    """
    Reads TREC topics from a file and returns a list of dictionaries with the topic_id and the query text.
    """
    topics = {}
    with open(topic_file, "r") as f:
        for line in f:
            topic_id, query = line.strip().split("\t")
            topics[int(topic_id.strip())] = query.strip()

    return topics


def read_documents(document_files: Set[str], corpus_path: str):
    """
    Reads TREC documents from a file and returns a list of dictionaries with the document_id and the document text.
    """

    files_to_index = set()

    for _, _, files in os.walk("../TREC/corpus/"):
        for file in files:
            if files_to_index.intersection(document_files) == document_files:
                break
            if not file.endswith(".json"):
                continue
            print(file)
            with open(os.path.join(corpus_path, file), "r") as f:
                for line in f:
                    doc = json.loads(line)
                    doc_id = re.split(r"#", doc["docid"])
                    if doc_id[0] in document_files:
                        files_to_index.add(file)

    return files_to_index


def get_qdrant_doc_store(embedding_dim: int = 768):
    doc_store = QdrantDocumentStore(
        url="localhost",
        index="trec2024",
        embedding_dim=embedding_dim,
        on_disk=True,
        recreate_index=True,
        hnsw_config={"m": 16, "ef_construct": 64},  # Optional
    )

    return doc_store


@component
class ParseTRECCorpus:
    @staticmethod
    def create_document(line: str):
        doc = json.loads(line)
        return Document(content=doc["segment"], meta={"docid": doc["docid"], "url": doc["url"]})

    @component.output_types(segments=List[Document])
    def run(self, files: List[str]):
        for file in files:
            with open(file, "r") as f:
                results = [self.create_document(line) for line in f]
        return {"segments": results}


def indexing(doc_store, model: str, chunk_size: int, files_to_index: Set[str]):
    pipeline = Pipeline()
    pipeline.add_component("converter", ParseTRECCorpus())
    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size, split_overlap=5))  # splitting by word
    pipeline.add_component("writer", DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model))
    pipeline.connect("converter", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run({"converter": {"files": files_to_index}})

    return doc_store


def built_basic_rag(document_store, embedding_model, top_k=2):
    template = """
        You have to answer the following question based on the given context information only.
        If the context is empty or just a '\n' answer with None, example: "None".

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

    basic_rag = Pipeline()
    basic_rag.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    )
    basic_rag.add_component("retriever", QdrantEmbeddingRetriever(document_store, top_k=top_k))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template))
    basic_rag.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    basic_rag.add_component("answer_builder", AnswerBuilder())

    basic_rag.connect("query_embedder", "retriever.query_embedding")
    basic_rag.connect("retriever", "prompt_builder.documents")
    basic_rag.connect("prompt_builder", "llm")
    basic_rag.connect("llm.replies", "answer_builder.replies")
    basic_rag.connect("llm.meta", "answer_builder.meta")
    basic_rag.connect("retriever", "answer_builder.documents")

    return basic_rag


def prepare():
    topics = read_topics("topics/topics.dl23.txt")
    queries = read_query_relevance("qrels/qrels.dl23-doc-msmarco-v2.1.txt")
    topics_id = set(topics.keys()).intersection(set(queries.keys()))
    docs = []
    for k in topics_id:
        docs.extend([entry.document_id for entry in queries[k]])
    files_to_index = read_documents(set(docs), "../TREC/corpus/")
    model = "all-MiniLM-L12-v2"
    embedding_dim = 384
    doc_store = get_qdrant_doc_store(embedding_dim)
    indexing(doc_store, model, 128, files_to_index)

    rag = built_basic_rag(doc_store, model, top_k=2)

    # choose a random entry from the topics
    random_key = choice(list(topics.keys()))
    question = topics[random_key]

    rag.run(
        {
            "query_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
    )

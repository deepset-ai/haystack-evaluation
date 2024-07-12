import json
from typing import List, Set

from haystack import Document, Pipeline, component
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


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


def built_basic_rag(document_store, embedding_model):
    template = (
        "You have to answer the following question based on the contexts given below. "
        "If all the contexts are empty answer with None, example: None. "
        "Otherwise, analyze all the contexts and build a coherent answer and complete answer. "
        "Split your answer into multiple sentences, and for each sentence please provide the context number "
        "that you used to generate that sentence."
        "{% for document in documents %}"
        "Context {{loop.index}}: {{document.content}}"
        "{%endfor %}"
        "Question: {{question}}"
        "Answer:"
    )

    basic_rag = Pipeline()
    basic_rag.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    )
    basic_rag.add_component("retriever", QdrantEmbeddingRetriever(document_store))
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


def pipeline_task_1(document_store, embedding_model):
    retrieval = Pipeline()
    retrieval.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    )
    retrieval.add_component("retriever", QdrantEmbeddingRetriever(document_store))
    retrieval.connect("query_embedder", "retriever.query_embedding")

    return retrieval

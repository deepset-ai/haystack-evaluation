import os
from pathlib import Path
from typing import Tuple, List, Dict

from haystack import Pipeline, component, Document
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


class DocumentChunks:

    def __init__(self, chunks: List[Document]):
        self.chunks: List[Document] = chunks
        self.mappings = {doc.id: idx for idx, doc in enumerate(chunks)}

    def get_window(self, doc_id: str, window_size: int) -> List[Document]:
        doc_idx = self.mappings[doc_id]
        return self.chunks[max(0, doc_idx - window_size):min(len(self.chunks), doc_idx + window_size + 1)]


class DocumentChunksStore:

    def __init__(self, documents: Dict[str, DocumentChunks]):
        self.mappings = documents


@component
class DocumentChunker:

    def __init__(self):
        self.documents = {}

    @component.output_types(doc_chunk_store=DocumentChunksStore)
    def run(self, documents: List[Document]):
        """
        Stores all the chunks of a document in the same DocumentChunks object.

        Each DocumentChunks object is stored in a dictionary with the source_id as key in a DocumentChunksStore object.
        """
        source_id = documents[0].meta['source_id']
        chunks = []
        for doc in documents[0:]:
            if source_id != doc.meta['source_id']:
                self.documents[source_id] = DocumentChunks(chunks)
                print(f"Stored {len(chunks)} chunks for document {source_id}")
                source_id = doc.meta['source_id']
                chunks = [doc]
            else:
                chunks.append(doc)
        if chunks:
            self.documents[source_id] = DocumentChunks(chunks)
            print(f"Stored {len(chunks)} chunks for document {source_id}")

        return {'doc_chunk_store': DocumentChunksStore(self.documents)}


def indexing(embedding_model: str, chunk_size: int, base_path: str) -> Tuple[InMemoryDocumentStore, DocumentChunksStore]:
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    doc_store = InMemoryDocumentStore()
    pipeline = Pipeline()

    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size))  # splitting by word
    pipeline.add_component("writer", DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.add_component("chunker", DocumentChunker())

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")

    pipeline.connect("splitter", "chunker")

    pipeline.connect("embedder", "writer")
    pdf_files = [full_path / "papers_for_questions" / f_name for f_name in os.listdir(files_path)]
    res = pipeline.run({"converter": {"sources": pdf_files}})

    source_ids_doc_store = set([d.meta['source_id'] for d in doc_store.storage.values()])
    source_ids_doc_chunk_store = res['chunker']['doc_chunk_store'].mappings.keys()

    assert source_ids_doc_store == set(source_ids_doc_chunk_store)

    return doc_store, res['chunker']['doc_chunk_store']


@component
class SentenceWindowRetriever:

    def __init__(self, doc_chunk_store: DocumentChunksStore, window_size: int = 2):
        self.doc_chunk_store = doc_chunk_store
        self.window_size = window_size

    @component.output_types(context_windows=List[DocumentChunksStore])
    def run(self, retrieved_documents: List[Document]):
        context_windows = []
        for doc in retrieved_documents:
            source_id = doc.meta['source_id']
            document_chunks = self.doc_chunk_store.mappings[source_id]
            context_windows.append(document_chunks.get_window(doc.id, self.window_size))
        return {"context_windows": context_windows}


def rag_sentence_window_retrieval(doc_store, doc_chunk_store, embedding_model, top_k=1):
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
    basic_rag.add_component("query_embedder", SentenceTransformersTextEmbedder(
        model=embedding_model, progress_bar=False
    ))
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(doc_store, top_k=top_k))
    basic_rag.add_component("sentence_window_retriever", SentenceWindowRetriever(doc_chunk_store=doc_chunk_store))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template))
    basic_rag.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    basic_rag.add_component("answer_builder", AnswerBuilder())

    basic_rag.connect("query_embedder", "retriever.query_embedding")
    basic_rag.connect("retriever", "prompt_builder.documents")
    basic_rag.connect("retriever", "sentence_window_retriever")
    basic_rag.connect("prompt_builder", "llm")
    basic_rag.connect("llm.replies", "answer_builder.replies")
    basic_rag.connect("llm.meta", "answer_builder.meta")
    basic_rag.connect("retriever", "answer_builder.documents")

    return basic_rag

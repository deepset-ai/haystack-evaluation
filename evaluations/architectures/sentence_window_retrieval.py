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

    def get_window(self, idx: int, window_size: int) -> List[Document]:
        pass


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


from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder


def get_extractive_qa_pipeline(doc_store, model):
    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
    extractive_qa_pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=doc_store), name="retriever")
    extractive_qa_pipeline.add_component(instance=ExtractiveReader(), name="reader")

    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


def get_extractive_qa_pipeline(document_store, embedding_model, top_k_retriever):
    """
    An extractive question answering pipeline.

    It uses a retriever to find relevant documents for a given question and a reader to extract the answer from
    these documents.
    """
    embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=top_k_retriever)
    reader = ExtractiveReader(top_k=1, no_answer=False)

    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=embedder, name="embedder")
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")
    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

    return extractive_qa_pipeline

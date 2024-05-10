from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever


def basic_rag(document_store, embedding_model, top_k=2):
    template = """
        You have to answer the following question based on the given context information only.
        If the context is empty or just a '\n' answer "I don't know".

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
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=top_k))
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

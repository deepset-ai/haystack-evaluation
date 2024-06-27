from typing import List

from haystack import Pipeline, component, Document
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class SentenceWindowRetriever:

    def __init__(self, document_store: InMemoryDocumentStore, window_size: int = 1):
        self.window_size = window_size
        self.document_store = document_store

    @staticmethod
    def get_window_content(docs: List[Document]):
        return ' '.join([doc.content for doc in docs])

    @staticmethod
    def merge_documents(documents):
        """
        Merge a list of doc chunks into a single doc by concatenating their content, eliminating overlapping content.
        """
        sorted_docs = sorted(documents, key=lambda doc: doc.meta["split_idx_start"])
        merged_text = ""
        last_idx_end = 0
        for doc in sorted_docs:
            start = doc.meta["split_idx_start"]  # start of the current content

            # if the start of the current content is before the end of the last appended content, adjust it
            if start < last_idx_end:
                start = last_idx_end

            # append the non-overlapping part to the merged text
            merged_text = merged_text.strip()
            merged_text += doc.content[start - doc.meta["split_idx_start"]:]

            # update the last end index
            last_idx_end = doc.meta["split_idx_start"] + len(doc.content)

        return merged_text

    @component.output_types(context_windows=List[Document])
    def run(self, retrieved_documents: List[Document]):
        """
        Based on the source_id and on the doc.meta['split_id'] get surrounding documents from the document store
        """
        context_windows = []
        for doc in retrieved_documents:
            source_id = doc.meta['source_id']
            split_id = doc.meta['split_id']
            min_before = min([i for i in range(split_id-1, split_id-self.window_size-1,-1)])
            max_after = max([i for i in range(split_id+1, split_id+self.window_size+1, 1)])
            context_docs = self.document_store.filter_documents(
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "source_id", "operator": "==", "value": source_id},
                        {"field": "split_id", "operator": ">=", "value": min_before},
                        {"field": "split_id", "operator": "<=", "value": max_after}
                    ],
                }
            )
            context_windows.append(self.merge_documents(context_docs))

        return {"context_windows": context_windows}


def rag_sentence_window_retrieval(doc_store, embedding_model, top_k=1):
    template = """
        You have to answer the following question based on the given context information only.
        If the context is empty or just a '\n' answer with None, example: "None".

        Context:
        {% for document in documents %}
            {{ document }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

    basic_rag = Pipeline()
    basic_rag.add_component("query_embedder", SentenceTransformersTextEmbedder(
        model=embedding_model, progress_bar=False
    ))
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(doc_store, top_k=top_k))
    basic_rag.add_component("sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template))
    basic_rag.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    basic_rag.add_component("answer_builder", AnswerBuilder())

    basic_rag.connect("query_embedder", "retriever.query_embedding")
    basic_rag.connect("retriever", "sentence_window_retriever")
    basic_rag.connect("sentence_window_retriever.context_windows", "prompt_builder.documents")
    basic_rag.connect("prompt_builder", "llm")
    basic_rag.connect("llm.replies", "answer_builder.replies")
    basic_rag.connect("llm.meta", "answer_builder.meta")

    # to see the retrieved documents in the answer
    basic_rag.connect("retriever", "answer_builder.documents")

    return basic_rag

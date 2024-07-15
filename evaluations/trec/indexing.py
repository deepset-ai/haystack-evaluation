from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pipelines import indexing


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


def main():
    # make sure you have Qdrant running on localhost
    print("Connecting to Qdrant...")
    doc_store = get_qdrant_doc_store()
    print("Indexing documents...")

    # we manually created a sample file of around 150MB for testing purposes
    files_to_index = {"../../datasets/TREC/corpus/msmarco_v2.1_doc_segmented_00_sample.json"}
    indexing(doc_store, "sentence-transformers/msmarco-distilroberta-base-v2", 128, files_to_index)


if __name__ == "__main__":
    main()

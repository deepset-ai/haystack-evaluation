import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from random import choice
from typing import Dict, List, Set

from haystack import Document, component
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from evaluations.trec.pipelines import built_basic_rag, indexing, pipeline_task_1


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

    return doc_store, topics, queries


def write_results(results, output_file):
    with open(output_file, "w") as f:
        for result in results:
            f.write(result)


def task_1():
    """
    RAG Task 1: Retrieval

    For each topic, the system needs to return the TREC runfile containing the ranked list containing the
    top 20 relevant segment IDs from the collection. The topics provided will be non-factoid and require
    long-form answer generation.

    Output Format (Ranked Results)
    Participants should provide their output in the standard TREC format containing top-k=20 MS MARCO v2.1 segments as
    TSV: <r_output_trec_rag_2024.tsv> for each individual topic.Each set of ranked results for a set of topics appears
    in a single file:

    Topic ID (taken from trec_rag_2024_queries.tsv)
    The fixed string “Q0”
    Segment ID (from the docid field in msmarco_v2.1_doc_segmented_XX.json.gz)
    Score (integer or float, selected by your system)
    Run ID where you should mention your team-name (e.g. my-team-name)
    """

    doc_store, topics, queries = prepare()

    retrieval = pipeline_task_1(doc_store, "all-MiniLM-L12-v2")
    run_results = []

    for topic_id, question in topics.items():
        print(topic_id, "\t", question)
        retrieved_docs = retrieval.run({"query_embedder": {"text": question}, "retriever": {"top_k": 20}})
        for result in retrieved_docs["retriever"]["documents"]:
            out = f"{topic_id}\tQ0\t{result.meta['docid']}\t{result.score}\tdeepset-trec2024\n"
            run_results.append(out)

    output_file = "task_1_trec_rag_2024_queries.tsv"
    with open(output_file, "w") as f:
        for result in run_results:
            f.write(result)

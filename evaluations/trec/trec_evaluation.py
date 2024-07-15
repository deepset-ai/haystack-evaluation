import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from evaluations.trec.pipelines import built_basic_rag, pipeline_task_1


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


def task_1(doc_store, topics):
    """
    RAG Task 1: Retrieval - https://trec-rag.github.io/annoucements/2024-track-guidelines/

    For each topic, the system needs to return the TREC runfile containing the ranked list containing the
    top 20 relevant segment IDs from the collection. The topics provided will be non-factoid and require
    long-form answer generation.

    Output Format (Ranked Results)
    Participants should provide their output in the standard TREC format containing top-k=20 MS MARCO v2.1 segments as
    TSV: <r_output_trec_rag_2024.tsv> for each individual topic. Each set of ranked results for a set of topics appears
    in a single file:

    - Topic ID
    - The fixed string “Q0”
    - Segment ID (from the docid field in msmarco_v2.1_doc_segmented_XX.json.gz)
    - Score (integer or float, selected by your system)
    - Run ID where you should mention your team-name (e.g. my-team-name)
    """

    retrieval = pipeline_task_1(doc_store, "all-MiniLM-L12-v2")
    run_results = []

    for topic_id, question in topics.items():
        print(topic_id, "\t", question)
        retrieved_docs = retrieval.run({"query_embedder": {"text": question}, "retriever": {"top_k": 20}})
        for result in retrieved_docs["retriever"]["documents"]:
            out = f"{topic_id}\tQ0\t{result.meta['docid']}\t{result.score}\tdeepset-trec2024\n"
            run_results.append(out)

    output_file = "r_output_trec_rag_2024.tsv"
    with open(output_file, "w") as f:
        for result in run_results:
            f.write(result)


def task_2(doc_store, topics):
    """
    RAG Task 2: Augmented Generation Task (AG) - https://trec-rag.github.io/annoucements/2024-track-guidelines/

    The Augmented Generation task emulates the modern-day RAG task to return the summarized answer ground based on the
    information available in the pre-determined list of top-k segments provided to the participant.

    Participating systems will receive:
     - a list of topics,
     - MS MARCO V2.1 segment collection
     - the ranked list of the top-k relevant segments for each individual topic.

    Output Format (AG Output)
    The final RAG answer should provided in the following JSON format. Each line of this JSONL file contains the
    following entries:

    - run_id (string) containing your team name (e.g. “my-awesome-team-name”)
    - topic_id (string) from the topic_id taken from trec_rag_2024_queries.tsv
    - topic (string) the sentence-level description of the topic taken from trec_rag_2024_queries.tsv
    - references (array) containing the ranked list of top-k segment IDs from the retrieval stage
        (a maximum of only 20 segments is allowed)
    - response_length (integer) containing the total words present in the overall RAG response.
    - answer (array) containing the list of sentences and citations from the references list. The text field contains
    the response and citations field contains the (zero-indexed) reference of the segment from the references list.

    see an example at: https://trec-rag.github.io/annoucements/2024-track-guidelines/
    """
    # ToDo: Implement Task 2 when baselines are released


def task_3(doc_store, topics):
    """
    Task 3: Retrieval-Augmented Generation Task (RAG) - https://trec-rag.github.io/annoucements/2024-track-guidelines/

    Given: Participants will be provided a list of topics and both the MS MARCO v2.1 document and
           MS MARCO v2.1 segment collections.

    Task: Return the summarized answer and ground based on information which you can either the MS MARCO v2.1
          document or segment collection. Develop your own retrieval system to fetch relevant information from the
          MS MARCO v2.1 segment/document collection.

    Output Format (RAG Output) - same as for Task 2
    """

    rag = built_basic_rag(doc_store, "all-MiniLM-L12-v2")

    for topic_id, question in topics.items():
        print(topic_id, "\t", question)
        data = {
            "query_embedder": {"text": question},
            "retriever": {"top_k": 20},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
        _ = rag.run(data, include_outputs_from=["retriever", "llm", "answer_builder", "prompt_builder"])

    """
    # ToDO. the expected output is a JSON file with the following format:

    output = {
        "run_id": "my-awesome-team-name",
        "topic_id": "2027497",
        "topic": "how often should you take your toddler to the potty when potty training",
        "references": [],
        "response_length": 0,
        "answer": [{"text": None, "citations": []}]
    }
    """


def run_tasks():
    topics_file = "../datasets/TREC/topics/topics.dl23.txt"
    queries = "../datasets/TREC/qrels/qrels.dl23-doc-msmarco-v2.1.txt"

    # ToDo: do some indexing of the documents here, we assume there's
    embedding_dim = 384
    doc_store = get_qdrant_doc_store(embedding_dim)

    # read topics and queries
    all_topics = read_topics(topics_file)
    query_relevance = read_query_relevance(queries)
    ground_truth = set(all_topics.keys()).intersection(set(query_relevance.keys()))  # topics w/ relevance judgements
    print(len(ground_truth))

    task_1(doc_store, all_topics)
    task_3(doc_store, all_topics[0:10])


if __name__ == "__main__":
    run_tasks()

# haystack-evaluation

This repository contains examples on how to use Haystack to evaluate systems build with Haystack for different tasks 
and datasets.

This repository is structured as:

- [Evaluations](evaluations/README.md)

- [Techniques/Architectures](evaluations/architectures/README.md)

- [Datasets](datasets/README.md)


## Evaluations

Here we provide full examples on how to use Haystack to evaluate systems build also with Haystack for different tasks and datasets.

Name                                                                      | Dataset       | Evaluation Metrics                                                                                                                                                                                                                                                                                                                                                                                               | 📚 Article|
--------------------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
[RAG with parameter search](evaluation_aragog.py)                         | ARAGOG        | [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator) , [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator), [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)                                                                                                                                                              |[Benchmarking Haystack Pipelines for Optimal Performance](https://haystack.deepset.ai/blog/benchmarking-haystack-pipelines)|
[Baseline RAG vs HyDE using Harness](evaluation_aragog_harness.py)        | ARAGOG        | [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator) , [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator), [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)                                                                                                                                                              | -                                                                                                                                                                                                                                              |
[Extractive QA with parameter search](evaluation_squad_extractive_qa.py)  | SQuAD         | [Answer Exact Match](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator), [DocumentMRR](https://docs.haystack.deepset.ai/docs/documentmrrevaluator), [DocumentMAP](https://docs.haystack.deepset.ai/docs/documentmapevaluator), [DocumentRecall](https://docs.haystack.deepset.ai/docs/documentrecallevaluator), [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)   | -                                                                                   |



## Techniques/Architectures

Name                    | Description                                                                                                                                                                                                              |
------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
[Baseline RAG](evaluations/architectures/basic_rag.py)            | Retrieve-and-Generate (RAG) is a model that combines a retriever and a generator to answer questions. The retriever finds relevant documents and the generator creates an answer based on the retrieved documents.       |
[HyDE](evaluations/architectures/hyde_rag.py)                    | Hybrid Document Embeddings (HyDE) is a HyDE generates a hypothetical document from the query and uses it to retrieve similar documents from the document embedding space.                                                | -                                                                                                                           |
[Extractive QA](evaluations/architectures/extractive_qa.py)           | Extractive Question Answering (QA) is a task where the model is given a question and a document and it has to find the answer to the question in the document. The answer is typically a span of text from the document. |
[Sentence-Window](evaluations/architectures/sentence_window_retrieval.py)         | Sentence-Window is a technique that uses a sliding window to extract chunks/sentences from a document. The extracted chunks/sentences are then used to generate answers to questions.                                    |


## Datasets

Name                    | Suitable Metrics                                                                                                                                                                                                                                                                                                                                                                                              | Description                                                                             
------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
ARAGOG                  | [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator), [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator), [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)                                                                                                                                                            |A collection of papers from ArXiv covering topics around Transformers and Large Language Models, all in PDF format.
SQuAD 2.0               | [Answer Exact Match](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator), [DocumentMRR](https://docs.haystack.deepset.ai/docs/documentmrrevaluator), [DocumentMAP](https://docs.haystack.deepset.ai/docs/documentmapevaluator), [DocumentRecall](https://docs.haystack.deepset.ai/docs/documentrecallevaluator) [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator) | A collection of questions and answers from Wikipedia articles, typically used for training and evaluating models for extractive question-answering tasks.


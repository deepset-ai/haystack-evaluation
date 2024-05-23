# haystack-evaluation

This repository contains examples on how to use Haystack to build different RAG architectures over different datasets, and evaluate the performance.

- [RAG Techniques/Architectures](architectures/README.md)
- [Datasets](datasets/README.md)


## Evaluations

### ARAGOG dataset

The ARAGOG dataset is a collection of AI/LLM-ArXiv papers in PDF format. The dataset is used for training and evaluating models for question-answering tasks.
The dataset contains annotations for the following metrics:

- [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator)
- [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator)
- [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)

### SQuAD dataset 

The SQuAD dataset is a collection of questions and answers from Wikipedia articles. The dataset is used for training and evaluating models for extractive question-answering tasks.
The dataset contains annotations for the following metrics:

- [Answer Exact Match](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator)
- [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)
- [DocumentMRR](https://docs.haystack.deepset.ai/docs/documentmrrevaluator)
- [DocumentMAP](https://docs.haystack.deepset.ai/docs/documentmapevaluator)
- [DocumentRecall](https://docs.haystack.deepset.ai/docs/documentrecallevaluator)

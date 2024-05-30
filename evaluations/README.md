# Evaluations

| Dataset and Evaluation  | Colab                                                                                                                                                                                                                                          |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RAG over ARAGOG dataset | <a href="https://colab.research.google.com/github/deepset-ai/haystack-evaluation/blob/main/evaluations/evaluation_aragog.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |



## ARAGOG

This dataset is based on the paper [Advanced Retrieval Augmented Generation Output Grading (ARAGOG)](https://arxiv.org/pdf/2404.01037).
It's a collection of papers from ArXiv covering topics around Transformers and Large Language Models, all in PDF format. 

The dataset contains:
- 13 PDF papers 
- 107 questions and answers generated with the assistance of GPT-4, and validated/corrected by humans.

It has human annotations for the following metrics:
- [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator)
- [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator)
- [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)

Check the [RAG over ARAGOG dataset notebook](aragog_evaluation.ipynb) for an example.


---

## SQuAD dataset 

The SQuAD dataset is a collection of questions and answers from Wikipedia articles. 
This dataset is typically used for training and evaluating models for extractive question-answering tasks.

The dataset contains:
- 490 Wikipedia articles in text format
- 98k questions whose answers are spans in the articles

It contains human annotations suitable for the following metrics:
- [Answer Exact Match](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator)
- [DocumentMRR](https://docs.haystack.deepset.ai/docs/documentmrrevaluator)
- [DocumentMAP](https://docs.haystack.deepset.ai/docs/documentmapevaluator)
- [DocumentRecall](https://docs.haystack.deepset.ai/docs/documentrecallevaluator)
- [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)


Check the [RAG over SQuAD notebook](squad_rag_evaluation.ipynb) for an example.

Check the [Extractive QA over SQuAD notebook](squad_extractive_qa_evaluation.ipynb) for an example.
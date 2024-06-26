# Datasets


## Overview 


Name                    | Suitable Metrics                                                                                                                                                                                                                                                                                                                                                                                              | Description                                                                             
------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
ARAGOG                  | [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator), [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator), [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)                                                                                                                                                            |A collection of papers from ArXiv covering topics around Transformers and Large Language Models, all in PDF format.
SQuAD 2.0               | [Answer Exact Match](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator), [DocumentMRR](https://docs.haystack.deepset.ai/docs/documentmrrevaluator), [DocumentMAP](https://docs.haystack.deepset.ai/docs/documentmapevaluator), [DocumentRecall](https://docs.haystack.deepset.ai/docs/documentrecallevaluator) [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator) | A collection of questions and answers from Wikipedia articles, typically used for training and evaluating models for extractive question-answering tasks.


## ARAGOG

This dataset is based on the paper [Advanced Retrieval Augmented Generation Output Grading (ARAGOG)](https://arxiv.org/pdf/2404.01037). It's a 
collection of papers from ArXiv covering topics around Transformers and Large Language Models, all in PDF format. 

The dataset contains:
- 13 PDF papers.
- 107 questions and answers generated with the assistance of GPT-4, and validated/corrected by humans.

The following metrics can be used:
- [ContextRelevance](https://docs.haystack.deepset.ai/docs/contextrelevanceevaluator)
- [Faithfulness](https://docs.haystack.deepset.ai/docs/faithfulnessevaluator)
- [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)




## SQuAD dataset 

The SQuAD 1.1 dataset is a collection of questions and answers from Wikipedia articles, and it's typically used for 
training and evaluating models for extractive question-answering tasks. You can find more about this dataset on the 
paper [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/) and on the 
official website [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

The dataset contains:
- 490 Wikipedia articles in text format.
- 98k questions whose answers are spans in the articles.

It contains human annotations suitable for the following metrics:
- [Answer Exact Match](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator)
- [DocumentMRR](https://docs.haystack.deepset.ai/docs/documentmrrevaluator)
- [DocumentMAP](https://docs.haystack.deepset.ai/docs/documentmapevaluator)
- [DocumentRecall](https://docs.haystack.deepset.ai/docs/documentrecallevaluator)
- [Semantic Answer Similarity](https://docs.haystack.deepset.ai/docs/sasevaluator)

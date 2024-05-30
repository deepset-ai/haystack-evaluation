# RAG Techniques/Architectures


## Baseline: basic RAG

This is the baseline RAG technique, that retrieves the top-k documents and then uses  the generator to generate the answer.


## Baseline: Extractive QA

Just like the basic RAG, this technique retrieves the top-k documents, but instead of using the generator to generate the answer, 
it uses an extractive QA model to extract the answer from the retrieved documents.


## Hypothetical Document Embeddings (HyDE)

Hypothetical Document Embeddings (HyDE) tries to tackle this problem of embedding retrievers generalize poorly to new, unseen domains. 

Given a query, HyDE first zero-shot prompts an instruction-following language model  to generate a “fake” hypothetical document that 
captures relevant textual patterns from the initial query - in practice, this is done five times. 

Then each hypothetical document is transformed into an embedding vector and averaged, resulting in a single embedding 
which can be used to identify a neighbourhood in the document embedding space from which similar actual documents are 
retrieved based on vector similarity.  As with any other retriever, these retrieved documents can then be used 
downstream in a pipeline (for example, in a Generator for RAG). 

Paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99.pdf)

## Sentence-Window Retrieval
## Document Summary Index
## Multi-Query
## Maximal Marginal Relevance (MMR) 
## Cohere Re-ranker 
## LLM-based Re-ranker
# RAG Techniques/Architectures


## Basic RAG

This is the baseline RAG technique, that retrieves the top-k document chunks and then uses them to generate the answer.
It uses the same text chunk for indexing/embedding as well as for generating answers.


## Extractive QA

This technique retrieves the top-_k_ documents, but instead of using the generator to generate the answer,  it uses an 
extractive QA model to extract the answer from the retrieved documents.


## Hypothetical Document Embeddings (HyDE)

HyDE first zero-shot prompts an instruction-following language model to generate a “fake” hypothetical document that 
captures relevant textual patterns from the initial query - in practice, this is done five times. 

Each document is transformed into an embedding vector and averaged, resulting in a single embedding 
which is used to identify a neighbourhood in the document embedding space from which similar actual documents are 
retrieved based on vector similarity.

- Paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99.pdf)
- Blog: [HyDE: Hypothetical Document Embeddings for Zero-Shot Dense Retrieval](https://huggingface.co/blog/hyde-zero-shot-dense-retrieval)


## Sentence-Window Retrieval 

The sentence-window approach breaks down documents into smaller chunks (sentences) and indexes them separately.

During retrieval, we retrieve the sentences that are most relevant to the query via similarity search and replace the 
sentence with the full surrounding context, using a static sentence-window around the context.

## Document Summary Index
## Multi-Query
## Maximal Marginal Relevance (MMR) 
## Cohere Re-ranker 
## LLM-based Re-ranker
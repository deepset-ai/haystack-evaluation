# RAG Techniques/Architectures

## Overview 

Here we provide full examples on how to use Haystack to evaluate systems build also with Haystack for different tasks and datasets.

Name                                    | Code                                                         | Description                                                                                                        
----------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
Basic RAG                               | [basic_rag.py](basic_rag.py)                                 | Retrieves the top-k document chunks and then passes them to an LLM generate the answer. 
Extractive QA                           | [extractive_qa.py](extractive_qa.py)                         | Retrieves the top-_k_ documents and uses an extractive QA model to extract the answer from the documents.
Hypothetical Document Embeddings (HyDE) | [hyde_rag.py](hyde_rag.py)                                   | HyDE generates a hypothetical document from the query and uses it to retrieve similar documents from the document embedding space.
Sentence-Window Retrieval               | [sentence_window_retrieval.py](sentence_window_retrieval.py) | Breaks down documents into smaller chunks (sentences) and indexes them separately. Retrieves the most relevant sentences and replaces them with the full surrounding context. 
Document Summary Index                  | ToDo                                                         | ToDo
Multi-Query                             | ToDo                                                         | ToDo
Maximal Marginal Relevance (MMR)        | ToDo                                                         | ToDo
Cohere Re-ranker                        | ToDo                                                         | ToDo
LLM-based Re-ranker                     | ToDo                                                         | ToDo




### Basic RAG

This is the baseline RAG technique, that retrieves the top-k document chunks and then uses them to generate the answer.
It uses the same text chunk for indexing/embedding as well as for generating answers.

---

### Extractive QA

This technique retrieves the top-_k_ documents, but instead of using the generator to provide the answer, it uses an 
extractive QA model to extract the answer from the retrieved documents.

---

### Hypothetical Document Embeddings (HyDE)

HyDE first zero-shot prompts an instruction-following language model to generate a “fake” hypothetical document that 
captures relevant textual patterns from the initial query - in practice, this is done five times. 

Each document is transformed into an embedding vector and averaged, resulting in a single embedding 
which is used to identify a neighbourhood in the document embedding space from which similar actual documents are 
retrieved based on vector similarity.

- Paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99.pdf)
- Blog: [HyDE: Hypothetical Document Embeddings for Zero-Shot Dense Retrieval](https://huggingface.co/blog/hyde-zero-shot-dense-retrieval)

---

### Sentence-Window Retrieval 

The sentence-window approach breaks down documents into smaller chunks (sentences) and indexes them separately.

During retrieval, we retrieve the sentences that are most relevant to the query via similarity search and replace the 
sentence with the full surrounding context, using a static sentence-window around the context.
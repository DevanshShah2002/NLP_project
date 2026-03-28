## Problem statment:
Enhancing Reliability and Contextual Reasoning in Retrieval-Augmented Generation (RAG)

### 1. The Context
Retrieval-Augmented Generation (RAG) has become the industry standard for grounding Large Language Models (LLMs) in external, private, or real-time data. By fetching relevant document chunks from a vector database, RAG minimizes the LLM’s reliance on internal training data, thereby reducing generic outputs. However, as the demand for autonomous "Agentic AI" grows, the limitations of traditional, "vanilla" RAG architectures have become a significant bottleneck for enterprise-grade applications.

### 2. The Core Problems
Computational Inefficiency & Latency (The "One-Size-Fits-All" Problem): Current RAG systems treat all queries with equal complexity. For a simple query, retrieving a high number of chunks ($k$) is computationally expensive and introduces unnecessary "noise" into the context window. Conversely, for complex queries, a small fixed $k$ often misses critical information. There is a lack of an intelligent routing mechanism to distinguish between a "Simple" lookup and a "Complex" multi-hop inquiry.

The Relational Gap in Vector Space: Standard RAG relies on semantic similarity (vector embeddings). While effective for finding similar text, it fails to understand the structural or hierarchical relationships between entities. It struggles with "global" questions or queries that require connecting disparate data points across a large corpus.

Logical Hallucinations & Faithfulness: Even with retrieved context, LLMs frequently hallucinate by "hallucinating between the lines"—misinterpreting the relationship between two retrieved chunks or generating answers that are not explicitly supported by the source material.

### 3. Proposed
Technical ObjectivesThis project aims to bridge these gaps by developing an Adaptive GraphRAG pipeline with a multi-layered evaluation strategy:

A. Adaptive Query Classification (SLM Layer):To optimize cost and time, we propose integrating a Small Language Model (SLM) classifier at the entry point of the pipeline.

Simple Queries: If the SLM identifies a direct fact-seeking query, the system triggers a lightweight, low-latency retrieval path.

Complex Queries: For multi-hop reasoning (e.g., "Who established the city where Maharana Pratap was born?"), the system recognizes that it must first identify the birth city and then search for its founder. This triggers an iterative or recursive retrieval process, adjusting the Top-$k$ and the depth of the search dynamically.

B. Relational Retrieval via GraphRAG:To solve the relational gap, we will implement GraphRAG. By representing the corpus as a Knowledge Graph (Entities and Relationships), the system can navigate logical paths that vector search alone would miss. This allows the model to "walk" through a data structure to find the connection between "Maharana Pratap," his "Birthplace," and the "Founder" of that location.

C. Hallucination Mitigation via the "Judge" Pipeline:To ensure faithfulness, the final stage will involve an Evaluator/Judge LLM. This model will perform a NLI (Natural Language Inference) check to ensure that every claim in the generated response is directly mapped to a specific retrieved chunk, discarding or regenerating any information that lacks evidentiary support.

## Reference:

Adaptive RAG:
https://arxiv.org/pdf/2403.14403

https://medium.com/@tuhinsharma121/understanding-adaptive-rag-smarter-faster-and-more-efficient-retrieval-augmented-generation-38490b6acf88

GraphRAG:
https://arxiv.org/pdf/2404.16130


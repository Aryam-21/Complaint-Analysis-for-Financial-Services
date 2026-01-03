# Complaint-Analysis-for-Financial-Services
Embedding-Ready Pipeline for Retrieval-Augmented Generation (RAG)

📌 Project Overview

This project focuses on analyzing consumer complaints in the financial services domain and preparing them for downstream machine learning and Retrieval-Augmented Generation (RAG) applications.

The work is divided into fore core tasks:

Task 1: Data Analysis & Preprocessing

Task 2: Text Chunking & Embedding Pipeline

Task 3: Building the RAG Core Logic and Evaluation

Task 4: Creating an Interactive Chat Interface

The final output of this repository is an embedding-ready dataset where complaint narratives are cleaned, stratified, chunked, embedded, and stored with rich metadata for traceability.

📂 Repository Structure
Complaint-Analysis-for-Financial-Services/
│
├── data/
│   ├── raw/                     # Original complaint datasets
│   └── processed/               # Cleaned & filtered datasets
│
├── notebooks/
│   ├── eda.ipynb
│   └── embedding.ipynb
│
├── src/
│   ├── embedding.py
│
├── vector_store/
│   ├── chunks_metadata.csv      # Chunk metadata
│   ├── chunks_embeddings.npy    # Stored embeddings
│   └── faiss.index              # FAISS index
│
├── requirements.txt
└── README.md

✅ Task 1: Data Analysis & Preprocessing (Completed)
Objective

Prepare a clean, feature-rich complaint dataset suitable for modeling and semantic retrieval.

Key Steps

Loaded raw CFPB complaint data

Removed duplicates and invalid records

Handled missing values

Performed exploratory data analysis (EDA)

Engineered features such as:

Word counts

Cleaned complaint narratives

Verified class imbalance across product categories

Output

Cleaned dataset stored in:

data/processed/filtered_complaints.csv


Task 1 fully satisfies the assignment requirements.

✅ Task 2: Text Chunking & Embedding Pipeline (Core Completed)
Objective

Transform complaint narratives into vector embeddings suitable for semantic search and RAG systems.

Pipeline Overview

Implemented as a reusable Python class:

ComplaintEmbeddingPipeline

Steps Implemented
1. Stratified Sampling

Preserves product-level distribution

Prevents bias during downstream retrieval

pipeline.stratified_sample()

2. Text Chunking

Uses RecursiveCharacterTextSplitter

Chunk size: 500 characters

Overlap: 100 characters

Each chunk retains metadata:

Complaint ID

Product category

Chunk index

Chunk text

pipeline.chunk_text()

3. Embedding Model Selection

Model: sentence-transformers/all-MiniLM-L6-v2

Why this model?

Lightweight and fast

Strong semantic performance

384-dimensional embeddings

Widely adopted in production RAG systems

pipeline.load_embedding_model()
pipeline.generate_embeddings()

4. Embedding Storage

Embeddings stored as float32

Metadata preserved for traceability

Saved as:

CSV for metadata

NumPy array for embeddings

pipeline.save_chunks_and_embeddings(
    metadata_path="../vector_store/chunks_metadata.csv",
    embedding_path="../vector_store/chunks_embeddings.npy"
)

Vector Store Integration (Planned / Partially Implemented)
Current Status

Embeddings are ready for indexing

FAISS integration is implemented in code but optional due to environment constraints

FAISS Index Support
pipeline.build_faiss_index("../vector_store/faiss.index")


Uses IndexFlatL2 (exact similarity search)

Supports persistence and reload

Fully compatible with RAG pipelines

⚠️ Note: On Windows, FAISS requires faiss-cpu.
ChromaDB is a planned alternative backend for broader compatibility.

 How to Run the Pipeline
from src.embedding import ComplaintEmbeddingPipeline

pipeline = ComplaintEmbeddingPipeline(
    csv_path="../data/processed/filtered_complaints.csv"
)

pipeline.load_data()
pipeline.stratified_sample()
pipeline.chunk_text()
pipeline.load_embedding_model()
pipeline.generate_embeddings()
pipeline.save_chunks_and_embeddings(
    metadata_path="../vector_store/chunks_metadata.csv",
    embedding_path="../vector_store/chunks_embeddings.npy"
)

Future Work

Based on instructor feedback and project roadmap:

1. Wire FAISS or ChromaDB as a default vector store

2. Add semantic search query interface

3.  Integrate with LLMs for full RAG workflow

4.  Improve documentation and usage examples

5. Add configuration support (YAML / env)

# Complaint-Analysis-for-Financial-Services

This repository contains the implementation for **Task 1 and Task 2** of the CrediTrust AI Mastery Challenge: building a **Retrieval-Augmented Generation (RAG) pipeline** for customer complaints. It focuses on **data preprocessing, stratified sampling, chunking, embedding, and FAISS vector indexing**, providing a foundation for a complaint-answering AI system.

---

## 🏢 Business Context

CrediTrust Financial is a digital finance company operating in East Africa. Internal teams handle thousands of customer complaints each month across five product categories:

- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

Currently, manual complaint analysis is slow and reactive. The goal is to **turn unstructured complaint narratives into actionable insights** quickly using AI.

---

## 📁 Repository Structure

```text
rag-complaint-chatbot/
├── data/
│   ├── raw/                    # Raw CFPB complaint datasets
│   └── processed/              # Cleaned/filtered complaints
├── vector_store/               # Persisted embeddings and FAISS index
├── notebooks/                  # Jupyter notebooks for EDA and pipeline demos
├── src/                        # Source code (pipeline classes)
├── tests/                      # Unit tests for pipeline methods
├── app.py                      # Future RAG chatbot UI (Gradio/Streamlit)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

 Task 1: EDA and Data Preprocessing
Objectives:

Load the CFPB complaint dataset.

Explore and summarize complaints by product, issue type, and narrative length.

Clean text narratives for embedding quality:

Lowercasing

Removing boilerplate or special characters

Filtering only relevant products

Save cleaned, filtered dataset for downstream tasks.

Output:

data/processed/filtered_complaints.csv – cleaned and filtered complaints.

Notebook with visualizations and analysis of complaint distributions.

 Task 2: Chunking, Embedding, and FAISS Indexing
Objectives:

Perform stratified sampling (10K–15K complaints) to ensure proportional representation of product categories.

Chunk long complaint narratives (default: 500 characters with 50 overlap) for effective embedding.

Generate vector embeddings using Sentence-Transformers all-MiniLM-L6-v2:

Small, fast, and suitable for semantic search.

Produces 384-dimensional embeddings.

Build a FAISS vector store to store embeddings for semantic search.

Save both metadata and embeddings for future RAG queries.

Pipeline Highlights:

Class-based, modular design: ComplaintEmbeddingPipeline

load_data()

stratified_sample()

chunk_text()

load_embedding_model()

generate_embeddings(batch_size=64)

save_chunks_and_embeddings(metadata_path, embedding_path)

build_faiss_index(index_path)

Error handling ensures proper method sequence and robust execution.

Outputs:

Metadata CSV: vector_store/chunks_metadata.csv

Embeddings array: vector_store/chunks_embeddings.npy

FAISS index: vector_store/faiss.index

⚙️ Installation
bash
Copy code
git clone <repo-url>
cd rag-complaint-chatbot

# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt

# FAISS installation
pip install faiss-cpu        # For CPU
# pip install faiss-gpu       # If GPU is available
Python dependencies:

pandas

numpy

tqdm

scikit-learn

sentence-transformers

faiss-cpu

langchain (for text splitting)

 Usage Example
python
Copy code
from src.pipeline import ComplaintEmbeddingPipeline

pipeline = ComplaintEmbeddingPipeline(
    csv_path="data/processed/filtered_complaints.csv"
)

# Task 1 + Task 2
pipeline.load_data()
pipeline.stratified_sample()
pipeline.chunk_text()
pipeline.load_embedding_model()
pipeline.generate_embeddings(batchsize=64)
pipeline.save_chunks_and_embeddings(
    metadata_path="vector_store/chunks_metadata.csv",
    embedding_path="vector_store/chunks_embeddings.npy"
)

# Build FAISS index
faiss_index = pipeline.build_faiss_index(index_path="vector_store/faiss.index")
 Proportional Sampling Verification
The stratified sampling ensures product categories in the sample reflect the original distribution:

python
Copy code
import pandas as pd

pd.concat([
    pipeline.df[pipeline.product_col].value_counts(normalize=True).rename("Original"),
    pipeline.df_sample[pipeline.product_col].value_counts(normalize=True).rename("Sample")
], axis=1)
## Next Steps (RAG Integration)
Load chunks_metadata.csv and faiss.index in a RAG retrieval pipeline.

Use retrieved chunks as context for a language model to answer natural-language questions.

Integrate with Gradio or Streamlit for an interactive chatbot.

Evaluate RAG answers with qualitative metrics and real complaints.

 Tests
Create basic unit tests in tests/:

Ensure CSV loads correctly.

Check chunking produces non-empty text.

Validate embedding dimensions.

Verify FAISS index stores the correct number of vectors.

 Notes & Best Practices
Batch embeddings to avoid memory overload.

Use GPU if available (device='cuda') for faster embedding.

Maintain metadata to trace each chunk back to the original complaint.

Keep vector store and metadata versioned for reproducibility.

References
FAISS Documentation

Sentence Transformers

LangChain Text Splitter

Gradio Docs

CFPB Dataset: Consumer Financial Protection Bureau

Author: Aryam Tesfay
Project: 10 Academy AI Mastery Challenge
Tasks: 1 (EDA & preprocessing), 2 (Embedding & FAISS indexing)
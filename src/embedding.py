import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
class ComplaintEmbeddingPipeline:
    """
    pipeline for:
    1. Stratified sampling of complaints
    2. Chunking complaint narratives
    3. Generating embeddings using sentence-transformers
    """
    def __init__(self, csv_path, product_col='Product', narrative_col='cleaned_narrative',
                 complaint_id_col='Complaint ID', sample_size=14000,
                 chunk_size=500, chunk_overlap=50, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.csv_path = csv_path
        self.product_col = product_col
        self.narrative_col = narrative_col
        self.complaint_id_col = complaint_id_col
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        # Attributes to be set later
        self.df = None
        self.df_sample = None
        self.chunk_df = None
        self.embedding_model = None
        
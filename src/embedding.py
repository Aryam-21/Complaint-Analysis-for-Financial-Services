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
        self.chunks_df = None
        self.embedding_model = None
    def load_data(self):
        """ Load clean complaint CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f'[INFO] Loaded {len(self.df)} complaints from {self.csv_path}')
        except FileNotFoundError:
            raise FileNotFoundError(f'File not Found: {self.csv_path}')
        except Exception as e:
            raise RuntimeError(f'Error loading CSV: {e}')
    def stratified_sample(self):
        """ Perform stratified sampling by product category"""
        if self.df is None:
            raise ValueError('Data not loaded. call load_data() first.')
        if self.sample_size > len(self.df):
            raise ValueError('Sample size cannot exceed dataset size.')
        self.df_sample,_ = train_test_split(self.df,
                                            stratify=self.df[self.product_col],
                                            train_size=self.sample_size,
                                            random_state=42)
        print(f'[INFO] Stratified sample created with {len(self.df_sample)} complaints.')
    def chunk_text(self):
        """Split complaints into text chunks."""
        if self.df_sample is None:
            raise ValueError('Sample not created. Call stratified_sample() first.')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        chunks = []
        for _,row in self.df_sample.iterrows():
            narrative = row[self.narrative_col]
            if pd.isna(narrative) or narrative.strip() == '':
                continue # skip empty narratives
            for idx, chunk in enumerate(text_splitter.split_text(narrative)):
                chunks.append({
                    'complaint_id': row[self.complaint_id_col],
                    'product_category':row[self.product_col],
                    'chunk_index': idx,
                    'chunk_text': chunk
                })
        self.chunks_df = pd.DataFrame(chunks)
        print(f'[INFO] created {len(self.chunks_df)} text chunks')
    def load_embedding_model(self):
        """Load sentence-transformers embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f'[INFO] lOADED EMBEDDING MODEL: {self.embedding_model_name}')
        except Exception as e:
            raise RuntimeError(f'Error loading embeddingmodel: {e}')
    def generate_embeddings(self, batchsize=64):
        """generate embeddings for all chunks."""
        if self.chunks_df is None:
            raise ValueError('Chunks not created. call chunk_text() first.')
        if self.embedding_model is None:
            raise ValueError('Embedding model not loaded. Call load_embedding_model(0 first.)')
        # Get chunk texts
        texts = self.chunks_df['chunk_text'].tolist()
        # Generate embeddings in one call
        embeddings = self.embedding_model.encode(texts)
        # Convert to float32 for vector databases
        embeddings = np.array(embeddings, dtype='float32')
        # store embeddings
        self.chunks_df['embedding'] = list(embeddings)
        print(f"[INFO] Generated embeddings for {len(self.chunks_df)} chunks")
    def save_chunks_and_embeddings(self, metadata_path, embedding_path=None):
        """Save chunks metadata and optionally embeddings"""
        if self.chunks_df is None:
            raise ValueError('chunks not created. call chunk_text() frist.')
        metadata_cols = ["complaint_id", "product_category", "chunk_index", "chunk_text"]
        self.chunks_df[metadata_cols].to_csv(metadata_path, index=False)
        print(f'[INFO] Saved chunks metadata to {metadata_path}')
        if embedding_path:
            np.save(embedding_path,np.vstack(self.chunks_df['embedding'].values))
            print(f"[INFO] Saved embeddings array to {embedding_path}")
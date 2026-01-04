import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
class ComplaintRetriever:
    """task 3 FAISS Retriever using CSV metadata"""
    def __init__(self,
                 faiss_index_path:str,
                 metadata_csv_path:str,
                 embedding_model_name:str="sentence-transformers/all-MiniLM-L6-v2",
                 top_k:int=5):
        # Load FAISS index
        try:
            self.index = faiss.read_index(faiss_index_path)
        except Exception as e:
            raise RuntimeError(f'Failed to load FAISS index: {e}')
        # Load metadata CSV
        try:
            self.metadata = pd.read_csv(metadata_csv_path)
        except Exception as e:
            raise RuntimeError(f'Failed to load metadata model: {e}')
        # Load embedding model
        try:
            self.embedder = SentenceTransformer(embedding_model_name)
        except Exception as e:
            raise RuntimeError(f'Faile to load embedding model: {e}')
        self.top_k = top_k
        # Safty check
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index size does not match metadata rows.")
    def retrieve(self, question:str):
        """Embeded user question and retrieve top-k complaint chunks"""
        if not question or not question.strip():
            raise ValueError('Question must be non-empty string.')
        # Embed question
        query_vector = self.embedder.encode([question],normalize_embeddings=True).astype('float32')
        # FAISS similarity search
        distances, indices = self.index.search(query_vector, self.top_k)
        # collect retrieved chunks
        results = []
        for idx in indices[0]:
            row = self.metadata.iloc[int(idx)]
            results.append({
                "chunk_text": row['chunk_text'],
                "complaint_id": row['complaint_id'],
                'product_category': row['product_category'],
                'chunk_index': row['chunk_index'],
            })
        return results
class RAGPromptBuilder:
    """Build a robust, grounded prompt for Task-3"""
    PROMPT_TEMPLATE = """
    You are a financial analyst assistant for CrediTrust.
    Your task is to answer questions about customer complaints.
    Use ONLY the retrieved complaint excerpts below to formulate your answer.
    If the context does not contain enough information, state clearly that
    you do not have enough information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """.strip()
    def build_prompt(self, retrieved_chunks, question:str) -> str:
        if not retrieved_chunks:
            context = 'No relevant complaint excerpts were retrieved.'
        else:
            context = '\n\n'.join(f"-  {chunk['chunk_text']}" for chunk in retrieved_chunks)
        return self.PROMPT_TEMPLATE.format(context=context,question=question)
class RAGGenerator:
    """Sends the prompt to an LLm and returns generated responce"""
    def __init__(self,
                 model_name:str="google/flan-t5-small", max_new_tokens=200):
        try:
            self.generator = pipeline(
                task="text2text-generation",
                model=model_name,
                device=-1)
            self.max_new_tokens = max_new_tokens
        except Exception as e:
            raise RuntimeError(f'Failed to load LLM: {e}')
    def generate_answer(self, prompt:str) -> str:
            """Send the combined prompt to the LLM and returns the answer."""
            if not prompt or not prompt.strip():
                raise ValueError('Prompt must be a non-empty string.')
            try:
                output = self.generator(prompt,
                                        max_new_tokens=self.max_new_tokens,
                                        do_sample=False)
                return output[0]['generated_text']
            except Exception as e:
                raise RuntimeError(f'LLM generation failed: {e}')























































































































































































































































































































































































































            
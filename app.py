import gradio as gr
from src.faiss_retriever import ComplaintRetriever, RAGPromptBuilder, RAGGenerator

# Initialize RAG components
retriever = ComplaintRetriever(faiss_index_path='../vector_store/faiss.index',
                               metadata_csv_path='../vector_store/chunks_metadata.csv')
prompt_builder = RAGPromptBuilder()
generator = RAGGenerator(model_name="google/flan-t5-small")
# Core RAG pipeline function
def answer_question(question):
    if not question.strip():
        return "please enter a question.",""
    # Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve(question)

    #Build prompt
    prompt = prompt_builder.build_prompt(retrieved_chunks=retrieved_chunks, question=question)

    # Generate answer
    answer = generator.generate_answer(prompt)

    #Format sources for display
    sources = "\n\n".join(f"• Product: {c['product_category']} | Complaint ID: {c['complaint_id']}\n{c['chunk_text']}"
        for c in retrieved_chunks)
    return answer, sources
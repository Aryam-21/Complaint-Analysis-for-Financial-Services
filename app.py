import gradio as gr
from src.faiss_retriever import ComplaintRetriever, RAGPromptBuilder, RAGGenerator

# Initialize RAG components
retriever = ComplaintRetriever(faiss_index_path='vector_store/faiss.index',
                               metadata_csv_path='vector_store/chunks_metadata.csv')
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
# Gradio UI
with gr.Blocks(title="CrediTrust Complaint Intelligence Assistant") as demo:
    gr.Markdown(
        """
        # 💬 CrediTrust Complaint Intelligence Assistant
        Ask questions about customer complaints across financial products.
        The answers are generated **only from real complaint data**.
        """
    )
    question_input = gr.Textbox(
        label='Enter your question',
        placeholder='Why are customers unhappy with credit cards?',
        lines=2
    )
    ask_button = gr.Button('Ask')
    clear_button = gr.Button('Clear')
    answer_output = gr.Textbox(
        label='AI Generated Answer',
        lines=6
    )
    sources_output = gr.Textbox(
        label='Sources (Retrieved Complaint Excerpts)',
        lines=10
    )
    ask_button.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    clear_button.click(
        fn=lambda: ("","",""),
        inputs=None,
        outputs=[question_input, answer_output, sources_output]
    )
    if __name__ == "__main__":
        demo.launch()
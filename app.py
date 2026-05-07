from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile



from src.ingestion.pdf_loader import load_pdf_text
from src.ingestion.chunker import chunk_text
from src.embeddings.openai_embedder import embed_texts, embed_query
from src.vector_store.faiss_store import FAISSStore
from src.qa.rag_chat import generate_answer



st.set_page_config(page_title="RAG Study Chatbot", layout="wide")
st.title("📚 RAG‑Based Study Chatbot")

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.session_state.faiss_store is None:
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)

        embeddings = embed_texts(chunks)
        st.session_state.faiss_store = FAISSStore(embeddings, chunks)

    st.success("PDF processed successfully ✅")

question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.faiss_store is None:
        st.warning("Upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            query_emb = embed_query(question)
            context = st.session_state.faiss_store.search(query_emb)
            answer = generate_answer(question, context)

        st.subheader("Answer")
        st.write(answer)

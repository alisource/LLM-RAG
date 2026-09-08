import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Multi-Source Medical RAG Assistant", page_icon="🤖")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:
    st.error("GROQ_API_KEY tidak ditemukan!")
    st.stop()

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    db_pdf = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db_pdf"), embedding_function=embeddings)
    db_json = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db_json"), embedding_function=embeddings)
    db_csv = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db_csv"), embedding_function=embeddings)
    
    # Model Groq
    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile", # Disarankan memakai llama-3.3-70b jika gpt-oss bermasalah
        temperature=0.0
    )

    return db_pdf, db_json, db_csv, llm_groq

db_pdf, db_json, db_csv, llm_groq = load_resources()

# --- PROMPT TEMPLATE YANG DIPERKETAT ---
template = """Kamu adalah Asisten Informasi Kesehatan.

Jawablah pertanyaan berdasarkan aturan berikut:
1. JIKA pertanyaan berkaitan dengan kesehatan/medis, gunakan Konteks di bawah ini untuk menjawabnya secara akurat.
2. JIKA pertanyaan SAMA SEKALI TIDAK berkaitan dengan kesehatan/medis (contoh: pertanyaan politik, umum, sejarah, geografi):
   - Jawab pertanyaan tersebut secara langsung berdasarkan pengetahuan umummu.
   - DILARANG KERAS menyertakan istilah medis, saran kesehatan, atau disclaimer medis/dokter sama sekali.

Konteks Medis:
{context}

Pertanyaan:
{question}

Jawaban:"""

PROMPT = PromptTemplate.from_template(template)

retriever_pdf = db_pdf.as_retriever(search_kwargs={"k": 2})
retriever_json = db_json.as_retriever(search_kwargs={"k": 2})
retriever_csv = db_csv.as_retriever(search_kwargs={"k": 2})

def retrieve_multi_source_docs(query):
    docs_p = retriever_pdf.invoke(query)
    docs_j = retriever_json.invoke(query)
    docs_c = retriever_csv.invoke(query)
    return docs_p + docs_j + docs_c

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- UI STREAMLIT ---
st.title("Medical RAG Assistant (Groq Cloud)")
st.write("Tanyakan informasi kesehatan berdasarkan basis data lokal Anda (PDF, JSON, & CSV).")

# Fitur Tambahan: Reset Session agar memori lama tidak mengendap
if st.sidebar.button("Clear Chat / Reset"):
    st.session_state.clear()
    st.rerun()

user_query = st.text_input("Masukkan pertanyaan Anda:")

if user_query:
    with st.spinner("Sedang memproses..."):
        # 1. Ambil dokumen sekali saja agar efisien
        retrieved_docs = retrieve_multi_source_docs(user_query)
        context_text = format_docs(retrieved_docs)
        
        # 2. Format prompt
        formatted_prompt = PROMPT.format(context=context_text, question=user_query)
        
        # 3. Panggil LLM
        response_bio = llm_groq.invoke(formatted_prompt).content
        
        st.subheader("Jawaban:")
        st.write(response_bio)
        
        # Tampilkan sumber dokumen
        with st.expander("Lihat Dokumen Konteks yang Ditarik dari Database"):
            for i, doc in enumerate(retrieved_docs):
                source_name = doc.metadata.get('source') or doc.metadata.get('file_name') or 'Database Lokal'
                st.markdown(f"**Dokumen {i+1}** (*{os.path.basename(source_name)}*)")
                st.text(doc.page_content[:300] + "...")

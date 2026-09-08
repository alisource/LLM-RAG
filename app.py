import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Multi-Source Medical RAG Assistant", page_icon="🤖")

# Memuat token API dari file .env lokal
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:
    st.error("GROQ_API_KEY tidak ditemukan! Buat file .env di lokal atau isi Secrets di Streamlit Cloud.")
    st.stop()

@st.cache_resource
def load_resources():
    # 1. Inisialisasi Embeddings (tetap menggunakan HuggingFace karena sangat ringan & lokal)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Mendapatkan direktori tempat file app.py berada
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Memuat kembali vector database terpisah dari direktori lokal dengan absolute path
    db_pdf = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db_pdf"), embedding_function=embeddings)
    db_json = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db_json"), embedding_function=embeddings)
    db_csv = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db_csv"), embedding_function=embeddings)
    
    # 3. Inisialisasi LLM menggunakan Groq dengan model yang diminta
    llm_groq = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0
    )

    return db_pdf, db_json, db_csv, llm_groq

# Memuat resource (database & Groq LLM di-cache agar efisien)
db_pdf, db_json, db_csv, llm_groq = load_resources()

# 4. Konfigurasi Prompt & Retriever (Memperketat Prompt Behavior untuk Penanganan Non-Medis)
template = """Gunakan konteks berikut untuk menjawab pertanyaan. Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu.

Aturan Tambahan:
1. Jika pertanyaan BUKAN tentang kesehatan/medis (seperti politik, pengetahuan umum, sejarah):
   - Jawab pertanyaan tersebut secara langsung berdasarkan pengetahuan umum Anda.
   - DILARANG KERAS menyertakan istilah medis, saran kesehatan, atau disclaimer medis/dokter sama sekali.
2. Jika pertanyaan tentang kesehatan/medis, gunakan konteks di bawah untuk menjawab.

Konteks:
{context}

Pertanyaan:
{question}
Jawaban:"""

PROMPT = PromptTemplate.from_template(template)

retriever_pdf = db_pdf.as_retriever(search_kwargs={"k": 4})
retriever_json = db_json.as_retriever(search_kwargs={"k": 4})
retriever_csv = db_csv.as_retriever(search_kwargs={"k": 4})

def retrieve_multi_source_docs(query):
    docs_p = retriever_pdf.invoke(query)
    docs_j = retriever_json.invoke(query)
    docs_c = retriever_csv.invoke(query)
    return docs_p + docs_j + docs_c

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Antarmuka Pengguna (Streamlit UI) ---
st.title("Medical RAG Assistant (Groq Cloud)")
st.write("Tanyakan informasi kesehatan berdasarkan basis data lokal Anda (PDF, JSON, & CSV).")

user_query = st.text_input("Masukkan pertanyaan Anda (Contoh: What are the symptoms of Glaucoma?):")

if user_query:
    with st.spinner("Sedang mencari jawaban..."):
        # Menghindari duplikasi pencarian ke ChromaDB: panggil sekali lalu pakai hasilnya untuk LLM dan Tampilan UI
        retrieved_docs = retrieve_multi_source_docs(user_query)
        context_text = format_docs(retrieved_docs)
        
        # Format prompt dan panggil LLM secara langsung tanpa eksekusi ulang retrieval
        formatted_prompt = PROMPT.format(context=context_text, question=user_query)
        response_bio = llm_groq.invoke(formatted_prompt).content
        
        st.subheader("Jawaban:")
        st.write(response_bio)
        
        st.subheader("Sumber Dokumen:")
        for i, doc in enumerate(retrieved_docs):
            source_name = doc.metadata.get('source') or doc.metadata.get('file_name') or doc.metadata.get('source_file') or 'unknown'
            display_name = os.path.basename(source_name) if source_name != 'unknown' else 'Database Lokal'
            
            with st.expander(f"Dokumen {i+1} (Sumber: {display_name})"):
                st.write(doc.page_content)

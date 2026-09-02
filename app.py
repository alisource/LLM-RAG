import os
import gc
import torch
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import pipeline, BitsAndBytesConfig

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Multi-Source Medical RAG Assistant", page_icon="🤖")

# Memuat token API dari file .env lokal
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("LLMtoken")

@st.cache_resource
def load_resources():
    # 1. Inisialisasi Embeddings (harus sama dengan saat pembuatan database)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # 2. Memuat kembali vector database terpisah dari direktori lokal
    db_pdf = Chroma(persist_directory=r"C:\Users\Nur Ali Astaguna\Downloads\chroma_db_pdf", embedding_function=embeddings)
    db_json = Chroma(persist_directory=r"C:\Users\Nur Ali Astaguna\Downloads\chroma_db_json", embedding_function=embeddings)
    db_csv = Chroma(persist_directory=r"C:\Users\Nur Ali Astaguna\Downloads\chroma_db_csv", embedding_function=embeddings)

    # 3. Inisialisasi Model Qwen dengan Kuantisasi 4-bit
    torch.cuda.empty_cache()
    gc.collect()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    llm_qwen = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        model_kwargs={"quantization_config": quantization_config},
        max_new_tokens=150,
        pad_token_id=151643
    )

    return db_pdf, db_json, db_csv, llm_qwen

# Memuat resource (database & LLM di-cache agar tidak meload ulang terus)
db_pdf, db_json, db_csv, llm_qwen = load_resources()

# 4. Konfigurasi Prompt & Retriever
template = """Gunakan konteks berikut untuk menjawab pertanyaan. Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu.

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

def call_huggingface_pipeline(prompt_value):
    prompt_str = prompt_value.to_string()
    result = llm_qwen(
        prompt_str, 
        max_new_tokens=150, 
        pad_token_id=151643,
        repetition_penalty=1.2,
        do_sample=False  # Menjaga agar hasil respons stabil dan deterministik
    )
    generated_text = result[0]['generated_text']
    if prompt_str in generated_text:
        return generated_text[len(prompt_str):].strip()
    return generated_text.strip()

# 5. Susun RAG Chain menggunakan LCEL
rag_chain_bio = (
    {
        "context": RunnableLambda(retrieve_multi_source_docs) | RunnableLambda(format_docs), 
        "question": RunnablePassthrough()
    }
    | PROMPT
    | RunnableLambda(call_huggingface_pipeline)
)

# --- Antarmuka Pengguna (Streamlit UI) ---
st.title("Medical RAG Assistant (Qwen 2.5)")
st.write("Tanyakan informasi kesehatan berdasarkan basis data lokal Anda (PDF, JSON, & CSV).")

user_query = st.text_input("Masukkan pertanyaan Anda (Contoh: What are the symptoms of Glaucoma?):")

if user_query:
    with st.spinner("Sedang mencari jawaban..."):
        # Jalankan RAG chain
        response_bio = rag_chain_bio.invoke(user_query)
        
        # Tampilkan Jawaban
        st.subheader("Jawaban:")
        st.write(response_bio)
        
        # Tampilkan Dokumen Sumber
        st.subheader("Sumber Dokumen:")
        retrieved_docs = retrieve_multi_source_docs(user_query)
        
        for i, doc in enumerate(retrieved_docs):
            source_name = doc.metadata.get('source') or doc.metadata.get('file_name') or doc.metadata.get('source_file') or 'unknown'
            display_name = os.path.basename(source_name) if source_name != 'unknown' else 'Database Lokal'
            
            with st.expander(f"Dokumen {i+1} (Sumber: {display_name})"):
                st.write(doc.page_content)
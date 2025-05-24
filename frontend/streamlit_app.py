# filepath: /root/Binky/frontend/streamlit_app.py
import sys
import os

# Tambahkan path root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

# Import langsung dari file tanpa melalui package
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
from helpers import setup_rag_pipeline

from src.chatbot import LlamaRagChatbot
import config

st.set_page_config(
    page_title="Binky - Library Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
body {
    background-color: #F7F9FA;
    color: #212121;
    font-family: 'Segoe UI', sans-serif;
}

header, .st-emotion-cache-zq5wmm.ezrtsby0 {
    background-color: #E6F4F4;
}

.css-18e3th9 {
    padding: 1rem;
    background-color: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

h1, h2, h3, h4 {
    color: #1A8181;
}

.stButton button {
    background-color: rgb(59,165,165);
    color: white;
    font-weight: bold;
    border-radius: 6px;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background-color: rgb(36,154,154);
    color: white;
}

.stChatMessage.user {
    background-color: #E6F4F4;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.stChatMessage.assistant {
    background-color: #FFFFFF;
    border-left: 4px solid rgb(59,165,165);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.stSidebar {
    background-color: #E6F4F4;
    border-right: 1px solid #ccc;
}

</style>
""", unsafe_allow_html=True)

# --- Load logo ---

# --- Language Selector ---
language_options = {
    "Bahasa Indonesia": "bahasa_indonesia",
    "English": "english"
}
selected_language = st.sidebar.selectbox(
    "Pilih Bahasa / Select Language",
    options=list(language_options.keys()),
    index=0
)
language_code = language_options[selected_language]

# --- Main Title ---
st.title("🦙 Binky - Library Assistant Chatbot")
if selected_language == "Bahasa Indonesia":
    st.subheader("Ajukan pertanyaan tentang perpustakaan atau koleksi akademik Anda")
    placeholder_text = "Tulis pertanyaan di sini..."
    button_text = "Kirim"
    processing_text = "Memproses pertanyaan..."
    setup_text = "Menyiapkan chatbot..."
    no_docs_text = "Tidak ada dokumen ditemukan. Periksa data Anda."
else:
    st.subheader("Ask questions about the library or your academic documents")
    placeholder_text = "Type your question here..."
    button_text = "Send"
    processing_text = "Processing your query..."
    setup_text = "Setting up chatbot..."
    no_docs_text = "No documents found. Please check your data."

# --- Initialize Chatbot ---
if "chatbot" not in st.session_state:
    with st.spinner(setup_text):
        success, message, vectorstore = setup_rag_pipeline()
        if not success:
            st.error(no_docs_text)
            st.stop()
        st.session_state.chatbot = LlamaRagChatbot(vectorstore)
        st.session_state.chatbot.set_language(language_code)

# --- Handle language switching ---
if "current_language" not in st.session_state:
    st.session_state.current_language = language_code
elif st.session_state.current_language != language_code:
    st.session_state.chatbot.set_language(language_code)
    st.session_state.current_language = language_code

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input and response ---
if prompt := st.chat_input(placeholder_text):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(processing_text):
            result = st.session_state.chatbot.process_query(prompt)
            response = result["response"]
        
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
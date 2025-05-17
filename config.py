"""
Configuration settings for the Llama RAG chatbot.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model settings
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # Llama model
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
DEVICE = "cuda:0" if os.environ.get("USE_CUDA", "False").lower() == "true" else "cpu"

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llama/data")
PAPERS_JSON_PATH = os.path.join(DATA_DIR, "paper.json")
LIBRARY_INFO_PATH = os.path.join(DATA_DIR, "documents.txt")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K_RESULTS = 5

# Language setting - Bahasa Indonesia
OUTPUT_LANGUAGE = "bahasa_indonesia"

# Prompt engineering settings
TEMPERATURE = 0.5  # Lower temperature for more deterministic answers
TOP_P = 0.9        # Nucleus sampling
TOP_K = 50         # Top-k sampling
REPETITION_PENALTY = 1.2  # Higher repetition penalty to avoid repetitions
MAX_NEW_TOKENS = 512  # Maximum length of generated response
MAX_INPUT_TOKENS = 2048  # Maximum length of input prompt

# Prompt templates in Bahasa Indonesia
SYSTEM_PROMPT_INDONESIA = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.
Selalu jawab dalam Bahasa Indonesia dengan jelas dan ringkas.
Jika Anda tidak tahu jawabannya atau tidak ada dalam konteks, jangan mencoba mengarang jawaban. 
Katakan dengan jujur "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut" atau "Informasi tersebut tidak terdapat dalam dokumen yang tersedia."

Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna:
{context}

Pertanyaan: {question}
Jawaban:"""

# Default prompt in English (as fallback)
SYSTEM_PROMPT_ENGLISH = """You are a helpful assistant that answers questions based on the retrieved context. 
Always answer in Bahasa Indonesia clearly and concisely.
If you don't know the answer or it's not in the context, don't try to make up an answer.
Honestly say "I don't have enough information to answer that question" or "That information is not in the available documents."

Use the following pieces of context to answer the user's question:
{context}

Question: {question}
Answer:"""

# Use Indonesian prompt by default
SYSTEM_PROMPT = SYSTEM_PROMPT_INDONESIA

# Default answers when information is not found (in Bahasa Indonesia)
DEFAULT_NO_INFO_ANSWERS = [
    "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut.",
    "Berdasarkan dokumen yang tersedia, saya tidak dapat menemukan jawaban untuk pertanyaan Anda.",
    "Informasi tersebut tidak terdapat dalam konteks yang diberikan.",
    "Saya tidak memiliki data yang diperlukan untuk menjawab pertanyaan ini dengan akurat.",
    "Mohon maaf, tetapi pertanyaan Anda berada di luar cakupan informasi yang tersedia."
]
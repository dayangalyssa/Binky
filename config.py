"""
Configuration settings for the Llama RAG chatbot.
"""
import os
from dotenv import load_dotenv
import torch
import platform
import glob

load_dotenv()

DEVICE = "cpu"
GPU_LAYERS = 0
print(f"[INFO] Model akan dijalankan di: {DEVICE}")

# Model settings
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILE = "llama-2-7b-chat.Q5_K_S.gguf"  
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Binky/data")
PAPERS_JSON_PATH = os.path.join(DATA_DIR, "paper.json")
BUKU_JSON_PATH = os.path.join(DATA_DIR, "buku.json")
LIBRARY_INFO_PATH = os.path.join(DATA_DIR, "documents.txt")
FASILITAS_PATH = os.path.join(DATA_DIR, "fasilitas.txt")
LAYANAN_PATH = os.path.join(DATA_DIR, "layanan.txt")
PROFIL_PATH = os.path.join(DATA_DIR, "profil.txt")
SOP_PATH = os.path.join(DATA_DIR, "sop.txt")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")


# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K_RESULTS = 3

# Language setting - Bahasa Indonesia
OUTPUT_LANGUAGE = "bahasa_indonesia"

# CPU Threading settings
import multiprocessing
CPU_CORES = multiprocessing.cpu_count()
THREADS = min(CPU_CORES, 8) 

TEMPERATURE = 0.2
TOP_P = 0.9       
TOP_K = 5
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 512
MAX_INPUT_TOKENS = 2048  

def find_model_path():
    base = os.path.join(os.getenv("HOME"), ".cache", "huggingface", "hub")
    pattern = os.path.join(
        base,
        "models--TheBloke--Llama-2-7B-Chat-GGUF",
        "snapshots",
        "*",
        MODEL_FILE
    )
    files = glob.glob(pattern)
    if files:
        return files[0]
    else:
        raise FileNotFoundError("Model file not found in Hugging Face cache.")

MODEL_PATH = find_model_path()

# Prompt templates in Bahasa Indonesia
SYSTEM_PROMPT_INDONESIA = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.
Selalu jawab dalam Bahasa Indonesia dengan jelas dan ringkas. Jawablah berdasarkan konteks yang diberikan dari pertanyaan atau prompt. 
Jika Anda tidak tahu jawabannya atau tidak ada dalam konteks, jangan mencoba mengarang jawaban. 
Jika pengguna mengucap "terima kasih" atau "makasih" atau sejenisnya, balaslah dengan "Sama-sama, senang bisa membantu!". 
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

# Default answers when information is not found 
DEFAULT_NO_INFO_ANSWERS = [
    "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut.",
    "Berdasarkan dokumen yang tersedia, saya tidak dapat menemukan jawaban untuk pertanyaan Anda.",
    "Informasi tersebut tidak terdapat dalam konteks yang diberikan.",
    "Saya tidak memiliki data yang diperlukan untuk menjawab pertanyaan ini dengan akurat.",
    "Mohon maaf, tetapi pertanyaan Anda berada di luar cakupan informasi yang tersedia."
]
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils.helpers import setup_rag_pipeline
from src.chatbot import LlamaRagChatbot
import config

app = FastAPI(title="Binky RAG Chatbot API")

# Enable CORS (for public frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request format
class QueryRequest(BaseModel):
    query: str
    language: str = "bahasa_indonesia"

# Response format
class ChatResponse(BaseModel):
    response: str
    num_docs_retrieved: int
    has_relevant_context: bool

# Global chatbot instance
chatbot: LlamaRagChatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    print("[INIT] Setting up RAG pipeline...")
    success, message, vectorstore = setup_rag_pipeline()
    if not success:
        raise RuntimeError(f"[ERROR] {message}")
    chatbot = LlamaRagChatbot(vectorstore)
    chatbot.set_language(config.OUTPUT_LANGUAGE)
    print("[READY] Chatbot initialized.")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: QueryRequest):
    chatbot.set_language(req.language)
    result = chatbot.process_query(req.query)
    return ChatResponse(
        response=result["response"],
        has_relevant_context=result["has_relevant_context"]
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}
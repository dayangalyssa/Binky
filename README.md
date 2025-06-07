# 🦙 Llama RAG Chatbot

This project is an AI-powered chatbot system designed specifically for library services. It is built using the LLaMA 2 large language model and enhanced with RAG (Retrieval-Augmented Generation) to provide more accurate and context-aware responses based on real library data.

To achieve this, we use a FAISS vector database to store and search through documents. When a user asks a question, the system retrieves relevant information from the database and feeds it to the model to generate informed and helpful answers.

## Quick Start

### Setup

1. **Create and Activate Conda Environment**

   Make sure you have Anaconda or Miniconda installed.

   Then, open a terminal and run:
   ```bash
   conda create -n llama-chat python=3.10 -y
   conda activate llama-chat
   ```
   
3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

### Config Model
Open config.py to set the model:
```bash
USE_GPU = False  # True if GPU
```

# Model settings 
```bash
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILE = "llama-2-7b-chat.Q5_K_S.gguf"
```
📌 Note: Make sure the model file llama-2-7b-chat.Q5_K_S.gguf is already downloaded and placed in the correct directory.

### Testing

1. **Test document loading**
   ```bash
   python -c "from src.document_loader import load_all_documents; print(load_all_documents())"
   ```

2. **Test RAG pipeline**
   ```bash
   python -c "from utils import setup_rag_pipeline; success, message, vectorstore = setup_rag_pipeline(); print(message)"
   ```

3. **Run the chatbot**
   ```bash
   # CLI interface
   python app.py
   
   # Web interface without bakcend
   streamlit run app.py
   ```
3. **Running with Docker + FastAPI Backend**

   This project is containerized using Docker for easier deployment and integrates with a FastAPI backend for API-based inference.

   Step 1: start the backend with Docker
   ```bash
   docker run -p 8000:8000 backend
   ```

      Step  2: Open the frontend
   ```bash
   streamlit run frontend/streamlit_app.py --server.port 8501 &
   ```
   Alternative (Without Docker): Run Backend with Uvicorn
   If you prefer not to use Docker, you can run the FastAPI backend manually:
   
      Step 1: start the backend with uvicorn
      ```bash
      uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
      ```
      Step  2: Open the frontend
      ```bash
      streamlit run frontend/streamlit_app.py --server.port 8501 &
      ```
## Features

- Loads and processes JSON and text documents
- Uses FAISS for store and search
- Answers in Bahasa Indonesia
- Reduces hallucination when information isn't found
- Offers both CLI and web interfaces

## Troubleshooting

- **Hugging Face Access**: Make sure you have a format in `.cache`
- **Performance**: Enable GPU by setting `USE_CUDA=True`

## Structure

```
llama-rag-chatbot/
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── config.py               # Configuration settings
├── .env                    # Environment variables (create this file)
├── data/
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading utilities
│   ├── embedding.py        # Embedding generation
│   ├── indexing.py         # Vector indexing with FAISS
│   ├── chunking.py         # Text splitting logic
│   ├── retriever.py        # Document retrieval
│   ├── llm.py              # LLM interface
│   └── chatbot.py          # Chatbot implementation
└── utils/
    ├── __init__.py
    └── helpers.py          # Helper functions
```



> 🦙 **LLaMA 2 Chatbot for Library Services**  
> A lightweight AI assistant for library services using **LLaMA 2** with **RAG (Retrieval-Augmented Generation)**.  
> Built for the **Capstone Project** course by Arache, Arion, Danna, Dayang, Ryan.

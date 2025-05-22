# Llama RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot using Llama to answer questions in Bahasa Indonesia based on your documents.

## Quick Start

### Setup

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place JSON data in `binky/data/paper.json`
   - Place text data in `binky/data/documents.txt`

### Config Model
Open config.py to set the model:
```bash
USE_GPU = False  # True if GPU
```

# Model settings 
```bash
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILE = "llama-2-7b-chat.Q4_K_M.gguf" 
```

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
   
   # Web interface
   streamlit run app.py
   ```

## Features

- Loads and processes JSON and text documents
- Uses FAISS for semantic search
- Answers in Bahasa Indonesia
- Reduces hallucination when information isn't found
- Offers both CLI and web interfaces

## Troubleshooting

- **Hugging Face Access**: Make sure you have a valid token in `.env`
- **Memory Issues**: Use a smaller model like "google/flan-t5-base" in `config.py`
- **Performance**: Enable GPU by setting `USE_CUDA=True` in `.env`

## Structure

```
llama-rag-chatbot/
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── config.py               # Configuration settings
├── .env                    # Environment variables (create this file)
├── data/
│   ├── paper.json          # Your JSON data from web scraping
│   ├── documents.txt       # Your TXT data about the library
│   └── faiss_index/        # Directory for storing the vector index
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
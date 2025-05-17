"""
Indexing module to create and manage FAISS vector indexes.
"""
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.embedding import get_embedding_model
import config


def create_faiss_index(documents: List[Document], save_path: Optional[str] = None) -> FAISS:
    """
    Create a FAISS index from documents.
    
    Args:
        documents: List of Document objects to index
        save_path: Optional path to save the index
        
    Returns:
        FAISS vector store
    """
    embedding_model = get_embedding_model()
    
    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    # Save the index if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectorstore.save_local(save_path)
        print(f"Saved FAISS index to {save_path}")
    
    return vectorstore


def load_faiss_index(load_path: str) -> Optional[FAISS]:
    """
    Load a FAISS index from disk.
    
    Args:
        load_path: Path to the saved index
        
    Returns:
        FAISS vector store or None if loading fails
    """
    try:
        embedding_model = get_embedding_model()
        vectorstore = FAISS.load_local(load_path, embedding_model)
        print(f"Loaded FAISS index from {load_path}")
        return vectorstore
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


def get_or_create_index(documents: Optional[List[Document]] = None) -> FAISS:
    """
    Get existing FAISS index or create a new one.
    
    Args:
        documents: List of Document objects to index if creating a new index
        
    Returns:
        FAISS vector store
    """
    if os.path.exists(config.FAISS_INDEX_PATH):
        # Try to load existing index
        index = load_faiss_index(config.FAISS_INDEX_PATH)
        if index:
            return index
    
    # If loading fails or index doesn't exist, create a new one
    if documents:
        print("Creating new FAISS index...")
        return create_faiss_index(documents, config.FAISS_INDEX_PATH)
    else:
        raise ValueError("Documents must be provided to create a new index")
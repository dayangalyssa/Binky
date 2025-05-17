"""
Helper utility functions.
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from src.document_loader import load_all_documents
from src.chunking import split_documents
from src.indexing import get_or_create_index


def setup_rag_pipeline() -> Tuple[bool, str, Any]:
    """
    Set up the complete RAG pipeline.
    
    Returns:
        Tuple of (success: bool, message: str, vectorstore: FAISS or None)
    """
    try:
        # Step 1: Load documents
        print("Loading documents...")
        documents = load_all_documents()
        
        if not documents:
            return False, "No documents were loaded. Please check your data files.", None
        
        # Step 2: Split documents into chunks
        print("Splitting documents into chunks...")
        chunks = split_documents(documents)
        
        # Step 3: Create or load vector index
        print("Creating/loading vector index...")
        vectorstore = get_or_create_index(chunks)
        
        return True, "RAG pipeline setup successfully", vectorstore
    
    except Exception as e:
        error_message = f"Error setting up RAG pipeline: {str(e)}"
        print(error_message)
        return False, error_message, None


def time_function(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper
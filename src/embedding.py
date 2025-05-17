"""
Embedding module to generate embeddings for documents.
"""
from typing import List

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

import config


def get_embedding_model():
    """
    Initialize and return the embedding model.
    
    Returns:
        HuggingFaceEmbeddings model
    """
    model_kwargs = {'device': config.DEVICE}
    
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_ID,
        model_kwargs=model_kwargs
    )
    
    return embeddings
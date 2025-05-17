"""
Retriever module to fetch relevant documents from the vector store.
"""
from typing import List, Dict, Any

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

import config


class DocumentRetriever:
    """
    Class to handle document retrieval from vector store.
    """
    
    def __init__(self, vectorstore: FAISS):
        """
        Initialize the retriever with a vector store.
        
        Args:
            vectorstore: FAISS vector store
        """
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The user's query string
            top_k: Number of documents to retrieve (default: from config)
            
        Returns:
            List of relevant Document objects
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Get similar documents from the vector store
        docs = self.vectorstore.similarity_search(query, k=top_k)
        
        return docs
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Extract source information
            source = doc.metadata.get("source", "Unknown")
            
            # Format the document content with source information
            formatted_doc = f"Document {i+1} (Source: {source}):\n{doc.page_content}\n"
            context_parts.append(formatted_doc)
        
        return "\n".join(context_parts)
"""
Chatbot implementation module that combines all components.
"""
from typing import Dict, List, Any, Optional

from langchain_community.vectorstores import FAISS

from src.llm import LlamaInterface
from src.retriever import DocumentRetriever
import config


class LlamaRagChatbot:
    """
    RAG-powered chatbot using Llama model.
    """
    
    def __init__(self, vectorstore: FAISS):
        """
        Initialize the chatbot with a vector store.
        
        Args:
            vectorstore: FAISS vector store for retrieval
        """
        self.llm = LlamaInterface()
        self.retriever = DocumentRetriever(vectorstore)
        self.chat_history = []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the response and related information
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query)
        has_relevant_context = len(retrieved_docs) > 0

        #Augmented: add retrieval context to the prompt
        context = self.retriever.format_context(retrieved_docs) if has_relevant_context else ""
        rag_prompt = self.llm.format_rag_prompt(query, context)
        
        # Generate the prompt
        response = self.llm.generate_response(rag_prompt, has_context=has_relevant_context)
        
        # Add to chat history
        self.chat_history.append({"query": query, "response": response})
        
        # Return the results
        return {
            "query": query,
            "response": response,
            "context_documents": retrieved_docs,
            "num_docs_retrieved": len(retrieved_docs),
            "has_relevant_context": has_relevant_context
        }
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history.
        
        Returns:
            List of query-response pairs
        """
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
        
    def translate_query(self, query: str) -> str:
        """
        Translate the query to English if needed for better retrieval.
        This is a simple stub - in a production system, you'd use a translation service.
        
        Args:
            query: The user's question, potentially in Bahasa Indonesia
            
        Returns:
            Query in English for retrieval purposes
        """
        return query
    
    def set_language(self, language: str):
        """
        Set the output language for the chatbot.
        
        Args:
            language: The language code (e.g., 'bahasa_indonesia', 'english')
        """
        if language == "bahasa_indonesia":
            config.SYSTEM_PROMPT = config.SYSTEM_PROMPT_INDONESIA
            config.OUTPUT_LANGUAGE = "bahasa_indonesia"
        elif language == "english":
            config.SYSTEM_PROMPT = config.SYSTEM_PROMPT_ENGLISH
            config.OUTPUT_LANGUAGE = "english"
        else:
            print(f"Unsupported language: {language}, defaulting to Bahasa Indonesia")
            config.SYSTEM_PROMPT = config.SYSTEM_PROMPT_INDONESIA
            config.OUTPUT_LANGUAGE = "bahasa_indonesia"
"""
Package initialization for the src module.
"""

# Version information
__version__ = '0.1.0'

# Import key components for easier access
from .chatbot import LlamaRagChatbot
from .document_loader import load_all_documents
from .indexing import get_or_create_index

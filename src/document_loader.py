"""
Document loader module to process different file types.
"""
import json
import os
from typing import Dict, List, Any, Union

import pandas as pd
from langchain.schema import Document
from langchain.document_loaders import TextLoader

import config


def load_json_papers(file_path: str) -> List[Document]:
    """
    Load JSON file containing papers data and convert to LangChain documents.
    Custom implementation for the specific JSON structure used in this project.
    
    Expected JSON structure:
    {
        "Judul": "Paper Title",
        "Penulis": ["Author 1", "Author 2"],
        "URL": "https://example.com/paper",
        "Abstrak": "Paper abstract text...",
        "Publisher": "Publisher name",
        "Tahun": "2023"
    }
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of LangChain Document objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
        
        documents = []
        
        # Handle different possible structures of the JSON data
        if isinstance(papers_data, list):
            # If it's a list of papers
            for i, paper in enumerate(papers_data):
                if isinstance(paper, dict):
                    # Extract content and metadata
                    content = ""
                    metadata = {"source": file_path, "index": i}
                    
                    # Add title if available
                    if "Judul" in paper:
                        content += f"Judul: {paper['Judul']}\n\n"
                        metadata["title"] = paper["Judul"]
                    
                    # Add authors if available
                    if "Penulis" in paper:
                        if isinstance(paper["Penulis"], list):
                            authors = ", ".join(paper["Penulis"])
                        else:
                            authors = str(paper["Penulis"])
                        content += f"Penulis: {authors}\n\n"
                        metadata["authors"] = authors
                    
                    # Add URL if available
                    if "URL" in paper:
                        content += f"URL: {paper['URL']}\n\n"
                        metadata["url"] = paper["URL"]
                    
                    # Add abstract if available
                    if "Abstrak" in paper:
                        content += f"Abstrak: {paper['Abstrak']}\n\n"
                        metadata["has_abstract"] = True
                    
                    # Add publisher if available
                    if "Publisher" in paper:
                        content += f"Publisher: {paper['Publisher']}\n\n"
                        metadata["publisher"] = paper["Publisher"]
                    
                    # Add year if available
                    if "Tahun" in paper:
                        content += f"Tahun: {paper['Tahun']}\n\n"
                        metadata["year"] = paper["Tahun"]
                    
                    # Add other fields that might be relevant
                    for key, value in paper.items():
                        if key not in ["Judul", "Penulis", "URL", "Abstrak", "Publisher", "Tahun"]:
                            if isinstance(value, (str, int, float, bool)):
                                metadata[key.lower()] = value
                                content += f"{key}: {value}\n\n"
                    
                    # Create document if content is not empty
                    if content.strip():
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
        
        elif isinstance(papers_data, dict):
            # Handle case where the JSON is a single object or a dictionary with papers
            
            # Check if it's a single paper object
            if "Judul" in papers_data:
                # This is a single paper
                paper = papers_data
                content = ""
                metadata = {"source": file_path}
                
                # Add title if available
                if "Judul" in paper:
                    content += f"Judul: {paper['Judul']}\n\n"
                    metadata["title"] = paper["Judul"]
                
                # Add authors if available
                if "Penulis" in paper:
                    if isinstance(paper["Penulis"], list):
                        authors = ", ".join(paper["Penulis"])
                    else:
                        authors = str(paper["Penulis"])
                    content += f"Penulis: {authors}\n\n"
                    metadata["authors"] = authors
                
                # Add URL if available
                if "URL" in paper:
                    content += f"URL: {paper['URL']}\n\n"
                    metadata["url"] = paper["URL"]
                
                # Add abstract if available
                if "Abstrak" in paper:
                    content += f"Abstrak: {paper['Abstrak']}\n\n"
                    metadata["has_abstract"] = True
                
                # Add publisher if available
                if "Publisher" in paper:
                    content += f"Publisher: {paper['Publisher']}\n\n"
                    metadata["publisher"] = paper["Publisher"]
                
                # Add year if available
                if "Tahun" in paper:
                    content += f"Tahun: {paper['Tahun']}\n\n"
                    metadata["year"] = paper["Tahun"]
                
                # Add other fields that might be relevant
                for key, value in paper.items():
                    if key not in ["Judul", "Penulis", "URL", "Abstrak", "Publisher", "Tahun"]:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key.lower()] = value
                            content += f"{key}: {value}\n\n"
                
                # Create document if content is not empty
                if content.strip():
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            else:
                # It's a dictionary with papers as values (with keys as IDs)
                for key, paper in papers_data.items():
                    if isinstance(paper, dict):
                        # Extract content and metadata
                        content = ""
                        metadata = {"source": file_path, "key": key}
                        
                        # Add title if available
                        if "Judul" in paper:
                            content += f"Judul: {paper['Judul']}\n\n"
                            metadata["title"] = paper["Judul"]
                        
                        # Add authors if available
                        if "Penulis" in paper:
                            if isinstance(paper["Penulis"], list):
                                authors = ", ".join(paper["Penulis"])
                            else:
                                authors = str(paper["Penulis"])
                            content += f"Penulis: {authors}\n\n"
                            metadata["authors"] = authors
                        
                        # Add URL if available
                        if "URL" in paper:
                            content += f"URL: {paper['URL']}\n\n"
                            metadata["url"] = paper["URL"]
                        
                        # Add abstract if available
                        if "Abstrak" in paper:
                            content += f"Abstrak: {paper['Abstrak']}\n\n"
                            metadata["has_abstract"] = True
                        
                        # Add publisher if available
                        if "Publisher" in paper:
                            content += f"Publisher: {paper['Publisher']}\n\n"
                            metadata["publisher"] = paper["Publisher"]
                        
                        # Add year if available
                        if "Tahun" in paper:
                            content += f"Tahun: {paper['Tahun']}\n\n"
                            metadata["year"] = paper["Tahun"]
                        
                        # Add other fields that might be relevant
                        for field_key, value in paper.items():
                            if field_key not in ["Judul", "Penulis", "URL", "Abstrak", "Publisher", "Tahun"]:
                                if isinstance(value, (str, int, float, bool)):
                                    metadata[field_key.lower()] = value
                                    content += f"{field_key}: {value}\n\n"
                        
                        # Create document if content is not empty
                        if content.strip():
                            doc = Document(page_content=content, metadata=metadata)
                            documents.append(doc)
        
        print(f"Loaded {len(documents)} documents from JSON file")
        return documents
    
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []


def load_text_file(file_path: str) -> List[Document]:
    """
    Load a plain text file and convert to LangChain documents.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of LangChain Document objects
    """
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # Add additional metadata
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = "text"
        
        print(f"Loaded {len(documents)} documents from text file")
        return documents
    
    except Exception as e:
        print(f"Error loading text file: {e}")
        return []


def load_all_documents() -> List[Document]:
    """
    Load all documents from the configured data sources.
    
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    # Load JSON papers
    if os.path.exists(config.PAPERS_JSON_PATH):
        json_docs = load_json_papers(config.PAPERS_JSON_PATH)
        documents.extend(json_docs)
    else:
        print(f"Warning: JSON file not found at {config.PAPERS_JSON_PATH}")
    
    # Load library info text file
    if os.path.exists(config.LIBRARY_INFO_PATH):
        text_docs = load_text_file(config.LIBRARY_INFO_PATH)
        documents.extend(text_docs)
    else:
        print(f"Warning: Text file not found at {config.LIBRARY_INFO_PATH}")
    
    return documents
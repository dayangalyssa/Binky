"""
LLM interface module to interact with the Llama model.
"""
import os
import random
from typing import Dict, List, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)

import config


class LlamaInterface:
    """
    Interface for the Llama language model.
    """
    
    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the Llama model.
        
        Args:
            model_id: Hugging Face model ID (default: from config)
        """
        self.model_id = model_id or config.MODEL_ID
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model, tokenizer, and pipeline."""
        try:
            print(f"Loading model {self.model_id}...")
            
            # Set up quantization configuration for more efficient memory usage
            bnb_config = None
            if config.DEVICE == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            
            # Create the text generation pipeline with enhanced parameters
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=config.MAX_INPUT_TOKENS + config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repetition_penalty=config.REPETITION_PENALTY,
                do_sample=True,
            )
            
            print("Model loaded successfully")
        
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def generate_response(self, prompt: str, has_context: bool = True) -> str:
        """
        Generate a response using the Llama model.
        
        Args:
            prompt: The full prompt including system prompt and user query
            has_context: Whether relevant context was found
            
        Returns:
            Generated text response
        """
        if not self.pipeline:
            raise ValueError("Model not initialized")
        
        try:
            # If no context was found, there's a chance to return a default no-info answer
            if not has_context and random.random() < 0.4:  # 80% chance to use default answer
                return random.choice(config.DEFAULT_NO_INFO_ANSWERS)
            
            # Generate text with the pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=config.MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the original prompt from the output
            response = generated_text[len(prompt):].strip()
            
            if config.OUTPUT_LANGUAGE == "bahasa_indonesia" and self._detect_english(response):
                retry_prompt = prompt + "\n\nPenting: Jawablah dalam Bahasa Indonesia."
                outputs = self.pipeline(
                    retry_prompt,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = outputs[0]['generated_text']
                response = generated_text[len(retry_prompt):].strip()
            
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Maaf, saya mengalami kesalahan saat mencoba menjawab. Silakan coba lagi."
    
    def _detect_english(self, text: str) -> bool:
        """
        Simple heuristic to detect if text is mainly in English instead of Bahasa Indonesia.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be in English
        """
        # Common Indonesian words
        indonesian_words = ["ini", "itu", "dan", "atau", "saya", "anda", "adalah", "tidak", 
                           "untuk", "dalam", "dengan", "yang", "dari", "akan", "pada", "ke",
                           "tersebut", "bisa", "dapat", "karena", "oleh", "jika", "ada", "sudah"]
        
        # Common English words
        english_words = ["the", "is", "and", "of", "to", "in", "that", "for", "it", "with", 
                        "as", "on", "at", "by", "from", "this", "an", "are", "was", "were",
                        "have", "has", "been", "would", "could", "should"]
        
        words = text.lower().split()
        indonesian_count = sum(1 for word in words if word in indonesian_words)
        english_count = sum(1 for word in words if word in english_words)
        
        # If there are significantly more English indicator words, consider it English
        if english_count > indonesian_count * 2:
            return True
        return False
    
    def format_rag_prompt(self, query: str, context: str) -> str:
        """
        Format a prompt for RAG using the retrieved context.
        
        Args:
            query: The user's question
            context: The retrieved context for RAG
            
        Returns:
            Formatted prompt string
        """
        # Format using the system prompt template from config
        prompt = config.SYSTEM_PROMPT.format(
            context=context,
            question=query
        )
        
        return prompt
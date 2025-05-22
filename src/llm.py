"""
LLM interface module to interact with the Llama model using GGUF format.
"""
import os
import random
import time
from typing import Dict, List, Any, Optional
import torch

from ctransformers import AutoModelForCausalLM

import config


class LlamaInterface:
    """
    Interface for the Llama GGUF model using ctransformers.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Llama model.
        
        Args:
            model_path: Path to GGUF model file (default: from config)
        """
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model using ctransformers."""
        try:
            start_time = time.time()
            
            print(f"Loading GGUF model from {self.model_path}...")
            print(f"Device: {config.DEVICE}, GPU Layers: {config.GPU_LAYERS}")
            print(f"Threads: {config.THREADS}")
            
            # Cek apakah CUDA tersedia dan apakah menggunakan GPU
            if config.DEVICE == "cuda:0" and torch.cuda.is_available():
                cuda_device = 0
                print(f"CUDA available, using device: {cuda_device}")
                print(f"GPU: {torch.cuda.get_device_name(cuda_device)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.2f} GB")
            else:
                print("Using CPU only")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found at {self.model_path}")
                print("Downloading model. This may take a while...")
                
                # For models that might not be downloaded yet
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_ID,
                    model_file=config.MODEL_FILE,
                    model_type="llama",
                    gpu_layers=config.GPU_LAYERS,  # 0 untuk CPU, >0 untuk GPU
                    context_length=config.MAX_INPUT_TOKENS,
                    threads=config.THREADS,        # CPU threads
                )
            else:
                # Load from local path
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    model_type="llama",
                    gpu_layers=config.GPU_LAYERS,  # 0 untuk CPU, >0 untuk GPU
                    context_length=config.MAX_INPUT_TOKENS,
                    threads=config.THREADS,        # CPU threads
                )
            
            end_time = time.time()
            load_time = end_time - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Print memory info
            if config.DEVICE == "cuda:0" and torch.cuda.is_available() and config.GPU_LAYERS > 0:
                cuda_device = 0
                allocated = torch.cuda.memory_allocated(cuda_device) / 1e9
                reserved = torch.cuda.memory_reserved(cuda_device) / 1e9
                print(f"GPU Memory Allocated: {allocated:.2f} GB")
                print(f"GPU Memory Reserved: {reserved:.2f} GB")
            else:
                print("Running on CPU - no GPU memory usage")
        
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
        if not self.model:
            raise ValueError("Model not initialized")
        
        try:
            # If no context was found, there's a chance to return a default no-info answer
            if not has_context and random.random() < 0.4:  # 40% chance to use default
                return random.choice(config.DEFAULT_NO_INFO_ANSWERS)
            
            start_time = time.time()
            
            # Generate text with the model
            generated_text = self.model(
                prompt,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repetition_penalty=config.REPETITION_PENALTY,
                stop=["</s>", "[/INST]", "\n\n"]  # Stop tokens untuk Llama 2
            )
            
            end_time = time.time()
            gen_time = end_time - start_time
            
            # Hitung jumlah token yang dihasilkan untuk menghitung kecepatan
            response_tokens = len(generated_text.split()) - len(prompt.split())
            tokens_per_second = response_tokens / gen_time if gen_time > 0 else 0
            
            print(f"Generation completed in {gen_time:.2f} seconds")
            print(f"Speed: {tokens_per_second:.1f} tokens/second")
            
            # Trim the prompt from the response
            response = generated_text[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            # If the generated response is in English, retry with explicit Indonesian instruction
            if config.OUTPUT_LANGUAGE == "bahasa_indonesia" and self._detect_english(response):
                print("Detected English response, retrying in Bahasa Indonesia...")
                retry_prompt = prompt + "\n\nPenting: Jawablah dalam Bahasa Indonesia."
                
                generated_text = self.model(
                    retry_prompt,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    top_k=config.TOP_K,
                    repetition_penalty=config.REPETITION_PENALTY,
                    stop=["</s>", "[/INST]", "\n\n"]
                )
                
                response = generated_text[len(retry_prompt):].strip()
                response = self._clean_response(response)
            
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Maaf, saya mengalami kesalahan saat mencoba menjawab. Silakan coba lagi."
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up the generated response.
        
        Args:
            response: Raw response from model
            
        Returns:
            Cleaned response
        """
        # Remove common unwanted tokens/patterns
        unwanted_patterns = [
            "</s>", "[/INST]", "<|im_end|>", "<|im_start|>",
            "User:", "Assistant:", "Jawaban:", "Answer:"
        ]
        
        for pattern in unwanted_patterns:
            response = response.replace(pattern, "")
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        return response.strip()
    
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
                           "tersebut", "bisa", "dapat", "karena", "oleh", "jika", "ada", "sudah",
                           "berdasarkan", "informasi", "dokumen", "maaf", "tidak ada"]
        
        # Common English words
        english_words = ["the", "is", "and", "of", "to", "in", "that", "for", "it", "with", 
                        "as", "on", "at", "by", "from", "this", "an", "are", "was", "were",
                        "have", "has", "been", "would", "could", "should", "based", "information"]
        
        words = text.lower().split()
        if len(words) < 3:  # Too short to determine
            return False
            
        indonesian_count = sum(1 for word in words if word in indonesian_words)
        english_count = sum(1 for word in words if word in english_words)
        
        # If there are significantly more English indicator words, consider it English
        if english_count > indonesian_count * 1.5:
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
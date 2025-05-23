"""
Main application entry point for the Llama RAG chatbot.
"""
import os
import sys
import argparse
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import login
import os
from src.chatbot import LlamaRagChatbot
from utils.helpers import setup_rag_pipeline, time_function
import config

def setup_cli_app():
    """Set up command-line interface application."""
    parser = argparse.ArgumentParser(description="Llama RAG Chatbot")
    parser.add_argument("--setup-only", action="store_true", help="Only set up the RAG pipeline without starting interactive mode")
    parser.add_argument("--language", choices=["bahasa_indonesia", "english"], default="bahasa_indonesia", 
                       help="Output language (default: bahasa_indonesia)")
    args = parser.parse_args()
    
    # Set up the RAG pipeline
    success, message, vectorstore = setup_rag_pipeline()
    print(message)
    
    if not success:
        sys.exit(1)
    
    if args.setup_only:
        print("RAG pipeline setup complete. Exiting.")
        sys.exit(0)
    
    # Initialize the chatbot
    chatbot = LlamaRagChatbot(vectorstore)
    
    # Set language
    chatbot.set_language(args.language)
    print(f"Chatbot language set to: {args.language}")
    
    # Run CLI interactive mode
    if args.language == "bahasa_indonesia":
        welcome_msg = "\nLlama RAG Chatbot - Mode Interaktif"
        prompt_msg = "Anda: "
        exit_msg = "Selamat tinggal!"
        processing_msg = "\nMemproses pertanyaan Anda..."
        bot_prefix = "\nChatbot: "
        retrieved_msg = "\n[Ditemukan {n} dokumen yang relevan]"
    else:
        welcome_msg = "\nLlama RAG Chatbot - Interactive Mode"
        prompt_msg = "You: "
        exit_msg = "Goodbye!"
        processing_msg = "\nProcessing your query..."
        bot_prefix = "\nChatbot: "
        retrieved_msg = "\n[Retrieved {n} relevant documents]"
    
    print(welcome_msg)
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        user_input = input(f"\n{prompt_msg}").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print(exit_msg)
            break
        
        if not user_input:
            continue
        
        # Process the query
        print(processing_msg)
        result = time_function(chatbot.process_query)(user_input)
        
        # Print the response
        print(f"{bot_prefix}{result['response']}")
        print(retrieved_msg.format(n=result['num_docs_retrieved']))



def run_cli_chatbot(language: str):
    """Run the chatbot in CLI interactive mode."""
    # Setup pipeline
    success, message, vectorstore = setup_rag_pipeline()
    print(f"\n[INFO] {message}")
    if not success:
        sys.exit(1)

    # Initialize chatbot
    chatbot = LlamaRagChatbot(vectorstore)
    chatbot.set_language(language)
    print(f"[INFO] Bahasa output disetel ke: {language}\n")

    # CLI interface
    print("🦙  Llama RAG Chatbot - CLI Mode")
    print("Ketik 'exit' atau 'quit' untuk keluar\n")

    while True:
        try:
            user_input = input("Anda: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Terima kasih telah menggunakan Binky. Sampai jumpa!")
                break
            if not user_input:
                continue
            print("\n⏳ Memproses pertanyaan Anda...")
            result = time_function(chatbot.process_query)(user_input)
            print("\n🦙 Binky:", result["response"])
            print(f"📄 ({result['num_docs_retrieved']} dokumen relevan ditemukan)")
            print("\n---")
        except KeyboardInterrupt:
            print("\n👋 Keluar dari sesi. Sampai jumpa!")
            break

def main():
    parser = argparse.ArgumentParser(description="Llama RAG Chatbot CLI")
    parser.add_argument(
        "--language",
        choices=["bahasa_indonesia", "english"],
        default="bahasa_indonesia",
        help="Pilih bahasa output (default: bahasa_indonesia)"
    )
    args = parser.parse_args()
    run_cli_chatbot(args.language)

if _name_ == "_main_":
    main()